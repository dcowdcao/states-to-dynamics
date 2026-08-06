"""`load_sae(checkpoint_dir)` — every SAE checkpoint this study evaluates, on raw activations.

Three provenances reach the same evaluation loop: Karvonen's released OthelloGPT sweeps
(standard / gated / p_anneal / gated_anneal / top_k, both expansion factors), this study's
SpaDE and CosineSpaDE runs, and this study's baseline re-trains. The returned `ae` always
consumes RAW layer-5 activations, so the two local families — trained on standardized
activations — come back wrapped in StandardizedSAE and his never do.

That gate reads `config["buffer"]["standardization_scale"]`, which only this study's
ActivationBuffer writes, and never filesystem adjacency: one of his checkpoints put through
the wrapper scores a plausible but wrong baseline (variance explained 0.9944 -> 0.7167, L0
409 -> 206). A checkpoint whose config says raw but whose directory carries a
standardization_params.json is rejected rather than guessed at.

Shapes come from the weights, not the config: his 80-checkpoint standard sweep records
neither activation_dim nor dict_size nor dict_class. Loading is a strict `load_state_dict`,
never `from_pretrained`, whose `normalize_decoder=True` default rescales encoder weights.

The SpaDE families are instantiated as the `trace_safe_spade` subclasses, whose encode()
survives the nnsight trace his classifier stage runs it in; nothing else differs.

Usage:
    ae, meta = load_sae(checkpoint_dir, device="cuda")
"""

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from spade.common.paths import add_karvonen_to_path
from spade.sae_loader.standardized_sae import StandardizedSAE

add_karvonen_to_path()
from circuits.dictionary_learning.dictionary import AutoEncoder, Dictionary, GatedAutoEncoder
from circuits.dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from circuits.dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from circuits.dictionary_learning.trainers.top_k import AutoEncoderTopK

from spade.sae_loader.trace_safe_spade import TraceSafeCosineSpaDE, TraceSafeSpaDE

# Decoder columns drift by ~1e-6 from unit norm under ConstrainedAdam.
NORM_ATOL = 1e-4
# AutoEncoderTopK's own untrained default, and what the pre-2024-07-10 checkpoints imply.
THRESHOLD_SENTINEL = -1.0
STANDARDIZED_CONFIG_KEY = "standardization_scale"
PARAMS_FILE = "standardization_params.json"


class CheckpointError(Exception):
    """A checkpoint the loader refuses to guess at."""


@dataclass(frozen=True)
class SAEMeta:
    trainer_class: str
    dict_class: str
    activation_dim: int
    dict_size: int
    standardized: bool
    checkpoint_dir: Path
    config: dict


@dataclass(frozen=True)
class _Family:
    cls: type
    shape_key: str                                  # state-dict key carrying both dims
    dict_size_first: bool = True                    # ...as (dict_size, activation_dim)
    unit_norm: tuple[str, int] | None = None        # decoder key + dim to reduce over
    extra: Callable[[dict, dict], tuple] = field(default=lambda config, state: ())
    topk: bool = False


# SpaDE-family decoders are unconstrained (observed column norms 10-154), so no unit_norm.
_FAMILIES = {
    "AutoEncoder": _Family(AutoEncoder, "encoder.weight", unit_norm=("decoder.weight", 0)),
    "GatedAutoEncoder": _Family(GatedAutoEncoder, "encoder.weight", unit_norm=("decoder.weight", 0)),
    "AutoEncoderTopK": _Family(AutoEncoderTopK, "encoder.weight", unit_norm=("decoder.weight", 0),
                               extra=lambda config, state: (config["k"],), topk=True),
    "BatchTopKSAE": _Family(BatchTopKSAE, "encoder.weight", unit_norm=("decoder.weight", 0),
                            extra=lambda config, state: (config["k"],), topk=True),
    "MatryoshkaBatchTopKSAE": _Family(MatryoshkaBatchTopKSAE, "W_enc", dict_size_first=False,
                                      unit_norm=("W_dec", 1), topk=True,
                                      extra=lambda config, state: (config["k"],
                                                                   state["group_sizes"].tolist())),
    "SpaDEAutoEncoder": _Family(TraceSafeSpaDE, "Ae"),
    "CosineSpaDEAutoEncoder": _Family(TraceSafeCosineSpaDE, "Ae"),
}

# Only for the configs with no dict_class (his 2024-05-23 standard sweep). Most specific
# first: "TopK" is a substring of "BatchTopK" is a substring of "MatryoshkaBatchTopK", and
# "SpaDE" is a substring of "CosineSpaDE".
_TRAINER_CLASS_FALLBACK = (
    ("Matryoshka", "MatryoshkaBatchTopKSAE"),
    ("BatchTopK", "BatchTopKSAE"),
    ("TopK", "AutoEncoderTopK"),
    ("CosineSpaDE", "CosineSpaDEAutoEncoder"),
    ("SpaDE", "SpaDEAutoEncoder"),
    ("Gated", "GatedAutoEncoder"),
    ("Standard", "AutoEncoder"),
    ("PAnneal", "AutoEncoder"),
)


def dispatch(trainer_config: dict) -> tuple[str, _Family]:
    """(dict_class, family) for a trainer config — equality on dict_class, as Karvonen does."""
    dict_class = trainer_config.get("dict_class")
    if dict_class is None:
        trainer_class = trainer_config["trainer_class"]
        for pattern, name in _TRAINER_CLASS_FALLBACK:
            if pattern in trainer_class:
                dict_class = name
                break
        else:
            raise CheckpointError(
                f"trainer_class {trainer_class!r} has no dict_class and matches no known "
                f"family; refusing to guess"
            )
    if dict_class == "Identity":
        raise CheckpointError(
            "Identity checkpoints are Karvonen's passthrough baseline over MLP activations "
            "(d_model-wide, 0-byte ae.pt), not a layer-5 residual SAE; out of scope"
        )
    if dict_class not in _FAMILIES:
        raise CheckpointError(f"dict_class {dict_class!r} unsupported; known: {sorted(_FAMILIES)}")
    return dict_class, _FAMILIES[dict_class]


def _shapes(state: dict, family: _Family, trainer_config: dict, path: Path) -> tuple[int, int]:
    """(activation_dim, dict_size) from the weights; the config is only a cross-check."""
    if family.shape_key not in state:
        raise CheckpointError(
            f"{path}: {family.cls.__name__} needs {family.shape_key!r} to size itself, "
            f"state dict has {sorted(state)}"
        )
    rows, cols = state[family.shape_key].shape
    dict_size, activation_dim = (rows, cols) if family.dict_size_first else (cols, rows)
    for key, from_weights in (("activation_dim", activation_dim), ("dict_size", dict_size)):
        if key in trainer_config and trainer_config[key] != from_weights:
            raise CheckpointError(
                f"{path}: config {key}={trainer_config[key]} but weights say {from_weights}"
            )
    return activation_dim, dict_size


def _prepare_topk_buffers(state: dict, trainer_config: dict, family: _Family, path: Path) -> None:
    """Backfill the k/threshold buffers his 2024-07-10 top_k checkpoints predate.

    The sentinel threshold is only safe for a class whose encode() ignores it by default:
    under `use_threshold=True`, `acts > -1.0` keeps every feature and the measured L0 jumps
    from k to the whole dictionary (10 -> 457 on his top_k trainer_0). Everything else in
    the load stays strict.
    """
    thresholds_by_default = inspect.signature(family.cls.encode).parameters["use_threshold"].default

    if "k" not in state:
        state["k"] = torch.tensor(trainer_config["k"], dtype=torch.int)
    elif state["k"].item() != trainer_config["k"]:
        raise CheckpointError(
            f"{path}: config k={trainer_config['k']} but weights carry k={state['k'].item()}"
        )

    if "threshold" not in state:
        if thresholds_by_default:
            raise CheckpointError(
                f"{path}: {family.cls.__name__}.encode() thresholds by default but the "
                f"checkpoint has no trained threshold; it would return every feature"
            )
        state["threshold"] = torch.tensor(THRESHOLD_SENTINEL, dtype=torch.float32)
    elif thresholds_by_default and state["threshold"].item() < 0:
        raise CheckpointError(
            f"{path}: {family.cls.__name__}.encode() thresholds by default but the stored "
            f"threshold is the untrained sentinel {state['threshold'].item()}"
        )


def _check_decoder_norms(ae: Dictionary, family: _Family, path: Path) -> None:
    key, dim = family.unit_norm
    norms = ae.state_dict()[key].norm(dim=dim)
    if not torch.allclose(norms, torch.ones_like(norms), atol=NORM_ATOL):
        raise CheckpointError(
            f"{path}: {key} atoms are not unit-norm (min {norms.min():.4f}, max "
            f"{norms.max():.4f}). F1 thresholds are fractions of each feature's max "
            f"activation, so a rescaled dictionary is not comparable"
        )


def _params_path(checkpoint_dir: Path) -> Path | None:
    """trainSAE writes the params once at the save dir, above the per-trainer dirs."""
    for d in (checkpoint_dir, checkpoint_dir.parent):
        if (d / PARAMS_FILE).is_file():
            return d / PARAMS_FILE
    return None


def _standardization(checkpoint_dir: Path, config: dict,
                     activation_dim: int) -> tuple[torch.Tensor, float] | None:
    """(mean, scale) if this checkpoint was trained on standardized activations, else None.

    Intrinsic gate: only this study's ActivationBuffer writes standardization_scale into the
    saved buffer config. A raw-activation checkpoint that happens to sit next to a params
    file is an error, so the two provenances can never be mixed up by directory layout.
    """
    scale = config["buffer"].get(STANDARDIZED_CONFIG_KEY)
    params_path = _params_path(checkpoint_dir)

    if scale is None:
        if params_path is not None:
            raise CheckpointError(
                f"{checkpoint_dir}: config records raw activations but {params_path} is "
                f"adjacent. Standardizing a raw-activation SAE halves its L0 and wrecks its "
                f"variance explained; move the params file or fix the config"
            )
        return None

    if params_path is None:
        raise CheckpointError(
            f"{checkpoint_dir}: config records {STANDARDIZED_CONFIG_KEY}={scale} but no "
            f"{PARAMS_FILE} in it or its parent, so the mean vector is unrecoverable"
        )
    params = json.loads(params_path.read_text())
    if not math.isclose(params["scale"], scale, rel_tol=1e-9):
        raise CheckpointError(
            f"{checkpoint_dir}: {params_path} scale {params['scale']} != config {scale}"
        )
    mean = torch.tensor(params["mean"], dtype=torch.float32)
    if mean.shape != (activation_dim,):
        raise CheckpointError(
            f"{checkpoint_dir}: {params_path} mean is {tuple(mean.shape)}, expected "
            f"({activation_dim},)"
        )
    mean_norm = config["buffer"]["standardization_mean_norm"]
    if not math.isclose(mean.norm().item(), mean_norm, rel_tol=1e-4):
        raise CheckpointError(
            f"{checkpoint_dir}: {params_path} mean has norm {mean.norm().item()} but this "
            f"run recorded {mean_norm}; the params file belongs to a different run"
        )
    return mean, scale


def load_sae(checkpoint_dir, device: str = "cpu") -> tuple[Dictionary, SAEMeta]:
    """Load the SAE at `checkpoint_dir` onto `device`, in eval mode, on raw activations."""
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())
    trainer_config = config["trainer"]

    dict_class, family = dispatch(trainer_config)
    state = torch.load(checkpoint_dir / "ae.pt", map_location="cpu", weights_only=True)
    activation_dim, dict_size = _shapes(state, family, trainer_config, checkpoint_dir)
    if family.topk:
        _prepare_topk_buffers(state, trainer_config, family, checkpoint_dir)

    ae = family.cls(activation_dim, dict_size, *family.extra(trainer_config, state))
    ae.load_state_dict(state)  # not from_pretrained: normalize_decoder=True rescales encoders
    if family.unit_norm is not None:
        _check_decoder_norms(ae, family, checkpoint_dir)

    standardization = _standardization(checkpoint_dir, config, activation_dim)
    if standardization is not None:
        ae = StandardizedSAE(ae, *standardization)

    ae.to(device).eval()
    return ae, SAEMeta(
        trainer_class=trainer_config["trainer_class"],
        dict_class=dict_class,
        activation_dim=activation_dim,
        dict_size=dict_size,
        standardized=standardization is not None,
        checkpoint_dir=checkpoint_dir,
        config=config,
    )
