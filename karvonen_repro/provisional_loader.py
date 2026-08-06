"""PROVISIONAL. Stands in for the real `load_sae` until that lands.

Handles only Karvonen's released Othello checkpoints (standard / gated / p_anneal /
gated_anneal / top_k) by calling his own `Dictionary.from_pretrained` classmethods —
for his SAEs those are ground truth, and anything else here would be a divergence.

It does not call `dictionary_learning.utils.load_dictionary`, which cannot read two
of the five released types: the `standard` configs have no `dict_class` key, and the
top_k `ae.pt` predates the `k` / `threshold` buffers so a strict load fails.

Swapping in the real loader is one line in `evaluate.py`. Delete this file then.
"""

import contextlib
import functools
import json
from pathlib import Path

import torch

from circuits.dictionary_learning.dictionary import (
    AutoEncoder,
    AutoEncoderNew,
    GatedAutoEncoder,
)
from circuits.dictionary_learning.trainers.top_k import AutoEncoderTopK

# The released `standard` configs omit dict_class; the trainer that wrote them fixes it.
DICT_CLASS_BY_TRAINER = {
    "StandardTrainer": "AutoEncoder",
    "GatedSAETrainer": "GatedAutoEncoder",
    "PAnnealTrainer": "AutoEncoder",
    "GatedAnnealTrainer": "GatedAutoEncoder",
    "TrainerTopK": "AutoEncoderTopK",
}

FROM_PRETRAINED = {
    "AutoEncoder": AutoEncoder.from_pretrained,
    "AutoEncoderNew": AutoEncoderNew.from_pretrained,
    "GatedAutoEncoder": GatedAutoEncoder.from_pretrained,
}


@contextlib.contextmanager
def _map_location(device: str):
    """His `from_pretrained` calls `torch.load(path)` with no map_location and the
    released ae.pt were all saved from cuda:0, so a CPU run cannot read them at all."""
    original = torch.load
    torch.load = functools.partial(original, map_location=device)
    try:
        yield
    finally:
        torch.load = original


def _load_top_k(ae_path: str, k: int, device: str) -> AutoEncoderTopK:
    """`from_pretrained` loads strictly, and the released top_k ae.pt has neither the
    `k` nor the `threshold` buffer. Both come from the constructor instead: k from the
    config, threshold unused (encode() only reads it under use_threshold=True)."""
    state_dict = torch.load(ae_path, map_location=device)
    dict_size, activation_dim = state_dict["encoder.weight"].shape
    ae = AutoEncoderTopK(activation_dim, dict_size, k)
    missing, unexpected = ae.load_state_dict(state_dict, strict=False)
    assert not unexpected and set(missing) <= {"k", "threshold"}, (missing, unexpected)
    return ae.to(device)


def load_sae(checkpoint_dir: str) -> tuple[torch.nn.Module, dict]:
    """Returns an `ae` that consumes RAW model activations, plus its metadata.

    Karvonen's SAEs are trained on raw residual-stream activations, so there is no
    transform to fold in here. A loader for standardized-input SAEs must fold the
    standardization into the returned module — the harness never applies one.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(checkpoint_dir)

    with open(path / "config.json") as f:
        trainer = json.load(f)["trainer"]

    trainer_class = trainer["trainer_class"]
    dict_class = (
        trainer["dict_class"] if "dict_class" in trainer else DICT_CLASS_BY_TRAINER[trainer_class]
    )
    ae_path = str(path / "ae.pt")

    if dict_class == "AutoEncoderTopK":
        ae = _load_top_k(ae_path, trainer["k"], device)
    else:
        with _map_location(device):
            ae = FROM_PRETRAINED[dict_class](ae_path, device=device)

    ae = ae.eval()
    meta = {
        "trainer_class": trainer_class,
        "dict_size": ae.dict_size,
        "activation_dim": ae.activation_dim,
    }
    return ae, meta
