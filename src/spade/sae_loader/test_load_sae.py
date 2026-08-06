"""Tests for load_sae: one per checkpoint format on disk, plus the failures it must be loud about.

The synthetic tests need no weights and always run. The rest discover checkpoints under
SPADE_SAE_ROOTS (colon-separated, defaults to the in-repo checkpoints/ tree) and load one
representative per distinct format, skipping when a root holds none.

Usage:
    python -m spade.sae_loader.test_load_sae            # one checkpoint per format
    python -m spade.sae_loader.test_load_sae --all      # plus every checkpoint under the roots
    SPADE_SAE_ROOTS=/a/karvonen_released:/b/study pytest src/spade/sae_loader/
"""

import json
import os
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import torch

from spade.common.paths import add_karvonen_to_path, repo_root
from spade.sae_loader.load_sae import (
    PARAMS_FILE,
    STANDARDIZED_CONFIG_KEY,
    THRESHOLD_SENTINEL,
    CheckpointError,
    dispatch,
    load_sae,
)
from spade.sae_loader.standardized_sae import StandardizedSAE

add_karvonen_to_path()
from circuits.dictionary_learning.dictionary import AutoEncoder, Dictionary


class Skip(Exception):
    pass


# A checkpoint on disk may only ever be refused for one of these; anything else is a bug.
# Both are deliberate: an overwritten standardization_params.json, and Karvonen's identity
# passthrough baselines (0-byte ae.pt over MLP activations).
EXPECTED_REFUSALS = ("belongs to a different run", "!= config", "out of scope")


def roots() -> list[Path]:
    default = str(repo_root() / "checkpoints")
    return [Path(p) for p in os.environ.get("SPADE_SAE_ROOTS", default).split(":")
        if Path(p).is_dir()]


def checkpoints() -> list[Path]:
    return sorted(p.parent for root in roots() for p in root.rglob("ae.pt"))


def format_key(checkpoint_dir: Path) -> str:
    """What makes two checkpoints the same loading problem: sweep, classes, declared dims, gate.

    Read from config.json alone so discovery does not touch 39 GB of weights.
    """
    config = json.loads((checkpoint_dir / "config.json").read_text())
    trainer, buffer = config["trainer"], config["buffer"]
    return "|".join([
        checkpoint_dir.parent.name,
        trainer["trainer_class"],
        trainer.get("dict_class", "no_dict_class"),
        f"d{trainer.get('dict_size', '?')}",
        "std" if STANDARDIZED_CONFIG_KEY in buffer else "raw",
        "v2" if checkpoint_dir.name.endswith("-v2") else "v1",
    ])


def formats() -> dict[str, Path]:
    found = {}
    for d in checkpoints():
        found.setdefault(format_key(d), d)
    if not found:
        raise Skip(f"no ae.pt under {[str(r) for r in roots()]}; set SPADE_SAE_ROOTS")
    return found


def _write_checkpoint(d: Path, trainer: dict, buffer: dict, state: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"trainer": trainer, "buffer": buffer}))
    torch.save(state, d / "ae.pt")
    return d


def _autoencoder_state(activation_dim=8, dict_size=16) -> dict:
    ae = AutoEncoder(activation_dim, dict_size)
    with torch.no_grad():
        ae.decoder.weight /= ae.decoder.weight.norm(dim=0, keepdim=True)
    return ae.state_dict()


_RAW_TRAINER = {"trainer_class": "PAnnealTrainer", "dict_class": "AutoEncoder",
                "activation_dim": 8, "dict_size": 16}
_RAW_BUFFER = {"d_submodule": 8, "io": "out", "ctx_len": 59}


def _standardized(mean: torch.Tensor, scale: float) -> tuple[dict, dict]:
    buffer = {**_RAW_BUFFER, STANDARDIZED_CONFIG_KEY: scale,
              "standardization_mean_norm": mean.norm().item()}
    params = {"mean": mean.tolist(), "scale": scale, "d_model": len(mean),
              "formula": "act_standardized = scale * (act - mean)"}
    return buffer, params


def _raises(message: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except CheckpointError as e:
        assert message in str(e), f"expected {message!r} in {e}"
        return
    raise AssertionError(f"expected CheckpointError containing {message!r}")


# ---------------------------------------------------------------------------
# Dispatch — no weights needed
# ---------------------------------------------------------------------------

def test_dict_class_wins_over_trainer_class():
    """Karvonen's own utils.load_dictionary dispatches on dict_class equality; so do we."""
    dict_class, _ = dispatch({"trainer_class": "BatchTopKTrainer", "dict_class": "AutoEncoder"})
    assert dict_class == "AutoEncoder"


def test_trainer_class_fallback_is_most_specific_first():
    """The substring chain the released standard sweep needs, and the traps in it.

    "TopK" is a substring of "BatchTopK" is a substring of "MatryoshkaBatchTopK"; a chain in
    the wrong order silently evaluates a Matryoshka SAE as a plain BatchTopK one.
    """
    expected = {
        "StandardTrainer": "AutoEncoder",
        "PAnnealTrainer": "AutoEncoder",
        "GatedSAETrainer": "GatedAutoEncoder",
        "GatedAnnealTrainer": "GatedAutoEncoder",
        "TrainerTopK": "AutoEncoderTopK",
        "TopKTrainer": "AutoEncoderTopK",
        "BatchTopKTrainer": "BatchTopKSAE",
        "MatryoshkaBatchTopKTrainer": "MatryoshkaBatchTopKSAE",
        "SpaDETrainer": "SpaDEAutoEncoder",
        "CosineSpaDETrainer": "CosineSpaDEAutoEncoder",
    }
    for trainer_class, dict_class in expected.items():
        got, _ = dispatch({"trainer_class": trainer_class})
        assert got == dict_class, f"{trainer_class} dispatched to {got}, expected {dict_class}"


def test_unknown_trainer_class_is_refused():
    _raises("refusing to guess", dispatch, {"trainer_class": "SomeNewTrainer"})
    _raises("unsupported", dispatch, {"trainer_class": "X", "dict_class": "JumpReluAutoEncoder"})


# ---------------------------------------------------------------------------
# Standardization gate — the wrong case must be impossible, not unlikely
# ---------------------------------------------------------------------------

def test_raw_checkpoint_next_to_params_file_is_refused():
    """Adjacency must never standardize one of Karvonen's SAEs (0.9944 -> 0.7167 VE)."""
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        d = _write_checkpoint(save_dir / "trainer0", _RAW_TRAINER, _RAW_BUFFER, _autoencoder_state())

        ae, meta = load_sae(d)
        assert meta.standardized is False
        assert not isinstance(ae, StandardizedSAE)

        _, params = _standardized(torch.arange(8, dtype=torch.float32), 0.5)
        (save_dir / PARAMS_FILE).write_text(json.dumps(params))
        _raises("config records raw activations", load_sae, d)


def test_standardized_config_without_params_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        buffer, _ = _standardized(torch.arange(8, dtype=torch.float32), 0.5)
        d = _write_checkpoint(Path(tmp) / "trainer0", _RAW_TRAINER, buffer, _autoencoder_state())
        _raises("mean vector is unrecoverable", load_sae, d)


def test_params_file_from_another_run_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        mean = torch.arange(8, dtype=torch.float32)
        buffer, params = _standardized(mean, 0.5)
        d = _write_checkpoint(save_dir / "trainer0", _RAW_TRAINER, buffer, _autoencoder_state())

        (save_dir / PARAMS_FILE).write_text(json.dumps({**params, "mean": (mean * 2).tolist()}))
        _raises("belongs to a different run", load_sae, d)

        (save_dir / PARAMS_FILE).write_text(json.dumps({**params, "scale": 0.25}))
        _raises("!= config", load_sae, d)


def test_wrapper_consumes_raw_activations():
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        mean, scale = torch.arange(8, dtype=torch.float32), 0.5
        buffer, params = _standardized(mean, scale)
        state = _autoencoder_state()
        d = _write_checkpoint(save_dir / "trainer0", _RAW_TRAINER, buffer, state)
        (save_dir / PARAMS_FILE).write_text(json.dumps(params))

        ae, meta = load_sae(d)
        assert meta.standardized is True and isinstance(ae, StandardizedSAE)

        inner = AutoEncoder(8, 16)
        inner.load_state_dict(state)
        inner.eval()
        x = torch.randn(4, 8) + mean
        assert torch.allclose(ae.encode(x), inner.encode(scale * (x - mean)))
        assert torch.allclose(ae(x), inner(scale * (x - mean)) / scale + mean)
        # The wrapper is not a no-op: dropping it would change every downstream number.
        assert not torch.allclose(ae(x), inner(x), atol=1e-3)


# ---------------------------------------------------------------------------
# Weights: shapes, decoder norms, top-k buffers
# ---------------------------------------------------------------------------

def test_shapes_come_from_weights_when_config_omits_them():
    """His 80-checkpoint standard sweep records no dict_class, activation_dim or dict_size."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = {"trainer_class": "StandardTrainer", "lr": 3e-4, "layer": 5}
        d = _write_checkpoint(Path(tmp) / "trainer0", trainer, _RAW_BUFFER, _autoencoder_state())
        ae, meta = load_sae(d)
        assert (meta.activation_dim, meta.dict_size) == (8, 16)
        assert meta.dict_class == "AutoEncoder" and ae.encoder.weight.shape == (16, 8)


def test_config_shape_disagreeing_with_weights_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = {**_RAW_TRAINER, "dict_size": 32}
        d = _write_checkpoint(Path(tmp) / "trainer0", trainer, _RAW_BUFFER, _autoencoder_state())
        _raises("but weights say 16", load_sae, d)


def test_rescaled_decoder_is_refused():
    """F1 thresholds are fractions of a feature's max activation, so scaling is not neutral."""
    with tempfile.TemporaryDirectory() as tmp:
        state = _autoencoder_state()
        state["decoder.weight"] = state["decoder.weight"] * 2
        d = _write_checkpoint(Path(tmp) / "trainer0", _RAW_TRAINER, _RAW_BUFFER, state)
        _raises("not unit-norm", load_sae, d)


def test_missing_threshold_refused_for_a_thresholding_class():
    """BatchTopKSAE.encode thresholds by default: a sentinel threshold returns everything."""
    with tempfile.TemporaryDirectory() as tmp:
        add_karvonen_to_path()
        from circuits.dictionary_learning.trainers.batch_top_k import BatchTopKSAE

        sae = BatchTopKSAE(8, 16, 4)
        trainer = {"trainer_class": "BatchTopKTrainer", "dict_class": "BatchTopKSAE", "k": 4}
        state = {k: v for k, v in sae.state_dict().items() if k != "threshold"}
        d = _write_checkpoint(Path(tmp) / "trainer0", trainer, _RAW_BUFFER, state)
        _raises("no trained threshold", load_sae, d)

        state["threshold"] = torch.tensor(THRESHOLD_SENTINEL)
        _write_checkpoint(d, trainer, _RAW_BUFFER, state)
        _raises("untrained sentinel", load_sae, d)


# ---------------------------------------------------------------------------
# Every checkpoint format found on disk
# ---------------------------------------------------------------------------

def check_checkpoint(d: Path) -> str:
    """Load one checkpoint and assert every invariant the study depends on. Returns a summary."""
    ae, meta = load_sae(d)
    state = torch.load(d / "ae.pt", map_location="cpu", weights_only=True)

    assert isinstance(ae, Dictionary), type(ae)
    assert not ae.training, f"{d} loaded in train mode"
    assert (meta.activation_dim, meta.dict_size) == (ae.activation_dim, ae.dict_size)

    trainer = meta.config["trainer"]
    for key, value in (("activation_dim", meta.activation_dim), ("dict_size", meta.dict_size)):
        assert trainer.get(key, value) == value, f"{d}: config {key} != {value}"

    # The gate, checked against the independent filesystem signal it must never read.
    has_params = any((p / PARAMS_FILE).is_file() for p in (d, d.parent))
    assert meta.standardized == (STANDARDIZED_CONFIG_KEY in meta.config["buffer"])
    assert meta.standardized == has_params, f"{d}: gate {meta.standardized} vs params {has_params}"
    assert isinstance(ae, StandardizedSAE) == meta.standardized

    x = torch.randn(32, meta.activation_dim)
    x_hat, f = ae(x, output_features=True)
    assert x_hat.shape == x.shape and f.shape == (32, meta.dict_size)
    assert torch.isfinite(x_hat).all() and torch.isfinite(f).all()

    l0 = (f != 0).float().sum(dim=-1).mean().item()
    if meta.dict_class == "AutoEncoderTopK":
        # Exactly k, whatever the threshold buffer says: the 2024-07-10 checkpoints have no
        # trained threshold, and a sentinel one read as real gives L0 = dict_size.
        assert l0 == trainer["k"], f"{d}: L0 {l0} != k {trainer['k']}"

    buffers = "".join(c for c, present in zip("kt", ("k" in state, "threshold" in state)) if present)
    dims = f"d={meta.activation_dim} n={meta.dict_size}"
    return (f"{meta.trainer_class:26} {meta.dict_class:24} {dims:18} "
            f"std={str(meta.standardized):5} L0={l0:8.1f} buffers={buffers or '-':2} {d}")


def test_every_format_loads():
    """One checkpoint per format; a format may only ever be refused for EXPECTED_REFUSALS.

    autoencoders_spade_sweep/trainer_{4,6} are the live provenance case: a later sweep run
    overwrote their save dir's standardization_params.json, so the mean that would be
    applied to them is not the one they were trained under.
    """
    for key, d in sorted(formats().items()):
        try:
            print(f"  {key}\n    {check_checkpoint(d)}")
        except CheckpointError as e:
            print(f"  {key}\n    REFUSED {e}")
            assert any(r in str(e) for r in EXPECTED_REFUSALS), e


def test_released_topk_predates_the_k_and_threshold_buffers():
    """Karvonen's 2024-07-10 top_k state dicts have neither buffer; the load stays strict."""
    seen = 0
    for key, d in sorted(formats().items()):
        if "AutoEncoderTopK" not in key or "std" in key.split("|"):
            continue
        state = torch.load(d / "ae.pt", map_location="cpu", weights_only=True)
        assert "k" not in state and "threshold" not in state, f"{d}: expected no buffers"
        ae, meta = load_sae(d)
        assert ae.k.item() == meta.config["trainer"]["k"]
        assert ae.threshold.item() == THRESHOLD_SENTINEL
        seen += 1
    if not seen:
        raise Skip("no released top_k checkpoints under SPADE_SAE_ROOTS")


def test_standardization_gate_agrees_with_params_files_everywhere():
    """Every checkpoint on disk, config gate vs adjacent params file — the two must never split."""
    found = checkpoints()
    if not found:
        raise Skip("no checkpoints under SPADE_SAE_ROOTS")
    for d in found:
        config = json.loads((d / "config.json").read_text())
        gate = STANDARDIZED_CONFIG_KEY in config["buffer"]
        has_params = any((p / PARAMS_FILE).is_file() for p in (d, d.parent))
        assert gate == has_params, f"{d}: config gate {gate} but params file {has_params}"
    print(f"  {len(found)} checkpoints agree")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed, skipped = 0, 0
    for t in tests:
        try:
            t()
        except Skip as e:
            print(f"SKIP {t.__name__}: {e}")
            skipped += 1
        except AssertionError:
            traceback.print_exc()
            failed += 1
        else:
            print(f"ok   {t.__name__}")
    if "--all" in sys.argv:
        found = checkpoints()
        print(f"\nloading all {len(found)} checkpoints under {[str(r) for r in roots()]}")
        loaded, refused = [], []
        for d in found:
            try:
                loaded.append(check_checkpoint(d))
            except CheckpointError as e:
                if not any(r in str(e) for r in EXPECTED_REFUSALS):
                    print(f"FAIL {d}: {e}")
                    failed += 1
                refused.append(f"{d}: {e}")
            except (AssertionError, RuntimeError, KeyError) as e:
                print(f"FAIL {d}: {type(e).__name__}: {e}")
                failed += 1
        shapes = Counter(" ".join(line.split()[:5]) for line in loaded)
        print(f"\n{len(loaded)} loaded, {len(refused)} refused")
        print("\n".join(f"  {n:4} {s}" for s, n in sorted(shapes.items())))
        print("\n".join(f"  REFUSED {r}" for r in refused))
    print(f"\n{len(tests) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
