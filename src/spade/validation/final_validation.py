#!/usr/bin/env python3
"""Fast pre-deployment self-tests for the spade package (no GPU, no checkpoints).

Everything here runs on synthetic data in seconds. The checks that need real weights
or extracted activations report SKIP with the reason rather than passing quietly, and
main() prints what was not checked so a green run is not mistaken for a full one.

Usage:
    python -m spade.validation.final_validation
"""

import importlib
import json
import py_compile
import re
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

PKG_DIR = Path(__file__).resolve().parents[1]        # src/spade
SRC_DIR = PKG_DIR.parent                             # src

EXPECTED_D_MODEL = 512
WIDTH = 64

# The eight trainer families of the L0-matched roster (results/l0_operating_point/picks.csv),
# and the substring each one must hit in load_karvonen_sae's dispatch chain.
EXPECTED_DISPATCH = {
    "StandardTrainer": "Standard",
    "PAnnealTrainer": "PAnneal",
    "GatedSAETrainer": "Gated",
    "GatedAnnealTrainer": "Gated",
    "TopKTrainer": "TopK",
    "BatchTopKTrainer": "BatchTopK",
    "SpaDETrainer": "SpaDE",
    "CosineSpaDETrainer": "CosineSpaDE",
}


class Skip(Exception):
    """Raised by a test that needs data this environment does not have."""


results = []
_TESTS = []


def test(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, "PASS", ""))
                print(f"  PASS  {name}")
            except Skip as e:
                results.append((name, "SKIP", str(e)))
                print(f"  SKIP  {name}: {e}")
            # A test harness reports a broken test; it does not die on one.
            except Exception as e:
                results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
        _TESTS.append(wrapper)
        return wrapper
    return decorator


def _package_modules():
    """Every spade subpackage (a dir with __init__.py), as importable module names."""
    packages = ["spade"]
    for init in sorted(PKG_DIR.rglob("__init__.py")):
        rel = init.parent.relative_to(SRC_DIR)
        if rel != Path("spade"):
            packages.append(".".join(rel.parts))
    return packages


def _dispatch_order():
    """The ordered trainer_class substrings load_karvonen_sae branches on."""
    src = (PKG_DIR / "evaluation" / "karvonen_adapter.py").read_text()
    body = src.split("def load_karvonen_sae", 1)[1].split("\ndef ", 1)[0]
    return re.findall(r'"([A-Za-z]+)" in trainer_class', body)


def _first_match(order, trainer_class):
    for substring in order:
        if substring in trainer_class:
            return substring
    return None


class _IdentityDictionary(nn.Module):
    """Minimal Dictionary stand-in so the wrapper's math can be checked without an SAE."""

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W = nn.Parameter(torch.randn(activation_dim, dict_size) / dict_size**0.5)

    def encode(self, x):
        return x @ self.W

    def decode(self, f):
        return f @ self.W.T

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# ── Static structure ────────────────────────────────────────────────────────

@test("every .py under src/spade compiles")
def test_compilation():
    files = sorted(PKG_DIR.rglob("*.py"))
    assert files, f"no .py files under {PKG_DIR}"
    for path in files:
        py_compile.compile(str(path), doraise=True)


@test("every spade subpackage imports")
def test_subpackages_import():
    packages = _package_modules()
    assert len(packages) > 10, f"expected the full subpackage set, found {packages}"
    for name in packages:
        importlib.import_module(name)


@test("every package dir has an __init__.py")
def test_init_files_present():
    missing = [d.relative_to(SRC_DIR) for d in sorted(PKG_DIR.rglob("*"))
        if d.is_dir()
        and not d.name.startswith((".", "__"))
        and any(d.glob("*.py"))
        and not (d / "__init__.py").exists()]
    assert not missing, f"package dirs without __init__.py: {missing}"


@test("no legacy import paths or intra-repo sys.path injection")
def test_no_legacy_imports():
    legacy = [
        re.compile(r"^\s*from (common|models|evaluation|extraction|multiseed|figures|training|rules)[. ]"),
        re.compile(r"^\s*import (common|models)\b"),
        re.compile(r"sys\.path\.insert\(0, str\(Path\(__file__\)"),
    ]
    for path in sorted(PKG_DIR.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in legacy:
                assert not pattern.search(line), \
                    f"{path.relative_to(SRC_DIR)}:{lineno} legacy import: {line.strip()}"


# ── The load_karvonen_sae dispatch contract ─────────────────────────────────

@test("dispatch table covers all 8 trainer classes")
def test_dispatch_covers_8():
    order = _dispatch_order()
    assert order, "no dispatch branches found in load_karvonen_sae"
    assert len(EXPECTED_DISPATCH) == 8, "the trainer roster is 8 types"
    for trainer_class, expected in EXPECTED_DISPATCH.items():
        hit = _first_match(order, trainer_class)
        assert hit == expected, \
            f"{trainer_class} dispatches to {hit!r}, expected {expected!r}"


@test("dispatch ordering: BatchTopK<TopK, CosineSpaDE<SpaDE, Gated first")
def test_dispatch_ordering():
    order = _dispatch_order()
    for earlier, later in [("BatchTopK", "TopK"), ("CosineSpaDE", "SpaDE"),
                           ("Gated", "Standard"), ("Gated", "PAnneal")]:
        assert earlier in order, f"dispatch lost its {earlier} branch"
        assert later in order, f"dispatch lost its {later} branch"
        assert order.index(earlier) < order.index(later), \
            f"{earlier} must be matched before {later} (substring collision)"


@test("load_karvonen_sae keeps the wandb_name fallback")
def test_wandb_name_fallback():
    src = (PKG_DIR / "evaluation" / "karvonen_adapter.py").read_text()
    body = src.split("def load_karvonen_sae", 1)[1].split("\ndef ", 1)[0]
    assert 'elif "wandb_name" in trainer_config:' in body, \
        "old checkpoints have no trainer_class; the wandb_name fallback is required"


@test("load_standardization_params falls back to the parent dir")
def test_standardization_parent_fallback():
    from spade.evaluation.karvonen_adapter import load_standardization_params

    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp) / "checkpoints" / "main"
        trainer_dir = save_dir / "trainer_0"
        trainer_dir.mkdir(parents=True)
        params = {"mean": [0.0] * EXPECTED_D_MODEL, "scale": 2.0, "d_model": EXPECTED_D_MODEL}
        (save_dir / "standardization_params.json").write_text(json.dumps(params))

        loaded = load_standardization_params(str(trainer_dir))
        assert loaded["scale"] == 2.0, "parent-dir standardization_params.json not found"
        assert loaded["d_model"] == EXPECTED_D_MODEL


@test("checkpoint dispatch round-trip on real weights")
def test_checkpoint_roundtrip():
    from spade.common.paths import repo_root

    weights = sorted((repo_root() / "checkpoints").rglob("ae.pt"))
    if not weights:
        raise Skip("no checkpoints/*/trainer_*/ae.pt in this tree (weights are gitignored)")

    from spade.evaluation.karvonen_adapter import load_karvonen_sae
    ae, trainer_config = load_karvonen_sae(str(weights[0].parent))
    assert ae.activation_dim == trainer_config["activation_dim"]
    assert ae.dict_size == trainer_config["dict_size"]


# ── StandardizedDictionary ──────────────────────────────────────────────────

@test("StandardizedDictionary standardization math is correct")
def test_standardization_math():
    from spade.evaluation.karvonen_adapter import StandardizedDictionary

    wrapped = StandardizedDictionary(_IdentityDictionary(EXPECTED_D_MODEL, WIDTH),
                                     torch.randn(EXPECTED_D_MODEL), 0.5)
    x = torch.randn(8, EXPECTED_D_MODEL)
    x_std = wrapped._standardize(x)
    assert torch.allclose(x_std, 0.5 * (x - wrapped.mean), atol=1e-5), \
        "standardization is scale * (x - mean)"
    assert torch.allclose(wrapped._unstandardize(x_std), x, atol=1e-5), \
        "unstandardize must invert standardize"


@test("StandardizedDictionary encode/decode/forward shapes")
def test_standardized_dictionary_shapes():
    from spade.evaluation.karvonen_adapter import StandardizedDictionary

    wrapped = StandardizedDictionary(_IdentityDictionary(EXPECTED_D_MODEL, WIDTH),
                                     torch.zeros(EXPECTED_D_MODEL), 1.0)
    x = torch.randn(16, EXPECTED_D_MODEL)
    f = wrapped.encode(x)
    assert f.shape == (16, WIDTH)
    assert wrapped.decode(f).shape == (16, EXPECTED_D_MODEL)
    x_hat, f2 = wrapped(x, output_features=True)
    assert f2.shape == f.shape
    # 3D input is the loss_recovered path
    assert wrapped(torch.randn(4, 8, EXPECTED_D_MODEL)).shape == (4, 8, EXPECTED_D_MODEL)


# ── The SpaDE family ────────────────────────────────────────────────────────

@test("SpaDEAutoEncoder encode/decode round-trips")
def test_spade_roundtrip():
    from spade.saes.spade import SpaDEAutoEncoder

    ae = SpaDEAutoEncoder(activation_dim=EXPECTED_D_MODEL, dict_size=WIDTH, lambda_init=9.77e-4)
    x = torch.randn(16, EXPECTED_D_MODEL)
    f = ae.encode(x)
    assert f.shape == (16, WIDTH)
    assert ae.decode(f).shape == (16, EXPECTED_D_MODEL)
    _, f2 = ae(x, output_features=True)
    assert f2.shape == f.shape


@test("SpaDEAutoEncoder lambda is trainable and initialized")
def test_spade_lambda():
    from spade.saes.spade import SpaDEAutoEncoder

    ae = SpaDEAutoEncoder(activation_dim=EXPECTED_D_MODEL, dict_size=WIDTH, lambda_init=9.77e-4)
    assert hasattr(ae, "lambda_pre"), "lambda_pre is a state_dict key loaders depend on"
    assert ae.lambda_val.requires_grad, "SpaDE lambda must be trainable"
    assert abs(ae.lambda_val.item() - 9.77e-4) < 1e-5


@test("CosineSpaDEAutoEncoder encode/decode round-trips")
def test_cosine_spade_roundtrip():
    from spade.saes.spade_cosine import CosineSpaDEAutoEncoder

    ae = CosineSpaDEAutoEncoder(activation_dim=EXPECTED_D_MODEL, dict_size=WIDTH)
    x = torch.randn(16, EXPECTED_D_MODEL)
    f = ae.encode(x)
    assert f.shape == (16, WIDTH)
    assert ae.decode(f).shape == (16, EXPECTED_D_MODEL)


@test("class names SpaDEAutoEncoder / CosineSpaDEAutoEncoder are unchanged")
def test_spade_class_names():
    from spade.saes.spade import SpaDEAutoEncoder
    from spade.saes.spade_cosine import CosineSpaDEAutoEncoder

    # The adapter dispatches on trainer_class strings and the viewer exporter keys on
    # type(ae).__name__; module paths may move, these names may not.
    assert SpaDEAutoEncoder.__name__ == "SpaDEAutoEncoder"
    assert CosineSpaDEAutoEncoder.__name__ == "CosineSpaDEAutoEncoder"


# ── Single-sourced configuration ────────────────────────────────────────────

@test("spade.common.config is internally consistent")
def test_config_consistent():
    from spade.common import config

    assert config.DICT_SIZE == config.EXPANSION * config.D_MODEL
    assert config.HOOK_POINT == f"blocks.{config.LAYER}.hook_resid_post"


@test("model constants are defined only in spade.common.config")
def test_constants_single_sourced():
    owned = ("D_MODEL", "DICT_SIZE", "HOOK_POINT", "LAYER", "EXPANSION",
             "MODEL_NAME", "DATASET", "CTX_LEN")
    pattern = re.compile(rf"^({'|'.join(owned)})\s*=")
    for path in sorted(PKG_DIR.rglob("*.py")):
        if path.resolve() in (Path(__file__).resolve(),
                              (PKG_DIR / "common" / "config.py").resolve()):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = pattern.match(line)
            assert not match, (f"{path.relative_to(SRC_DIR)}:{lineno} redefines "
                               f"{match.group(1)}; it belongs to spade.common.config")


NOT_CHECKED = """
  - no eval is run here. The acceptance eval is in docs/finish_notes/acceptance-eval.md,
    with its rows in results/acceptance/ (three checkpoints reproduce their reference exactly)
  - no GPU kernel, no nnsight trace, no loss_recovered path is exercised
  - the F1 / coverage / board-reconstruction math is not checked numerically
  - training loops are not run
  - checkpoint state_dict key compatibility is only checked if weights are present
"""


def main():
    print("=" * 70)
    print("  spade — pre-deployment validation (no GPU, synthetic data)")
    print("=" * 70)

    for t in _TESTS:
        t()

    passed = sum(1 for _, status, _ in results if status == "PASS")
    skipped = sum(1 for _, status, _ in results if status == "SKIP")
    failed = sum(1 for _, status, _ in results if status == "FAIL")

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed}/{len(results)} PASSED, {skipped} SKIPPED, {failed} FAILED")

    if failed:
        print("\n  FAILURES:")
        for name, status, msg in results:
            if status == "FAIL":
                print(f"    {name}: {msg}")
    if skipped:
        print("\n  SKIPPED:")
        for name, status, msg in results:
            if status == "SKIP":
                print(f"    {name}: {msg}")

    print("\n  NOT CHECKED BY THIS SUITE:" + NOT_CHECKED)
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
