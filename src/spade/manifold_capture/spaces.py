"""Space loading and decoder access shared by all manifold-capture analyses.

Every analysis works in standardized space (scale * (x - mean), the transform
the SAEs were trained on): "raw" is the identity view of that space, SAE codes
come from StandardizedDictionary.encode on raw activations, and decoder atoms
(columns of D) live in the same space — which is what makes SpaDE
landmark-to-activation distances directly meaningful. Never compare raw code
magnitudes across families: SpaDE-family codes are simplex-valued.

Distinct from the top-level `spade.spaces`, which owns the shared encode API
this module wraps; imports here are absolute so the two never shadow.
"""

import torch

from spade.common.paths import repo_root
from spade.evaluation.karvonen_adapter import load_karvonen_sae
from spade.spaces import EXCLUDED_SAE, batched_apply, load_space
# private registry internals: seed42_entries() verifies every config at once and
# crashes on any in-flight (registered, not yet trained) checkpoint; this study
# builds and verifies its entries per config instead.
from spade.saes.registry import _SEED42_DIRS, _verify, SAEEntry

POINT_BASED = {cfg for cfg in _SEED42_DIRS if "SpaDE" in cfg}
# paper drops VE<0.85; only the baseline distance-SpaDE (VE~0.43) trips that —
# CosineSpaDE VE~0.98, revived SpaDE-lam1e-03-gam1e-04 VE~0.995 (L0~1808).
#
# This is the same checkpoint as spade.spaces.EXCLUDED_SAE, named from the other side:
# every other study drops it as degenerate, while this one deliberately keeps and marks
# it, because the shattering result is a claim *about* it. Aliased rather than restated
# so the two cannot drift, and so the divergence is visible rather than accidental.
# Whether that result survives the exclusion is an open question in PROCESS.md.
LOW_VE = EXCLUDED_SAE


def spaces() -> list[tuple[str, SAEEntry | None]]:
    """(name, entry): standardized raw + untrained controls + every seed-42
    checkpoint present on disk. A registered config without ae.pt (training
    in flight elsewhere) is skipped loudly and joins on a later resumable run."""
    out = [(n, None) for n in ["raw", "random_proj", "random_topk"]]
    for cfg, rel in _SEED42_DIRS.items():
        if (repo_root() / rel / "ae.pt").exists():
            out.append((cfg, _verify(SAEEntry(cfg, 42, repo_root() / rel))))
        else:
            print(f"spaces: SKIPPING {cfg} — no ae.pt at {rel} yet")
    return out


def encode_fn(name: str, entry: SAEEntry | None, device: str):
    """Raw-activation batch on `device` -> features (standardized space for "raw")."""
    return load_space(name if entry is None else "sae", entry, device)


def decoder(entry: SAEEntry, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """(D [512, F], bias [512]) in standardized space: x_std_hat = bias + f @ D.T."""
    ae, _ = load_karvonen_sae(str(entry.ckpt_dir), device=device)
    if hasattr(ae, "Ad"):  # SpaDE family: prototype decoder, no bias
        return ae.Ad.detach(), torch.zeros(ae.activation_dim, device=device)
    for attr in ["bias", "decoder_bias", "b_dec"]:  # Standard/PAnneal, Gated, TopK/BatchTopK
        if hasattr(ae, attr):
            return ae.decoder.weight.detach(), getattr(ae, attr).detach()
    raise AttributeError(f"no known decoder bias on {type(ae).__name__}")


def check_codes(name: str, codes: torch.Tensor) -> None:
    """Fail-loud invariants: nonnegative codes; SpaDE codes on the simplex."""
    assert (codes >= -1e-5).all(), f"{name}: negative code values"
    if "SpaDE" in name:
        sums = codes.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3), \
            f"{name}: sparsemax codes do not sum to 1 (got mean {sums.mean():.4f})"
