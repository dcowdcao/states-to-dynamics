"""Per-position representation norm for the divergence branch tree.

`branch_geometry.py` writes `d` and `theta`. The straightness npz also carries `norm`, and
`quantity_ablation.py` needs it on THIS tree to ask whether the trajectory statistic beats the
magnitude it is computed from -- the study README's standing "treat this as a magnitude claim"
caveat, which cannot be checked on the spliced suffix without it.

Non-destructive on purpose: this writes `<space>_norm.npz` beside the existing file rather
than rewriting it. Re-running `branch_geometry.py` to add a key would recompute `d` and
`theta` through a GPU encode that is not bitwise reproducible, and every committed number in
`changepoint_localization` is byte-identical against the current files.

Rows come from the committed `meta.npz`'s `row` column, never re-derived. `extend_branches`
grew the tree after that npz was written -- re-deriving through `eval_games` now yields 2207
games where the npz holds 575 -- so the stored column is the only thing that keeps
`<space>_norm.npz` aligned with `<space>.npz`.

Usage: python analysis/violation_divergence/curvature_discriminant/branch_norms.py [--variant spliced]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.compute_divergence import sae_encoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branch_geometry import GAME_BATCH, VARIANTS, out_dir  # noqa: E402  (sibling script)


def norms(Z: torch.Tensor, encode, device: str) -> np.ndarray:
    out = []
    for start in range(0, len(Z), GAME_BATCH):
        x = Z[start:start + GAME_BATCH].float().to(device)
        if encode is not None:
            with torch.no_grad():
                x = encode(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:2], -1)
        out.append(x.norm(dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), default="spliced")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    device = select_device()
    base = variant_dir("divergence", variant=variant)
    out = out_dir(variant)
    assert (out / "meta.npz").exists(), "run branch_geometry.py first; this aligns to its rows"
    rows = torch.from_numpy(np.load(out / "meta.npz")["row"])

    spaces = [("resid_post_5", None)] + [(f"sae_{e.config}", e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    todo = [s for s in spaces if not (out / f"{s[0]}_norm.npz").exists()]
    if not todo:
        print("all norms present")
        return

    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    Z = acts[rows][:, 1:, :]        # drop the attention sink, as branch_geometry does
    del acts
    print(f"{len(rows)} rows, {Z.shape[1]} positions", flush=True)
    for space, entry in todo:
        norm = norms(Z, None if entry is None else sae_encoder(entry, device), device)
        expected = np.load(out / f"{space}.npz")["d"].shape
        assert norm.shape == (expected[0], expected[1] + 1), \
            f"{space}: norm {norm.shape} does not match d {expected}"
        np.savez(out / f"{space}_norm.npz", norm=norm)
        print(f"{space}: norm {norm.shape} mean {norm.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
