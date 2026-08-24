"""Step lengths for the eight SAE types seated at one shared L0.

`branch_geometry.py` encodes the canonical seed-42 roster, whose members sit wherever their
own default hyperparameters put them on the sparsity axis -- so a difference between two of
those spaces confounds the architecture with its L0. This encodes the `l0_operating_point`
picks instead: one checkpoint per type, each the nearest-L0 member of its own grid to the
shared target, so the architectures are compared at matched sparsity.

**Each SAE is fed the activations it was trained on.** Ours were trained on
`scale*(x - mean)` and are wrapped in `StandardizedDictionary`; Karvonen's five were trained
on the raw residual stream and are not. Feeding either the other's convention is silently
wrong rather than merely inaccurate (`spade.spaces`), so `checkpoints.Matched.standardize`
carries the rule per checkpoint. The consequence a reader is owed: the two families' latents
are not on one input scale, so this is a comparison of architectures at matched L0 and not of
one quantity measured eight ways. The per-move-index standardization downstream absorbs a
global scale difference but not a difference in what a dictionary learned to reconstruct.

Rows come from `curvature_discriminant/meta.npz`, never re-derived, so these npz files are
aligned with the study's own `d`/`theta` by construction.

Writes `d` AND `theta`, the pair `branch_geometry.py` writes, because the R^21 and curvature
studies build the ReStraV descriptor out of both -- saving only `d` would quietly restrict the
L0-matched roster to the step-length figures.

Writes study_dir("divergence")/curvature_discriminant/l0_<config>.npz, one per space.

Usage: PYTHONPATH=src python analysis/violation_divergence/curvature_discriminant/l0_matched_geometry.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.l0_operating_point.checkpoints import matched
from spade.straightness.geometry import temporal_geometry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branch_geometry import GAME_BATCH, out_dir  # noqa: E402  (sibling script)


def main() -> None:
    device = select_device()
    out = out_dir()
    rows = torch.from_numpy(np.load(out / "meta.npz")["row"])
    acts = torch.load(study_dir("divergence") / "acts" / "resid_post_5.pt",
                      weights_only=True, mmap=True)
    print(f"{len(rows)} branch rows", flush=True)

    for entry in matched():
        path = out / f"{entry.space}.npz"
        if path.exists() and "theta" in np.load(path):
            print(f"{path.name} present, skipping", flush=True)
            continue
        encode = entry.encode(device)
        ds, thetas = [], []
        for start in range(0, len(rows), GAME_BATCH):
            z = acts[rows[start:start + GAME_BATCH]][:, 1:, :].float().to(device)
            with torch.no_grad():
                z = encode(z.reshape(-1, z.shape[-1])).reshape(*z.shape[:2], -1)
            d_chunk, theta_chunk = temporal_geometry(z)
            ds.append(d_chunk.cpu().numpy())
            thetas.append(theta_chunk.cpu().numpy())
        d = np.concatenate(ds).astype(np.float32)
        theta = np.concatenate(thetas).astype(np.float32)
        np.savez(path, d=d, theta=theta, config=entry.config, l0=entry.l0,
                 standardized=entry.standardize)
        print(f"wrote {path.name}  d {d.shape} theta {theta.shape}  "
              f"({entry.config}, L0 {entry.l0:.1f})", flush=True)


if __name__ == "__main__":
    main()
