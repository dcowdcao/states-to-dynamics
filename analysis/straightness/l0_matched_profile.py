"""Step lengths over the straightness legal block, for the eight L0-matched picks.

The change-point score is `-(d_t - mu(t)) / sigma(t)`: how much shorter than usual a step is,
where "usual" is a per-move-index profile fit on legal games. `compute_geometry.py` writes that
profile for the canonical seed-42 roster only, so an L0-matched space has no baseline to be
scored against and `changepoint_localization` / `illegal_move_detector` cannot read it.

Only the legal block is written -- the profile is the sole consumer, and the block is the whole
of it. The window selection those studies do on the straightness *violation* block is not
redone for this roster: the window is chosen once, on the canonical spaces, and applied
untouched (see `changepoint_localization/README.md`).

Each SAE is fed the activations it was trained on -- ours standardized, Karvonen's raw --
via `checkpoints.Matched.standardize`, the same rule
`curvature_discriminant/l0_matched_geometry.py` follows.

Writes study_dir("straightness")/geometry/l0_<config>_legal.npz, one per pick.

Usage: PYTHONPATH=src python analysis/straightness/l0_matched_profile.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.l0_operating_point.checkpoints import matched
from spade.straightness.build_games import CLASSES
from spade.straightness.geometry import temporal_geometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "violation_divergence"
                      / "curvature_discriminant"))
from l0_matched_geometry import encoder  # noqa: E402  (sibling script, not a package)

GAME_BATCH = 32


def legal_block() -> torch.Tensor:
    """The trajectory of every legal straightness game: [n, 58, 512], sink dropped."""
    cls = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)["cls"]
    n = len(cls) // len(CLASSES)
    assert (cls[:n] == 0).all(), "the legal games are not the leading block of games.pt"
    traj = torch.load(study_dir("straightness") / "traj" / "resid_post_5.pt",
                      weights_only=True, mmap=True)
    return traj[:n, 1:, :]


def main() -> None:
    device = select_device()
    out = study_dir("straightness") / "geometry"
    z = legal_block()
    print(f"{len(z)} legal games on {device}", flush=True)

    for entry in matched():
        path = out / f"{entry.space}_legal.npz"
        if path.exists():
            print(f"{path.name} present, skipping", flush=True)
            continue
        encode = encoder(entry, device)
        started, ds = time.time(), []
        for start in range(0, len(z), GAME_BATCH):
            batch = z[start:start + GAME_BATCH].float().to(device)
            with torch.no_grad():
                lat = encode(batch.reshape(-1, batch.shape[-1])).reshape(*batch.shape[:2], -1)
            ds.append(temporal_geometry(lat)[0].cpu().numpy())
        d = np.concatenate(ds).astype(np.float32)
        assert np.isfinite(d).all() and (d > 0).all(), f"{entry.space}: a zero-length step"
        np.savez(path, d=d, config=entry.config, l0=entry.l0, standardized=entry.standardize)
        print(f"wrote {path.name}  d {d.shape}  ({entry.config}, L0 {entry.l0:.1f}, "
              f"{time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
