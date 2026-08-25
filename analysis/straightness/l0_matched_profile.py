"""Straightness-corpus geometry for the eight L0-matched picks: the profile, and the twin.

The change-point score is `-(d_t - mu(t)) / sigma(t)`: how much shorter than usual a step is,
where "usual" is a per-move-index profile fit on legal games. `compute_geometry.py` writes that
profile for the canonical seed-42 roster only, so an L0-matched space has no baseline to be
scored against and `changepoint_localization` / `illegal_move_detector` cannot read it.

Two artifacts, because two studies need different amounts of the corpus:

- `l0_<config>_legal.npz` -- `d` over the legal block. The profile is its only consumer and
  the block is the whole of it. **Never rewritten**: a GPU encode is not bitwise reproducible
  and every committed L0-matched number in `changepoint_localization` is byte-identical
  against the current file.
- `l0_<config>_raw.npz` -- `d`, `theta` and `norm` over all five classes, a drop-in twin of
  the canonical `<stem>_raw.npz` with the same row layout. `quantity_ablation.py` scores the
  *violation* block and compares three quantities, neither of which the legal-only file can
  serve. Its `d` on the legal rows duplicates the profile file to within encode noise; that
  is the price of not rewriting a file other studies are pinned to.

The window selection those studies do on the straightness violation block is not redone for
this roster: the window is chosen once, on the canonical spaces, and applied untouched (see
`changepoint_localization/README.md`).

Each SAE is fed the activations it was trained on -- ours standardized, Karvonen's raw --
via `checkpoints.Matched.standardize`, the same rule
`curvature_discriminant/l0_matched_geometry.py` follows.

Writes study_dir("straightness")/geometry/l0_<config>_{legal,raw}.npz, one pair per pick.

Usage: PYTHONPATH=src python analysis/straightness/l0_matched_profile.py
"""

import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.l0_operating_point.checkpoints import matched
from spade.straightness.build_games import CLASSES
from spade.straightness.geometry import temporal_geometry

GAME_BATCH = 32


def trajectories() -> tuple[torch.Tensor, int]:
    """Every straightness game as [n, 58, 512] with the sink dropped, and the class size."""
    cls = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)["cls"]
    n = len(cls) // len(CLASSES)
    assert (cls[:n] == 0).all(), "the legal games are not the leading block of games.pt"
    traj = torch.load(study_dir("straightness") / "traj" / "resid_post_5.pt",
                      weights_only=True, mmap=True)
    return traj[:, 1:, :], n


def encode_all(z: torch.Tensor, encode, device: str) -> tuple[np.ndarray, ...]:
    """(d, theta, norm) over `z`, batched by game."""
    ds, thetas, norms = [], [], []
    for start in range(0, len(z), GAME_BATCH):
        batch = z[start:start + GAME_BATCH].float().to(device)
        with torch.no_grad():
            lat = encode(batch.reshape(-1, batch.shape[-1])).reshape(*batch.shape[:2], -1)
        d, theta = temporal_geometry(lat)
        ds.append(d.cpu().numpy())
        thetas.append(theta.cpu().numpy())
        norms.append(lat.norm(dim=-1).cpu().numpy())
    return tuple(np.concatenate(a).astype(np.float32) for a in (ds, thetas, norms))


def main() -> None:
    device = select_device()
    out = study_dir("straightness") / "geometry"
    z, n_legal = trajectories()
    print(f"{len(z)} games ({n_legal} legal) on {device}", flush=True)

    for entry in matched():
        card = {"config": entry.config, "l0": entry.l0, "standardized": entry.standardize}
        legal, raw = out / f"{entry.space}_legal.npz", out / f"{entry.space}_raw.npz"
        if legal.exists() and raw.exists():
            print(f"{entry.space} present, skipping", flush=True)
            continue
        encode, started = entry.encode(device), time.time()

        if not legal.exists():
            d, _, _ = encode_all(z[:n_legal], encode, device)
            assert np.isfinite(d).all() and (d > 0).all(), f"{entry.space}: a zero-length step"
            np.savez(legal, d=d, **card)
            print(f"wrote {legal.name}  d {d.shape}", flush=True)

        if not raw.exists():
            d, theta, norm = encode_all(z, encode, device)
            np.savez(raw, d=d, theta=theta, norm=norm, **card)
            print(f"wrote {raw.name}  d {d.shape} theta {theta.shape} norm {norm.shape}",
                  flush=True)
        print(f"  {entry.config}, L0 {entry.l0:.1f}, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
