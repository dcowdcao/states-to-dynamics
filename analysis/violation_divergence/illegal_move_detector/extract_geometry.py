"""Step 1: raw layer-5 step lengths for every eligible branch of the spliced tree.

`curvature_discriminant/branch_geometry.py` already writes this, for nine spaces, but its
committed npz holds the 575 games that study selected before `extend_branches` grew the tree,
and these figures want ~1000. Raw layer 5 needs no dictionary encode, so re-extracting it for
every eligible game is cheap and leaves that study's artifacts untouched -- which matters,
because every number in `changepoint_localization` is verified byte-identical against them.

Eligibility is `branch_geometry`'s rule with one change: branch-complete, an odd-parity source
game so the per-move-index profile (fit on the even-index straightness legal games) never saw a
scored trajectory, and t0 inside T0_RANGE -- which here is (14, 34), not that module's (16, 34).

The wider floor is so this study's population figures scan the same move indices as
`move_legality.detection_auroc`'s bar chart, which they are read beside. They still cannot
match it exactly: that chart reads one position's activations, so its range is the tree's whole
[10, 34], while this score is a MOSUM over M_MAX = 12 steps either side of a candidate and does
not exist below t0 = 14. `branch_geometry`'s own range is left alone -- `changepoint_localization`
verifies byte-identical against its artifacts.

Writes study_dir("divergence")/illegal_move_detector/{d,meta}.npz.

Usage: PYTHONPATH=src python analysis/violation_divergence/illegal_move_detector/extract_geometry.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.straightness.geometry import temporal_geometry
from spade.violation_divergence.branches import complete_games

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "curvature_discriminant"))
from branch_geometry import (  # noqa: E402  (sibling script, not an installed package)
    GAME_BATCH, straightness_legal_index)

OUT = study_dir("divergence") / "illegal_move_detector"
T0_RANGE = (14, 34)   # the floor is the MOSUM's, see above


def eligible(meta: dict, legal_index: np.ndarray) -> list[int]:
    """Branch-complete, held out of the legal profile, and scannable at its own t0."""
    t0 = {int(g): int(t) for g, t in zip(meta["game_id"], meta["t0"])}
    return [g for g in complete_games(meta)
            if legal_index[g] % 2 == 1                       # -1 (absent) also passes
            and T0_RANGE[0] <= t0[g] <= T0_RANGE[1]]


def main() -> None:
    device = select_device()
    branches = study_dir("divergence") / "branches" / "branches.pt"
    meta = torch.load(branches, weights_only=True)
    legal_index = straightness_legal_index(meta)
    games = eligible(meta, legal_index)
    rows = torch.cat([(meta["game_id"] == g).nonzero().flatten() for g in games])
    print(f"{len(games)} games, {len(rows)} branch rows", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    # The straightness legal game each scored game IS, so the profile can exclude them: a
    # trajectory must not contribute to the baseline it is scored against.
    np.savez(OUT / "meta.npz", row=rows.numpy(),
             held_out_legal=legal_index[np.array(games)],
             **{k: meta[k][rows].numpy() for k in ("game_id", "t0", "category", "is_control")})

    acts = torch.load(study_dir("divergence") / "acts" / "resid_post_5.pt",
                      weights_only=True, mmap=True)
    out = []
    for start in range(0, len(rows), GAME_BATCH):
        z = acts[rows[start:start + GAME_BATCH]][:, 1:, :].float().to(device)  # drop the sink
        out.append(temporal_geometry(z)[0].cpu().numpy())
    d = np.concatenate(out).astype(np.float32)
    np.savez(OUT / "d.npz", d=d)
    print(f"wrote {OUT}/d.npz  d {d.shape}")


if __name__ == "__main__":
    main()
