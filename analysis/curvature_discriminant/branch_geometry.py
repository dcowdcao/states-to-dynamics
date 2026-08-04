"""Step 1: trajectory geometry of every divergence branch, in all nine spaces.

The straightness study ships d/theta for its own 10000 games; the fork figure
needs the same two arrays for the divergence branches, in the same indexing
(position 0 is the attention sink and is dropped, so d[i] is the step from
position i+1 to i+2 and theta[k] pivots at position k+2).

Games kept: those carrying all nine branch types, with t0 in [16, 34] so the
study's window r in [-8, 24] and the trailing 8-ply descriptor window both fit
inside positions 1..58, and whose straightness legal-game index is odd or absent
-- plus PLOT_GAME whatever its parity, since the figure's game is chosen for the
figure and not for its index.

The mapping is the honesty guard, and it needs the real thing, not a guess: both
studies draw legal games from the same seeded stream, so game_id k IS
straightness game k -- until divergence skips a prefix at k=696 and the two
indices shift apart. Every branch's reference tokens are therefore looked up in
the straightness token table, and `held_out_legal` is handed to fit_axis, which
drops exactly those games from the probe's training set.

Writes $SPADE_DIVERGENCE_DIR/curvature_discriminant/{meta,<space>}.npz.
Resumable: a space whose npz exists is skipped.

Usage: python analysis/curvature_discriminant/branch_geometry.py
"""

from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.straightness.build_games import CLASSES
from spade.straightness.geometry import temporal_geometry
from spade.violation_divergence.compute_divergence import sae_encoder

OUT_DIR = study_dir("divergence") / "curvature_discriminant"
T0_RANGE = (16, 34)
PLOT_GAME = 1235  # the game the figure draws; kept whatever its straightness parity
N_BRANCH_TYPES = len(CATEGORIES) + 2  # 6 illegal + reference + 2 legal controls
GAME_BATCH = 48


def straightness_legal_index(meta: dict) -> np.ndarray:
    """Straightness legal-game index of each divergence game_id, -1 if absent."""
    g = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    n = len(g["cls"]) // len(CLASSES)
    key = {tuple(row): i for i, row in enumerate(g["tokens"][:n].long().tolist())}
    is_ref = (meta["category"] == 0) & (meta["is_control"] == 0)
    out = np.full(int(meta["game_id"].max()) + 1, -1)
    for i in is_ref.nonzero().flatten().tolist():
        out[int(meta["game_id"][i])] = key.get(tuple(meta["tokens"][i].long().tolist()), -1)
    return out


def eval_games(meta: dict, legal_index: np.ndarray) -> list[int]:
    t0 = {int(g): int(t) for g, t in zip(meta["game_id"], meta["t0"])}
    keep = [g for g in meta["game_id"].unique().tolist()
        if (legal_index[g] % 2 == 1 or g == PLOT_GAME)  # -1 (absent) also passes
        and T0_RANGE[0] <= t0[g] <= T0_RANGE[1]
        and int((meta["game_id"] == g).sum()) == N_BRANCH_TYPES]
    assert PLOT_GAME in keep, f"plotted game {PLOT_GAME} is not branch-complete in range"
    return keep


def geometry(Z: torch.Tensor, encode, device: str) -> tuple[np.ndarray, np.ndarray]:
    ds, thetas = [], []
    for start in range(0, len(Z), GAME_BATCH):
        x = Z[start:start + GAME_BATCH].float().to(device)
        if encode is not None:
            with torch.no_grad():
                x = encode(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:2], -1)
        d, theta = temporal_geometry(x)
        ds.append(d.cpu().numpy())
        thetas.append(theta.cpu().numpy())
    return np.concatenate(ds).astype(np.float32), np.concatenate(thetas).astype(np.float32)


def main():
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    legal_index = straightness_legal_index(meta)
    games = eval_games(meta, legal_index)
    assert games, "no eligible games"

    rows = torch.cat([(meta["game_id"] == g).nonzero().flatten() for g in games])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "meta.npz",
             row=rows.numpy(),
             game_id=meta["game_id"][rows].numpy(),
             t0=meta["t0"][rows].numpy(),
             category=meta["category"][rows].numpy(),
             is_control=meta["is_control"][rows].numpy(),
             held_out_legal=legal_index[np.array(games)])
    print(f"{len(games)} games, {len(rows)} branches")

    spaces = [("resid_post_5", None)] + [(f"sae_{e.config}", e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    todo = [s for s in spaces if not (OUT_DIR / f"{s[0]}.npz").exists()]
    if not todo:
        print("all spaces present")
        return

    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)
    Z = acts[rows][:, 1:, :]  # drop the attention sink, as the straightness study does
    del acts
    for space, entry in todo:
        d, theta = geometry(Z, None if entry is None else sae_encoder(entry, device), device)
        np.savez(OUT_DIR / f"{space}.npz", d=d, theta=theta)
        print(f"{space}: d {d.shape} theta {theta.shape} "
              f"nan_theta {np.isnan(theta).mean():.2%}")


if __name__ == "__main__":
    main()
