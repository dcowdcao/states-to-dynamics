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

--variant rollout does the same over the resampled-continuation copy of the study,
where a branch is K=4 rows sharing every column but `tokens`. Nothing is reduced
here: d/theta are per row, and the mean over K is taken downstream, after the
descriptor (evaluate.fold_rollouts). The straightness mapping is still read from
the SPLICED branches.pt -- a rollout reference has resampled its own suffix, so
its tokens are no longer in the straightness table and the lookup would return -1
for every game, silently emptying the exclusion list the guard depends on.

Writes $SPADE_DIVERGENCE_DIR/[rollout/]curvature_discriminant/{meta,<space>}.npz.
Resumable: a space whose npz exists is skipped.

Usage: python analysis/violation_divergence/curvature_discriminant/branch_geometry.py
           [--variant {spliced,rollout}]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir, variant_dir
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.straightness.geometry import temporal_geometry
from spade.violation_divergence.branches import complete_games
from spade.violation_divergence.compute_divergence import sae_encoder

VARIANTS = {"spliced": "", "rollout": "rollout"}   # CLI name -> variant_dir() suffix
T0_RANGE = (16, 34)
PLOT_GAME = 1235  # the game the figure draws; kept whatever its straightness parity
GAME_BATCH = 48


def out_dir(variant: str = "") -> Path:
    return variant_dir("divergence", variant=variant) / "curvature_discriminant"


def straightness_legal_index(meta: dict) -> np.ndarray:
    """Straightness legal-game index of each divergence game_id, -1 if absent."""
    g = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    # Counted, not len // 5: the data release's games.pt holds the legal block alone.
    n = int((g["cls"] == 0).sum())
    assert (g["cls"][:n] == 0).all(), "the legal games are not the leading block of games.pt"
    key = {tuple(row): i for i, row in enumerate(g["tokens"][:n].long().tolist())}
    is_ref = (meta["category"] == 0) & (meta["is_control"] == 0)
    out = np.full(int(meta["game_id"].max()) + 1, -1)
    for i in is_ref.nonzero().flatten().tolist():
        out[int(meta["game_id"][i])] = key.get(tuple(meta["tokens"][i].long().tolist()), -1)
    return out


def eval_games(meta: dict, legal_index: np.ndarray) -> list[int]:
    t0 = {int(g): int(t) for g, t in zip(meta["game_id"], meta["t0"])}
    keep = [g for g in complete_games(meta)
        if (legal_index[g] % 2 == 1 or g == PLOT_GAME)  # -1 (absent) also passes
        and T0_RANGE[0] <= t0[g] <= T0_RANGE[1]]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="spliced")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    device = select_device()
    base = variant_dir("divergence", variant=variant)
    out = out_dir(variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # Always the spliced tokens: a rollout reference is not in the straightness table.
    source = meta if not variant else torch.load(
        study_dir("divergence") / "branches" / "branches.pt", weights_only=True)
    legal_index = straightness_legal_index(source)
    games = eval_games(meta, legal_index)
    assert games, "no eligible games"

    rows = torch.cat([(meta["game_id"] == g).nonzero().flatten() for g in games])
    out.mkdir(parents=True, exist_ok=True)
    cols = {"row": rows.numpy(),
            "game_id": meta["game_id"][rows].numpy(),
            "t0": meta["t0"][rows].numpy(),
            "category": meta["category"][rows].numpy(),
            "is_control": meta["is_control"][rows].numpy(),
            "held_out_legal": legal_index[np.array(games)]}
    if "rollout" in meta:
        cols["rollout"] = meta["rollout"][rows].numpy()
    np.savez(out / "meta.npz", **cols)
    print(f"{len(games)} games, {len(rows)} rows")

    spaces = [("resid_post_5", None)] + [(f"sae_{e.config}", e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    todo = [s for s in spaces if not (out / f"{s[0]}.npz").exists()]
    if not todo:
        print("all spaces present")
        return

    # mmap: the rollout activation file is 4 GB and only the eligible rows are read.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    Z = acts[rows][:, 1:, :]  # drop the attention sink, as the straightness study does
    del acts
    for space, entry in todo:
        d, theta = geometry(Z, None if entry is None else sae_encoder(entry, device), device)
        np.savez(out / f"{space}.npz", d=d, theta=theta)
        print(f"{space}: d {d.shape} theta {theta.shape} "
              f"nan_theta {np.isnan(theta).mean():.2%}")


if __name__ == "__main__":
    main()
