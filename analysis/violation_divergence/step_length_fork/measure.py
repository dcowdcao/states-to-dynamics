"""The step length at the fork over a population of source games, not over one.

`make_figures.py` without `--n_games` draws `s_b(t) / s_ref(t)` for one selected
game. This is the same quantity over the first 100 branch-complete games forking
at t0 = 20, so the figure that reads this npz can put an interval over games on
every curve. The reduction happens inside the loop: a game's features are turned
into 16 step lengths and dropped, so the sweep never holds a field for all games.

Spliced branches only. The rollout tree resamples each branch's suffix, so a step
length there also averages over which continuation was sampled.

    $SPADE_DIVERGENCE_DIR/step_length_fork/n{N}.npz
      game        [G]           source game ids, ascending
      n_pool      scalar        branch-complete games forking at t0, before the
                                first G were taken
      space       [S]           fork_plane.spaces() order, raw layer 5 first
      role        [B]           population.ROLES order, both legal controls in
      category    [B] / is_control [B]
      t0          scalar        20
      pos         [T]           the positions a step is attributed to, positions(t0)[1:]
      step        [S, G, B, T]  ||z_b(t) - z_b(t - 2)|| in that space's own units
      ratio       [S, G, B, T]  that / the same game's reference branch's at the
                                same position -- what the panels draw
      spread_game [S, G]        rule_groups.spread_ratio on the drawn ratio over
                                t >= t0, per game

Usage: python analysis/violation_divergence/step_length_fork/measure.py [--n_games 100]
"""

import argparse
import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import POPULATION_T0, games_at_t0
from spade.violation_divergence.fork_plane import positions, spaces
from spade.violation_divergence.population import (
    REF, ROLES, encode, encoders, rows_at)
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

POS = positions(POPULATION_T0)   # 12, 14, ..., 44; a step needs the position before it
DRAWN = POS[1:]                  # a step is attributed to the position it ends on
CHUNK = 5                        # games per encode call
# `spread_ratio` and `rule_rows` read a branch's taxonomy off its row, and in
# ROLES order those are constants rather than a per-game lookup.
ROWS = ([{"category": 0, "is_control": k} for k in range(3)]
        + [{"category": c, "is_control": 0} for c in range(1, len(CATEGORIES))])


def steps(F: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """([c, B, T], [c, B, T]) for a chunk of games: each branch's step length over
    the sampled positions, and that length divided by its own game's reference
    branch's at the same position.

    A step is branch-free exactly when the position it ends on precedes t0 -- and
    there the two rows are bitwise equal, which is what the assert checks.
    """
    s = np.linalg.norm(F[:, :, 1:] - F[:, :, :-1], axis=-1)
    ratio = s / s[:, REF:REF + 1]
    assert (ratio[:, :, free] == 1).all(), \
        "branches do not step identically over the shared prefix"
    return s, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=100)
    args = ap.parse_args()

    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    assert "rollout" not in meta, "spliced only: the rollout tree is K rows per branch"
    pool = games_at_t0(meta, POPULATION_T0)
    games = pool[:args.n_games]
    assert len(games) == args.n_games, f"only {len(pool)} games fork at t0 = {POPULATION_T0}"
    rows = torch.stack([rows_at(meta, g) for g in games])

    # mmap: the activation tensor is 2 GB and the population is 900 of its rows.
    # Those rows at the sampled positions are 31 MB and every space reads them.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    Z = acts[rows.flatten()][:, POS, :].float()

    G, S, B, T = len(games), len(spaces()), len(ROLES), len(DRAWN)
    free = DRAWN < POPULATION_T0
    post = DRAWN >= POPULATION_T0
    z = {"step": np.zeros((S, G, B, T), np.float32),
         "ratio": np.zeros((S, G, B, T), np.float32),
         "spread_game": np.zeros((S, G))}

    names, started = [], time.time()
    for si, (name, enc) in enumerate(encoders(select_device())):
        names.append(name)
        for lo in range(0, G, CHUNK):
            X = Z[lo * B:(lo + CHUNK) * B]
            F = X.numpy() if enc is None else encode(X, enc)
            s, ratio = steps(F.reshape(-1, B, len(POS), F.shape[-1]), free)
            z["step"][si, lo:lo + CHUNK] = s
            z["ratio"][si, lo:lo + CHUNK] = ratio
        for g in range(G):
            z["spread_game"][si, g] = spread_ratio(z["ratio"][si, g][:, post], ROWS)
        rule = z["ratio"][si][:, rule_rows(ROWS)["R3 turn"]].mean(axis=1)
        print(f"{name}: R3 {rule[:, DRAWN == POPULATION_T0].mean():.3f} at r = 0, "
              f"{rule[:, post].mean():.3f} over the window ({time.time() - started:.0f}s)",
              flush=True)

    assert all(np.isfinite(v).all() for v in z.values()), "a NaN reached the output"
    out = base / "step_length_fork"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"n{G}.npz"
    np.savez_compressed(
        path, game=np.array(games), n_pool=len(pool), space=names, role=ROLES,
        category=[r["category"] for r in ROWS], is_control=[r["is_control"] for r in ROWS],
        t0=POPULATION_T0, pos=DRAWN, **z)
    print(f"wrote {path}: {G} games x {S} spaces, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
