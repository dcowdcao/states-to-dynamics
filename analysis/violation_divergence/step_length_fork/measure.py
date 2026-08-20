"""The step length at the fork over a population of source games, not over one.

`make_figures.py` without `--n_games` draws `s_b(t) / s_ref(t)` for one selected
game. This is the same quantity over the first 100 branch-complete games forking
at t0 = 20, so the figure that reads this npz can put an interval over games on
every curve. The reduction happens inside the loop: a game's features are turned
into 16 step lengths and dropped, so the sweep never holds a field for all games.

`--variant rollout` runs the same sweep over the tree where every branch -- the
reference included -- samples its own continuation from OthelloGPT rather than
replaying the source game's, and writes under that tree. A branch is K = 4 rows
there, so every row is encoded and turned into a step length on its own and the
mean over K is taken last: the encode is nonlinear, and so is the ratio.

**The ratio is formed within a rollout index, then averaged over K.** A branch's
k-th rollout and the reference's k-th are driven by the same uniform stream
(`rollout_branches`), so s_b^k / s_ref^k holds the sampled suffix fixed and only
the branch token varies; mean-of-ratios is the mean of 4 such paired
measurements. Ratio-of-means would divide two independently averaged lengths and
is a different quantity -- and the one that cannot keep the shared prefix at
exactly 1 branch by branch.

    $SPADE_DIVERGENCE_DIR[/rollout]/step_length_fork/n{N}.npz
      game        [G]           source game ids, ascending
      n_pool      scalar        branch-complete games forking at t0, before the
                                first G were taken
      space       [S]           fork_plane.spaces() order, raw layer 5 first
      role        [B]           population.ROLES order, both legal controls in
      category    [B] / is_control [B]
      t0          scalar        20
      pos         [T]           the positions a step is attributed to, positions(t0)[1:]
      n_rollouts  scalar        K, rows per branch, which `step` and `ratio` are
                                the mean over; 1 on the spliced tree
      step        [S, G, B, T]  ||z_b(t) - z_b(t - 2)|| in that space's own units
      ratio       [S, G, B, T]  that / the same game's reference branch's at the
                                same position -- what the panels draw
      spread_game [S, G]        rule_groups.spread_ratio on the drawn ratio over
                                t >= t0, per game

Usage: python analysis/violation_divergence/step_length_fork/measure.py
       [--n_games 100] [--variant spliced|rollout]
"""

import argparse
import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import POPULATION_T0, games_at_t0
from spade.violation_divergence.fork_plane import positions, spaces
from spade.violation_divergence.population import (
    REF, ROLES, encode, encoders, rows_at)
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

VARIANTS = {"spliced": "", "rollout": "rollout"}
POS = positions(POPULATION_T0)   # 12, 14, ..., 44; a step needs the position before it
DRAWN = POS[1:]                  # a step is attributed to the position it ends on
CHUNK = 5                        # rows-worth of games per encode call: a rollout
                                 # branch is K rows, so K games' features at once
                                 # is what the shared 8 GB cgroup affords
# `spread_ratio` and `rule_rows` read a branch's taxonomy off its row, and in
# ROLES order those are constants rather than a per-game lookup.
ROWS = ([{"category": 0, "is_control": k} for k in range(3)]
        + [{"category": c, "is_control": 0} for c in range(1, len(CATEGORIES))])


def steps(F: np.ndarray, free: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """([c, B, T], [c, B, T]) for a chunk of games: each branch's step length over
    the sampled positions, and that length divided by its own game's reference
    branch's at the same position, both averaged over the branch's K rollouts.

    `F` is [c * B * K, len(POS), dim] with the K rollouts of a branch contiguous.
    The division happens before the mean over K and pairs a rollout with the
    reference's rollout of the same index, which is the one drawn on the same
    uniform stream. K = 1 is the spliced tree and the mean is a no-op.

    A step is branch-free exactly when the position it ends on precedes t0 -- and
    there every row is bitwise equal, which is what the assert checks. It is an
    exact test on purpose: a K reshape that crossed branches would fail it here
    rather than shift a curve nobody can check.
    """
    d = np.linalg.norm(F[:, :, 1:] - F[:, :, :-1], axis=-1)
    s = d.reshape(d.shape[0], -1, K, d.shape[-1])
    ratio = s / s[:, REF:REF + 1]
    assert (ratio[:, :, :, free] == 1).all(), \
        "branches do not step identically over the shared prefix"
    return s.mean(axis=2), ratio.mean(axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=100)
    ap.add_argument("--variant", default="spliced", choices=list(VARIANTS))
    args = ap.parse_args()

    base = variant_dir("divergence", variant=VARIANTS[args.variant])
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1
    pool = games_at_t0(meta, POPULATION_T0)
    games = pool[:args.n_games]
    assert len(games) == args.n_games, f"only {len(pool)} games fork at t0 = {POPULATION_T0}"
    rows = torch.stack([rows_at(meta, g, K) for g in games])

    # mmap: the activation tensor is 2 GB spliced and 8.3 GB rollout, and the
    # population is 900 of its rows (3600 at K = 4). Those rows at the sampled
    # positions are 31 MB (125 MB at K = 4), and every space reads them.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    Z = acts[rows.flatten()][:, POS, :].float()

    G, S, B, T = len(games), len(spaces()), len(ROLES), len(DRAWN)
    chunk = max(1, CHUNK // K)
    free = DRAWN < POPULATION_T0
    post = DRAWN >= POPULATION_T0
    z = {"step": np.zeros((S, G, B, T), np.float32),
         "ratio": np.zeros((S, G, B, T), np.float32),
         "spread_game": np.zeros((S, G))}

    names, started = [], time.time()
    for si, (name, enc) in enumerate(encoders(select_device())):
        names.append(name)
        for lo in range(0, G, chunk):
            X = Z[lo * B * K:(lo + chunk) * B * K]
            F = X.numpy() if enc is None else encode(X, enc)
            s, ratio = steps(F.reshape(-1, B * K, len(POS), F.shape[-1]), free, K)
            z["step"][si, lo:lo + chunk] = s
            z["ratio"][si, lo:lo + chunk] = ratio
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
        n_rollouts=K,
        category=[r["category"] for r in ROWS], is_control=[r["is_control"] for r in ROWS],
        t0=POPULATION_T0, pos=DRAWN, **z)
    print(f"wrote {path}: {G} games x {S} spaces, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
