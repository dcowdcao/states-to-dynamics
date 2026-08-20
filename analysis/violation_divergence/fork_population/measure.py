"""The nine-space fork, averaged over 100 source games instead of drawn from one.

`../fork_and_board` is this figure for a selected game: an illustration of the
construction, n = 1, whose panels are cleaner than a typical fork's because the
game is the argmax of a separation criterion. Same nine spaces, same three planes,
same ratio-to-the-legal-control -- taken over a population, so the trajectory a
panel draws is a mean over games and the ratio under it carries an interval.

The population is the first `--n_games` branch-complete games forking at t0 = 20,
ascending by game id -- no draw, no criterion (1696 exist;
docs/finish_notes/t0-20-population.md). One fixed t0 is what makes the mean a
trajectory at all: every game then shares one `positions(20)`, so games align by
array index and the drawn axis stays move index with the branch at t = 20.

Everything nonlinear happens per game, before the mean: the SAE encode, the
difference from that game's own reference, and -- for the descriptor planes --
phi over the W = 8 window, applied to the trajectory and differenced afterwards.
phi of a mean over games would be the curvature of a path no game took, and phi
of the displacement is undefined: D is identically zero over the shared prefix,
where a zero-length step leaves theta undefined. `--plane d7` is `r21` with the
descriptor cut to its 7 raw step-length columns, the half `../r21_shared_basis`
finds disagreeing with the turning angles.

The loop is space-outer because the plane is not known until every game has been
seen: one pass over the space's games gives the mean field, the plane is fit on
that, and a second pass projects every game into THAT plane, so the CI drawn
around a mean marker and the bar under the panel measure the plane the panel
draws. Only the denominator is per game.

Two estimators of "how far the illegal branches run", and they are different
quantities: `ratio_plane`/`ratio_space` measure the distance OF the mean field,
which is what the drawn panel shows, while `game_plane`/`game_space` are the mean
OF each game's own distance, which is what carries a CI.

`--variant rollout` runs the same sweep over the rollout branch set, where every
branch -- the reference included -- samples its own continuation from OthelloGPT
instead of replaying the source game's. A branch is K rows there, so a mean over
games is also a mean over which continuation was sampled: `population.paired_mean`
does that reduction, paired on the rollout index because a rollout's reference is
the one drawn from the same uniform stream, and after the encode and phi because
both are nonlinear. It draws to ROLLOUT_RMAX and not to the study window's r = 24,
the window `../fork_and_board` set on this same branch set. The two variants read
and write different trees, so neither can overwrite the other's npz.

    $DIR[/rollout]/fork_population/{plane}_n{N}.npz
      game        [G]              source game ids, ascending
      n_pool      scalar           branch-complete games forking at t0, before the
                                   first G were taken
      space       [S]              fork_plane.spaces() order, raw layer 5 first
      role        [B]              population.ROLES order, control 2 included
      category    [B] / is_control [B]
      t0          scalar           20
      pos         [T]              the drawn sampled positions
      coords      [S, B, T, 2]     the mean field on the plane fit to it
      var         [S, 2]           that plane's explained variance ratios
      coherence   [S, B, T]        per branch: ||mean_g D|| / mean_g ||D||, how much
                                   of one game's displacement survives the mean
      coherence_rule [S, R, T]     the same for the three RULE MEAN paths, in
                                   rule_rows order -- the path a panel draws, which
                                   also cancels across a rule's member categories,
                                   so it is not the mean of its members' coherences
      ratio_plane [S, B]           distance of the mean field over r >= 0, in the
                                   plane, / the legal control's
      ratio_space [S, B]           the same in that cell's full space
      game_coords [S, G, B, T, 2]  each game's field in the POPULATION plane
      game_plane  [S, G, B]        per game, in that same plane, / its own control
      game_space  [S, G, B]        per game, in that cell's full space
      spread_mean [S]              rule_groups.spread_ratio on the mean field
      spread_game [S, G]           the same per game

Usage: python analysis/violation_divergence/fork_population/measure.py
       [--plane displacement|r21|d7] [--n_games 100]
"""

import argparse
import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.figures.branch_panels import plane_displacement
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import FIGURE_GAME, games_at_t0
from spade.violation_divergence.curvature_plane import (
    PLANE_COLUMNS, W, fit_plane, windowed)
from spade.violation_divergence.fork_plane import positions, spaces
from spade.violation_divergence.fork_r21 import top2
from spade.violation_divergence.population import (
    C1, C2, REF, ROLES, encode, encoders, paired_mean, rows_at)
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

T0 = 20                       # the one fork position the population is drawn from
FULL = positions(T0, W - 1)   # 2, 4, ..., 44: the study window plus descriptor context
N_CONTEXT = 5                 # FULL[N_CONTEXT:] is the study window itself
PLANES = ("displacement", "r21", "d7")
VARIANTS = {"spliced": "", "rollout": "rollout"}
# Past r = +8 the rollout set's illegal and control paths agree to within 0.5% and
# cross over by r = +13, so the extra offsets draw separation that is not there.
# `../fork_and_board` sets the same window over the same branch set.
ROLLOUT_RMAX = 8
CHUNK = 5                     # games per encode call
# `spread_ratio` and `rule_rows` read a branch's taxonomy off its row, and in
# ROLES order those are constants rather than a per-game lookup.
ROWS = ([{"category": 0, "is_control": k} for k in range(3)]
        + [{"category": c, "is_control": 0} for c in range(1, len(CATEGORIES))])
RULES = rule_rows(ROWS)


def window(plane: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """(the positions encoded, the positions drawn). A descriptor plane reads W-1
    sampled positions of context the study window does not carry, and the rollout
    set stops at ROLLOUT_RMAX."""
    full = FULL[FULL - T0 <= ROLLOUT_RMAX] if variant else FULL
    return full, full[N_CONTEXT if plane == "displacement" else W - 1:]


def fields(plane: str, acts: torch.Tensor, rows: torch.Tensor, enc,
           full: np.ndarray):
    """One game's field at a time, in `rows` order, over the drawn positions.

    The sweep encodes each space twice -- once for the mean field, once to place
    every game in the plane fit on it -- rather than holding the fields between
    the two passes: 100 games of the displacement plane are 500 MB per space, and
    the login node's 8 GB user cgroup does not have that beside the encoders.
    """
    B, K = rows.shape[1:]
    for lo in range(0, len(rows), CHUNK):
        Z = acts[rows[lo:lo + CHUNK].flatten()][:, full, :].float()
        F = Z.numpy() if enc is None else encode(Z, enc)
        yield from (field(plane, Fg, K)
                    for Fg in F.reshape(-1, B * K, len(full), F.shape[-1]))


def field(plane: str, F: np.ndarray, K: int) -> np.ndarray:
    """[B, T, k]: one game's displacement from its own reference over the drawn
    positions, reduced over the K rollouts of each branch.

    A descriptor plane windows the trajectory and differences afterwards, so the
    shared prefix collapses onto the origin instead of feeding phi a zero-length
    step; `PLANE_COLUMNS` then keeps all 21 of it or only the raw step lengths.
    The encode and phi are both nonlinear, so the K rollouts are reduced here,
    after them, and paired on the rollout index.
    """
    if plane == "displacement":
        return paired_mean(F[:, N_CONTEXT:], REF, K)
    dphi = paired_mean(windowed(F, W), REF, K)
    return dphi[:, W - 1:, PLANE_COLUMNS[plane]]


def population_plane(plane: str, M: np.ndarray):
    """(project, column scale, [2] variance ratios) of the uncentred top-2 plane of
    the mean field, built as `fork_and_board.project` builds a cell's plane.

    Uncentred in both cases: zero displacement is a real point, and centring would
    move the reference off the origin the panel draws it at. A descriptor is
    column-scaled first -- R^21 mixes degrees with activation-scale lengths, so a
    raw covariance reports units, and the raw step lengths differ ~580x in scale
    between raw layer 5 and SpaDE -- and `Plane.project` keeps an undefined row NaN.
    """
    flat = M.reshape(-1, M.shape[-1])
    if plane == "displacement":
        basis, var = top2(flat)
        return (lambda X: X @ basis.T), np.float32(1), var
    p = fit_plane(flat, centred=False)
    return p.project, p.scale, p.var


def coherence(mean_field: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """||mean_g D|| / mean_g ||D||, and 0 where the field is identically zero: the
    reference row and the shared prefix have no displacement to survive the mean."""
    mean_norm = np.nanmean(norms, axis=0)
    return np.divide(np.linalg.norm(mean_field, axis=-1), mean_norm,
                     out=np.zeros_like(mean_norm), where=mean_norm > 0)


def rule_paths(X: np.ndarray) -> np.ndarray:
    """[R, ...]: one mean path per rule, over its member categories."""
    return np.stack([X[m].mean(0) for m in RULES.values()])


def ratios(X: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """[B]: distance from the reference over the drawn r >= 0, / the control's."""
    d = plane_displacement(X, pos, T0, REF)
    return d / d[C1]


def half_width(x: np.ndarray) -> float:
    return 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x))


def report(names: list[str], games: list[int], pos: np.ndarray, z: dict) -> None:
    keep = [i for i, g in enumerate(games) if g != FIGURE_GAME]
    print(f"\ndistance from the reference over the drawn r >= 0, / the cell's legal "
          f"control\n  mean field plane|space, then the per-game full-space mean "
          f"±95% CI and median (n = {len(games)})")
    for s, name in enumerate(names):
        print(f"  {name:<12} " + "  ".join(
            f"{rule.split()[0]} {z['ratio_plane'][s, m].mean():4.2f}|"
            f"{z['ratio_space'][s, m].mean():4.2f} "
            f"g {z['game_space'][s][:, m].mean():5.3f}"
            f"±{half_width(z['game_space'][s][:, m].mean(1)):.3f} "
            f"md {np.median(z['game_space'][s][:, m].mean(1)):5.3f}"
            for rule, m in RULES.items())
            + f"  c2 {z['game_space'][s][:, C2].mean():5.3f}"
              f"±{half_width(z['game_space'][s][:, C2]):.3f}")

    if len(keep) < len(games):
        print(f"\nthe same per-game full-space means without game {FIGURE_GAME}, the "
              f"single-game figure's selected argmax (rank {games.index(FIGURE_GAME) + 1} "
              f"of {len(games)})")
        for s, name in enumerate(names):
            print(f"  {name:<12} " + "  ".join(
                f"{rule.split()[0]} {z['game_space'][s][keep][:, m].mean():5.3f}"
                f"±{half_width(z['game_space'][s][keep][:, m].mean(1)):.3f}"
                for rule, m in RULES.items()))

    print("\nwithin/between rule spread ratio: per game median [IQR] | mean field")
    for s, name in enumerate(names):
        q = np.percentile(z["spread_game"][s], [25, 50, 75])
        print(f"  {name:<12} {q[1]:.3f} [{q[0]:.3f}-{q[2]:.3f}] | {z['spread_mean'][s]:.3f}")

    rule = next(iter(RULES))
    at = [int(np.flatnonzero(pos - T0 == r)[0]) for r in (0, pos[-1] - T0)]
    print(f"\ncoherence ||mean D|| / mean ||D||, the {rule} mean path | the legal "
          f"control (1/sqrt(n) = {1 / np.sqrt(len(games)):.3f} is cancellation)")
    for s, name in enumerate(names):
        print(f"  {name:<12} " + "  ".join(
            f"r{pos[t] - T0:>3} {z['coherence_rule'][s, 0, t]:.3f} | "
            f"{z['coherence'][s, C1, t]:.3f}" for t in at))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plane", default="displacement", choices=PLANES)
    ap.add_argument("--n_games", type=int, default=100)
    ap.add_argument("--variant", default="spliced", choices=list(VARIANTS),
                    help="rollout: every branch samples its own continuation, K rows "
                         "per branch, paired on the rollout index and reduced after "
                         "the encode and phi")
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    base = variant_dir("divergence", variant=variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1
    pool = games_at_t0(meta, T0)
    games = pool[:args.n_games]
    assert len(games) == args.n_games, f"only {len(pool)} games fork at t0 = {T0}"
    full, pos = window(args.plane, variant)
    post = pos >= T0
    # At t0 = 20 the sampled geometry is fixed, so it is asserted and not discovered.
    assert full[0] == 2 and full[9] == T0, "the sampled positions moved off t0 = 20"
    assert (full[N_CONTEXT:] == positions(T0)[:len(full) - N_CONTEXT]).all(), \
        "N_CONTEXT does not line the displacement window up with the study window"
    assert pos[0] < T0, "no branch-free offset is drawn, so the prefix check is vacuous"

    # mmap: the rollout activation tensor is 8.3 GB (spliced 2.1) and one game is
    # K * 9 of its rows.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    rows = torch.stack([rows_at(meta, g, K).reshape(len(ROLES), K) for g in games])
    # The reshape already catches a wrong row count; what it cannot catch is a
    # k-major order, which `paired_mean` would silently pair across branches.
    if "rollout" in meta:
        assert (meta["rollout"][rows] == torch.arange(K)).all(), \
            "rows are k-major: paired_mean would pair a branch with another rollout"
    names = []
    G, S, B, T = len(games), len(spaces()), len(ROLES), len(pos)
    z = {"coords": np.zeros((S, B, T, 2)), "var": np.zeros((S, 2)),
         "coherence": np.zeros((S, B, T)), "coherence_rule": np.zeros((S, len(RULES), T)),
         "ratio_plane": np.zeros((S, B)),
         "ratio_space": np.zeros((S, B)), "game_coords": np.zeros((S, G, B, T, 2), np.float32),
         "game_plane": np.zeros((S, G, B)), "game_space": np.zeros((S, G, B)),
         "spread_mean": np.zeros(S), "spread_game": np.zeros((S, G))}

    started = time.time()
    for s, (name, enc) in enumerate(encoders(select_device())):
        names.append(name)
        total, seen = None, None
        for g, x in enumerate(fields(args.plane, acts, rows, enc, full)):
            assert not x[REF].any() and not x[:, ~post].any(), \
                f"game {games[g]}, {name}: the reference or the prefix is not zero"
            if total is None:
                total, seen = np.zeros(x.shape), np.zeros(x.shape, np.int16)
            ok = np.isfinite(x)          # a streaming nanmean, in float64
            total += np.where(ok, x, 0)
            seen += ok

        M = total / seen
        project, scale, z["var"][s] = population_plane(args.plane, M)
        space = M / scale
        z["coords"][s] = project(M)
        z["ratio_plane"][s] = ratios(z["coords"][s], pos)
        z["ratio_space"][s] = ratios(space, pos)
        z["spread_mean"][s] = spread_ratio(space[:, post], ROWS)
        norm, rule_norm = np.empty((G, B, T)), np.empty((G, len(RULES), T))
        for g, x in enumerate(fields(args.plane, acts, rows, enc, full)):
            Xg = x / scale
            z["game_coords"][s, g] = project(x)
            z["game_plane"][s, g] = ratios(z["game_coords"][s, g], pos)
            z["game_space"][s, g] = ratios(Xg, pos)
            z["spread_game"][s, g] = spread_ratio(Xg[:, post], ROWS)
            norm[g] = np.linalg.norm(Xg, axis=-1)
            rule_norm[g] = np.linalg.norm(rule_paths(Xg), axis=-1)
        z["coherence"][s] = coherence(space, norm)
        z["coherence_rule"][s] = coherence(rule_paths(space), rule_norm)
        print(f"{name}: {time.time() - started:.0f}s", flush=True)

    assert max(z["coherence"].max(), z["coherence_rule"].max()) <= 1 + 1e-9, \
        "coherence above 1"
    assert all(np.isfinite(v).all() for v in z.values()), "a NaN reached the output"
    out = base / "fork_population"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.plane}_n{G}.npz"
    np.savez_compressed(
        path, game=np.array(games), space=names, n_pool=len(pool), role=ROLES,
        category=[r["category"] for r in ROWS], is_control=[r["is_control"] for r in ROWS],
        t0=T0, pos=pos, **z)
    report([name.split(" (")[0] for name in names], games, pos, z)
    print(f"\nwrote {path}: {G} games x {S} spaces, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
