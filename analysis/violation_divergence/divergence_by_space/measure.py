"""How far each branch runs from its reference at every offset r, in the plane the
fork figures draw and in the full space that plane was fit in, for all nine
representation spaces and every branch-complete source game.

`../fork_and_board`'s panel (c) is this quantity averaged over r >= 0, for one
game. Everything here is that panel resolved along the event-locked axis and run
over the population, so the figure it feeds carries an interval over games.

`--plane` picks the space the reference is subtracted in, and nothing else.

`displacement` is the default: per game and per space the plane is the uncentred
top-2 PCA of D(t) = z_b(t) - z_ref(t), refit exactly as `fork_plane.pca2` fits
it, over the same STRIDE=2 positions at t0's parity. Both distances then come out
of one Gram matrix G = D D^T: the full-space distance is sqrt(diag G), and the
in-plane one is the row norm of the top-2 eigenvectors scaled by their
eigenvalues. Neither needs the 8192-wide basis, which is what makes 1553 games x
9 spaces cheap.

`r21` routes each branch through ReStraV's 21 components first --
dphi(t) = phi(z_b(t)) - phi(z_ref(t)), phi the W = 8 trailing window of
`curvature_plane.windowed` -- and takes both distances there, in the plane
`curvature_plane.fit_plane` fits per game. phi is nonlinear, so it is applied per
branch and differenced afterwards; its trailing window costs the first two
offsets, so the axis is r = -4 ... +24, 15 offsets rather than 17.

Written raw, in each space's own units -- except in R^21, whose 21 columns mix
degrees with activation-scale lengths and are column-scaled per game before any
distance is taken. Normalising by the game's legal control is the figure's job,
because the raw scales are the thing the README has to quote: they differ ~585x
across the grid and are not a quantity nine panels share.

    $DIR[/rollout]/divergence_by_space/curves[_r21][_nN].npz
      space  [S]        panel labels, fork_plane.spaces() order
      window [T]        offsets r, stride 2
      role   [B]        branch roles, ROLES order
      game   [G]        source game ids
      plane  [S,G,B,T]  in-plane distance from the reference
      full   [S,G,B,T]  same-space distance from the reference

`--l0-matched` swaps the eight SAE cells for `l0_operating_point`'s picks -- one
checkpoint per type at the shared L0 -- and writes `curves..._l0_matched.npz`
beside the canonical file. Raw layer 5 is not an SAE and stays the reference cell.

Usage: python analysis/violation_divergence/divergence_by_space/measure.py
       [--plane displacement|r21] [--variant spliced|rollout] [--n_games N]
       [--l0-matched]
"""

import argparse
import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import (
    POPULATION_T0, complete_games, games_at_t0, load_game, role)
from spade.violation_divergence.curvature_plane import W, fit_plane, windowed
from spade.violation_divergence.fork_plane import encoder, positions, spaces
from spade.violation_divergence.rule_groups import spread_ratio

VARIANTS = {"spliced": "", "rollout": "rollout"}
# The eight branches every fork figure draws: control_legal_2 is the null *for*
# the control and is dropped there, so dropping it here keeps the plane the same
# plane. Order is fixed, so a game is one slice of a [G, B, T] array.
ROLES = ["reference", "control_legal"] + CATEGORIES[1:]
CONTROL = "control_legal"
# t0 in [10, 34] and the window in [-8, 24], so `positions` never clips: every
# game contributes to every offset and n is constant across the x-axis.
WINDOW = np.arange(-8, 25, 2)
# ROLES as `rule_groups` reads them: it looks at the illegal rows only, but it
# indexes the array this sweep builds, so the two orders have to be the same one.
ROLE_ROWS = ([{"category": 0, "is_control": 0}, {"category": 0, "is_control": 1}]
             + [{"category": i, "is_control": 0} for i in range(1, len(CATEGORIES))])
# The units the run reports its control distances in. R^21 is column-scaled per
# game, so the ~585x that separates the nine cells is already out of it there.
CONTROL_NOTE = {
    "displacement": "raw full-space control distance per space (the ~585x the figure "
                    "divides out):",
    "r21": "column-scaled R^21 control distance per space (unit-variance columns per "
           "game, so the ~585x is already out):",
}


def sampling(plane: str, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """(sampled positions to read, offsets r kept in the output).

    R^21's descriptor covers the W samples ENDING at t, so it reads W-1 sampled
    positions of context before the study window and leaves the first W-1 read
    positions undefined. `positions` cannot always supply that much context: at
    t0 = 20 two of the seven fall off the front of the board, so the axis that
    survives is r = -4 ... +24 rather than the study window's -8 ... +24.
    """
    if plane == "displacement":
        pos = positions(t0)
        assert np.array_equal(pos - t0, WINDOW), f"t0 {t0}: window {pos - t0}"
        return pos, WINDOW
    pos = positions(t0, W - 1)
    return pos, pos[W - 1:] - t0


def paired_mean(X: np.ndarray, ref: int, K: int) -> np.ndarray:
    """[n*K, ...] -> [n, ...]: each row minus the reference drawn on the same
    rollout, averaged over the branch's K rollouts. K = 1 is X - X[ref].

    Copied in behaviour from ../fork_and_board, deliberately: it is the reduction
    the fork panels draw, so the plane fit here is the plane drawn there. It
    averages displacement *vectors*, not the scalar δ̂ that `make_figures.py`'s
    `reduce_k` averages one directory up, so the incoherent part of a sampled
    suffix cancels here and does not there. README.md carries the size of that.
    """
    Y = X.reshape(-1, K, *X.shape[1:])
    return (Y - Y[ref]).mean(axis=1)


def distances(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """([n, T] in-plane, [n, T] full-space) distance from the reference.

    One Gram matrix serves both: the plane is the uncentred top-2 PCA of D over
    every branch and position, so the in-plane coordinates are the leading
    eigenvectors of D D^T scaled by sqrt of their eigenvalues, and the full-space
    distance is the diagonal. The reference's row of D is exactly zero, so
    distance-from-the-reference and distance-from-the-origin coincide and
    `branch_panels.plane_displacement`'s rule is what this computes.
    """
    n, T = D.shape[:2]
    X = D.reshape(n * T, -1)
    G = X @ X.T
    val, vec = np.linalg.eigh(G)
    plane = np.sqrt(np.maximum(vec[:, -2:] ** 2 * val[-2:], 0).sum(1))
    return plane.reshape(n, T), np.sqrt(np.maximum(np.diag(G), 0)).reshape(n, T)


def curvature_distances(F: np.ndarray, ref: int, K: int,
                        n_branch_free: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """([n, T] in-plane, [n, T] full-space, [n, T, 21] the field both are taken in).

    The same two distances as `distances`, with the reference subtracted in R^21
    instead: dphi(t) = phi(z_b(t)) - phi(z_ref(t)). phi is nonlinear, so it is
    applied per branch and differenced afterwards -- and the K rollouts are
    averaged after it, never before -- because phi of a mean is the curvature of a
    path no branch takes and phi of D is undefined over the shared prefix, where D
    is a zero-length step.

    Both distances are taken in the COLUMN-SCALED field, not the raw one: R^21
    mixes degrees with activation-scale lengths whose variances differ ~450x
    across dictionaries, so a raw norm reports whichever component carries the
    biggest units. The scaling and the uncentred top-2 plane are
    `curvature_plane.fit_plane`'s, so a distance here is a distance in the plane
    `../fork_and_board` and `../fork_population` draw.

    Not the Gram route: `windowed` leaves NaN wherever a descriptor window runs
    off the board, and one NaN poisons every entry of G = D D^T, while `fit_plane`
    fits the finite rows and projects the rest. 21 columns make it moot anyway --
    the trick exists for the 8192-wide bases.
    """
    dphi = paired_mean(windowed(F, W), ref, K)[:, W - 1:]
    assert not dphi[ref].any() and not dphi[:, :n_branch_free].any(), \
        "the reference row or the shared prefix does not collapse onto the origin"
    p = fit_plane(dphi.reshape(-1, dphi.shape[-1]), centred=False)
    scaled = dphi / p.scale
    return np.linalg.norm(p.project(dphi), axis=-1), np.linalg.norm(scaled, axis=-1), scaled


def encoders(device: str, l0_matched: bool) -> list[tuple[str, object]]:
    """(label, encode or None) per grid cell, built once.

    `fork_plane.space_features` rebuilds the encoder on every call, which costs a
    checkpoint load per game; a population sweep cannot pay that 1553 times.
    """
    return [(label, encoder(e, device)) for label, e in spaces(l0_matched)]


def features(Z: torch.Tensor, encode) -> np.ndarray:
    with torch.no_grad():
        f = encode(Z.reshape(-1, Z.shape[2]))
    return f.reshape(Z.shape[0], Z.shape[1], -1).cpu().numpy()


def game_rows(g, K: int) -> list[int]:
    """Row indices of the eight ROLES branches, in ROLES order, K rows each."""
    by_role = {}
    for i, r in enumerate(g.rows):
        by_role.setdefault(role(r["category"], r["is_control"]), []).append(i)
    picked = [by_role[name] for name in ROLES]
    assert all(len(rows) == K for rows in picked), f"game {g.game} is not branch-complete"
    return [i for rows in picked for i in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plane", default="displacement", choices=["displacement", "r21"],
                    help="the space the reference is subtracted in")
    ap.add_argument("--variant", default="spliced", choices=list(VARIANTS))
    ap.add_argument("--n_games", type=int, default=None,
                    help=f"the shared n-game population at t0 = {POPULATION_T0}; "
                         "the whole branch-complete tree when omitted")
    ap.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                    help="seat the eight SAE cells at the shared l0_operating_point "
                         "target instead of at each type's own default sparsity")
    args = ap.parse_args()

    device = select_device()
    base = variant_dir("divergence", variant=VARIANTS[args.variant])
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # mmap: the rollout tensor is 4.2 GB and is read eight branches at a time.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1
    games = (games_at_t0(meta, POPULATION_T0, args.n_games) if args.n_games
             else complete_games(meta))
    cells = encoders(device, args.l0_matched)
    ref = ROLES.index("reference")

    window = sampling(args.plane, load_game(meta, games[0]).t0)[1]
    # Every kept r < 0 is an offset whose whole descriptor window ends before t0,
    # so it is branch-free and dphi there is exactly zero -- the assertion below
    # is what pins where the branch actually is once R^21 smears it over W steps.
    n_branch_free = int((window < 0).sum())
    plane = np.full((len(cells), len(games), len(ROLES), len(window)), np.nan, np.float32)
    full = np.full_like(plane, np.nan)
    spread = np.full((len(cells), len(games)), np.nan, np.float32)
    started = time.time()
    for gi, gid in enumerate(games):
        g = load_game(meta, gid)
        pos, w = sampling(args.plane, g.t0)
        assert np.array_equal(w, window), f"game {gid}: window {w}"
        Z = acts[g.idx[game_rows(g, K)]][:, pos, :].float()
        for si, (_, encode) in enumerate(cells):
            F = Z.numpy() if encode is None else features(Z, encode)
            if args.plane == "displacement":
                D = paired_mean(F, ref, K)
                plane[si, gi], full[si, gi] = distances(D)
            else:
                plane[si, gi], full[si, gi], D = curvature_distances(
                    F, ref, K, n_branch_free)
            spread[si, gi] = spread_ratio(D, ROLE_ROWS)
        if gi % 100 == 0:
            print(f"{gi}/{len(games)} games, {time.time() - started:.0f}s", flush=True)

    # The prefix is bitwise identical across branches, so the full-space distance
    # is exactly 0 there. The in-plane one is an eigenvector component of a matrix
    # whose corresponding row is zero: zero in exact arithmetic, and reported here
    # as a share of the post-branch scale so a real leak could not hide in it.
    assert (full[..., window < 0] == 0).all(), "the shared prefix is not exactly zero"
    scale = full[..., window >= 0].mean()
    print(f"in-plane prefix leak: {np.abs(plane[..., window < 0]).max() / scale:.2e} "
          "of the mean post-branch distance (float noise; the full space gives exact 0)")
    control = full[:, :, ROLES.index(CONTROL)][..., window >= 0].mean(axis=(1, 2))
    print(CONTROL_NOTE[args.plane])
    for (name, _), c in zip(cells, control):
        print(f"  {name:<32} {c:.4g}")

    out = base / "divergence_by_space"
    out.mkdir(parents=True, exist_ok=True)
    # The n-game sweep is a different population, not a shorter run of the same one:
    # it writes beside the whole-tree curves rather than over them.
    tag = "" if args.plane == "displacement" else f"_{args.plane}"
    name = (f"curves{tag}" + (f"_n{args.n_games}" if args.n_games else "")
            + ("_l0_matched" if args.l0_matched else "") + ".npz")
    np.savez_compressed(out / name, space=[name for name, _ in cells],
                        window=window.astype(np.int16), role=ROLES,
                        game=np.array(games), plane=plane, full=full, spread=spread)
    print(f"wrote {out / name}: {len(games)} games, {len(cells)} spaces")


if __name__ == "__main__":
    main()
