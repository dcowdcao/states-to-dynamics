"""How far each branch runs from its reference at every offset r, in the plane the
fork figures draw and in the full space that plane was fit in, for all nine
representation spaces and every branch-complete source game.

`../fork_and_board`'s panel (c) is this quantity averaged over r >= 0, for one
game. Everything here is that panel resolved along the event-locked axis and run
over the population, so the figure it feeds carries an interval over games.

Per game and per space the plane is the uncentred top-2 PCA of the displacement
D(t) = z_b(t) - z_ref(t), refit exactly as `fork_plane.pca2` fits it, over the
same STRIDE=2 positions at t0's parity. Both distances then come out of one Gram
matrix G = D D^T: the full-space distance is sqrt(diag G), and the in-plane one
is the row norm of the top-2 eigenvectors scaled by their eigenvalues. Neither
needs the 8192-wide basis, which is what makes 1553 games x 9 spaces cheap.

Written raw, in each space's own units. Normalising by the game's legal control
is the figure's job, because the raw scales are the thing the README has to
quote: they differ ~585x across the grid and are not a quantity nine panels
share.

    $DIR[/rollout]/divergence_by_space/curves.npz
      space  [S]        panel labels, fork_plane.spaces() order
      window [T]        offsets r, stride 2
      role   [B]        branch roles, ROLES order
      game   [G]        source game ids
      plane  [S,G,B,T]  in-plane distance from the reference
      full   [S,G,B,T]  same-space distance from the reference

Usage: python analysis/violation_divergence/divergence_by_space/measure.py
       [--variant spliced|rollout] [--n_games N]
"""

import argparse
import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import complete_games, load_game, role
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.fork_plane import positions, spaces

VARIANTS = {"spliced": "", "rollout": "rollout"}
# The eight branches every fork figure draws: control_legal_2 is the null *for*
# the control and is dropped there, so dropping it here keeps the plane the same
# plane. Order is fixed, so a game is one slice of a [G, B, T] array.
ROLES = ["reference", "control_legal"] + CATEGORIES[1:]
CONTROL = "control_legal"
# t0 in [10, 34] and the window in [-8, 24], so `positions` never clips: every
# game contributes to every offset and n is constant across the x-axis.
WINDOW = np.arange(-8, 25, 2)


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


def encoders(device: str) -> list[tuple[str, object]]:
    """(label, encode or None) per grid cell, built once.

    `fork_plane.space_features` rebuilds the encoder on every call, which costs a
    checkpoint load per game; a population sweep cannot pay that 1553 times.
    """
    return [(label, None if e is None else sae_encoder(e, device))
            for label, e in spaces()]


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
    ap.add_argument("--variant", default="spliced", choices=list(VARIANTS))
    ap.add_argument("--n_games", type=int, default=None, help="cap the sweep")
    args = ap.parse_args()

    device = select_device()
    base = variant_dir("divergence", variant=VARIANTS[args.variant])
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # mmap: the rollout tensor is 4.2 GB and is read eight branches at a time.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1
    games = complete_games(meta)[:args.n_games]
    cells = encoders(device)
    ref = ROLES.index("reference")

    plane = np.full((len(cells), len(games), len(ROLES), len(WINDOW)), np.nan, np.float32)
    full = np.full_like(plane, np.nan)
    started = time.time()
    for gi, gid in enumerate(games):
        g = load_game(meta, gid)
        pos = positions(g.t0)
        assert np.array_equal(pos - g.t0, WINDOW), f"game {gid}: window {pos - g.t0}"
        Z = acts[g.idx[game_rows(g, K)]][:, pos, :].float()
        for si, (_, encode) in enumerate(cells):
            F = Z.numpy() if encode is None else features(Z, encode)
            plane[si, gi], full[si, gi] = distances(paired_mean(F, ref, K))
        if gi % 100 == 0:
            print(f"{gi}/{len(games)} games, {time.time() - started:.0f}s", flush=True)

    # The prefix is bitwise identical across branches, so the full-space distance
    # is exactly 0 there. The in-plane one is an eigenvector component of a matrix
    # whose corresponding row is zero: zero in exact arithmetic, and reported here
    # as a share of the post-branch scale so a real leak could not hide in it.
    assert (full[..., WINDOW < 0] == 0).all(), "the shared prefix is not exactly zero"
    scale = full[..., WINDOW >= 0].mean()
    print(f"in-plane prefix leak: {np.abs(plane[..., WINDOW < 0]).max() / scale:.2e} "
          "of the mean post-branch distance (float noise; the full space gives exact 0)")
    control = full[:, :, ROLES.index(CONTROL)][..., WINDOW >= 0].mean(axis=(1, 2))
    print("raw full-space control distance per space (the ~585x the figure divides out):")
    for (name, _), c in zip(cells, control):
        print(f"  {name:<32} {c:.4g}")

    out = base / "divergence_by_space"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "curves.npz", space=[name for name, _ in cells],
                        window=WINDOW.astype(np.int16), role=ROLES,
                        game=np.array(games), plane=plane, full=full)
    print(f"wrote {out / 'curves.npz'}: {len(games)} games, {len(cells)} spaces")


if __name__ == "__main__":
    main()
