"""How far an illegal branch runs from the reference in R^21, over the study's
shared `--n_games` population.

`../r21_shared_basis/run_from_reference` is this quantity for one game, with no
interval: 8 non-reference branch trajectories are all one game has. Same
descriptor, same planes, same anchors, same ratio-to-the-legal-control -- run
over the population, so the figure it feeds carries a mean over games and its CI.

Everything is refit per game: the z-scoring, the per-space top-2 planes, the
shared plane pooled over that game's nine spaces, and the activation plane. A
plane fit across games would make the games one estimate rather than n of them,
and the interval would be a lie.

Spliced only. The rollout twin resamples every branch's suffix, which changes
where the mean over K is taken, and every anchor here is at r >= +6, where the
rollout study found illegal and control already saturated.

    $DIR/r21_run_population/runs.npz
      space    [S]         fork_plane.spaces() order, raw layer 5 first
      game     [G]         source game ids, the shared population at t0 = 20
      series   [P]         SERIES, in figure order
      ratio    [S, G, P]   run from the reference / the legal control's run
      category [6]         the illegal categories `member` is over
      member   [S, G, 6]   each illegal category alone, full R^21

`--l0-matched` sweeps the eight `l0_operating_point` picks instead of the seed-42
roster, whose L0s span 32.5 to 1,808, so a difference between two cells is the
architecture and not its sparsity. Same games, same planes, same descriptor; it
writes `runs_l0_matched.npz` and leaves the canonical sweep alone.

Usage: python analysis/violation_divergence/r21_run_population/measure.py [--n_games 100]
           [--l0-matched]
"""

import argparse
import time

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.paper_style import PLANE_STYLE
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import POPULATION_T0, games_at_t0, role
from spade.violation_divergence.fork_plane import encoder, positions, spaces, tag
from spade.violation_divergence.fork_r21 import (
    BLOCKS, W, activation_plane, displacement21, named_basis, run_ratio, top2)

# The nine branches of a complete game, in a fixed order, so a game is one slice
# of a [G, B, ...] array and the roles below are constants and not lookups.
ROLES = ["reference", "control_legal", "control_legal_2"] + CATEGORIES[1:]
REF, C1, C2 = 0, 1, 2
ILL = np.arange(3, len(ROLES))
# The plane series, then the unprojected space and its two halves, then the null.
SERIES = list(PLANE_STYLE) + ["full", "d", "theta", "control 2"]
N_POS = 17  # positions(t0) over t0 in [10, 34]: the window never clips


def game_rows(meta: dict, game: int) -> torch.Tensor:
    """The nine branch rows of `game`, in ROLES order."""
    idx = (meta["game_id"] == game).nonzero().flatten()
    by_role = {role(int(meta["category"][i]), int(meta["is_control"][i])): int(i)
               for i in idx}
    assert len(by_role) == len(ROLES) == len(idx), f"game {game} is not branch-complete"
    return torch.tensor([by_role[r] for r in ROLES])


def encoders(device: str, l0_matched: bool) -> list[tuple[str, object]]:
    """(label, encode or None) per space, built once: `fork_plane.space_features`
    reloads the checkpoint on every call, which a population sweep cannot pay.

    Each encode takes and returns a CPU tensor. The activations are mmap'd on the host and
    the weights live on `device`, so the round trip belongs with the closure that knows both
    -- on CPU it is the identity, which is why feeding a host tensor straight in survived
    until the first GPU run.
    """
    def on_device(entry):
        enc = encoder(entry, device)
        return None if enc is None else lambda x: enc(x.to(device)).cpu()
    return [(label, on_device(e)) for label, e in spaces(l0_matched)]


def features(Z: torch.Tensor, encode) -> np.ndarray:
    with torch.no_grad():
        f = encode(Z.flatten(0, 1))
    return f.unflatten(0, Z.shape[:2]).numpy()


def game_ratios(Z: torch.Tensor, cells: list, t0: int, pos: np.ndarray):
    """([S, P] series ratios, [S, 6] per-category ratios) for one game."""
    X, act = {}, {}
    for label, encode in cells:
        F = Z.numpy() if encode is None else features(Z, encode)
        X[label] = displacement21(F, REF)
        act[label] = activation_plane(F, REF)[:, W - 1:]
    names = list(X)
    shared = top2(np.stack(list(X.values())))[0]
    coords = {"activation refit": act,
              "refit": {n: X[n] @ top2(X[n])[0].T for n in names},
              "shared": {n: X[n] @ shared.T for n in names},
              "named": {n: X[n] @ named_basis().T for n in names}}
    where = dict(ref=REF, anchors=pos[W - 1:], t0=t0)

    ratio, member = [], []
    for n in names:
        row = [run_ratio(coords[p][n], ILL, C1, **where) for p in PLANE_STYLE]
        row += [run_ratio(X[n][:, :, c], ILL, C1, **where)
                for c in (slice(None), BLOCKS["d"], BLOCKS["theta"])]
        row.append(run_ratio(X[n], np.array([C2]), C1, **where))
        ratio.append(row)
        member.append([run_ratio(X[n], np.array([i]), C1, **where) for i in ILL])
    return np.array(ratio), np.array(member)


def main():
    ap = argparse.ArgumentParser()
    # The study's shared population: 1696 branch-complete games fork at t0 = 20 and
    # the first 100 are what every n-game figure here draws. ~1.6 s/game.
    ap.add_argument("--n_games", type=int, default=100, help="games to sweep")
    ap.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                    help="sweep the eight l0_operating_point picks instead of the "
                         "seed-42 roster, so a difference between two cells is the "
                         "architecture and not its sparsity")
    args = ap.parse_args()

    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    assert "rollout" not in meta, "spliced only: the rollout tree is one row per suffix"
    # mmap: the activation tensor is 2 GB and one game is nine of its rows.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    games = games_at_t0(meta, POPULATION_T0, args.n_games)
    cells = encoders(select_device(), args.l0_matched)

    ratio = np.full((len(cells), len(games), len(SERIES)), np.nan, np.float32)
    member = np.full((len(cells), len(games), len(ILL)), np.nan, np.float32)
    started = time.time()
    for gi, game in enumerate(games):
        rows = game_rows(meta, game)
        t0 = int(meta["t0"][rows[REF]])
        pos = positions(t0)
        assert len(pos) == N_POS, f"game {game}: {len(pos)} positions, the window clipped"
        Z = acts[rows][:, pos, :].float()
        ratio[:, gi], member[:, gi] = game_ratios(Z, cells, t0, pos)
        if gi % 50 == 0:
            print(f"{gi}/{len(games)} games, {time.time() - started:.0f}s", flush=True)

    out = base / "r21_run_population"
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"runs{tag(l0_matched=args.l0_matched)}.npz"
    np.savez_compressed(npz, space=[name for name, _ in cells],
                        game=np.array(games), series=SERIES, ratio=ratio,
                        category=np.array(CATEGORIES[1:]), member=member)
    print(f"wrote {npz}: {len(games)} games x {len(cells)} spaces, "
          f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
