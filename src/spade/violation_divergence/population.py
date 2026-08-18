"""Which games a population sweep runs over, which rows a game is, and the
encoders it reads every space through.

A sweep over the nine-space grid needs three things a single-game figure does
not. A FIXED branch-row order, so a game is one slice of a [G, B, ...] array and
the roles are constants rather than lookups -- branches.pt order is not that
order and is not guaranteed to be the same order twice. The games that fork at
one t0, so a mean over games is a mean over aligned positions. And encoders built
once: `fork_plane.space_features` reloads the checkpoint on every call, which a
100-game sweep cannot pay.
"""

import numpy as np
import torch

from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import complete_games, role
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.fork_plane import spaces

ROLES = ["reference", "control_legal", "control_legal_2"] + CATEGORIES[1:]
REF, C1, C2 = 0, 1, 2


def rows_at(meta: dict, game: int) -> torch.Tensor:
    """The nine branch rows of `game`, in ROLES order."""
    idx = (meta["game_id"] == game).nonzero().flatten()
    by_role = {role(int(meta["category"][i]), int(meta["is_control"][i])): int(i)
               for i in idx}
    assert sorted(by_role) == sorted(ROLES) and len(idx) == len(ROLES), \
        f"game {game} is not branch-complete"
    return torch.tensor([by_role[r] for r in ROLES])


def at_t0(meta: dict, t0: int, n: int | None = None) -> list[int]:
    """The first `n` branch-complete games forking at `t0`, ascending by game id."""
    games = [g for g in complete_games(meta)
             if int(meta["t0"][(meta["game_id"] == g).nonzero()[0]]) == t0]
    return games[:n]


def encoders(device: str):
    """(label, encode or None) per space, one checkpoint resident at a time.

    A generator, and consumed by a space-outer sweep, so every checkpoint is still
    loaded exactly once -- holding all nine at once is 0.7 GB beside the 0.5 GB of
    fields such a sweep holds, and the login node's cgroup does not have both.
    """
    for label, e in spaces():
        yield label, None if e is None else sae_encoder(e, device)


def encode(Z: torch.Tensor, fn) -> np.ndarray:
    with torch.no_grad():
        f = fn(Z.flatten(0, 1))
    return f.unflatten(0, Z.shape[:2]).numpy()
