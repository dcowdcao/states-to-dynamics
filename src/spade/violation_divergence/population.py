"""Which games a population sweep runs over, which rows a game is, and the
encoders it reads every space through.

A sweep over the nine-space grid needs three things a single-game figure does
not. A FIXED branch-row order, so a game is one slice of a [G, B, ...] array and
the roles are constants rather than lookups -- branches.pt order is not that
order and is not guaranteed to be the same order twice. And encoders built
once: `fork_plane.space_features` reloads the checkpoint on every call, which a
100-game sweep cannot pay.
"""

import numpy as np
import torch

from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import role
from spade.violation_divergence.fork_plane import encoder, spaces

ROLES = ["reference", "control_legal", "control_legal_2"] + CATEGORIES[1:]
REF, C1, C2 = 0, 1, 2


def rows_at(meta: dict, game: int, K: int = 1) -> torch.Tensor:
    """The branch rows of `game` in ROLES order, each role's K rollouts contiguous.

    K = 1 is the spliced tree, one row per role. The rollout tree stores each
    spliced row K times as `source_row * K + k` (rollout_branches), so a role's
    rows are already ascending in rollout index and a [B * K, ...] array reshapes
    to [B, K, ...] with the branch axis still in ROLES order.
    """
    idx = (meta["game_id"] == game).nonzero().flatten()
    by_role = {}
    for i in idx:
        by_role.setdefault(role(int(meta["category"][i]), int(meta["is_control"][i])),
                           []).append(int(i))
    assert sorted(by_role) == sorted(ROLES), f"game {game} is not branch-complete"
    assert all(len(r) == K for r in by_role.values()), \
        f"game {game} does not carry K = {K} rows per branch"
    return torch.tensor([i for r in ROLES for i in by_role[r]])


def paired_mean(X: np.ndarray, ref: int, K: int) -> np.ndarray:
    """[B*K, ...] -> [B, ...]: each row's difference from the reference drawn on
    the same rollout, averaged over the K rollouts of its branch.

    Rows are K-contiguous per branch, and a rollout's reference is the one sharing
    its uniform stream, so the difference is paired before it is averaged -- and
    taken after the encode and phi, never before, since both are nonlinear. K = 1
    is the spliced set, where this is exactly X - X[ref]."""
    Y = X.reshape(-1, K, *X.shape[1:])
    return (Y - Y[ref]).mean(axis=1)


def encoders(device: str, l0_matched: bool = False):
    """(label, encode or None) per space, one checkpoint resident at a time.

    A generator, and consumed by a space-outer sweep, so every checkpoint is still
    loaded exactly once -- holding all nine at once is 0.7 GB beside the 0.5 GB of
    fields such a sweep holds, and the login node's cgroup does not have both.
    """
    for label, e in spaces(l0_matched):
        yield label, encoder(e, device)


def encode(Z: torch.Tensor, fn) -> np.ndarray:
    with torch.no_grad():
        f = fn(Z.flatten(0, 1))
    return f.unflatten(0, Z.shape[:2]).numpy()
