"""Turning angle over the board for one game's branches, grouped onto the three
constraint rules they violate.

The geometry convention here is the *straightness study's*, deliberately: position
0 (the attention sink) is dropped, steps run between consecutive plies, and theta
index k is the turn at board position k+2, whose window spans positions k+1..k+3.
That is what lets the many-game class baselines in `$SPADE_STRAIGHTNESS_DIR/
geometry/` be drawn on the same axis as a fork measured here -- the two pipelines
then compute the same number, checked game-for-game in the study README. The fork
figures of this study sample every STRIDE=2 plies instead (`fork_plane`), to
suppress the mover-parity sawtooth; a stride-2 curve must never be overlaid on a
straightness class.

phi is nonlinear, so a rule's mean is the mean over its member categories of each
member's own theta curve. theta of an averaged trajectory is the curvature of a
path no branch takes -- `test_curvature_plane.py` pins that phi does not commute
with the mean, and R1/R2/R3 average over 2, 3 and 1 members.

Two trees, one reader. `branches` splices the source game's `tokens[t0+1:]` onto
every branch; `rollout` resamples each branch's continuation from OthelloGPT K
times. A branch is therefore K sequences either way, and `Fork.theta` is
[n_branch, K, n_turn] with K = 1 on the spliced tree, so nothing downstream has to
know which tree it is reading.
"""

from dataclasses import dataclass

import numpy as np
import torch

from spade.common.config import LAYER
from spade.common.study import variant_dir
from spade.figures.straightness_panels import TURN_OFFSET, align_to_event
from spade.straightness.geometry import temporal_geometry
from spade.violation_divergence.branches import (
    FIGURE_GAME, load_game, reference_row)
from spade.violation_divergence.fork_plane import space_features
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

TREE_VARIANT = {"branches": "", "rollout": "rollout"}
SINK = 0        # position 0 is the attention sink and is dropped, as in straightness
FIRST_TURN = 2  # theta index k is the turn at board position k + FIRST_TURN


@dataclass(frozen=True)
class Fork:
    """One game's branches in one representation space, as turning-angle curves."""
    game: int
    t0: int
    rows: list[dict]     # one per branch, branches.pt order
    theta: np.ndarray    # [n_branch, K, n_turn], degrees
    tokens: list[int]    # the reference sequence, for replaying the board at t0

    @property
    def ref(self) -> int:
        return reference_row(self.rows)

    @property
    def n_rollouts(self) -> int:
        return self.theta.shape[1]

    @property
    def positions(self) -> np.ndarray:
        """Board position of each theta column."""
        return np.arange(self.theta.shape[-1]) + FIRST_TURN

    @property
    def last_shared(self) -> int:
        """Last theta column whose window is inside the shared prefix: the turn at
        t0-2, since the turn at t0-1 already reaches the branch token."""
        return self.t0 - FIRST_TURN - 2


def _branch_index(meta: dict, game: int):
    """(one row dict per branch, [n_branch, K] row indices into the act tensor).

    `rollout_branches` writes row = source_row * K + k, so a game's rows are
    branch-major and reshape cleanly; the `rollout` column is asserted, not assumed.
    """
    g = load_game(meta, game)
    if "rollout" not in meta:
        return g, g.rows, g.idx[:, None]
    k = int(meta["rollout"].max()) + 1
    idx = g.idx.reshape(-1, k)
    assert torch.equal(meta["rollout"][idx].long(),
                       torch.arange(k).expand(len(idx), k)), \
        f"game {game}: rows are not branch-major in the rollout index"
    return g, g.rows[::k], idx


def load_fork(tree: str, game: int = FIGURE_GAME, entry=None,
              device: str = "cpu", layer: int = LAYER) -> Fork:
    """One game's branches, encoded into `entry`'s space (None = raw layer `layer`).

    The activation tensor is memory-mapped and only this game's rows are read: the
    rollout tree's layer-5 file is 4 GB and the login node caps at 8.
    """
    base = variant_dir("divergence", False, TREE_VARIANT[tree])
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    g, rows, idx = _branch_index(meta, game)
    acts = torch.load(base / "acts" / f"resid_post_{layer}.pt",
                      weights_only=True, mmap=True)
    Z = acts[idx.flatten()][:, SINK + 1:, :].float().numpy()
    F = space_features(Z, entry, device)
    _, theta = temporal_geometry(torch.from_numpy(F))
    return Fork(game, g.t0, rows,
                theta.numpy().reshape(idx.shape + theta.shape[-1:]), g.tokens)


def branch_curves(fork: Fork) -> np.ndarray:
    """[n_branch, n_turn]: each branch's turning angle, averaged over its rollouts."""
    return np.nanmean(fork.theta, axis=1)


@dataclass(frozen=True)
class Rule:
    """One rule's curve and the sequences it is the mean of.

    `sem` is None when the mean is over a single sequence -- R3 has one member
    category, so on the spliced tree it has no spread at all and a zero-width band
    would read as a measured one.
    """
    mean: np.ndarray                # [n_turn] or [n_offset]
    members: np.ndarray             # [n_member * K, same length], drawn faint
    n_members: int

    @property
    def n_sequences(self) -> int:
        return len(self.members)

    @property
    def sem(self) -> np.ndarray | None:
        if self.n_sequences < 2:
            return None
        return np.nanstd(self.members, 0) / np.sqrt(self.n_sequences)


def rule_profiles(fork: Fork) -> dict[str, Rule]:
    """Per rule, the mean turning-angle curve over its member categories."""
    curves = branch_curves(fork)
    return {rule: Rule(curves[members].mean(0),
                       fork.theta[members].reshape(-1, fork.theta.shape[-1]),
                       len(members))
            for rule, members in rule_rows(fork.rows).items()}


def aligned_delta(fork: Fork) -> tuple[np.ndarray, np.ndarray]:
    """(offsets r = t - t0, [n_branch, K, n_offset] Δ turning angle vs. the reference).

    Pairing is per rollout: the k-th sample of a branch and the k-th of the
    reference are drawn from one uniform stream (`rollout_branches`), so k pairs
    with k and the branch effect is read against a shared noise draw -- which is
    what the broadcast below does.
    """
    delta = fork.theta - fork.theta[fork.ref]
    n_turn = delta.shape[-1]
    rels, flat = align_to_event(delta.reshape(-1, n_turn),
                                np.full(delta.size // n_turn, fork.t0), TURN_OFFSET)
    return rels, flat.reshape(delta.shape[:2] + (-1,))


def locked(fork: Fork, branches: list[int]) -> tuple[np.ndarray, Rule]:
    """The event-locked Δ for one group of branches.

    The mean is over member categories of each member's own Δ. Pooling the
    sequences gives the same number only while every member carries the same K,
    and the order is what phi's nonlinearity constrains, so it is written out.
    """
    rels, delta = aligned_delta(fork)
    group = delta[branches]
    return rels, Rule(np.nanmean(np.nanmean(group, axis=1), axis=0),
                      group.reshape(-1, group.shape[-1]), len(branches))


def rule_locked(fork: Fork) -> tuple[np.ndarray, dict[str, Rule]]:
    """The event-locked Δ per rule, and the offsets they share."""
    out, rels = {}, None
    for rule, members in rule_rows(fork.rows).items():
        rels, out[rule] = locked(fork, members)
    return rels, out


def spread(fork: Fork) -> float:
    """Within-rule over between-rule mean pairwise distance, on the quantity the
    figure draws: each branch's Δ turning angle over t >= t0.

    FIGURE_GUIDELINES 1.7 -- three means over 2, 3 and 1 members owe the reader
    whether a rule's members travel together at all. Near 1 they do not, and the
    faint member lines rather than the means are the evidence. Offsets where any
    branch is undefined (a sparse code can give a zero-length step) are dropped,
    never the branch.
    """
    rels, delta = aligned_delta(fork)
    post = np.nanmean(delta, axis=1)[:, rels >= 0]
    return spread_ratio(post[:, np.isfinite(post).all(0)], fork.rows)


def check_prefix(fork: Fork) -> None:
    """Every branch's turning angle equals the reference's over the shared prefix.

    Zero by construction, so a nonzero value means the alignment or the tree is
    wrong rather than that the model responded early.
    """
    delta = fork.theta - fork.theta[fork.ref]
    assert np.nanmax(np.abs(delta[:, :, :fork.last_shared + 1])) == 0, \
        f"game {fork.game}: branches differ before t0={fork.t0}"
