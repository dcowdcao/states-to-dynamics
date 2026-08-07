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

One game is not a population, so the same three rules are also measured over a
population of games (`build_population`). The straightness study cannot supply
them: its violation class is built from empty-square categories only and carries
no R1 at all. The population is the **rule-complete** set -- the 2000 games
carrying at least one member of every rule, listed in `rule_complete_games.json`
so the membership rule has one definition. Of the original 2000 source games only
1997 qualified, three t0 offering no opponent_only square at all; `extend_branches`
appended three more rather than let R3 run short, so all three rules share n =
2000. A rule's member composition still varies by game (R2 averages under its
three categories), which the figure caption states. Written to
`$DIR/[rollout/]rule_population/<space>.npz` by this module's main; the figure
script only reads them.

Usage: python -m spade.violation_divergence.rule_curvature [--tree branches|rollout]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spade.common.config import LAYER
from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.figures.straightness_panels import TURN_OFFSET, align_to_event
from spade.othello.taxonomy import RULE_MEMBERS
from spade.straightness.geometry import temporal_geometry
from spade.violation_divergence.branches import (
    FIGURE_GAME, ROLE_NAMES, load_game, reference_row)
from spade.violation_divergence.fork_plane import space_features, spaces
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

TREE_VARIANT = {"branches": "", "rollout": "rollout"}
SINK = 0        # position 0 is the attention sink and is dropped, as in straightness
FIRST_TURN = 2  # theta index k is the turn at board position k + FIRST_TURN
ROW_BATCH = 128  # 128 seqs x 58 positions x 8192 codes x 4 B = 245 MB of codes


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


def _theta(acts, idx: torch.Tensor, entry, device: str) -> np.ndarray:
    """[n_branch, K] row indices -> [n_branch, K, n_turn] turning angles, degrees."""
    Z = acts[idx.flatten()][:, SINK + 1:, :].float().numpy()
    F = space_features(Z, entry, device)
    _, theta = temporal_geometry(torch.from_numpy(F))
    return theta.numpy().reshape(idx.shape + theta.shape[-1:])


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
    return Fork(game, g.t0, rows, _theta(acts, idx, entry, device), g.tokens)


def branch_curves(fork: Fork) -> np.ndarray:
    """[n_branch, n_turn]: each branch's turning angle, averaged over its rollouts."""
    return np.nanmean(fork.theta, axis=1)


@dataclass(frozen=True)
class Rule:
    """One rule's curve and the sequences it is the mean of.

    The figure draws `mean` alone; `members` is what `sem` is taken over, and the
    study README quotes those tail SEMs as printed numbers. `sem` is None when the
    mean is over a single sequence -- R3 has one member category, so on the spliced
    tree it has no spread at all and a zero-width band would read as a measured one.
    """
    mean: np.ndarray                # [n_turn] or [n_offset]
    members: np.ndarray             # [n_member * K, same length]
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
    whether a rule's members travel together at all. Near 1 they do not, and this
    number rather than the three means is the evidence. Offsets where any branch
    is undefined (a sparse code can give a zero-length step) are dropped, never
    the branch.
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


RULES = list(RULE_MEMBERS)


@dataclass(frozen=True)
class Population:
    """The three rules' turning-angle curves over every game that carries them.

    `n_games` and `members` are per rule and not one number: a rule takes every
    game with at least one of its member categories available at that game's t0,
    so R3 is short by the three games where no opponent_only square exists, and a
    rule's member composition varies from game to game. `sem` is over games.
    """
    mean: np.ndarray      # [n_rule, n_turn]
    sem: np.ndarray       # [n_rule, n_turn]
    n_games: np.ndarray   # [n_rule]
    members: np.ndarray   # [n_rule], mean member categories per game
    n_rollouts: int
    t0_range: np.ndarray  # [2], the board positions the branches fork on

    @property
    def positions(self) -> np.ndarray:
        return np.arange(self.mean.shape[-1]) + FIRST_TURN


def _game_rule_means(theta: np.ndarray,
                     rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """One game's ([n_rule, n_turn] curves, [n_rule] member counts), NaN where absent.

    phi is nonlinear, so the order is fixed and written out: over a branch's
    rollouts, then over the rule's member categories. Across games comes after.
    """
    groups = rule_rows(rows)
    curves = np.nanmean(theta, axis=1)
    mean = np.full((len(RULES), theta.shape[-1]), np.nan)
    n = np.full(len(RULES), np.nan)
    for j, rule in enumerate(RULES):
        if rule in groups:
            mean[j] = curves[groups[rule]].mean(0)
            n[j] = len(groups[rule])
    return mean, n


def rule_complete() -> list[int]:
    """The games carrying at least one member of every rule, written by
    `extend_branches`. Reading the list rather than re-deriving it keeps one
    definition of the population and every rule on the same n.

    Always the spliced tree's list: which rules a game admits is decided at t0 by
    the board, so it is the same games whichever suffix follows.
    """
    base = variant_dir("divergence", False, "")
    listing = json.loads((base / "branches" / "rule_complete_games.json").read_text())
    return listing["game_ids"]


def build_population(tree: str, entry, device: str = "cpu",
                     layer: int = LAYER) -> Population:
    """Every game's per-rule curve, streamed a chunk of games at a time.

    Only the [n_game, n_rule, n_turn] table of per-game means survives a chunk:
    the rollout tree's layer-5 activations are 4 GB and its codes are 16x wider.
    """
    base = variant_dir("divergence", False, TREE_VARIANT[tree])
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / f"resid_post_{layer}.pt",
                      weights_only=True, mmap=True)
    games = rule_complete()
    k = 1 if "rollout" not in meta else int(meta["rollout"].max()) + 1
    step = max(1, ROW_BATCH // (len(ROLE_NAMES) * k))

    per_game, counts = [], []
    for start in range(0, len(games), step):
        chunk = [_branch_index(meta, g) for g in games[start:start + step]]
        theta = _theta(acts, torch.cat([i for _, _, i in chunk]), entry, device)
        at = 0
        for _, rows, i in chunk:
            mean, n = _game_rule_means(theta[at:at + len(i)], rows)
            per_game.append(mean)
            counts.append(n)
            at += len(i)
        if start % (100 * step) == 0:
            print(f"  {start + len(chunk)}/{len(games)} games", flush=True)

    G, C = np.array(per_game), np.array(counts)
    n = np.isfinite(G).sum(0)
    return Population(np.nanmean(G, 0), np.nanstd(G, 0) / np.sqrt(np.maximum(n, 1)),
                      np.isfinite(C).sum(0), np.nanmean(C, 0), k,
                      np.array([int(meta["t0"].min()), int(meta["t0"].max())]))


def space_key(entry) -> str:
    """One space's file stem: this study's cache, and the straightness geometry npz
    the class baselines drawn beside it are read from."""
    return f"resid_post_{LAYER}" if entry is None else f"sae_{entry.config}"


def population_path(tree: str, entry) -> Path:
    return (variant_dir("divergence", False, TREE_VARIANT[tree])
            / "rule_population" / f"{space_key(entry)}.npz")


def population(tree: str, entry) -> Population:
    path = population_path(tree, entry)
    assert path.exists(), (
        f"{path} is missing -- run `python -m spade.violation_divergence."
        f"rule_curvature --tree {tree}` (SLURM: submit_rule_population.sbatch)")
    z = np.load(path)
    return Population(z["mean"], z["sem"], z["n_games"], z["members"],
                      int(z["n_rollouts"]), z["t0_range"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="branches", choices=list(TREE_VARIANT))
    args = parser.parse_args()

    device = select_device()
    for label, entry in spaces():
        path = population_path(args.tree, entry)
        if path.exists():
            print(f"{label:34} cached")
            continue
        print(f"{label:34} building", flush=True)
        pop = build_population(args.tree, entry, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=pop.mean, sem=pop.sem, n_games=pop.n_games,
                 members=pop.members, n_rollouts=pop.n_rollouts, t0_range=pop.t0_range)
        print(f"{label:34} " + "  ".join(
            f"{r.split()[0]} {np.nanmean(m):.2f}deg n={g} members={c:.2f}"
            for r, m, g, c in zip(RULES, pop.mean, pop.n_games, pop.members)), flush=True)


if __name__ == "__main__":
    main()
