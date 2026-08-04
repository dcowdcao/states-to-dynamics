"""The six illegal categories, grouped into the three rules they violate.

A figure that draws one mean trajectory per rule averages over groups of 2, 3
and 1. Two things it owes the reader: n per group, and evidence that the members
of a group actually travel together. `spread_ratio` is that evidence -- mean
within-rule distance over mean between-rule distance, on the branches' full
displacement trajectories and never on a 2-D projection of them. Near 1 the
grouping hides as much structure as it shows; well below 1 the rules are the
coherent unit and the mean is a fair summary.

Usage: python -m spade.violation_divergence.rule_groups [--layer 5]
       prints the ratio over every branch-complete game, and for FIGURE_GAME.
"""

import argparse

import numpy as np
import torch

from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES, RULE_MEMBERS

# What `python -m spade.violation_divergence.rule_groups` prints at layer 5. The
# grouping is weak: a rule's members travel ~5% closer to each other than to
# another rule's, and in three games in ten they travel further. Any figure that
# draws one mean per rule has to say this next to the means.
POPULATION_SPREAD = "median 0.945 [IQR 0.872–1.020] over all 1550 branch-complete games, above 1 in 30% of them"


def rule_rows(rows: list[dict]) -> dict[str, list[int]]:
    """Row indices per rule, in RULE_MEMBERS order; rules with no member drop out."""
    groups = {rule: [i for i, r in enumerate(rows) if CATEGORIES[r["category"]] in members]
        for rule, members in RULE_MEMBERS.items()}
    return {rule: idx for rule, idx in groups.items() if idx}


def spread_ratio(D: np.ndarray, rows: list[dict]) -> float:
    """Within-rule over between-rule mean pairwise distance. `D` is [n_branches,
    ..., dim] displacements from the reference; only the illegal rows are used,
    each flattened over everything but the branch axis."""
    groups = rule_rows(rows)
    idx = [i for g in groups.values() for i in g]
    rule = np.array([k for k, g in enumerate(groups.values()) for _ in g])
    X = D[idx].reshape(len(idx), -1)
    d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    same = rule[:, None] == rule[None, :]
    iu = np.triu_indices(len(idx), 1)
    return float(d[iu][same[iu]].mean() / d[iu][~same[iu]].mean())


def main():
    # branches.py pulls the karvonen engine in at import; the grouping needs none
    # of it, and spade.figures imports this module.
    from spade.violation_divergence.branches import (
        FIGURE_GAME, complete_games, load_game, reference_row)
    from spade.violation_divergence.fork_plane import positions

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=5)
    args = parser.parse_args()

    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / f"resid_post_{args.layer}.pt", weights_only=True)

    ratios = {}
    for game in complete_games(meta):
        g = load_game(meta, game)
        pos = positions(g.t0)
        Z = acts[g.idx][:, pos[pos >= g.t0], :].float().numpy()
        ratios[game] = spread_ratio(Z - Z[reference_row(g.rows)], g.rows)

    r = np.array(list(ratios.values()))
    print(f"layer {args.layer}, {len(r)} branch-complete games, "
          f"full {acts.shape[-1]}-d displacements over t >= t0")
    print(f"  within/between spread ratio: median {np.median(r):.3f} "
          f"[IQR {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f}], "
          f"{(r < 1).mean():.1%} of games below 1")
    print(f"  game {FIGURE_GAME} (the figures' default): {ratios[FIGURE_GAME]:.3f}, "
          f"the {(r < ratios[FIGURE_GAME]).mean():.0%} percentile")


if __name__ == "__main__":
    main()
