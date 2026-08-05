"""The six illegal categories, grouped into the three rules they violate.

A figure that draws one mean trajectory per rule averages over groups of 2, 3
and 1. Two things it owes the reader: n per group, and evidence that the members
of a group actually travel together. `spread_ratio` is that evidence -- mean
within-rule distance over mean between-rule distance, on the branches' full
displacement trajectories and never on a 2-D projection of them. Near 1 the
grouping hides as much structure as it shows; well below 1 the rules are the
coherent unit and the mean is a fair summary.

At layer 5 it is weak in both fields the figures use, and by the same margin:
displacement median 0.945 [IQR 0.872-1.020] over all 1550 branch-complete games,
above 1 in 30%; R^21 median 0.936 [IQR 0.827-0.998] over 60 games with t0 >= 18,
above 1 in 25%. The two populations differ (see `r21_ratios`), so the medians are
not a paired comparison -- what replicates is that a rule's members travel barely
closer to each other than to another rule's, in either field. A single game
orders the two fields either way and should not be read as one beating the other.
Every figure drawing one mean per rule owes the reader this in its caption, and
the study READMEs carry it.

Usage: python -m spade.violation_divergence.rule_groups [--layer 5]
                  [--field displacement|r21] [--n_games N]
"""

import argparse

import numpy as np
import torch

from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES, RULE_MEMBERS

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


def r21_ratios(n_games: int | None) -> dict[int, float]:
    """The same statistic on the R^21 curvature descriptor, layer 5.

    A smaller population than the displacement one and not a subset of it:
    `curvature_plane.usable` keeps only games whose first full-width window still
    ends before the branch (t0 >= 18 at W=8), so the figures can show the prefix
    collapsing rather than assert it. Quote the two medians as what they are.
    """
    from spade.violation_divergence import curvature_plane as cp
    from spade.violation_divergence.fork_plane import RAW_SPACE

    ratios = {}
    for g in cp.stream(cp.usable(n_games)):
        category, is_control = np.asarray(g["category"]), np.asarray(g["is_control"])
        ref = int(np.flatnonzero((category == 0) & (is_control == 0))[0])
        rows = [{"category": int(c), "is_control": int(i)}
            for c, i in zip(category, is_control)]
        D = cp.windowed(g[RAW_SPACE]) - cp.windowed(g[RAW_SPACE])[ref]
        # the earliest windows run off the board and are NaN by construction:
        # drop those window slots, never the game.
        defined = np.isfinite(D).all(axis=0).all(axis=-1)
        X = D[:, defined, :].reshape(len(D), -1)
        ratios[int(g["game"])] = spread_ratio(X / (np.sqrt((X ** 2).mean(0)) + 1e-12), rows)
    return ratios


def main():
    # branches.py pulls the karvonen engine in at import; the grouping needs none
    # of it, and spade.figures imports this module.
    from spade.violation_divergence.branches import (
        FIGURE_GAME, complete_games, load_game, reference_row)
    from spade.violation_divergence.fork_plane import positions

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--field", default="displacement", choices=["displacement", "r21"])
    parser.add_argument("--n_games", type=int, default=None,
                        help="r21 only: cap the sweep, which streams activations per game")
    args = parser.parse_args()

    if args.field == "r21":
        assert args.layer == 5, "the curvature plane is built on layer 5"
        ratios = r21_ratios(args.n_games)
        where = f"{len(ratios)} games with t0 >= 18, R^21 curvature descriptor"
    else:
        base = study_dir("divergence")
        meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
        acts = torch.load(base / "acts" / f"resid_post_{args.layer}.pt", weights_only=True)
        ratios = {}
        for game in complete_games(meta):
            g = load_game(meta, game)
            pos = positions(g.t0)
            Z = acts[g.idx][:, pos[pos >= g.t0], :].float().numpy()
            ratios[game] = spread_ratio(Z - Z[reference_row(g.rows)], g.rows)
        where = f"{len(ratios)} branch-complete games, full {acts.shape[-1]}-d displacements"

    r = np.array(list(ratios.values()))
    print(f"layer {args.layer}, {where}, over t >= t0")
    print(f"  within/between spread ratio: median {np.median(r):.3f} "
          f"[IQR {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f}], "
          f"{(r < 1).mean():.1%} of games below 1")
    if FIGURE_GAME in ratios:
        print(f"  game {FIGURE_GAME} (the figures' default): {ratios[FIGURE_GAME]:.3f}, "
              f"the {(r < ratios[FIGURE_GAME]).mean():.0%} percentile")
    else:
        print(f"  game {FIGURE_GAME} is outside this sweep")


if __name__ == "__main__":
    main()
