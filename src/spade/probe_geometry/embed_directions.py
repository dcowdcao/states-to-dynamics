"""Coordinates for the probe-geometry figure: a PCA of the directions, and their cosines.

Points are per-square directions, one per board square: two board contrasts, and
the rule probe's classes refit on the rows where that square was the candidate
move. Arrows are the committed probes' own directions, drawn in the same basis —
the board arrows are the mean of their per-square points, since the board probe
has no pooled direction of its own.

The PCA is uncentred (FIGURE_GUIDELINES 4.10): these are unit vectors, so the
origin is a real point and the mean direction is itself a result.

Usage: python src/probe_geometry/embed_directions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


from spade.common.paths import repo_root
from spade.probe_geometry.probes import (
    Direction,
    board_directions,
    board_probe,
    committed_rule_class_directions,
    rules_corpus,
    square_rule_class_directions,
    square_rules_directions,
    stack,
    unit,
)

RESULTS = repo_root() / "analysis" / "probe_geometry" / "results"
LAMBDA = 10.0
ORDER = ("legal", "illegal", "R1", "R2", "R3", "colour", "occupancy")


def points() -> list[Direction]:
    rules = rules_corpus()
    rows = np.arange(len(rules["acts"]))
    legality = [Direction("illegal", d.name, d.vec, d.square)
        for d in square_rules_directions(rules, rows, LAMBDA)]
    return (board_directions(board_probe()) + legality
            + square_rule_class_directions(rules, rows, LAMBDA))


def arrows(pts: list[Direction]) -> list[Direction]:
    """Committed rule directions; board directions have no pooled probe, so take the mean."""
    board = [Direction(fam, fam, unit(stack([d for d in pts if d.family == fam]).mean(0)))
        for fam in ("colour", "occupancy")]
    return committed_rule_class_directions() + board


def pca(points_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, s, vt = np.linalg.svd(points_matrix, full_matrices=False)
    basis = vt[:2]
    # SVD sign is arbitrary and a refit that flips it mirrors the panel for no reason a
    # reader can act on. Pin each component so the point cloud's mean coordinate on it is
    # positive: the PCA is uncentred, so that mean is a drawn quantity, not a basis choice.
    sign = np.sign((points_matrix @ basis.T).mean(0))
    return basis * sign[:, None], s**2 / (s**2).sum()


def main():
    pts = points()
    pts = [d for d in pts if d.family in ORDER]
    arr = arrows(pts)

    basis, ratio = pca(stack(pts))
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"family": d.family, "square": d.square,
                   "pc1": xy[0], "pc2": xy[1]}
        for d, xy in zip(pts, stack(pts) @ basis.T)]).to_csv(
            RESULTS / "pca_points.csv", index=False)
    pd.DataFrame([{"family": d.family, "pc1": xy[0], "pc2": xy[1]}
        for d, xy in zip(arr, stack(arr) @ basis.T)]).to_csv(
            RESULTS / "pca_arrows.csv", index=False)
    pd.DataFrame({"component": [1, 2], "variance_ratio": ratio[:2]}).to_csv(
        RESULTS / "pca_variance.csv", index=False)

    by_family = {d.family: d.vec for d in arr}
    gram = np.array([[by_family[a] @ by_family[b] for b in ORDER] for a in ORDER])
    pd.DataFrame(gram, index=ORDER, columns=ORDER).to_csv(RESULTS / "type_cosines.csv")

    counts = {f: sum(d.family == f for d in pts) for f in ORDER}
    print("points per family:", counts)
    print(f"PC1 {ratio[0]:.3%}  PC2 {ratio[1]:.3%}")
    print(pd.DataFrame(gram, index=ORDER, columns=ORDER).round(3))


if __name__ == "__main__":
    main()
