"""Coordinates for the probe-geometry figure: a nonmetric MDS of the directions, and their cosines.

Points are per-square directions, one per board square: two board contrasts, and
the rule probe's classes refit on the rows where that square was the candidate
move. The committed probes are placed in the same layout, fitted jointly with the
cloud rather than transformed into it afterwards.

Nonmetric MDS on the angular distances, not PCA. These directions are
near-mutually-orthogonal, so no plane holds their variance and PCA's top two
components carry 5.69% of it. What a plane can carry is the *order* of the
angles, which is what nonmetric MDS fits. Distance in the panel is therefore
monotone in angle and not proportional to it, so the axes carry no variance
ratio. README holds the fidelity, the ceiling, and the null that says how little
of it is earned.

Usage: python -m spade.probe_geometry.embed_directions
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.manifold import MDS

from spade.common.paths import results_dir
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

RESULTS = results_dir("fig02_probe_geometry_and_detection")
LAMBDA = 10.0
ORDER = ("legal", "illegal", "R1", "R2", "R3", "colour", "occupancy")

RULE_FAMILIES = ("legal", "illegal", "R1", "R2", "R3")

# `illegal@q` is exactly -`legal@q`: both are ridge onto centred indicators over the same
# rows, and 1[rule>0] = 1 - 1[rule==0] forces w2[1] = -w4[0]. The panel draws it -- the
# reflection is legible, and legality having two opposed poles is the thing the eye should
# see -- but it is scored out. Its 180 deg pairs are the easiest in the set and a plane
# renders them for free, so leaving them in lifts every fidelity number without any of the
# structure being better shown.
SCORED = tuple(f for f in ORDER if f != "illegal")
NULL_DRAWS = 5


def points() -> list[Direction]:
    rules = rules_corpus()
    rows = np.arange(len(rules["acts"]))
    legality = [Direction("illegal", d.name, d.vec, d.square)
        for d in square_rules_directions(rules, rows, LAMBDA)]
    return (board_directions(board_probe()) + legality
            + square_rule_class_directions(rules, rows, LAMBDA))


def committed(pts: list[Direction]) -> list[Direction]:
    """Committed rule directions; board directions have no pooled probe, so take the mean."""
    board = [Direction(fam, fam, unit(stack([d for d in pts if d.family == fam]).mean(0)))
        for fam in ("colour", "occupancy")]
    return committed_rule_class_directions() + board


def angular(x: np.ndarray) -> np.ndarray:
    """Geodesic distance on the unit sphere. A metric, which 1 - cos is not."""
    theta = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(theta, 0.0)
    return theta


def embed(x: np.ndarray) -> np.ndarray:
    """2-D nonmetric MDS of the angular distances, oriented against the PCA basis.

    SMACOF fixes a configuration only up to rigid motion, so a refit is free to rotate or
    mirror the panel for no reason a reader can act on. Orthogonal Procrustes onto the
    uncentred PCA coordinates pins it.
    """
    xy = MDS(n_components=2, metric_mds=False, metric="precomputed", init="classical_mds",
             n_init=1, max_iter=1000, normalized_stress=True).fit_transform(angular(x))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    rotation, _ = orthogonal_procrustes(xy, x @ vt[:2].T)
    return xy @ rotation


def rank_fidelity(x: np.ndarray, xy: np.ndarray) -> float:
    """Spearman between the 512-d angles and the drawn distances, over every pair."""
    iu = np.triu_indices(len(x), 1)
    drawn = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    return float(spearmanr(angular(x)[iu], drawn[iu]).statistic)


def structure_free_null(x: np.ndarray, family: np.ndarray) -> tuple[float, float]:
    """Fidelity when only the family mean directions survive and the rest is noise.

    The guard on reading the panel: a cloud carrying nothing but six mean directions
    scores at least as well as the real one, so the layout is evidence about families
    and never about individual squares.
    """
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(NULL_DRAWS):
        z = np.empty_like(x)
        for fam in np.unique(family):
            at = family == fam
            mean = x[at].mean(0)
            spread = np.linalg.norm(x[at] - mean, axis=1).mean() / np.sqrt(x.shape[1])
            z[at] = mean + rng.normal(scale=spread, size=(at.sum(), x.shape[1]))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        scores.append(rank_fidelity(z, embed(z)))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def ring(xy: np.ndarray, family: np.ndarray, x: np.ndarray) -> dict:
    """The radius that best splits the rule families from the board families.

    The panel draws the board contrasts inside and the rule contrasts outside. That is a
    property of the directions, not of the layout: a board contrast sits at a smaller mean
    angle to every other direction than a rule contrast does, 89.71 deg against 90.13.
    `centrality_rho` is how well the drawn radius tracks that 512-d quantity, per point,
    and is what licenses drawing the ring at all.
    """
    radii = np.linalg.norm(xy - xy.mean(0), axis=1)
    is_rule = np.isin(family, RULE_FAMILIES)
    # Balanced, not raw, accuracy: there are 300 rule points against 124 board ones, and a
    # cut chosen on raw accuracy buys itself the majority class and leaves a third of the
    # board points outside the ring it is supposed to enclose.
    balanced = lambda cut: ((radii[is_rule] > cut).mean() + (radii[~is_rule] <= cut).mean()) / 2
    grid = np.linspace(radii.min(), radii.max(), 2000)
    cut = grid[np.argmax([balanced(c) for c in grid])]

    theta = angular(x).astype(float)
    np.fill_diagonal(theta, np.nan)
    auc = mannwhitneyu(radii[is_rule], radii[~is_rule]).statistic / (is_rule.sum() * (~is_rule).sum())
    return {"ring_radius": float(cut),
            "ring_balanced_accuracy": float(balanced(cut)),
            "ring_auc": float(auc),
            "ring_centrality_rho": float(spearmanr(radii, np.nanmean(theta, 1)).statistic)}


def main():
    pts = points()
    drawn = committed(pts)

    cloud = stack(pts)
    xy = embed(np.vstack([cloud, stack(drawn)]))
    cloud_xy, probe_xy = xy[:len(pts)], xy[len(pts):]

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"family": d.family, "square": d.square, "x": p[0], "y": p[1]}
        for d, p in zip(pts, cloud_xy)]).to_csv(RESULTS / "embed_points.csv", index=False)
    pd.DataFrame([{"family": d.family, "x": p[0], "y": p[1]}
        for d, p in zip(drawn, probe_xy)]).to_csv(RESULTS / "embed_probes.csv", index=False)

    family = np.array([d.family for d in pts])
    keep = np.isin(family, SCORED)
    fidelity = rank_fidelity(cloud[keep], cloud_xy[keep])
    null_mean, null_sd = structure_free_null(cloud[keep], family[keep])
    pd.DataFrame([{"n_drawn": len(pts), "n_scored": int(keep.sum()),
                   "rank_fidelity": fidelity,
                   "null_mean": null_mean, "null_sd": null_sd,
                   **ring(cloud_xy, family, cloud)}]).to_csv(
        RESULTS / "embed_fidelity.csv", index=False)

    by_family = {d.family: d.vec for d in drawn}
    gram = np.array([[by_family[a] @ by_family[b] for b in ORDER] for a in ORDER])
    pd.DataFrame(gram, index=ORDER, columns=ORDER).to_csv(RESULTS / "type_cosines.csv")

    print("points per family:", {f: int((family == f).sum()) for f in ORDER})
    r = ring(cloud_xy, family, cloud)
    print(f"rank fidelity {fidelity:.3f}  null {null_mean:.3f} +- {null_sd:.3f}")
    print(f"ring r={r['ring_radius']:.3f} splits rule from board, balanced accuracy "
          f"{r['ring_balanced_accuracy']:.1%}, AUC {r['ring_auc']:.3f}, tracking 512-d "
          f"centrality at rho={r['ring_centrality_rho']:+.3f}")
    print(pd.DataFrame(gram, index=ORDER, columns=ORDER).round(3))


if __name__ == "__main__":
    main()
