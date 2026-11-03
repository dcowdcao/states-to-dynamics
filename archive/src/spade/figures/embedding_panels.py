"""Panels for 2-D embeddings of activations (PCA, Isomap).

A class panel is one shuffled point cloud plus, per class, a 50% probability
contour and a shape-coded centroid: the contour and the marker shape carry the
class identity when color is gone, which a color-only cloud cannot. Draw order
is randomized in a single scatter call so no class sits on top of the others.
See FIGURE_GUIDELINES.md §4.6 for the conventions these encode.
"""

import numpy as np
from matplotlib.patches import Ellipse

ELLIPSE_CHI2 = 1.3863  # 50% mass of a 2-d Gaussian: -2 ln(0.5)
CLOUD_SEED = 0
CLOUD_KW = {"s": 2.5, "alpha": 0.35, "lw": 0, "rasterized": True}


def class_ellipse(ax, points: np.ndarray, color: str, lw: float = 1.3) -> None:
    """50% probability contour of a Gaussian fit to one class — a dispersion
    summary, not a confidence interval."""
    values, vectors = np.linalg.eigh(np.cov(points.T))
    width, height = 2 * np.sqrt(ELLIPSE_CHI2 * values[::-1])
    angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
    ax.add_patch(Ellipse(points.mean(0), width, height, angle=angle, fill=False,
                         color=color, lw=lw, zorder=4))


def class_cloud(ax, xy: np.ndarray, values: np.ndarray, styles: dict,
                order: list, reference=None) -> None:
    """Cloud colored by class, with a contour and centroid marker per class.

    `styles` maps a class value to {color, marker, label}; `order` fixes the
    legend order, reference class first. Legend entries come from the centroids,
    so the legend reads by shape as well as by color.
    """
    g = np.random.default_rng(CLOUD_SEED).permutation(len(xy))
    ax.scatter(xy[g, 0], xy[g, 1], c=[styles[v]["color"] for v in values[g]],
               **CLOUD_KW)
    for v in order:
        pts = xy[values == v]
        if not len(pts):
            continue
        style, is_ref = styles[v], v == reference
        class_ellipse(ax, pts, style["color"], 2.0 if is_ref else 1.3)
        ax.plot(*pts.mean(0), ls="none", marker=style["marker"],
                ms=8 if is_ref else 7, color=style["color"],
                markeredgecolor="white", markeredgewidth=1.1, zorder=6,
                label=f"{style['label']} (n={len(pts)})")


def continuous_cloud(ax, fig, xy: np.ndarray, values: np.ndarray, label: str,
                     vmax: float | None = None) -> None:
    """Cloud colored by an ordered variable: sequential ramp plus a colorbar."""
    g = np.random.default_rng(CLOUD_SEED).permutation(len(xy))
    sc = ax.scatter(xy[g, 0], xy[g, 1], c=values[g], cmap="viridis", vmax=vmax,
                    **CLOUD_KW)
    fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.02)
