"""PCA of the 21-D ReStraV feature vector, one panel per representation space.

Each game is one point: the 21 components are its first 7 step lengths, first 6
turning angles, and the 4 moments of each (src/spade/straightness/geometry.py).
This is the space the LR-21 probe classifies in, viewed unsupervised.

Every space gets its own PCA, fit on that space's z-scored features. Raw
covariance PCA would answer a different question here: the components mix
degrees with activation-scale lengths, whose absolute variances differ by ~450x
across dictionaries (SpaDE step lengths ~0.05, raw layer 5 ~22.8), so PC1 lands
on turning-angle variance in every space and reports units, not structure.

Usage: python analysis/straightness/feature_pca/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.study import study_dir
from spade.figures.embedding_panels import class_ellipse
from spade.figures.paper_style import CLASS_STYLE, MUTED, apply_style, sae_label
from spade.straightness.build_games import CLASSES
from spade.straightness.geometry import RESTRAV_FEATURE_NAMES, restrav_features

apply_style()

FIG_DIR = Path(__file__).resolve().parent / "figures"
GEOM = study_dir("straightness") / "geometry"
RAW_SPACE = "resid_post_5"


def spaces() -> list[tuple[str, str]]:
    """(space, label) for raw layer 5 then every SAE family, sorted by family."""
    sae = sorted(f.name.removesuffix("_raw.npz") for f in GEOM.glob("sae_*_raw.npz"))
    return [(RAW_SPACE, "raw L5 (reference)")] + [
        (s, sae_label(s.removeprefix("sae_").removesuffix("Trainer-othello")))
        for s in sae]


def features(space: str) -> np.ndarray:
    g = np.load(GEOM / f"{space}_raw.npz")
    return restrav_features(torch.from_numpy(g["d"]),
                            torch.from_numpy(g["theta"])).numpy()


def pca(f21: np.ndarray, align: np.ndarray | None = None):
    """Top-2 PCs of the z-scored features: (scores, variance ratios, loadings).

    Component sign is arbitrary, so it is pinned: the reference space puts its
    largest loading positive, every other space matches the reference's sign, so
    panels are not mirrored against each other for no reason.
    """
    z = (f21 - f21.mean(0)) / f21.std(0)
    u, sv, vt = np.linalg.svd(z, full_matrices=False)
    v = vt[:2]
    sign = np.sign(v @ align.T).diagonal() if align is not None else \
        np.sign(v[np.arange(2), np.abs(v).argmax(1)])
    return u[:, :2] * sv[:2] * sign, (sv ** 2 / (sv ** 2).sum())[:2], v * sign[:, None]


def pca_panel(ax, scores: np.ndarray, blocks: dict) -> None:
    for c in CLASSES:
        style, is_ref = CLASS_STYLE[c], c == "legal"
        pts = scores[blocks[c]]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.16, lw=0,
                   color=style["color"], marker=style["marker"],
                   label=style["label"], zorder=3 if is_ref else 2, rasterized=True)
        class_ellipse(ax, pts, style["color"], 2.0 if is_ref else 1.3)
        ax.plot(*pts.mean(0), marker=style["marker"], ms=7, color=style["color"],
                markeredgecolor="white", markeredgewidth=1.1, zorder=6)


def restrav_pca(blocks: dict) -> dict[str, np.ndarray]:
    """Small multiples of the top-2 PC plane. Returns each space's loadings."""
    panels = spaces()
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 3.0 * nrow),
                             sharex=True, sharey=True)
    loadings, raw_loadings = {}, None
    for ax, (space, label) in zip(axes.flat, panels):
        scores, evr, vecs = pca(features(space), align=raw_loadings)
        if raw_loadings is None:
            raw_loadings = vecs
        loadings[label] = vecs
        pca_panel(ax, scores, blocks)
        lead = [RESTRAV_FEATURE_NAMES[i] for i in np.abs(vecs).argmax(1)]
        ax.set_title(f"{label}\nPC1 {evr[0]:.0%} ({lead[0]})  ·  PC2 {evr[1]:.0%} "
                     f"({lead[1]})", loc="left", fontsize=8.5)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
                        markerscale=3.0, bbox_to_anchor=(0.5, -0.015))
    for h in legend.legend_handles:
        h.set_alpha(1)
    fig.suptitle("Random-token games are the only class the 21-D descriptor "
                 "separates on its own — in every dictionary", fontsize=11, y=1.00)
    fig.text(0.5, 0.963, "one point per game in the top-2 principal plane of that "
             "space's z-scored ReStraV features; ellipses enclose 50% of a Gaussian "
             "fit per class\nlarge markers are class means; each space has its own "
             "basis (alignment with raw L5: README); raw variant, n = 2000 games per "
             "class\nPCA is unsupervised — the supervised LR-21 probe separates "
             "classes this plane does not",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    fig.savefig(FIG_DIR / "restrav_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return loadings


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    n = len(data["cls"]) // len(CLASSES)
    blocks = {c: slice(i * n, (i + 1) * n) for i, c in enumerate(CLASSES)}

    loadings = restrav_pca(blocks)
    raw = loadings["raw L5 (reference)"]
    print(f"{'space':34} {'|cos| with raw PC1':>18} {'PC2':>6}")
    for label, vecs in loadings.items():
        cos = np.abs((vecs @ raw.T).diagonal())
        print(f"{label:34} {cos[0]:18.2f} {cos[1]:6.2f}")
    print(f"figure written to {FIG_DIR}")


if __name__ == "__main__":
    main()
