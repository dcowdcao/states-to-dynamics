"""Figures for the rules manifold study.

Reads embeddings/ (written by spade.rules.embed) and renders, per space and
view, a 2-row grid — PCA on top, Isomap below — with view-specific colorings:
combined by rule / fine category / t, per-rule views by category / t, the
legal-only R5 view by discs-flipped / directions / t. Centered variant is the
main figure; the raw variant ships alongside (suffix `_raw`) because t
dominates uncentered geometry. Runs on CPU.

Usage: python analysis/rules_manifolds/make_figures.py [--camera-ready]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spade.figures.embedding_panels import class_cloud, continuous_cloud
from spade.figures.paper_style import (
    CATEGORY_STYLE, INK, RULE_STYLE, apply_style, rule_key, save_figure)
from spade.othello.taxonomy import CATEGORIES

HERE = Path(__file__).resolve().parent
EMB = HERE / "embeddings"
VIEWS = {"combined": ["R1", "R2", "R3"], "r1_placement": ["R1"],
         "r2_bracketing": ["R2"], "r3_turn": ["R3"], "r5_flips": ["R5"]}

RULES = list(RULE_STYLE)
RULE_OF_CATEGORY = np.array([0, 1, 1, 2, 2, 2, 3])  # legal, R1, R1, R2, R2, R2, R3
CATEGORY_BY_INDEX = {i: c for i, c in enumerate(CATEGORIES)}
RULE_BY_INDEX = dict(enumerate(RULES))
ELLIPSE_NOTE = "ellipse: 50% contour"

# 5 views x 18 spaces x 2 variants is 180 tracked PNGs, so the working figures
# stay at 150 dpi (~200 MB); the camera-ready path is vector PDF, not a bigger
# raster.
PNG_DPI = 150


def panels(view: str, labels: dict, idx: np.ndarray) -> list[tuple]:
    """(title, draw(ax, fig, xy)) per column, for the items selected by `idx`."""
    cat, t = labels["category"][idx], labels["t"][idx]
    out = []
    if view == "combined":
        out.append(("rule", lambda ax, fig, xy: class_cloud(
            ax, xy, RULE_OF_CATEGORY[cat], {i: RULE_STYLE[r] for i, r in
                                            RULE_BY_INDEX.items()},
            list(RULE_BY_INDEX), reference=0)))
    if view == "r5_flips":
        out.append(("discs flipped", lambda ax, fig, xy: continuous_cloud(
            ax, fig, xy, labels["n_flipped"][idx], "discs flipped", vmax=8)))
        out.append(("flip directions", lambda ax, fig, xy: continuous_cloud(
            ax, fig, xy, labels["n_directions"][idx], "flip directions")))
    else:
        present = [i for i in range(len(CATEGORIES)) if (cat == i).any()]
        out.append(("violation category", lambda ax, fig, xy: class_cloud(
            ax, xy, cat, {i: CATEGORY_STYLE[c] for i, c in
                          CATEGORY_BY_INDEX.items()}, present, reference=0)))
    out.append(("move number t (diagnostic)",
                lambda ax, fig, xy: continuous_cloud(ax, fig, xy, t, "move number t")))
    return out


def axis_labels(method: str, evr: list) -> tuple[str, str]:
    if method == "PCA":
        pct = [f" ({100 * v:.1f}% var)" for v in evr] if evr else ["", ""]
        return f"PC1{pct[0]}", f"PC2{pct[1]}"
    return "Isomap 1 (arb. units)", "Isomap 2 (arb. units)"


def render(space: str, view: str, variant: str, arrays, labels, evr,
           camera_ready: bool) -> None:
    rows = [("PCA", arrays[f"{view}_{variant}_pca"].astype(np.float32),
             labels[f"items_{view}"]),
            ("Isomap", arrays[f"{view}_{variant}_isomap"].astype(np.float32),
             labels[f"iso_{view}"])]
    ncol = len(panels(view, labels, labels[f"items_{view}"]))
    fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7.0), squeeze=False)

    for r, (method, xy, idx) in enumerate(rows):
        xlab, ylab = axis_labels(method, evr.get(f"{space}/{view}/{variant}", []))
        for c, (title, draw) in enumerate(panels(view, labels, idx)):
            ax = axes[r][c]
            draw(ax, fig, xy)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(xlab)
            ax.text(0.0, 1.02, f"({chr(97 + r * ncol + c)})", va="bottom",
                    transform=ax.transAxes, color=INK, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{method} — {ylab}")
            if r == 0:
                ax.set_title(title)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc="best", fontsize=7, title=ELLIPSE_NOTE,
                              title_fontsize=7, labelspacing=0.3, frameon=True,
                              framealpha=0.85, edgecolor="none")

    if not camera_ready:
        fig.suptitle(f"{space} — {view} ({variant})", y=0.995)
    fig.tight_layout()
    rule_key(fig, VIEWS[view])
    stem = HERE / "figures" / view / (space if variant == "centered" else f"{space}_raw")
    save_figure(fig, stem, camera_ready, dpi=PNG_DPI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)

    labels = dict(np.load(EMB / "labels.npz").items())
    evr = json.loads((EMB / "evr.json").read_text())
    spaces = sorted(f.stem for f in EMB.glob("*.npz") if f.name != "labels.npz")
    for space in spaces:
        arrays = np.load(EMB / f"{space}.npz")
        for view in VIEWS:
            for variant in ["centered", "raw"]:
                render(space, view, variant, arrays, labels, evr,
                       args.camera_ready)
        print(f"rendered {space}")


if __name__ == "__main__":
    main()
