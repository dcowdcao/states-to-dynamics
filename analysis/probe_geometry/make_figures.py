"""Probe directions in OthelloGPT's layer-5 residual space: a PCA and their cosines.

Draws the CSVs written by src/probe_geometry/embed_directions.py; recomputes nothing.

Usage: python analysis/probe_geometry/make_figures.py [--camera-ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from spade.figures.paper_style import (
    DIRECTION_ORDER, DIRECTION_STYLE, GRID, INK, MUTED,
    apply_style, contrast_key, save_figure)

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
DIVERGING = "RdBu_r"


def panel_letter(ax, letter: str) -> None:
    ax.text(0.0, 1.02, f"({letter})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def pca_panel(ax, points: pd.DataFrame, arrows: pd.DataFrame, ratio: pd.DataFrame) -> None:
    span = 1.08 * max(np.abs(points[["pc1", "pc2"]].to_numpy()).max(),
                      np.abs(arrows[["pc1", "pc2"]].to_numpy()).max())
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=GRID, lw=0.6, zorder=0)
    for family in DIRECTION_ORDER:
        style, at = DIRECTION_STYLE[family], points[points.family == family]
        ax.scatter(at.pc1, at.pc2, s=18, marker=style["marker"], color=style["color"],
                   alpha=0.45, lw=0, zorder=3,
                   label=f"{style['label']} (n={len(at)})")
    for family in DIRECTION_ORDER:
        style = DIRECTION_STYLE[family]
        head = arrows[arrows.family == family].iloc[0]
        ax.annotate("", xy=(head.pc1, head.pc2), xytext=(0, 0), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.8,
                                    shrinkA=0, shrinkB=0,
                                    path_effects=[pe.withStroke(linewidth=3.2,
                                                                foreground="white")]))
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({ratio.variance_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ratio.variance_ratio[1]:.1%} var)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=7,
              labelspacing=0.3, columnspacing=1.1, handletextpad=0.3,
              title="point = one board square; arrow = the committed probe, same origin",
              title_fontsize=7)


def cosine_panel(fig, ax, gram: pd.DataFrame) -> None:
    """Lower triangle only: cos is symmetric, so the other half is the same numbers."""
    labels = [DIRECTION_STYLE[f]["label"].replace(" (reference)", "") for f in gram.index]
    shown = gram.to_numpy().astype(float)
    shown[np.triu_indices_from(shown, k=1)] = np.nan
    cmap = plt.get_cmap(DIVERGING).copy()
    cmap.set_bad("white")

    im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
    for i in range(len(gram)):
        for j in range(i + 1):
            v = shown[i, j]
            ax.text(j, i, f"{v:+.2f}".replace("+1.00", "1.00"), ha="center", va="center",
                    fontsize=6.8, color="white" if abs(v) > 0.55 else INK)
    ax.set_xticks(range(len(gram)), labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(gram)), labels, fontsize=7.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[-1, 0, 1])
    bar.set_label("cos  (+1 aligned, 0 orthogonal, $-$1 opposed)", fontsize=7.5)
    bar.ax.tick_params(labelsize=7.5, length=0)
    bar.outline.set_visible(False)


def figure(camera_ready: bool) -> Path:
    points = pd.read_csv(RESULTS / "pca_points.csv")
    arrows = pd.read_csv(RESULTS / "pca_arrows.csv")
    ratio = pd.read_csv(RESULTS / "pca_variance.csv")
    gram = pd.read_csv(RESULTS / "type_cosines.csv", index_col=0).loc[
        list(DIRECTION_ORDER), list(DIRECTION_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3))
    pca_panel(axes[0], points, arrows, ratio)
    cosine_panel(fig, axes[1], gram)
    for ax, letter in zip(axes, "ab"):
        panel_letter(ax, letter)
    contrast_key(fig, ["colour", "occupancy", "rules"], [], y=-0.03)
    fig.tight_layout()
    return save_figure(fig, FIGURES / "fig1_direction_geometry", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    print(figure(args.camera_ready))


if __name__ == "__main__":
    main()
