"""Probe directions in OthelloGPT's layer-5 residual space: a nonmetric MDS and their cosines.

Draws the CSVs written by spade.probe_geometry.embed_directions; recomputes nothing.

Usage: python analysis/probe_geometry/make_figures.py [--camera-ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

from spade.figures.paper_style import (
    DIRECTION_ORDER, DIRECTION_STYLE, INK, apply_style, save_figure)

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent / "figures"
DIVERGING = "RdBu_r"


def embed_panel(ax, points: pd.DataFrame, probes: pd.DataFrame) -> None:
    """Nonmetric MDS: distance is monotone in angle, not proportional to it.

    No origin line and no variance ratio -- SMACOF fixes a configuration only up to
    rigid motion, so the centre is the cloud's own centroid and both axes are arbitrary.
    The committed probe is drawn open so it cannot be read as the family centroid: they
    are different objects, and for R1 they differ by more than a within-family radius.
    """
    for family in DIRECTION_ORDER:
        style, at = DIRECTION_STYLE[family], points[points.family == family]
        xy = at[["x", "y"]].to_numpy()
        ax.scatter(*xy.T, s=18, marker=style["marker"], color=style["color"],
                   alpha=0.6, lw=0, zorder=3, label=f"{style['label']} (n={len(at)})")
        ax.plot(*xy.mean(0), ls="none", marker=style["marker"], ms=7,
                color=style["color"], markeredgecolor="white", markeredgewidth=1.1,
                zorder=6)
    for family in DIRECTION_ORDER:
        style = DIRECTION_STYLE[family]
        at = probes[probes.family == family].iloc[0]
        ax.plot(at.x, at.y, ls="none", marker="o", ms=10, mfc="none",
                markeredgecolor=style["color"], markeredgewidth=1.8, zorder=7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("MDS axis 1")
    ax.set_ylabel("MDS axis 2")
    # No legend title: what the markers mean, the missing origin, the monotone-not-
    # proportional distance and the fidelity all live in the README's Figure notes.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7,
              labelspacing=0.3, columnspacing=1.1, handletextpad=0.3)


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
    bar.set_label("cos", fontsize=7.5)
    bar.ax.tick_params(labelsize=7.5, length=0)
    bar.outline.set_visible(False)


def figure(camera_ready: bool) -> Path:
    points = pd.read_csv(RESULTS / "embed_points.csv")
    probes = pd.read_csv(RESULTS / "embed_probes.csv")
    gram = pd.read_csv(RESULTS / "type_cosines.csv", index_col=0).loc[
        list(DIRECTION_ORDER), list(DIRECTION_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3))
    embed_panel(axes[0], points, probes)
    cosine_panel(fig, axes[1], gram)
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
