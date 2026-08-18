"""The headline figure for the probe-geometry study: board read-out and rule
legality look orthogonal only while legality is pooled over candidate squares.

Drawing only: everything is read from the committed CSVs in results/ (written by
src/probe_geometry/embed_directions.py and src/probe_geometry/cosine_tables.py).
Conventions: FIGURE_GUIDELINES.md 4.4, 4.6, 4.8.

Usage: python analysis/probe_geometry/make_figures.py [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from spade.figures.paper_style import (
    DIRECTION_STYLE, INK, MUTED, NULL, apply_style, contrast_key, save_figure)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

D_MODEL = 512
ISO_SD = 1 / np.sqrt(D_MODEL)  # SD of cos between two random directions in 512-d
POOLED = ["illegal", "R1", "R2", "R3"]
BOARD = ["colour", "occupancy"]
# Rows and columns of the cosine matrix, in taxonomy order, each block sorted by
# engine square. Never clustered: clustering would manufacture the blocks, and
# the shared square order is what puts the matched pairs on a visible line.
BLOCKS = [("colour", "board\ncolour"), ("occupancy", "board\noccupancy"),
          ("rules_square", "legality\nper square"), ("rules4", "R1$-$R3"),
          ("rules2", "illegal")]
JITTER_SEED = 0
DIVERGING = "RdBu_r"   # signed cosine, centred at zero
MAGNITUDE = "Greys"    # mean |cos|, a non-negative quantity


def read(name: str) -> pd.DataFrame:
    """`null` is a family name in these tables, not a missing value."""
    return pd.read_csv(RESULTS / f"{name}.csv", keep_default_na=False, na_values=[""])


def panel_letter(ax, i: int) -> None:
    ax.text(-0.13, 1.05, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def scatter_panel(ax, coords: pd.DataFrame, variance: pd.DataFrame) -> None:
    """Where the 188 directions land in the first two components of an uncentred
    joint PCA. The four pooled rules directions are drawn large and labelled;
    everything else is a cloud."""
    null = coords[coords.family == "null"]
    ax.scatter(null.pc1, null.pc2, s=9, lw=0, alpha=0.55, zorder=1,
               color=NULL["color"], marker=NULL["marker"],
               label=f"random direction (null, n={len(null)})")
    for family in BOARD + ["rules_square"]:
        d, style = coords[coords.family == family], DIRECTION_STYLE[family]
        ax.scatter(d.pc1, d.pc2, s=30 if family == "rules_square" else 22,
                   lw=0.4, alpha=0.85, zorder=3, color=style["color"],
                   marker=style["marker"], edgecolor="white",
                   label=f"{style['label']} (n={len(d)})")
    for name in POOLED:
        d, style = coords[coords.name == name].iloc[0], DIRECTION_STYLE[name]
        ax.plot(d.pc1, d.pc2, ls="none", marker=style["marker"], ms=11, zorder=5,
                color=style["color"], markeredgecolor="white", markeredgewidth=1.1)
        ax.annotate(name, (d.pc1, d.pc2), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8, zorder=6,
                    color=style["color"])
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)
    joint = variance[variance.fit == "joint"].variance_ratio.to_numpy()
    ax.set_xlabel(f"PC1 ({100 * joint[0]:.1f}% var)\n"
                  "uncentred PCA of the 188 unit directions; the null is\n"
                  "projected into that basis, not fitted to it", fontsize=8)
    ax.set_ylabel(f"PC2 ({100 * joint[1]:.1f}% var)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.08, y=0.12)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.34 * (hi - lo), hi)  # room for the legend under the cloud
    ax.set_title("the pooled rules axes sit off every cloud", fontsize=9, pad=8)
    ax.legend(loc="lower left", fontsize=6.8, labelspacing=0.3, borderpad=0.2)


def capture_panel(ax, capture: pd.DataFrame) -> None:
    """How much of a legality direction the board's own subspace already holds.
    A random direction is the thing to beat, and it is not at zero."""
    null = capture[capture.name == "null"]
    ax.fill_between(null.k, null.lo, null.hi, color=NULL["color"], alpha=0.25,
                    lw=0, zorder=1)
    ax.plot(null.k, null.captured, color=NULL["color"], ls=NULL["ls"],
            lw=NULL["lw"], marker=NULL["marker"], ms=4, markevery=25,
            markeredgecolor="white", markeredgewidth=0.5, zorder=2,
            label=f"random direction in {D_MODEL}-d (null, n={int(null.n.iloc[0])})")
    for i, name in enumerate(POOLED + ["rules_square"]):
        d, style = capture[capture.name == name], DIRECTION_STYLE[name]
        n = int(d.n.iloc[0])
        if n > 1:
            ax.fill_between(d.k, d.lo, d.hi, color=style["color"], alpha=0.18,
                            lw=0, zorder=3)
        ax.plot(d.k, d.captured, color=style["color"], lw=2.4 if n > 1 else 1.5,
                marker=style["marker"], ms=5 if n > 1 else 4,
                markevery=(i * 5 + 8, 25), markeredgecolor="white",
                markeredgewidth=0.5, zorder=4,
                label=f"{style['label']}: {d.captured.iloc[-1]:.2f}"
                      + (f" (n={n})" if n > 1 else ""))
    ax.set_xlim(1, capture.k.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("k, principal components of the 124 board directions\n"
                  "(the board subspace is rank 124 of 512)")
    ax.set_ylabel("captured fraction of a legality direction\n"
                  r"($\downarrow$ more orthogonal to the board subspace)")
    ax.set_title("pooled legality is captured less than a random\n"
                 "direction; per-square legality is not", fontsize=9, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="upper left",
              fontsize=7, labelspacing=0.3,
              title=f"value at k={int(capture.k.max())}. Bands: 95% interval, over\n"
                    "the squares and over the draws; the four pooled curves\n"
                    "are one fit each, so they carry no band.",
              title_fontsize=6.8)


def square_pairs(cos: pd.DataFrame) -> pd.DataFrame:
    """Per-square legality against a board direction, tagged matched vs. other square."""
    d = cos[(cos.family_a == "rules_square") & (cos.family_b.isin(BOARD))
            | (cos.family_b == "rules_square") & (cos.family_a.isin(BOARD))].copy()
    d["family"] = np.where(d.family_a == "rules_square", d.family_b, d.family_a)
    d["matched"] = d.square_a == d.square_b
    return d


def matched_panel(ax, cos: pd.DataFrame) -> None:
    """One point per square: how well legality at square q lines up with the board
    read-out at the same square, against the same cosines at every other square."""
    d = square_pairs(cos)
    other = d[~d.matched]["cos"]
    ax.axhspan(*np.percentile(other, [2.5, 97.5]), color=NULL["color"], alpha=0.25,
               lw=0, zorder=1, label=f"same pairs, other squares (n={len(other):,})")
    for sign in (1, -1):
        ax.axhline(sign * 1.96 * ISO_SD, color=NULL["color"], ls=NULL["ls"],
                   lw=1.0, zorder=2,
                   label="isotropic random 95%" if sign == 1 else None)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)

    rng = np.random.default_rng(JITTER_SEED)
    matched = {f: d[d.matched & (d.family == f)]["cos"].to_numpy() for f in BOARD}
    for x, family in enumerate(BOARD):
        v, style = matched[family], DIRECTION_STYLE[family]
        # The two families are named on the x axis, so they stay out of the legend.
        ax.plot(x + rng.uniform(-0.13, 0.13, len(v)), v, ls="none",
                marker=style["marker"], ms=5, alpha=0.75, zorder=4,
                color=style["color"], markeredgecolor="white", markeredgewidth=0.4)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([x - 0.3, x + 0.3], [med, med], color=INK, lw=1.6, zorder=6)
        ax.plot([x + 0.34, x + 0.34], [q1, q3], color=INK, lw=1.6, zorder=6)
        ax.annotate(f"median {med:+.2f}\nIQR {q1:+.2f} to {q3:+.2f}", (x + 0.38, med),
                    fontsize=7, va="center", color=INK)
    ax.set_xlim(-0.55, 1.95)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.24 * (hi - lo), hi)  # room for the legend under the points
    ax.set_xticks(range(len(BOARD)), [f"{DIRECTION_STYLE[f]['label'].split(' (')[0]}\n"
                                      "at the same square" for f in BOARD])
    ax.set_ylabel(r"cos(legality at square $q$, board read-out at $q$)"
                  "\n" r"$+$ = illegal looks like 'mine' / 'occupied'")
    ax.set_title("at its own square, legality is aligned with\nthe board read-out",
                 fontsize=9, pad=8)
    ax.legend(loc="lower left", fontsize=7, labelspacing=0.3,
              title=f"one marker per square (n={len(matched[BOARD[0]])} squares); "
                    "bar = median, whisker = IQR", title_fontsize=6.8)


def cosine_matrix(cos: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """The symmetric cosine matrix and its taxonomy order, from the upper triangle."""
    columns = ["family", "name", "square"]
    ends = pd.concat([cos[[f"{c}_a" for c in columns]].set_axis(columns, axis=1),
                      cos[[f"{c}_b" for c in columns]].set_axis(columns, axis=1)]
                     ).drop_duplicates("name")
    ends["rank"] = ends.family.map({f: i for i, (f, _) in enumerate(BLOCKS)})
    ends = ends.sort_values(["rank", "square", "name"], na_position="last")
    order = list(ends.name)
    index = {name: i for i, name in enumerate(order)}
    m = np.eye(len(order))
    i, j = cos.name_a.map(index).to_numpy(), cos.name_b.map(index).to_numpy()
    m[i, j] = m[j, i] = cos["cos"].to_numpy()
    return m, order, list(ends.family)


def block_means(cos: pd.DataFrame) -> np.ndarray:
    """Mean |cos| per family pair; the diagonal is within-family, distinct pairs only."""
    families = [f for f, _ in BLOCKS]
    a = cos.family_a.map(families.index).to_numpy()
    b = cos.family_b.map(families.index).to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    means = np.full((len(families), len(families)), np.nan)
    for p in range(len(families)):
        for q in range(p, len(families)):
            v = np.abs(cos["cos"].to_numpy()[(lo == p) & (hi == q)])
            means[p, q] = means[q, p] = v.mean() if len(v) else np.nan
    return means


def block_inset(ax, means: np.ndarray) -> None:
    """Annotated family-vs-family mean |cos|, in the triangle the main matrix leaves
    empty. Grayscale sends +0.6 and -0.6 to the same ink, so the numbers ship."""
    inset = ax.inset_axes([0.20, 0.04, 0.40, 0.40])
    top = np.nanmax(means)
    inset.imshow(means, cmap=MAGNITUDE, vmin=0, vmax=top)
    for p in range(len(means)):
        for q in range(len(means)):
            text = "--" if np.isnan(means[p, q]) else f"{means[p, q]:.3f}".lstrip("0")
            inset.text(q, p, text, ha="center", va="center", fontsize=5.8,
                       color="white" if means[p, q] > 0.5 * top else INK)
    inset.set_xticks([])
    inset.set_yticks(range(len(means)), [label for _, label in BLOCKS], fontsize=5.2)
    inset.tick_params(length=0, pad=1.5)
    inset.grid(False)
    inset.set_title("mean $|\\cos|$ per family pair, rows = columns.\n"
                    "Pooling hides the matched square: see (c).",
                    fontsize=6.2, pad=3)


def cosine_panel(ax, fig, cos: pd.DataFrame) -> None:
    m, order, families = cosine_matrix(cos)
    upper = np.where(np.triu(np.ones_like(m), 1) > 0, m, np.nan)
    vmax = np.nanmax(np.abs(upper))
    im = ax.imshow(upper, cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    bounds = [families.index(f) for f, _ in BLOCKS[1:]]
    for b in bounds:
        ax.axhline(b - 0.5, color=MUTED, lw=0.5)
        ax.axvline(b - 0.5, color=MUTED, lw=0.5)
    # The two pooled blocks are 3 and 1 column wide, so they share one tick.
    edges = [0] + bounds[:3] + [len(order)]
    centres = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    names = ["colour", "occupancy", "per square", "pooled"]
    ax.set_xticks(centres, names, fontsize=6, rotation=30, ha="right")
    ax.set_yticks(centres, names, fontsize=6, rotation=90, va="center")
    ax.tick_params(length=0)
    ax.grid(False)
    ax.annotate("matched square", (bounds[1] + 30, 30), xytext=(-58, -40),
                textcoords="offset points", fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK,
                                shrinkA=0, shrinkB=2))
    ax.set_title("every direction pair, in taxonomy order", fontsize=9, pad=8)

    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                       ticks=[-vmax, -0.5, 0, 0.5, vmax])
    bar.ax.set_yticklabels([f"$-${vmax:.2f}\nanti-aligned", "$-$0.5", "0", "$+$0.5",
                            f"$+${vmax:.2f}\naligned"], fontsize=6.5)
    bar.set_label(r"cos($u$, $v$)", fontsize=7.5)
    bar.outline.set_visible(False)
    block_inset(ax, block_means(cos))


def figure_one(camera_ready: bool) -> None:
    cos = read("direction_cosines")
    # 9.0 in is the guidelines' width for a two-panel-wide working figure, and
    # the fonts are sized for it: drawn any wider, camera-ready shrinks them out.
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2),
                             gridspec_kw={"wspace": 0.36, "hspace": 0.46})
    scatter_panel(axes[0, 0], read("pca_coords"), read("pca_variance"))
    capture_panel(axes[0, 1], read("pca_capture"))
    matched_panel(axes[1, 0], cos)
    cosine_panel(axes[1, 1], fig, cos)
    for i, ax in enumerate(axes.ravel()):
        panel_letter(ax, i)
    if not camera_ready:
        fig.suptitle("Board read-out and rule legality look orthogonal only while "
                     "legality is pooled over candidate squares", y=0.97)
    contrast_key(fig, BOARD + ["rules_square"], ["R1", "R2", "R3"], y=0.02)
    save_figure(fig, FIGURES / "fig1_direction_geometry", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    figure_one(args.camera_ready)


if __name__ == "__main__":
    main()
