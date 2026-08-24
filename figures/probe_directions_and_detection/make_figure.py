"""Where the probe's rule directions sit, and how well those probes separate a spliced move.

Three panels of one argument, which is why they are one figure: (a) and (b) say the five rule
directions are near-mutually-orthogonal and that `legal` and R1 are the pair the probe cannot
separate; (c) says what that costs at the spliced move, where R1 is the short bar. Read apart,
(c) looks like a probe that works; read under (a)-(b), its R1 gap has a geometric reason.

Draws committed CSVs and recomputes nothing, so it cannot disagree with the two studies it
combines. The panels keep their own drawing code too -- (a) and (b) are imported from
`analysis/probe_geometry/make_figures`, not reimplemented -- so a change there lands here.

**The two panels do not share a corpus, on purpose.** (a) and (b) are the probe's own weights,
refit per candidate square on the move-legality fit set (800 games); (c) evaluates those
probes on the violation-divergence branch tree (3,923 games), which is disjoint from the fit
set by exact token match. Pointing them at one corpus would mean scoring a probe on the rows
it was fitted on.

Usage: python figures/probe_directions_and_detection/make_figure.py [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from spade.common.paths import repo_root
from spade.figures.paper_style import (
    CLASS_STYLE, DIRECTION_ORDER, INK, MUTED, NULL, PROBE_CLASS_STYLE, apply_style,
    save_figure)

sys.path.insert(0, str(repo_root() / "analysis" / "probe_geometry"))
from make_figures import cosine_panel, embed_panel  # noqa: E402  (sibling study, not a package)

HERE = Path(__file__).resolve().parent
GEOMETRY = repo_root() / "analysis" / "probe_geometry" / "results"
DETECTION = repo_root() / "analysis" / "move_legality" / "results"
RULES = ["R1", "R2", "R3"]
GAP = 0.8                    # blank slot between the two probe heads, as in the source figure
DRAWN = "disjoint groups"    # the t0-stratified even split; the CSV also holds the paired form

BAR_STYLE = {"illegal": {**CLASS_STYLE["violation"], "label": "illegal / legal"},
             **{r: {**PROBE_CLASS_STYLE[r], "label": r} for r in RULES}}


def letters(fig, rows: list[list]) -> None:
    """Panel letters UNDER each plot, placed off the rendered extents.

    A fixed axes-fraction offset cannot align them: (a) carries a two-row legend below its
    axes and (b) carries tick labels, so the same offset lands in different places. Measuring
    each panel's tight bbox puts every letter just below whatever that panel actually occupies,
    and a row shares one baseline so (a) and (b) line up.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = lambda ax: ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    i = 0
    for row in rows:
        boxes = [box(ax) for ax in row]
        base = min(b.y0 for b in boxes) - 0.018
        for b in boxes:
            fig.text((b.x0 + b.x1) / 2, base, f"({chr(97 + i)})", ha="center", va="top",
                     color=INK, fontsize=10)
            i += 1


def detection_panel(ax, bars: pd.DataFrame) -> None:
    """(c) redrawn from `detection_auroc_raw.csv`, the even-split rows only."""
    x = np.array([i + (GAP if r.head == "4-class" else 0.0) for i, r in enumerate(bars.itertuples())])
    split = int(np.argmax(bars["head"].to_numpy() == "4-class"))

    ax.axhline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=1)
    ax.text(x[split - 1:split + 1].mean(), 0.515, "chance", color=NULL["color"], fontsize=7.5,
            ha="center", va="bottom")
    for pos, row in zip(x, bars.itertuples()):
        style = BAR_STYLE[row.arm]
        ax.bar(pos, row.auroc, width=0.72, color=style["color"], edgecolor="white",
               linewidth=0.6, zorder=3)
        ax.errorbar(pos, row.auroc, yerr=[[row.auroc - row.lo], [row.hi - row.auroc]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(pos, row.hi + 0.02, f"{row.auroc:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=MUTED, zorder=4)

    for head in ["2-class", "4-class"]:
        at = x[bars["head"].to_numpy() == head]
        ax.text(at.mean(), 1.10, f"{head} probe", ha="center", va="bottom", fontsize=9)
        ax.plot([at[0] - 0.45, at[-1] + 0.45], [1.07, 1.07], color=MUTED, lw=0.8, clip_on=False)

    ax.set_xticks(x, [BAR_STYLE[r.arm]["label"] for r in bars.itertuples()], fontsize=8.4)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    # No x label: the evaluation set (1000 games split evenly over the five conditions and
    # stratified by t0, 200 per group) is the caption's job, and it is stated in
    # analysis/move_legality/README.md next to the numbers it describes.
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    apply_style(args.camera_ready)

    points = pd.read_csv(GEOMETRY / "embed_points.csv")
    probes = pd.read_csv(GEOMETRY / "embed_probes.csv")
    gram = pd.read_csv(GEOMETRY / "type_cosines.csv", index_col=0).loc[
        list(DIRECTION_ORDER), list(DIRECTION_ORDER)]
    detection = pd.read_csv(DETECTION / "detection_auroc_raw.csv")
    bars = detection[detection.pairing == DRAWN].reset_index(drop=True)
    assert len(bars) == 4, f"expected 4 even-split bars, found {len(bars)}"

    # (c) spans the width because its x axis is four bars and a sentence; the top row is two
    # square-ish panels. Heights favour the top row, where the MDS cloud needs the room.
    fig = plt.figure(figsize=(9.0, 8.0))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.86], hspace=0.62, wspace=0.26)
    top_left, top_right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    bottom = fig.add_subplot(grid[1, :])

    embed_panel(top_left, points, probes)
    cosine_panel(fig, top_right, gram)
    detection_panel(bottom, bars)
    letters(fig, [[top_left, top_right], [bottom]])

    return print(save_figure(fig, HERE / "probe_directions_and_detection", args.camera_ready))


if __name__ == "__main__":
    main()
