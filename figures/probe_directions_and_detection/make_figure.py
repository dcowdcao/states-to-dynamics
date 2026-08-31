"""Where the probe's rule directions sit, and how well those probes separate a spliced move.

Three panels of one argument, which is why they are one figure: A and B say the five rule
directions are near-mutually-orthogonal and that `legal` and R1 are the pair the probe cannot
separate; C says what that costs at the spliced move, where R1 is the short bar. Read apart,
C looks like a probe that works; read beside A and B, its R1 gap has a geometric reason.

Draws committed CSVs and recomputes nothing, so it cannot disagree with the two studies it
combines. The panels keep their own drawing code too -- A and B are imported from
`analysis/probe_geometry/make_figures`, not reimplemented -- so a change there lands here.

**The two panels do not share a corpus, on purpose.** A and B are the probe's own weights,
refit per candidate square on the move-legality fit set (800 games); C evaluates those
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
# The paired rows over the 500 games `illegal_move_detector`'s population figures score --
# `detection_auroc` reads that study's `eval_games.csv`, so the panels of this figure and those
# two are one evaluation set. The CSV also holds those pairs over all 3,923 games, and the
# t0-stratified even split, which deals the corpus into
# five disjoint groups of 200 and pays about twice the interval width for mutual independence
# between bars that nothing here tests. `control` is a row of both heads -- the legal-vs-legal
# null -- and is not a bar: this panel draws the four arms the split one drew.
DRAWN = "paired on game, n = 500"
BARS = ["illegal", *RULES]

BAR_STYLE = {"illegal": {**CLASS_STYLE["violation"], "label": "legal / illegal"},
             **{r: {**PROBE_CLASS_STYLE[r], "label": r} for r in RULES}}


def letters(fig, axes: list) -> None:
    """Bold A. B. C. at each panel's top-left, on one baseline.

    Each panel supplies its own left edge, but the three share the highest top: the heatmap is
    squared by its aspect and the MDS cloud gives room to a legend, so their drawn boxes are
    unequal and a per-axes offset would step the letters down across the row.
    """
    fig.canvas.draw()
    boxes = [ax.get_window_extent().transformed(fig.transFigure.inverted()) for ax in axes]
    top = max(b.y1 for b in boxes) + 0.015
    for i, b in enumerate(boxes):
        fig.text(b.x0, top, f"{chr(65 + i)}.", ha="left", va="bottom", color=INK,
                 fontsize=12, fontweight="bold")


def detection_panel(ax, bars: pd.DataFrame) -> None:
    """C redrawn from `detection_auroc_raw.csv`, the even-split rows only."""
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
    bars = detection[(detection.pairing == DRAWN)
                     & detection.arm.isin(BARS)].reset_index(drop=True)
    assert len(bars) == len(BARS), f"expected {len(BARS)} paired bars, found {len(bars)}"

    # One row of three. C's x axis is four bars and reads at any width, so it sits beside the
    # two square-ish panels rather than under them; A's legend hangs below its own axes and
    # bbox_inches="tight" keeps it.
    fig = plt.figure(figsize=(14.0, 4.4))
    grid = GridSpec(1, 3, figure=fig, wspace=0.30)
    left, middle, right = (fig.add_subplot(grid[0, i]) for i in range(3))

    embed_panel(left, points, probes)
    cosine_panel(fig, middle, gram)
    detection_panel(right, bars)
    letters(fig, [left, middle, right])

    return print(save_figure(fig, HERE / "probe_directions_and_detection", args.camera_ready))


if __name__ == "__main__":
    main()
