"""F7 of the violation-divergence study: what the layer-5 linear probe decodes off an illegal branch.

(a) does the illegal branch write the illegal disc, and for how long
(b) the occupied pair: does an illegal move on an occupied square read as a
    no-op or as a recolor

Every branch is scored against the reference game's board in the reference
game's frame; decode.py's docstring states the convention and RESULTS.md spells
it out. Conventions: FIGURE_GUIDELINES.md 4.2 (event-locked panels) and 3
(reference styled as a reference, null gray/dashed/last).

`--variant rollout` draws the same three read-outs off `probe_rollout_*.csv`,
where every branch samples its own continuation. The panels are read the same
way, with one caveat they carry: the reference's board is itself sampled there
and ends at its first illegal move, so the anchors thin out with r and the
per-offset n in the CSV is not constant.

Usage: python analysis/violation_divergence/probe_readout/make_figures.py
                                            [--variant rollout] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spade.common.paths import results_dir
from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, REFERENCE, apply_style, save_figure)
from spade.violation_divergence.divergence import WINDOW

RESULTS = results_dir("violation_divergence")
FIGURES = Path(__file__).resolve().parent / "figures"
VARIANTS = ("spliced", "rollout")

EMPTY_SQUARE_CATEGORIES = ["empty_detached", "empty_own_only", "empty_no_bracket", "opponent_only"]
OCCUPIED_CATEGORIES = ["occupied_own", "occupied_opp"]
# Six to eight curves per panel, so the CI is reported rather than shaded: eight
# overlapping ribbons hid the anchors the panels are read against.
ESTIMATOR = {"spliced": "mean over source games; 95% bootstrap CIs in the CSV",
             "rollout": "mean over source branches, K=4 averaged; "
                        "95% bootstrap CIs in the CSV"}


def curve(ax, df: pd.DataFrame, style: dict, label: str, offset: int = 0, **kw) -> None:
    d = df.sort_values("r")
    kw = {"color": style["color"], "marker": style["marker"], "markersize": 4,
          "markevery": (offset, 6), "markeredgecolor": "white", "markeredgewidth": 0.5,
          "label": label, **kw}
    ax.plot(d.r, d["mean"], **kw)


def event_axes(ax, ylabel: str, letter: str, title: str, ylim=(-0.03, 1.05)) -> None:
    ax.axvspan(WINDOW[0], 0, color=GRID, alpha=0.45, zorder=0)
    ax.axvline(0, color=MUTED, ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.set_xlabel("position relative to the branch move, $r = s - t_0$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(WINDOW[0], WINDOW[-1])
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.text(-0.14, 1.02, f"({letter})", va="bottom", transform=ax.transAxes, color=INK, fontsize=9)


def label_with_n(df: pd.DataFrame, key: str, category: str) -> str:
    n = int(df[(df[key] == category) & (df.r == 0)].iloc[0].n)
    return f"{CATEGORY_STYLE[category]['label']} (n={n})"


def curve_legend(ax, **kw) -> None:
    ax.legend(title=ESTIMATOR["spliced"], fontsize=7.4, title_fontsize=7,
              alignment="left", **kw)


def occupancy_panel(ax, square: pd.DataFrame, in_panel_legend: bool) -> None:
    p = square[square.metric == "p_disc"]
    illegal = p[p.role == "illegal"]
    for i, category in enumerate(EMPTY_SQUARE_CATEGORIES):
        curve(ax, illegal[illegal.category == category], CATEGORY_STYLE[category],
              label_with_n(illegal, "category", category), offset=i, zorder=5)
    ceiling = p[p.category == "reference_move"]
    curve(ax, ceiling, REFERENCE, f"reference branch, its own move (n={int(ceiling.iloc[0].n)})",
          offset=2, ls=(0, (1, 1.4)), lw=1.6, zorder=4)
    pooled = p[p.category == "empty_pooled"]
    curve(ax, pooled[pooled.role == "reference"], REFERENCE,
          f"reference branch, same square (n={int(pooled.iloc[0].n)} games)",
          offset=0, lw=REFERENCE["lw"], zorder=3)
    curve(ax, pooled[pooled.role == "control"], NULL,
          "control_legal branch, same square (coincides)", offset=3,
          ls=NULL["ls"], lw=NULL["lw"], zorder=4)
    if in_panel_legend:
        curve_legend(ax, loc="center left", bbox_to_anchor=(0.28, 0.62))


def recolor_panel(ax, square: pd.DataFrame, in_panel_legend: bool) -> None:
    p = square[square.metric == "p_mover_colour"]
    for i, category in enumerate(OCCUPIED_CATEGORIES):
        short = CATEGORY_STYLE[category]["label"].removeprefix("R1 ")
        illegal = p[(p.category == category) & (p.role == "illegal")]
        curve(ax, illegal, CATEGORY_STYLE[category],
              f"{short} (n={int(illegal[illegal.r == 0].iloc[0].n)})", offset=i, zorder=5)
        curve(ax, p[(p.category == category) & (p.role == "reference")],
              CATEGORY_STYLE[category], f"{short}, reference branch",
              offset=i + 3, ls=(0, (1, 1.4)), lw=1.3, zorder=3)
    if in_panel_legend:
        curve_legend(ax, loc="lower left", bbox_to_anchor=(0.02, 0.30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="spliced", choices=VARIANTS,
                        help="rollout: every branch samples its own continuation")
    parser.add_argument("--camera-ready", action="store_true")
    args = parser.parse_args()
    apply_style(camera_ready=args.camera_ready)
    stem = "probe" if args.variant == "spliced" else f"probe_{args.variant}"
    figure = "f7_probe_readout" + ("" if args.variant == "spliced" else f"_{args.variant}")

    square = pd.read_csv(RESULTS / f"{stem}_illegal_square.csv")

    # The parity panel is dropped: engine-replay mover and t%2 disagree on 0.20%
    # of cells in this window and the probe is at chance on exactly those, so the
    # curve sits flat at ~1 and carries no result. probe_parity.csv keeps it.
    spliced = args.variant == "spliced"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    occupancy_panel(axes[0], square, spliced)
    event_axes(axes[0], "P(illegally played square decodes as occupied)", "a",
               "" if args.camera_ready else "the illegal disc is written, and it stays")
    recolor_panel(axes[1], square, spliced)
    event_axes(axes[1], "P(that square decodes the mover's own colour)", "b",
               "" if args.camera_ready else "occupied squares: no-op or recolor")

    fig.tight_layout()
    if not spliced:
        # The in-panel legends are placed against the spliced curves; here the
        # anchors move, so both legends go under the figure.
        for i, ax in enumerate(axes):
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper left", fontsize=7,
                       bbox_to_anchor=(0.04 + 0.5 * i, -0.02), title=ESTIMATOR["rollout"],
                       title_fontsize=7, alignment="left")
    print(save_figure(fig, FIGURES / figure, camera_ready=args.camera_ready))


if __name__ == "__main__":
    main()
