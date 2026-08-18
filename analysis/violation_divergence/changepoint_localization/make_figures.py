"""Which representation space lets a reference-free change-point score catch the splice?

One bar per space: the paired AUROC of an illegal branch against the *same game's* matched
legal branch, both scored by the peak of the change-point statistic over a blind scan of the
whole game. Pairing on the legal splice is what makes the bar about illegality rather than
about a move having changed at all.

Sorted, because the ordering is the claim; raw layer 5 keeps reference styling wherever it
lands, so the reader can see how many dictionaries clear it.

Usage: python analysis/violation_divergence/changepoint_localization/make_figures.py
           [--camera-ready]
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.figures.paper_style import (
    MUTED, NULL, REFERENCE_SPACE, SPACE_STYLE, apply_style, save_figure)

HERE = Path(__file__).resolve().parent
BAR_WIDTH = 0.72


def draw(table: pd.DataFrame, camera_ready: bool) -> Path:
    table = table.sort_values("auroc_vs_control", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    ax.axhline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=1)
    for x, row in table.iterrows():
        style = SPACE_STYLE[row.space]
        # SPACE_STYLE separates the two dictionaries of a family by line style, which a bar
        # cannot carry; the hatch is that same dimension, so the pairs stay separable in
        # grayscale and the hue keeps meaning "architecture family".
        ax.bar(x, row.auroc_vs_control, width=BAR_WIDTH, color=style["color"],
               hatch=None if style["ls"] == "-" else "///",
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(x, row.auroc_vs_control,
                    yerr=[[row.auroc_vs_control - row.auroc_vs_control_lo],
                          [row.auroc_vs_control_hi - row.auroc_vs_control]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(x, row.auroc_vs_control_hi + 0.015, f"{row.auroc_vs_control:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=MUTED, zorder=4)

    label = lambda s: "raw L5 (ref.)" if s == REFERENCE_SPACE else SPACE_STYLE[s]["short"]
    ax.set_xticks(range(len(table)), [label(s) for s in table.space], fontsize=7.5,
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.7, len(table) + 0.55)
    ax.set_ylabel("paired AUROC, illegal vs matched legal splice\n($\\uparrow$ better)")
    ax.text(len(table) - 0.35, 0.5, "chance", color=NULL["color"], fontsize=7.5,
            ha="left", va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    ax.text(0.0, 1.045, "hue = architecture family, hatch = its second dictionary; "
            "raw layer 5 is the reference space", fontsize=7, color=MUTED, va="top")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

    row = table.iloc[0]
    fig.tight_layout()
    fig.text(0.005, -0.005,
             f"Score: how much shorter than usual the step through the space is, standardized per "
             f"move index against legal games,\nthen contrasted across each candidate move ({row.statistic}, "
             f"m = {row.m:.0f} plies each side) and scanned blind over the whole game -- no\nknowledge "
             f"that a move was spliced, and no counterfactual branch. The window was chosen on the "
             f"disjoint\nstraightness corpus. One illegal branch per game at seed 42; whiskers are 95% "
             f"percentile bootstrap CIs over\n{row.n_games:.0f} paired games (1,000 resamples). The "
             f"chance line is empirical: the legal control scored against the second\nlegal control each "
             f"game carries gives {table.auroc_null.min():.3f}-{table.auroc_null.max():.3f}. AUROCs are "
             f"never flipped, so a bar at 0.5 is the finding.",
             ha="left", va="top", fontsize=7, color=MUTED)
    return save_figure(fig, HERE / "figures" / "changepoint_auroc_by_space", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    table = pd.read_csv(HERE / "results" / "changepoint_localization.csv")
    print(draw(table, args.camera_ready))


if __name__ == "__main__":
    main()
