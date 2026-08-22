"""Which representation space lets a reference-free change-point score catch the splice?

`changepoint_by_space` -- two panels over the same nine spaces and the same sort order, so a
space is one column in both: (a) does the score know a game contains an illegal move at all,
(b) does its argmax say which ply. They are different questions and a space can be good at one
and not the other, which is why they are two panels and not one number.

`threshold_detection` -- the same score read as a detector that has to commit. (a) is the
per-position AUROC, illegal move against legal move, so it asks whether the score orders
positions at all; (b) is what a fitted cut actually buys out of fold. A space can rank well
and still have no cut that transfers, which is why these are also two panels.

Sorted by (a), because the ordering is the claim; raw layer 5 keeps reference styling wherever
it lands, so the reader can see how many dictionaries clear it, and keeping one order across
both panels is what makes the two rankings comparable.

No note under the axes: the estimator, n, seed and null semantics are the caption, and the
caption is the paragraph under "Results" in README.md. These figures are not standalone.

Usage: python analysis/violation_divergence/changepoint_localization/make_figures.py
           [--camera-ready]
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.figures.paper_style import (
    INK, MUTED, NULL, REFERENCE_SPACE, SPACE_STYLE, apply_style, save_figure)

HERE = Path(__file__).resolve().parent
BAR_WIDTH = 0.72
FALSE_ALARM = ["hit_control", "hit_control2", "hit_reference"]


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def panel(ax, table: pd.DataFrame, value: str, ylabel: str, chance: float,
          band: tuple[float, float] | None, index: int, top: float,
          chance_label: str) -> None:
    if band is not None:
        ax.axhspan(*band, color=NULL["color"], alpha=0.28, lw=0, zorder=1)
    ax.axhline(chance, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=2)
    for x, row in table.iterrows():
        style = SPACE_STYLE[row.space]
        # SPACE_STYLE separates the two dictionaries of a family by line style, which a bar
        # cannot carry; the hatch is that same dimension, so the pairs stay separable in
        # grayscale and the hue keeps meaning "architecture family".
        ax.bar(x, row[value], width=BAR_WIDTH, color=style["color"],
               hatch=None if style["ls"] == "-" else "///",
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(x, row[value], yerr=[[row[value] - row[f"{value}_lo"]],
                                         [row[f"{value}_hi"] - row[value]]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(x, row[f"{value}_hi"] + top * 0.02, f"{row[value]:.3f}", ha="center",
                va="bottom", fontsize=6.6, color=MUTED, zorder=4, rotation=90)

    label = lambda s: "raw L5 (ref.)" if s == REFERENCE_SPACE else SPACE_STYLE[s]["short"]
    ax.set_xticks(range(len(table)), [label(s) for s in table.space], fontsize=6.8,
                  rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.7, len(table) + 0.5)
    ax.set_ylabel(ylabel)
    ax.text(len(table) - 0.35, chance, chance_label, color=NULL["color"], fontsize=6.8,
            ha="left", va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    panel_letter(ax, index)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def finish(fig, stem: str, camera_ready: bool) -> Path:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.005, 0.985, "hue = architecture family, hatch = its second dictionary; "
             "raw layer 5 is the reference space", fontsize=7, color=MUTED, va="top")
    return save_figure(fig, HERE / "figures" / stem, camera_ready)


def draw_by_space(table: pd.DataFrame, camera_ready: bool) -> Path:
    table = table.sort_values("auroc_vs_control", ascending=False).reset_index(drop=True)
    row = table.iloc[0]
    alarms = table[FALSE_ALARM]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    panel(axes[0], table, "auroc_vs_control",
          "AUROC, illegal vs matched legal splice\n($\\uparrow$ better)",
          0.5, (table.auroc_null.min(), table.auroc_null.max()), 0, 1.05,
          "chance (empirical)")
    panel(axes[1], table, "hit_illegal",
          "localization: argmax lands on $t_0$\n($\\uparrow$ better)",
          row.chance_uniform, (alarms.values.min(), alarms.values.max()), 1, 0.50,
          "chance")

    return finish(fig, "changepoint_by_space", camera_ready)


def threshold_table(table: pd.DataFrame) -> pd.DataFrame:
    """Sorted by panel (a). Both corpora sort the same way, so a space is one column."""
    return table.sort_values("auroc_position", ascending=False).reset_index(drop=True)


def threshold_top(table: pd.DataFrame) -> float:
    """One ceiling for both panels: two probability scales sharing a chance line at 0.5."""
    hi = table[["auroc_position_hi", "balanced_accuracy_hi"]].to_numpy().max()
    return max(1.05, float(hi) * 1.12)


def draw_threshold(table: pd.DataFrame, camera_ready: bool) -> Path:
    table = threshold_table(table)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    top = threshold_top(table)

    panel(axes[0], table, "auroc_position",
          "per-position AUROC, illegal move vs legal move\n($\\uparrow$ better)",
          0.5, None, 0, top, "chance")
    panel(axes[1], table, "balanced_accuracy",
          "balanced accuracy at the fitted cut, out of fold\n($\\uparrow$ better)",
          0.5, None, 1, top, "chance")

    return finish(fig, "threshold_detection", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    results = HERE / "results"
    print(draw_by_space(pd.read_csv(results / "changepoint_localization.csv"),
                        args.camera_ready))
    print(draw_threshold(pd.read_csv(results / "threshold_localization.csv"),
                         args.camera_ready))


if __name__ == "__main__":
    main()
