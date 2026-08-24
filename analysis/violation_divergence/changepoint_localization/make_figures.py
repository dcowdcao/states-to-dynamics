"""Which representation space lets a reference-free change-point score catch the splice?

`changepoint_by_space` -- two panels over the same nine spaces and the same sort order, so a
space is one column in both: (a) does the score know a game contains an illegal move at all,
(b) does its argmax say which ply. They are different questions and a space can be good at one
and not the other, which is why they are two panels and not one number.

`threshold_detection` -- (a) the per-position AUROC of illegal move against legal move, over
every space, beside (b) the five score distributions that AUROC summarizes, in the space that
tops it. The bar chart says which space orders positions best; the densities say what "best"
still looks like up close, which is a cut sitting inside the overlap rather than between two
separated classes. (b) is drawn by `illegal_move_detector.figure_histograms.panel`, the same
code its own figure uses, so the two cannot drift.

The out-of-fold balanced accuracy that used to be (b) is not lost: it is a column of
`results/threshold_localization.csv`, and its caveat -- a space can rank well and still have
no cut that transfers -- is in README.md.

Sorted by (a), because the ordering is the claim; raw layer 5 keeps reference styling wherever
it lands, so the reader can see how many dictionaries clear it, and keeping one order across
both panels is what makes the two rankings comparable.

No note under the axes: the estimator, n, seed and null semantics are the caption, and the
caption is the paragraph under "Results" in README.md. These figures are not standalone.

Usage: python analysis/violation_divergence/changepoint_localization/make_figures.py
           [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.figures.paper_style import (
    INK, MUTED, NULL, REFERENCE_SPACE, SPACE_STYLE, apply_style, save_figure)

HERE = Path(__file__).resolve().parent
BAR_WIDTH = 0.72
FALSE_ALARM = ["hit_control", "hit_control2", "hit_reference"]
# The space (a) ranks first. Its densities are what (b) draws, so the two panels are the same
# claim at two magnifications rather than two unrelated read-outs.
HISTOGRAM_SPACE = "GatedAnnealTrainer-othello"

sys.path.insert(0, str(HERE.parent / "illegal_move_detector"))


def histogram_panel(ax, space: str) -> None:
    """(b): the five score distributions at t0, in `space`, drawn by their own figure's code."""
    import detector
    from figure_histograms import panel as densities, subset

    data = detector.load(space)
    cand = detector.band()
    score = detector.score(data, cand)
    sel = detector.arms(data)
    tau = detector.operating_point(score, cand, data, sel)
    rows = subset(data, sel)
    densities(ax, {k: detector.at_t0(score, cand, data["t0"], r) for k, r in rows.items()}, tau)


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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 0.92]})
    panel(axes[0], table, "auroc_position", "AUROC", 0.5, None, 0, threshold_top(table),
          "chance")
    histogram_panel(axes[1], HISTOGRAM_SPACE)
    axes[1].text(0.0, 1.02, "(b)", va="bottom", transform=axes[1].transAxes, color=INK,
                 fontsize=9)
    fig.tight_layout()
    return save_figure(fig, HERE / "figures" / "threshold_detection", camera_ready)


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
