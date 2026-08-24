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

Both panels are the SAME 500 games -- `illegal_move_detector/results/eval_games.csv`, 20 games
at each of 25 positions -- which is why (a) reads `results/threshold_detection.csv` and not the
`threshold_localization.csv` beside it. That table is the 575 games of `curvature_discriminant`
at t0 in [16, 34], a different corpus from the one (b) draws, and two panels on two corpora are
not two views of one dataset. `measure_detector.py` writes the one they share; both panels
check their own draw against that file before anything is drawn.

The out-of-fold balanced accuracy that used to be (b) is not lost: it is a column of
`results/threshold_detection.csv`, and its caveat -- a space can rank well and still have no
cut that transfers -- is in README.md.

Sorted by (a), because the ordering is the claim; raw layer 5 keeps reference styling wherever
it lands, so the reader can see how many dictionaries clear it, and keeping one order across
both panels is what makes the two rankings comparable.

No note under the axes: the estimator, n, seed and null semantics are the caption, and the
caption is the paragraph under "Results" in README.md. These figures are not standalone.

`--l0-matched` redraws both figures over the eight `l0_operating_point` picks, one checkpoint
per architecture seated at the shared L0, so a difference between two bars is the architecture
and not its sparsity. Same panels, same estimator, same styling -- a pick keeps its family's
hue and hatch and its tick label carries the L0 it was seated at. The canonical figures are
untouched by it.

Usage: python analysis/violation_divergence/changepoint_localization/make_figures.py
           [--camera-ready] [--l0-matched]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.figures.paper_style import (
    INK, MUTED, NULL, REFERENCE_SPACE, SPACE_STYLE, apply_style, save_figure)
from spade.l0_operating_point.roster import styles as l0_styles

HERE = Path(__file__).resolve().parent
BAR_WIDTH = 0.72
FALSE_ALARM = ["hit_control", "hit_control2", "hit_reference"]

sys.path.insert(0, str(HERE.parent / "illegal_move_detector"))


def histogram_panel(ax, space: str) -> int:
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
    return detector.assert_eval_set(data["game_id"][rows["legal"]], data["t0"][rows["legal"]])


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def panel(ax, table: pd.DataFrame, value: str, ylabel: str, chance: float,
          band: tuple[float, float] | None, index: int | None, top: float,
          chance_label: str, styles: dict = SPACE_STYLE) -> None:
    """`index` is the panel letter; None on a standalone panel, which owns no grid."""
    if band is not None:
        ax.axhspan(*band, color=NULL["color"], alpha=0.28, lw=0, zorder=1)
    ax.axhline(chance, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=2)
    for x, row in table.iterrows():
        style = styles[row.space]
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

    label = lambda s: "raw L5 (ref.)" if s == REFERENCE_SPACE else styles[s]["short"]
    ax.set_xticks(range(len(table)), [label(s) for s in table.space], fontsize=6.8,
                  rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.7, len(table) + 0.5)
    ax.set_ylabel(ylabel)
    ax.text(len(table) - 0.35, chance, chance_label, color=NULL["color"], fontsize=6.8,
            ha="left", va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    if index is not None:
        panel_letter(ax, index)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def finish(fig, stem: str, camera_ready: bool, directory: Path | None = None) -> Path:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.005, 0.985, "hue = architecture family, hatch = its second dictionary; "
             "raw layer 5 is the reference space", fontsize=7, color=MUTED, va="top")
    return save_figure(fig, (directory or HERE / "figures") / stem, camera_ready)


def draw_by_space(table: pd.DataFrame, styles: dict, camera_ready: bool,
                  suffix: str) -> Path:
    table = table.sort_values("auroc_vs_control", ascending=False).reset_index(drop=True)
    row = table.iloc[0]
    alarms = table[FALSE_ALARM]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    panel(axes[0], table, "auroc_vs_control",
          "AUROC, illegal vs matched legal splice\n($\\uparrow$ better)",
          0.5, (table.auroc_null.min(), table.auroc_null.max()), 0, 1.05,
          "chance (empirical)", styles)
    panel(axes[1], table, "hit_illegal",
          "localization: argmax lands on $t_0$\n($\\uparrow$ better)",
          row.chance_uniform, (alarms.values.min(), alarms.values.max()), 1, 0.50,
          "chance", styles)

    return finish(fig, f"changepoint_by_space{suffix}", camera_ready)


def draw_threshold(table: pd.DataFrame, styles: dict, camera_ready: bool,
                   suffix: str) -> Path:
    """Sorted by (a); (b) is whichever space that puts first, so the panels cannot disagree."""
    table = table.sort_values("auroc_position", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 0.92]})
    panel(axes[0], table, "auroc_position", "AUROC", 0.5, None, 0, 1.05, "chance", styles)
    n_games = histogram_panel(axes[1], table.space.iloc[0])
    assert (table.n_games == n_games).all(), "the two panels are not one evaluation set"
    axes[1].text(0.0, 1.02, "(b)", va="bottom", transform=axes[1].transAxes, color=INK,
                 fontsize=9)
    fig.tight_layout()
    return save_figure(fig, HERE / "figures" / f"threshold_detection{suffix}", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    parser.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                        help="draw the tables measure.py --l0-matched wrote")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    results = HERE / "results"
    suffix = "_l0_matched" if args.l0_matched else ""
    styles = l0_styles() if args.l0_matched else SPACE_STYLE
    print(draw_by_space(pd.read_csv(results / f"changepoint_localization{suffix}.csv"),
                        styles, args.camera_ready, suffix))
    print(draw_threshold(pd.read_csv(results / f"threshold_detection{suffix}.csv"),
                         styles, args.camera_ready, suffix))


if __name__ == "__main__":
    main()
