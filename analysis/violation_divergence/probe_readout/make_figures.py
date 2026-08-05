"""F7 of the violation-divergence study: what the layer-5 linear probe decodes off an illegal branch.

(a) does the illegal branch write the illegal disc, and for how long
(b) the occupied pair: does an illegal move on an occupied square read as a
    no-op or as a recolor
(c) does the branch's decoded mover track the reference's -- R3 turn confusion
    read straight off the representation

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
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, REFERENCE, apply_style, rule_key, save_figure)
from spade.violation_divergence.divergence import WINDOW

RESULTS = results_dir("violation_divergence")
FIGURES = Path(__file__).resolve().parent / "figures"
VARIANTS = ("spliced", "rollout")

EMPTY_SQUARE_CATEGORIES = ["empty_detached", "empty_own_only", "empty_no_bracket", "opponent_only"]
OCCUPIED_CATEGORIES = ["occupied_own", "occupied_opp"]
PARITY_BRANCHES = [c for c in CATEGORY_STYLE if c != "legal"]  # taxonomy order
BAND = "band: 95% bootstrap CI over source games"


def curve(ax, df: pd.DataFrame, style: dict, label: str, offset: int = 0, **kw) -> None:
    d = df.sort_values("r")
    kw = {"color": style["color"], "marker": style["marker"], "markersize": 4,
          "markevery": (offset, 6), "markeredgecolor": "white", "markeredgewidth": 0.5,
          "label": label, **kw}
    ax.plot(d.r, d["mean"], **kw)
    ax.fill_between(d.r, d.lo, d.hi, color=kw["color"], alpha=0.15, lw=0,
                    zorder=kw.get("zorder", 2) - 1)


def event_axes(ax, ylabel: str, letter: str, title: str, ylim=(-0.03, 1.05)) -> None:
    ax.axvspan(WINDOW[0], 0, color=GRID, alpha=0.45, zorder=0)
    ax.axvline(0, color=MUTED, ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.text(0.04, 0.5, "shared prefix — branches coincide by construction",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="center", ha="center", rotation=90)
    ax.set_xlabel("position relative to the branch move, $r = s - t_0$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(WINDOW[0], WINDOW[-1])
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.text(-0.14, 1.02, f"({letter})", va="bottom", transform=ax.transAxes, color=INK, fontsize=9)


def label_with_n(df: pd.DataFrame, key: str, category: str) -> str:
    n = int(df[(df[key] == category) & (df.r == 0)].iloc[0].n)
    return f"{CATEGORY_STYLE[category]['label']} (n={n})"


def band_legend(ax, note: str = "", **kw) -> None:
    ax.legend(title=BAND + note, fontsize=7.4, title_fontsize=7, alignment="left", **kw)


def occupancy_panel(ax, square: pd.DataFrame, notes: bool) -> None:
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
    if not notes:
        return
    ax.annotate("a disc that really was placed", xy=(18, 0.995), xytext=(14, 0.87),
                fontsize=7, color=MUTED, ha="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.annotate("nothing was played here; both anchors rise\nonly as the shared suffix fills the square",
                xy=(20, 0.55), xytext=(0.5, 0.33), fontsize=7, color=MUTED, va="bottom",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    band_legend(ax, loc="center left", bbox_to_anchor=(0.28, 0.62))


def recolor_panel(ax, square: pd.DataFrame, notes: bool) -> None:
    p = square[square.metric == "p_mover_colour"]
    for i, category in enumerate(OCCUPIED_CATEGORIES):
        short = CATEGORY_STYLE[category]["label"].removeprefix("R1 ")
        illegal = p[(p.category == category) & (p.role == "illegal")]
        curve(ax, illegal, CATEGORY_STYLE[category],
              f"{short} (n={int(illegal[illegal.r == 0].iloc[0].n)})", offset=i, zorder=5)
        curve(ax, p[(p.category == category) & (p.role == "reference")],
              CATEGORY_STYLE[category], f"{short}, reference branch",
              offset=i + 3, ls=(0, (1, 1.4)), lw=1.3, zorder=3)
    if not notes:
        return
    gap = p[(p.category == "occupied_opp") & (p.r == 0)].set_index("role")["mean"]
    ax.annotate("", xy=(0.7, gap["illegal"]), xytext=(0.7, gap["reference"]),
                arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 0.8})
    ax.annotate(f"recolor: at $r$ = 0 the illegal branch reads\nthe mover's colour {gap['illegal'] - gap['reference']:+.2f} more often than\nthe reference branch does on the same square",
                xy=(1.2, gap.mean()), xytext=(24, 0.24), fontsize=7, color=MUTED,
                ha="right", va="top", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.text(24, 0.99, "no-op: occupied (own) tracks\nits own reference anchor", fontsize=7,
            color=MUTED, ha="right", va="top")
    band_legend(ax, loc="lower left", bbox_to_anchor=(0.02, 0.30))


def parity_panel(ax, parity: pd.DataFrame) -> None:
    p = parity[parity.metric == "p_aligned"]
    for i, branch in enumerate(PARITY_BRANCHES):
        curve(ax, p[p.branch == branch], CATEGORY_STYLE[branch],
              label_with_n(p, "branch", branch), offset=i, zorder=5)
    curve(ax, p[p.branch == "reference"], REFERENCE, "reference branch (the probe's own frame accuracy)",
          offset=0, lw=REFERENCE["lw"], zorder=4)
    curve(ax, p[p.branch == "control_legal"], NULL, "control_legal branch", offset=3,
          ls=NULL["ls"], lw=NULL["lw"], zorder=NULL["zorder"])
    band_legend(ax, note="\na parity-confused branch swaps theirs and mine on\nevery untouched disc at once, and would sit near 0",
                loc="lower left", bbox_to_anchor=(0.20, 0.02))


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
        # The in-panel legends and annotations are placed against the spliced
        # curves; here the anchors move, so both legends go under the figure.
        for i, ax in enumerate(axes):
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper left", fontsize=7,
                       bbox_to_anchor=(0.04 + 0.5 * i, -0.02), title=BAND,
                       title_fontsize=7, alignment="left")
        fig.text(0.5, -0.45, "rollout branch set: every branch samples its own "
                 "continuation, K=4 averaged per branch, so a branch is one item.\n"
                 "The reference's board is sampled too and ends at its first illegal "
                 "move: 1.1% of window positions have no board and are dropped, so n "
                 "falls with $r$.\nBoth anchors now rise as each branch's own rollout "
                 "fills the square, not as a shared suffix does.",
                 ha="center", va="top", fontsize=7.5, color=MUTED, style="italic",
                 linespacing=1.5)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.01 if spliced else -0.62)
    print(save_figure(fig, FIGURES / figure, camera_ready=args.camera_ready))


if __name__ == "__main__":
    main()
