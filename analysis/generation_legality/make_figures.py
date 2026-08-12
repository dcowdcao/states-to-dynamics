"""The two generation-legality figures, one panel each. Drawing only; the statistics are
in results/.

F1 `generation_legality`         — by move t, has the model played an illegal move yet?
F2 `generation_legality_by_rule` — at move t, how likely is an illegal move of each rule?

The two answer different questions and their y axes are not the same quantity. F1 is
**cumulative** and per game: the fraction of games that have committed at least one illegal
move by ply t, which only rises. F2 is **per move** and is the model's next-move probability
mass, which is what "the model generates an illegal move here" means for a distribution.
Reading F2's curves as if they accumulated, or F1's as if it were a per-move rate, is the
one misreading the pair invites; each panel says which it is on its y axis.

Two conventions F2 states for itself. Its rule series are **sums** over their taxonomy
categories, not means, so the legend says `sum of 3` (FIGURE_GUIDELINES 1.7); the six
members are drawn faint behind their rule, and the within-rule ÷ between-rule spread is
printed, because a spread near 1 would mean the three rules are a label and not a finding.

Usage: python analysis/generation_legality/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from spade.figures.paper_style import (
    CATEGORY_STYLE, GENERATION_STYLE, MUTED, RULE_STYLE, apply_style, rule_key, save_figure)
from spade.othello.taxonomy import RULE_MEMBERS

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RULES = {rule.split()[0]: (rule, members) for rule, members in RULE_MEMBERS.items()}
MARKS = 8          # markers per curve; the move axis has 56 points
NOTE = {"fontsize": 7.5, "color": MUTED}
SIZE = (6.4, 4.4)  # one panel, drawn at the width a two-column figure is printed at


def read(results: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(results / f"{n}.csv") for n in ["by_move", "survival", "overall"]}


def band(ax, x, frame: pd.DataFrame, style: dict, label: str, value: str = "mean", **kw):
    """Point curve with its bootstrap band, markers thinned to MARKS."""
    every = max(1, len(x) // MARKS)
    ax.fill_between(x, frame.lo, frame.hi, color=style["color"], alpha=0.22, lw=0,
                    zorder=kw.get("zorder", 3) - 1)
    return ax.plot(x, frame[value], color=style["color"], ls=style.get("ls", "-"),
                   marker=style["marker"], ms=4, markevery=(every // 2, every),
                   label=label, **kw)[0]


def move_series(by_move: pd.DataFrame, metric: str) -> pd.DataFrame:
    return by_move[by_move.metric == metric].sort_values("t")


def figure_one(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    """Cumulative: by move t, what fraction of games have gone illegal?"""
    fig, ax = plt.subplots(figsize=SIZE)
    table = tables["survival"]
    # The two arms end four plies apart and their labels collide unless they go opposite
    # ways off the endpoint: teacher-forced right-and-down into the gap under the rollout.
    for arm, offset, align in [("teacher_forced", (7, -11), "left"),
                               ("free_running", (-6, 7), "right")]:
        rows, style = table[table.arm == arm].sort_values("plies"), GENERATION_STYLE[arm]
        end = rows.iloc[-1]
        band(ax, rows.plies, rows, style,
             f"{style['label']} ($n$ = {int(rows.n_games.iloc[0])} games)",
             value="value", zorder=5 if arm == "free_running" else 4)
        ax.annotate(f"{end.value:.1%}", (end.plies, end.value), textcoords="offset points",
                    xytext=offset, fontsize=8, color=style["color"], ha=align)

    ax.set_xlabel("game move number $t$")
    ax.set_ylabel("P(at least one illegal move by move $t$)   ↓ better")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.text(0.03, 0.97, "band: 95% percentile bootstrap CI over games (2000 resamples)\n"
            "cumulative, so the curve can only rise", transform=ax.transAxes, va="top", **NOTE)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=8)
    if not camera_ready:
        legal = tables["overall"].set_index("metric").value["p_legal"]
        end = table.query("arm == 'free_running'").value.iloc[-1]
        ax.set_title(f"OthelloGPT puts {legal:.4f} of its next-move mass on legal squares, "
                     f"and {end:.1%} of the games\nit generates for itself still go illegal",
                     fontsize=10)
    fig.tight_layout()
    return save_figure(fig, FIGURES / "generation_legality", camera_ready)


def figure_two(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    """Per move, split by the constraint rule the illegal move would break."""
    fig, ax = plt.subplots(figsize=SIZE)
    by_move = tables["by_move"]

    total = move_series(by_move, "p_illegal")
    band(ax, total.t, total, GENERATION_STYLE["mass"], GENERATION_STYLE["mass"]["label"],
         zorder=6, lw=2.2)
    for key, (rule, members) in RULES.items():
        for category in members:
            faint = move_series(by_move, f"p_{category}")
            ax.plot(faint.t, faint["mean"], color=CATEGORY_STYLE[category]["color"],
                    lw=0.7, alpha=0.55, zorder=3)
        rows = move_series(by_move, f"p_{key}")
        band(ax, rows.t, rows, RULE_STYLE[rule], f"{rule} (sum of {len(members)})", zorder=5)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))
    ax.set_xlabel("game move number $t$")
    ax.set_ylabel("P(illegal move at move $t$)   ↓ better")
    ax.set_xlim(left=0)
    # Floor from the rule curves, not the members: `empty_detached` decays four decades
    # below R2 by the endgame and would otherwise set the axis for the whole panel.
    floor = min(move_series(by_move, f"p_{k}").lo.min() for k in RULES)
    ax.set_ylim(bottom=floor * 0.6)
    spread = float(tables["overall"].set_index("metric").value["rule_mass_within_over_between"])
    note = ("faint: the 6 taxonomy categories, behind the rule that sums them; some leave the "
            "axis below.  band: 95% bootstrap CI over "
            f"$n$ = {int(total.n_games.iloc[0])} games.\n"
            f"within-rule ÷ between-rule spread = {spread:.2f} (SD of log$_{{10}}$ share), and "
            "it is all R2: its three members span 1.73 dex where R1's two span 0.11,\n"
            "so on this axis R2 is a label and R1 is a class.")
    ax.legend(loc="upper left", fontsize=8)
    if not camera_ready:
        ax.set_title("The residual illegal mass is not spread evenly: R3 (turn, one category)\n"
                     "dominates from the midgame on and R1 (placement) is suppressed",
                     fontsize=10)
    fig.tight_layout()
    # Under the panel, not in it: four lines of note inside a log axis sit on the faint
    # members it is telling the reader to look at.
    fig.text(0.005, -0.015, note, ha="left", va="top", linespacing=1.5, **NOTE)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.135)
    return save_figure(fig, FIGURES / "generation_legality_by_rule", camera_ready)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    tables = read(args.results)
    for path in (figure_one(tables, args.camera_ready), figure_two(tables, args.camera_ready)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
