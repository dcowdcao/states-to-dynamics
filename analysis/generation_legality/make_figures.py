"""The two generation-legality figures. Drawing only; the statistics are in results/.

F1 `generation_legality` — two panels of one study on one canvas:
    (a) by move t, has the model played an illegal move yet? Games it generated itself.
    (b) at move t, how likely is an illegal move of each rule? Corpus prefixes.
F2 `generation_legality_by_rule_rollout` — (b) on prefixes the model generated for itself

(a) and (b) sit in one figure because neither answers the other's question: (a) says the
model does emit illegal moves while it plays, (b) says which of the three constraint rules
the residual mass sits on. **Their y axes are not the same axis.** (a) is **cumulative** and
per game: the fraction of games that have committed at least one illegal move by ply t, which
only rises. (b) is **per move**, the model's next-move probability mass, which is what "the
model generates an illegal move here" means for a distribution. Reading (b) as if it
accumulated, or (a) as a per-move rate, is the misreading the pair invites, so each panel
states which it is on its y axis. Their prefix distributions differ too -- (a)'s games the
model wrote, (b)'s games handed to it -- and that is what each panel title says.

F2 is (b) exactly, drawn by the same function, and differs only in where the legal prefix
came from. It is conditional on survival -- a rollout leaves the table at its first illegal
move -- so its n falls along the move axis and the panel prints it. (b) conditions on a legal
prefix too, which is what makes the pair comparable; neither is P(illegal at ply t)
unconditionally.

The rule series are **sums** over their taxonomy categories, not means, so the legend says
`sum of 3` (FIGURE_GUIDELINES 1.7), and the within-rule ÷ between-rule spread is printed,
because a spread near 1 would mean the three rules are a label and not a finding. Each panel
prints the spread of its own source, never the other's.

Usage: python analysis/generation_legality/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from spade.figures.paper_style import (
    GENERATION_STYLE, MUTED, RULE_STYLE, apply_style, rule_key, save_figure)
from spade.othello.taxonomy import RULE_MEMBERS

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RULES = {rule.split()[0]: (rule, members) for rule, members in RULE_MEMBERS.items()}
MARKS = 8          # markers per curve; the move axis has 56 points
NOTE = {"fontsize": 7.5, "color": MUTED}
NOTE_CHAR = 0.053  # inches per note character at 7.5 pt, net of the mathtext a wrap overcounts
NOTE_Y = -0.22     # axes fraction: under the x label, never over the curves
KEY_PAD = 0.03     # figure fraction between the deepest note and the rule key
SIZE = (6.4, 4.4)  # one panel, drawn at the width a two-column figure is printed at
PAIR = (9.4, 4.4)  # two panels side by side (FIGURE_GUIDELINES 2)

ROLLOUT_TITLE = ("Conditioning on the model's own prefixes barely moves it: the same\n"
                 "R3 > R2 > R1 ordering, and the mass tracks the corpus scan to ~10%")


def read(results: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(results / f"{n}.csv")
            for n in ["by_move", "rollout_by_move", "survival", "overall"]}


def curve(ax, x, frame: pd.DataFrame, style: dict, label: str, value: str = "mean", **kw):
    """Point curve, markers thinned to MARKS. No band: the intervals are in the CSVs.

    Dropping the drawn interval is a deliberate departure from FIGURE_GUIDELINES 1.2, which
    wants every claim-supporting curve to carry its uncertainty. Every `lo`/`hi` column is
    still computed and written, and each panel says where; nothing was stopped being
    measured.
    """
    every = max(1, len(x) // MARKS)
    return ax.plot(x, frame[value], color=style["color"], ls=style.get("ls", "-"),
                   marker=style["marker"], ms=4, markevery=(every // 2, every),
                   label=label, **kw)[0]


def move_series(by_move: pd.DataFrame, metric: str) -> pd.DataFrame:
    return by_move[by_move.metric == metric].sort_values("t")


def panel_note(ax, text: str) -> None:
    """The estimator, its n and where its interval lives, under the panel that owes them.

    One logical line per input line, wrapped to this panel's own width -- a panel of the pair
    is a third of the canvas, and a note has to be read against its panel and not the other.
    """
    wrap = int(ax.get_position().width * ax.figure.get_figwidth() / NOTE_CHAR)
    ax.text(0, NOTE_Y, "\n".join(fill(line, wrap, break_on_hyphens=False)
                                 for line in text.splitlines()),
            transform=ax.transAxes, ha="left", va="top", linespacing=1.5, **NOTE)


def key_under(fig, panels: list) -> None:
    """The R1-R3 key under the panels, below the deepest note and left-aligned on the first.

    The notes are axes children, so `tight_layout` has already lifted the axes to make room
    for them and where they end is only knowable after a draw. One line per rule needs the
    whole canvas, which is why this is figure text and not a third note.
    """
    fig.canvas.draw()
    inverse = fig.transFigure.inverted()
    bottom = min(inverse.transform(text.get_window_extent())[0][1]
                 for panel in panels for text in panel.texts)
    rule_key(fig, ["R1", "R2", "R3"], x=panels[0].get_position().x0, y=bottom - KEY_PAD)


def survival_panel(ax, survival: pd.DataFrame) -> float:
    """Cumulative: by move t, what fraction of generated games have gone illegal?

    The rollout arm only, and it returns its endpoint for the figure's claim. The
    teacher-forced arm is still computed and still in results/survival.csv, but composing
    per-position mass from prefixes the model never wrote answers a different question, and
    drawing the two together read as one.
    """
    rows = survival.query("arm == 'free_running'").sort_values("plies")
    style = GENERATION_STYLE["free_running"]
    end = rows.iloc[-1]
    curve(ax, rows.plies, rows, style, style["label"], value="value")
    ax.annotate(f"{end.value:.1%}", (end.plies, end.value), textcoords="offset points",
                xytext=(-6, 7), fontsize=8, color=style["color"], ha="right")

    ax.set_xlabel("game move number $t$")
    ax.set_ylabel("P(at least one illegal move by move $t$)\n↓ better")
    ax.set_ylim(0, 1.2 * float(end.value))  # headroom for the endpoint label
    ax.set_xlim(left=0)
    ax.legend(loc="upper left", fontsize=8)
    panel_note(ax, "cumulative, so the curve can only rise.\n"
               f"$n$ = {int(rows.n_games.iloc[0])} whole games at $T$ = 1.0, judged by the "
               "engine.\n95% bootstrap CIs are in results/survival.csv, not drawn.")
    return float(end.value)


def rules_panel(ax, tables: dict[str, pd.DataFrame], source: str) -> None:
    """Per move, split by the constraint rule the illegal move would break.

    `source` picks the prefix distribution and nothing else: "by_move" is the teacher-forced
    corpus scan, "rollout_by_move" the model's own generated games.
    """
    rollout = source == "rollout_by_move"
    by_move = tables[source]

    total = move_series(by_move, "p_illegal")
    curve(ax, total.t, total, GENERATION_STYLE["mass"], GENERATION_STYLE["mass"]["label"],
          zorder=6, lw=2.2)
    for key, (rule, members) in RULES.items():
        rows = move_series(by_move, f"p_{key}")
        curve(ax, rows.t, rows, RULE_STYLE[rule], f"{rule} (sum of {len(members)})", zorder=5)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))
    ax.set_xlabel("game move number $t$")
    ax.set_ylabel("P(illegal move at move $t$)\n↓ better")
    ax.set_xlim(left=0)
    # Lower right: the rules all rise, so the late-game floor is the only region a legend
    # does not sit on, and four entries do not fit beside the curves in half a canvas.
    ax.legend(loc="lower right", fontsize=8)

    spread_key = ("rollout_mass_within_over_between" if rollout
                  else "rule_mass_within_over_between")
    spread = float(tables["overall"].set_index("metric").value[spread_key])
    n_note = (f"the games still legal at each $t$, $n$ falling {int(total.n_games.iloc[0])} "
              f"$\\rightarrow$ {int(total.n_games.iloc[-1])}: a rollout leaves the table at "
              "its first illegal move, so $n$ falls along the axis"
              if rollout else f"$n$ = {int(total.n_games.iloc[0])} corpus games")
    panel_note(ax, f"per move, not cumulative; means over {n_note}. 95% bootstrap CIs are in "
               f"results/{source}.csv, not drawn.\n"
               f"within-rule ÷ between-rule spread = {spread:.2f} (SD of log$_{{10}}$ share "
               "over the 6 taxonomy categories): it is R2 that drives it, and R3 has a single "
               "member, so on this axis R2 is closer to a label than a class.")


def legality_figure(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    """Does the model play illegal moves at all, and which rule do the ones it plays break?"""
    fig, (left, right) = plt.subplots(1, 2, figsize=PAIR)
    end = survival_panel(left, tables["survival"])
    rules_panel(right, tables, "by_move")
    left.set_title("(a) games the model generated: does one go illegal?", loc="left")
    right.set_title("(b) corpus prefixes: which rule the mass sits on", loc="left")
    if not camera_ready:
        legal = tables["overall"].set_index("metric").value["p_legal"]
        fig.suptitle(f"OthelloGPT keeps {legal:.4f} of its next-move mass legal, yet {end:.1%} "
                     "of the games it generates for itself go illegal;\nthe residual mass is "
                     "not spread evenly — R3 (turn) dominates from the midgame on, R1 "
                     "(placement) is suppressed", fontsize=10)
    fig.tight_layout()
    key_under(fig, [left, right])
    return save_figure(fig, FIGURES / "generation_legality", camera_ready)


def rollout_figure(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    """(b) on the prefixes the model wrote itself. Standalone, so it owns its whole key and
    drops the panel letter (FIGURE_GUIDELINES 5)."""
    fig, ax = plt.subplots(figsize=SIZE)
    rules_panel(ax, tables, "rollout_by_move")
    if not camera_ready:
        ax.set_title(ROLLOUT_TITLE, fontsize=10)
    fig.tight_layout()
    key_under(fig, [ax])
    return save_figure(fig, FIGURES / "generation_legality_by_rule_rollout", camera_ready)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    tables = read(args.results)
    for path in (legality_figure(tables, args.camera_ready),
                 rollout_figure(tables, args.camera_ready)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
