"""The two generation-legality figures. Drawing only; the statistics are in results/.

F1 `generation_legality` — two panels of one study on one canvas:
    (a) by move t, has the model played an illegal move yet? Games it generated itself.
    (b) at move t, how likely is an illegal move of each rule? Corpus prefixes.
F2 `generation_legality_by_rule_rollout` — by move t, how likely is it that the game has
    taken an illegal move of each rule already? Prefixes the model generated for itself.

(a) and (b) sit in one figure because neither answers the other's question: (a) says the
model does emit illegal moves while it plays, (b) says which of the three constraint rules
the residual mass sits on. **Their y axes are not the same axis.** (a) is **cumulative** and
per game: the fraction of games that have committed at least one illegal move by ply t, which
only rises. (b) is **per move**, the model's next-move probability mass, which is what "the
model generates an illegal move here" means for a distribution. Reading (b) as if it
accumulated, or (a) as a per-move rate, is the misreading the pair invites, so each panel
states which it is on its y axis. Their prefix distributions differ too -- (a)'s games the
model wrote, (b)'s games handed to it -- and that is what each panel title says.

F2 draws (b)'s mass on the model's own prefixes and **accumulated**: `cum_*` in
`rollout_by_move.csv` is `1 - prod(1 - p)` composed per game along t, so it answers "has this
game gone illegal yet", the same question (a) asks of sampled play, off the distribution
rather than off sampled events. It is conditional on survival -- a rollout leaves the table at
its first illegal move -- so its n falls along the move axis, and the curve dips where they
leave: a per-game cumulative only rises, but a mean over survivors loses the games carrying
the most accumulated mass. (b) conditions on a legal prefix too; neither is P(illegal at ply
t) unconditionally.

(b)'s rule series are **sums** over their taxonomy categories, not means, so its legend says
`sum of 3` (FIGURE_GUIDELINES 1.7). F2's are not sums -- a composition over plies of a sum
over categories is neither -- so they carry the rule name alone. The estimator, the falling n
and each source's within-rule / between-rule spread are in README.md's "Figure notes";
`overall.csv` carries the spread itself.

Usage: python analysis/generation_legality/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spade.figures.generation_panels import (
    RULES, cumulative_panel, curve, log_y, move_series)
from spade.figures.paper_style import (
    GENERATION_STYLE, RULE_STYLE, apply_style, save_figure)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
SIZE = (6.4, 4.4)  # one panel, drawn at the width a two-column figure is printed at
PAIR = (9.4, 4.4)  # two panels side by side (FIGURE_GUIDELINES 2)


def read(results: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(results / f"{n}.csv")
            for n in ["by_move", "rollout_by_move", "survival", "overall"]}


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
    return float(end.value)


def rules_panel(ax, tables: dict[str, pd.DataFrame], source: str) -> None:
    """Per move, split by the constraint rule the illegal move would break.

    `source` picks the prefix distribution and nothing else: "by_move" is the teacher-forced
    corpus scan, "rollout_by_move" the model's own generated games.
    """
    by_move = tables[source]

    total = move_series(by_move, "p_illegal")
    curve(ax, total.t, total, GENERATION_STYLE["mass"], GENERATION_STYLE["mass"]["label"],
          zorder=6, lw=2.2)
    for key, (rule, members) in RULES.items():
        rows = move_series(by_move, f"p_{key}")
        curve(ax, rows.t, rows, RULE_STYLE[rule], f"{rule} (sum of {len(members)})", zorder=5)

    log_y(ax)
    ax.set_xlabel("game move number $t$")
    ax.set_ylabel("P(illegal move at move $t$)\n↓ better")
    ax.set_xlim(left=0)
    # Lower right: the rules all rise, so the late-game floor is the only region a legend
    # does not sit on, and four entries do not fit beside the curves in half a canvas.
    ax.legend(loc="lower right", fontsize=8)


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
    return save_figure(fig, FIGURES / "generation_legality", camera_ready)


def rollout_figure(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    """Accumulated mass on the prefixes the model wrote itself. Standalone, so it owns its
    whole key and drops the panel letter (FIGURE_GUIDELINES 5)."""
    fig, ax = plt.subplots(figsize=SIZE)
    cumulative_panel(ax, tables["rollout_by_move"])
    fig.tight_layout()
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
