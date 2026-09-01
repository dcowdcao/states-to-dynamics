"""Generation-legality panels, for the study's own figures and for any figure that reuses one.

The statistics are `analysis/generation_legality/results/`; everything here is drawing.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from spade.figures.paper_style import GENERATION_STYLE, MUTED, NULL, RULE_STYLE
from spade.othello.taxonomy import RULE_MEMBERS

RULES = {rule.split()[0]: (rule, members) for rule, members in RULE_MEMBERS.items()}
MARKS = 8      # markers per curve; the move axis has 56 points
LAST_POSITION = 57  # t indexes the prediction site and logits[:, t] score ply t+1,
                    # so a 59-token context makes t = 57 the axis's hard right edge


def curve(ax, x, frame: pd.DataFrame, style: dict, label: str, value: str = "mean", **kw):
    """Point curve, markers thinned to MARKS, over its interval band.

    The band is the `lo`/`hi` pair the study already writes for every series, drawn rather
    than left in the CSV so each curve carries its own uncertainty (FIGURE_GUIDELINES 1.2).
    It is faint and unstroked because four series overlap on the cumulative panel, and the
    panels are on a log axis where every bound is positive.
    """
    every = max(1, len(x) // MARKS)
    if {"lo", "hi"} <= set(frame.columns):
        ax.fill_between(x, frame["lo"], frame["hi"], color=style["color"], alpha=0.15,
                        lw=0, zorder=kw.get("zorder", 2) - 1)
    return ax.plot(x, frame[value], color=style["color"], ls=style.get("ls", "-"),
                   marker=style["marker"], ms=4, markevery=(every // 2, every),
                   label=label, **kw)[0]


def move_series(by_move: pd.DataFrame, metric: str) -> pd.DataFrame:
    return by_move[by_move.metric == metric].sort_values("t")


def log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))


def cumulative_panel(ax, by_move: pd.DataFrame) -> None:
    """The mass composed along t: has the game gone illegal by move t, and on what.

    Rising by construction, and it does not fold into the study's per-move panel: a
    cumulative curve is a different quantity with a different label, and its series are
    compositions rather than the per-ply sums that panel's legend names.
    """
    total = move_series(by_move, "cum_illegal")
    curve(ax, total.t, total, GENERATION_STYLE["mass"], GENERATION_STYLE["mass"]["label"],
          zorder=6, lw=2.2)
    for key, (rule, _) in RULES.items():
        rows = move_series(by_move, f"cum_{key}")
        curve(ax, rows.t, rows, RULE_STYLE[rule], rule, zorder=5)

    log_y(ax)
    ax.set_xlabel("game move number $t$")
    ax.set_ylabel("P(Illegal move in game so far up to $t$)")
    # The axis runs to the last position the model can predict from, so the reader sees where
    # the measurement has to stop rather than where the curve happens to end.
    ax.set_xlim(0, LAST_POSITION + 1)
    ax.axvline(LAST_POSITION, color=NULL["color"], ls=NULL["ls"], lw=1.1, zorder=1)
    ax.text(LAST_POSITION, 0.5, f"$t = {LAST_POSITION}$: last predicted move",
            transform=blended_transform_factory(ax.transData, ax.transAxes), rotation=90,
            ha="right", va="center", fontsize=7.5, color=MUTED)
    ax.legend(loc="lower right", fontsize=8)
