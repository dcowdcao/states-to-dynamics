"""Generation-legality panels, for the study's own figures and for any figure that reuses one.

The statistics are `analysis/generation_legality/results/`; everything here is drawing.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from spade.figures.paper_style import GENERATION_STYLE, RULE_STYLE
from spade.othello.taxonomy import RULE_MEMBERS

RULES = {rule.split()[0]: (rule, members) for rule, members in RULE_MEMBERS.items()}
MARKS = 8   # markers per curve; the move axis has 56 points


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
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=8)
