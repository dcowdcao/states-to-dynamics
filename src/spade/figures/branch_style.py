"""How one violation-divergence branch is drawn, and where it sits in a legend.

Every figure of the study draws the same nine branches -- the move as played,
two legal controls, one move per illegal category -- so the styling and the
legend order live here rather than in any one figure. A reader moving between
the fork trajectories and the board they fork on must see the same colour, the
same marker and the same label for the same branch.
"""

from spade.figures.paper_style import CATEGORY_STYLE, NULL, REFERENCE
from spade.othello.taxonomy import CATEGORIES

CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}


def branch_style(category: int, is_control: int) -> dict:
    if is_control:
        return {"color": NULL["color"], "marker": NULL["marker"], "ls": NULL["ls"],
                "lw": NULL["lw"], "zorder": NULL["zorder"], "label": CONTROL_LABEL[is_control]}
    if category == 0:
        return {"color": REFERENCE["color"], "marker": REFERENCE["marker"], "ls": "-",
                "lw": REFERENCE["lw"], "zorder": REFERENCE["zorder"],
                "label": "reference (the game as played)"}
    s = CATEGORY_STYLE[CATEGORIES[category]]
    return {"color": s["color"], "marker": s["marker"], "ls": "-", "lw": 1.5,
            "zorder": 3, "label": s["label"]}


def legend_order(labels: list[str]) -> list[int]:
    """Reference first, nulls last (FIGURE_GUIDELINES 3); taxonomy order between."""
    rank = {}
    for i, lab in enumerate(labels):
        if lab.startswith("shared prefix"):
            rank[i] = 0
        elif lab.startswith("reference"):
            rank[i] = 1
        elif lab.startswith("branch point"):
            rank[i] = 8
        elif lab.startswith("control"):
            rank[i] = 9 + ("2" in lab)
        else:
            names = [CATEGORY_STYLE[c]["label"] for c in CATEGORIES[1:]]
            rank[i] = 2 + (names.index(lab) if lab in names else len(names))
    return sorted(range(len(labels)), key=lambda i: rank[i])


def ordered_legend(fig, ax, ncol: int, y: float, title: str):
    handles, labels = ax.get_legend_handles_labels()
    order = legend_order(labels)
    return fig.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="lower center", ncol=ncol, fontsize=8, frameon=False,
                      bbox_to_anchor=(0.5, y), title=title, title_fontsize=7.5)
