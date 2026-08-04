"""How one violation-divergence branch is drawn, and where it sits in a legend.

The figures of the study draw from the same nine branches -- the move as played,
two legal controls, one move per illegal category -- so the styling and the
legend order live here rather than in any one figure. A reader moving between
the fork trajectories and the board they fork on must see the same colour, the
same marker and the same label for the same branch. A figure may draw a subset
(fork_and_board drops the second control) or group the six illegal categories
onto their three rules (`by_rule`); either way the colours still agree.
"""

from spade.figures.paper_style import CATEGORY_STYLE, MUTED, NULL, REFERENCE, RULE_STYLE
from spade.othello.taxonomy import CATEGORIES, RULE_OF

CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}

# A rule's member categories, drawn behind their mean so the within-rule spread
# is visible rather than asserted. No hue of its own: a member takes its rule's.
MEMBER = {"lw": 0.7, "alpha": 0.38, "zorder": 2}
MEMBER_LABEL = "one category inside a rule (within-rule spread)"


def branch_style(category: int, is_control: int, by_rule: bool = False) -> dict:
    """Colour, marker and label for one branch. `mfc`/`mec` are the marker's
    face and edge: filled with a white halo, except control 2, which is open.

    `by_rule` collapses the six illegal categories onto the three rules they
    violate, so a figure drawing one series per rule and a figure drawing one
    per category still agree on what a colour means."""
    if is_control:
        # The two controls share the null styling and often land on adjacent
        # squares, where identical grey marks cannot be told apart without
        # reading the legend. Control 2 -- the null for control 1 -- opens up.
        opened = is_control == 2
        return {"color": NULL["color"], "marker": NULL["marker"], "ls": NULL["ls"],
                "lw": NULL["lw"], "zorder": NULL["zorder"],
                "label": CONTROL_LABEL[is_control],
                "mfc": "white" if opened else NULL["color"],
                "mec": NULL["color"] if opened else "white"}
    if category == 0:
        return {"color": REFERENCE["color"], "marker": REFERENCE["marker"], "ls": "-",
                "lw": REFERENCE["lw"], "zorder": REFERENCE["zorder"],
                "label": "reference (the game as played)",
                "mfc": REFERENCE["color"], "mec": "white"}
    name = CATEGORIES[category]
    s = RULE_STYLE[RULE_OF[name]] if by_rule else CATEGORY_STYLE[name]
    return {"color": s["color"], "marker": s["marker"], "ls": "-", "lw": 1.5,
            "zorder": 3, "label": s["label"], "mfc": s["color"], "mec": "white"}


def legend_proxy(ax) -> None:
    """The one legend entry the faint member lines share, drawn on nothing."""
    ax.plot([], [], [], color=MUTED, label=MEMBER_LABEL, **MEMBER)


# Series first (taxonomy order, then rule order), then the panel's furniture,
# then the nulls -- FIGURE_GUIDELINES 3 wants the reference first and nulls last.
_SERIES = [CATEGORY_STYLE[c]["label"] for c in CATEGORIES[1:]] + list(RULE_STYLE)[1:]


def _rank(lab: str) -> int:
    if lab.startswith("shared prefix"):
        return 0
    if lab.startswith("reference"):
        return 1
    if lab == MEMBER_LABEL:
        return 88
    if lab.startswith("branch point"):
        return 90
    if lab.startswith("control"):
        return 99 + ("2" in lab)
    return 2 + next((i for i, s in enumerate(_SERIES) if lab.startswith(s)), len(_SERIES))


def legend_order(labels: list[str]) -> list[int]:
    """Reference first, nulls last (FIGURE_GUIDELINES 3); taxonomy order between."""
    return sorted(range(len(labels)), key=lambda i: _rank(labels[i]))


def ordered_legend(owner, ax, ncol: int, title: str, **placement):
    """One legend for a whole figure. `owner` is the figure, or the blank axes
    the legend is parked in; `placement` is that owner's loc/bbox_to_anchor."""
    handles, labels = ax.get_legend_handles_labels()
    order = legend_order(labels)
    return owner.legend([handles[i] for i in order], [labels[i] for i in order],
                        ncol=ncol, fontsize=8, frameon=False, title=title,
                        title_fontsize=7.5, **placement)
