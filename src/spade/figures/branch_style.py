"""How one violation-divergence branch is drawn, and where it sits in a legend.

The figures of the study draw from the same nine branches -- the move as played,
two legal controls, one move per illegal category -- so the styling and the
legend order live here rather than in any one figure. A reader moving between
the fork trajectories and the board they fork on must see the same colour, the
same marker and the same label for the same branch. A figure may draw a subset
(fork_and_board drops the second control) or group the six illegal categories
onto their three rules (`by_rule`); either way the colours still agree.
"""

from matplotlib.axes import Axes

from spade.figures.paper_style import CATEGORY_STYLE, MUTED, NULL, REFERENCE, RULE_STYLE
from spade.othello.taxonomy import CATEGORIES, CATEGORY_GLOSS, RULE_MEMBERS, RULE_OF

CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}

# A rule's member categories, drawn behind their mean so the within-rule spread
# is visible rather than asserted. No hue of its own: a member takes its rule's.
MEMBER = {"lw": 0.7, "alpha": 0.38, "zorder": 2}
MEMBER_LABEL = "one category inside a rule (within-rule spread)"
CI_ELLIPSE_LABEL = "95% CI of the mean over games, at each path's end"
CONTOUR_LABEL = "50% Gaussian contour of {n} jackknife replicates (dispersion, not a CI)"


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
    """The one legend entry the faint member lines share, drawn on nothing.

    Two positional args, so it is the same call on a 2-D and a 3-D axes; a third
    empty list would be read as `zs` on one and as a second series on the other.
    """
    ax.plot([], [], color=MUTED, label=MEMBER_LABEL, **MEMBER)


def ci_ellipse_proxy(ax) -> None:
    """The population fork's ellipses, as one legend entry. An open ring stands in
    for the ellipse: a zero-size patch added to a 3-D axes moves the panel's limits."""
    ax.plot([], [], ls="none", marker="o", ms=6, markerfacecolor="none",
            markeredgecolor=MUTED, markeredgewidth=0.7, label=CI_ELLIPSE_LABEL)


# Series first (taxonomy order, then rule order), then the panel's furniture,
# then the nulls -- FIGURE_GUIDELINES 3 wants the reference first and nulls last.
_SERIES = [CATEGORY_STYLE[c]["label"] for c in CATEGORIES[1:]] + list(RULE_STYLE)[1:]


def _rank(lab: str) -> int:
    if lab.startswith("shared prefix"):
        return 0
    if lab.endswith("(baseline)"):  # a population baseline outranks one game's reference
        return 1
    if lab.startswith("reference"):
        return 2
    if lab == CI_ELLIPSE_LABEL:
        return 87
    if lab == MEMBER_LABEL:
        return 88
    if lab.startswith(CONTOUR_LABEL[:24]):
        return 89
    if lab.startswith("branch point"):
        return 90
    if lab.startswith("control"):
        return 99 + ("2" in lab)
    return 3 + next((i for i, s in enumerate(_SERIES) if lab.startswith(s)), len(_SERIES))


def legend_order(labels: list[str]) -> list[int]:
    """Reference first, nulls last (FIGURE_GUIDELINES 3); taxonomy order between."""
    return sorted(range(len(labels)), key=lambda i: _rank(labels[i]))


def ordered_legend(owner, ax, ncol: int, **placement):
    """One legend for a whole figure. `owner` is the figure, or the blank axes
    the legend is parked in; `placement` is that owner's loc/bbox_to_anchor.
    `ax` is the axes the entries are read from, or several of them when a
    figure's series are spread over more than one panel; a label drawn in two
    panels earns one entry.

    No title: what a figure of this study owes the reader in prose -- how the
    game was selected, where the rule mean is taken, the spread ratio, n=1, that
    cells are not comparable -- is its caption's job and lives in the study
    README (FIGURE_GUIDELINES 2, 5)."""
    by_label = {}
    for a in [ax] if isinstance(ax, Axes) else ax:
        handles, labels = a.get_legend_handles_labels()
        by_label.update(zip(labels, handles))
    labels = list(by_label)
    return owner.legend([by_label[labels[i]] for i in legend_order(labels)],
                        [labels[i] for i in legend_order(labels)],
                        ncol=ncol, fontsize=8, frameon=False, **placement)


def below(owner, leg, pad: float = 0.02) -> float:
    """y just under `leg`, in `owner`'s coordinates. The legend's height depends
    on how many of its entries wrapped, so a fixed offset collides."""
    owner.figure.canvas.draw()
    tr = owner.transAxes if isinstance(owner, Axes) else owner.transFigure
    return leg.get_window_extent().transformed(tr.inverted()).y0 - pad


def rule_member_key(owner, x: float = 0.005, y: float = -0.005) -> None:
    """One line per rule: its name, how many categories it averages, and what
    each of them is. The three means hide which violations sit inside them and
    the faint member lines are unlabelled, so the key names them (§5).

    Driven by RULE_MEMBERS and CATEGORY_GLOSS, so the counts cannot disagree with
    the legend's `(mean of n)` and the wording cannot drift from the taxonomy."""
    lines = [f"{rule} ({len(members)}): "
             + " · ".join(CATEGORY_GLOSS[c] for c in members)
             for rule, members in RULE_MEMBERS.items()]
    owner.text(x, y, "\n".join(lines), ha="left", va="top", fontsize=7.5,
               color=MUTED, linespacing=1.5)
