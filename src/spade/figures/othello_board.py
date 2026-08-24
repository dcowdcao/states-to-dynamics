"""The 8x8 Othello board a figure's overlay sits on.

Discs are absolute colour and the axes are algebraic (a-h left to right, 1-8 top
to bottom), so the same square lands in the same place in every board figure.
Each caller owns its own overlay: bracket_flip shades squares by role,
fork_and_board outlines the branch squares by violation category.
"""

import numpy as np
from matplotlib.patches import Rectangle

from spade.figures.paper_style import DISC_EDGE, DISC_STYLE, GRID, VERDICT_STYLE

DISC_SIZE = 13
CORNER = 0.34   # marker inset: clear of the disc at the centre, still inside the cell


def square_xy(square: int) -> tuple[int, int]:
    """Engine square 0-63 to (x, y) in the board axes' coordinates."""
    return square % 8, square // 8


def draw_board(ax, discs: np.ndarray, disc_size: float = DISC_SIZE) -> None:
    """`discs` is 64 long in engine order: +1 black, -1 white, 0 empty."""
    for square, disc in enumerate(discs):
        if disc == 0:
            continue
        style = DISC_STYLE["black" if disc > 0 else "white"]
        x, y = square_xy(square)
        ax.plot(x, y, marker=style["marker"], ms=disc_size, lw=0, zorder=3,
                markerfacecolor=style["color"], markeredgecolor=DISC_EDGE,
                markeredgewidth=1.0)
    ax.set_xticks(range(8), list("abcdefgh"))
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel("file")
    ax.set_ylabel("rank (1 at the top)")
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def mark_square(ax, square: int, style: dict, label: str | None = None, ls: str = "-",
                marker: bool = True, below: bool = False) -> None:
    """Outline one cell, carry its style's marker, and optionally name it.

    The outline sits on the cell edge and the marker inside a corner rather than at the
    centre, because the disc underneath is often the evidence -- an occupied square is *why*
    a move breaks R1. The marker is what survives grayscale once the hue is gone
    (FIGURE_GUIDELINES 1.1); drop it when one square is marked alone and there is nothing to
    tell it apart from. `below` puts the name under the cell, for when the square above it
    already holds a disc.
    """
    x, y = square_xy(square)
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, lw=2.0, ls=ls,
                           edgecolor=style["color"], zorder=4))
    if marker:
        ax.plot(x - CORNER, y - CORNER, marker=style["marker"], ms=6, lw=0, zorder=6,
                color=style["color"], markeredgecolor="white", markeredgewidth=0.6)
    if label:
        # draw_board inverts the y axis but ha/va do not follow it, so va="bottom" at the
        # smaller y puts the name above the cell.
        edge, va = (y + 0.52, "top") if below else (y - 0.52, "bottom")
        ax.annotate(label, (x, edge), ha="center", va=va, fontsize=9, fontweight="bold",
                    color=style["color"], zorder=6)


def candidate(ax, square: int, mover: int, disc_size: float = DISC_SIZE) -> None:
    """The disc the mover is trying to place on this square.

    Centred and half-size: centred because that is where the disc would go, half-size because
    the square may already be occupied and the disc on it is why the move breaks R1. It takes
    `draw_board`'s edge for `draw_board`'s reason -- a white candidate on the empty white cell
    of a white-to-move panel does not exist without it.
    """
    style = DISC_STYLE["black" if mover > 0 else "white"]
    x, y = square_xy(square)
    ax.plot(x, y, marker=style["marker"], ms=disc_size / 2, lw=0, zorder=6,
            markerfacecolor=style["color"], markeredgecolor=DISC_EDGE, markeredgewidth=0.8)


def verdict(ax, square: int, legal: bool, disc_size: float = DISC_SIZE) -> None:
    """Tick or cross in the square's lower-right corner, opposite `mark_square`'s marker.

    The corner, not the centre: the candidate disc is centred, and on an occupied square the
    disc underneath is the evidence. The tick is drawn in cell units, so it scales with the
    board on its own; the cross is a marker and takes the disc's scale.
    """
    x, y = square_xy(square)
    color = VERDICT_STYLE["legal" if legal else "illegal"]["color"]
    cx, cy, scale = x + CORNER, y + CORNER, disc_size / DISC_SIZE
    if legal:
        ax.plot([cx - 0.14, cx - 0.05, cx + 0.14], [cy - 0.01, cy + 0.09, cy - 0.15],
                color=color, lw=2.0 * scale, zorder=7, solid_capstyle="round")
    else:
        ax.plot(cx, cy, marker="x", ms=7 * scale, mew=2.2 * scale, color=color, zorder=7)
