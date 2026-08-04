"""The 8x8 Othello board a figure's overlay sits on.

Discs are absolute colour and the axes are algebraic (a-h left to right, 1-8 top
to bottom), so the same square lands in the same place in every board figure.
Each caller owns its own overlay: bracket_flip shades squares by role,
fork_and_board outlines the branch squares by violation category.
"""

import numpy as np

from spade.figures.paper_style import DISC_EDGE, DISC_STYLE, GRID

DISC_SIZE = 13


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
