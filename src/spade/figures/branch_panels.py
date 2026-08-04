"""The two panels the violation-divergence figures draw a branch set in.

`fork_panel` is one representation space: the branches as paths, with move index
on the third axis. `board_panel` is the position they fork on, with each branch
move on its square. Both take their colour, marker and label from
`branch_style`, so a branch reads the same in either. Layout -- how many panels,
where the legend goes -- belongs to the caller; these draw one axes each.
"""

import matplotlib.pyplot as plt
import numpy as np

from spade.figures.branch_style import branch_style
from spade.figures.othello_board import draw_board, square_xy
from spade.figures.paper_style import INK, MUTED

ELEV, AZIM = 18, -58  # fixed so the fork panel is reproducible
CORNER = 0.30         # marker offset from the cell centre, which may hold a disc
FLIP_RING = 0.38      # ring radius around a flipped disc, in cells


def axis_labels(method: str, var: np.ndarray) -> tuple[str, str]:
    if method == "isomap":
        return "Isomap-1 (arb. units)", "Isomap-2 (arb. units)"
    return f"PC1 ({var[0]:.0%} var)", f"PC2 ({var[1]:.0%} var)"


def fork_panel(ax, coords: np.ndarray, rows: list[dict], pos: np.ndarray, t0: int,
               legend: bool, zlabel: bool = True) -> None:
    """coords [n_branches, n_pos, 2]; move index is the vertical axis."""
    k = int(np.flatnonzero(pos == t0)[0])
    lo, hi = coords.min(), coords.max()
    ax.plot([0, 0], [0, 0], [pos[0], t0], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label="shared prefix (branches identical)" if legend else None)
    for i, r in enumerate(rows):
        st = branch_style(r["category"], r["is_control"])
        seg, z = coords[i, k:], pos[k:]
        ax.plot(seg[:, 0], seg[:, 1], z, ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.5, markevery=3, markerfacecolor=st["mfc"],
                markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"],
                label=st["label"] if legend else None)
        ax.plot([seg[-1, 0]], [seg[-1, 1]], [z[-1]], marker=st["marker"], ms=6,
                color=st["color"], markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
                markeredgewidth=0.7, zorder=st["zorder"] + 1)
    ax.plot([0], [0], [t0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    for a, lim in [(ax.set_xlim, (lo, hi)), (ax.set_ylim, (lo, hi))]:
        a(lim)
    ax.set_zlim(pos[0], pos[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    if zlabel:
        ax.set_zlabel("move index $t$", labelpad=-4)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def board_panel(ax, board, rows: list[dict], flipped: list[int]) -> None:
    """The shared prefix's last position, with every branch move on its square."""
    draw_board(ax, board.state.flatten())
    for square in flipped:
        x, y = square_xy(square)
        ax.add_patch(plt.Circle((x, y), FLIP_RING, fill=False, lw=1.1, zorder=5,
                                ls=(0, (3, 2)), edgecolor=INK))
    for r in rows:
        st = branch_style(r["category"], r["is_control"])
        x, y = square_xy(r["square"])
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, zorder=4,
                                   lw=st["lw"], ls=st["ls"], edgecolor=st["color"]))
        ax.plot(x - CORNER, y - CORNER, marker=st["marker"], ms=8, lw=0,
                color=st["color"], markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
                markeredgewidth=1.0, zorder=6, label=st["label"])


def board_note(board) -> str:
    """The board panel's grammar, for the caller to place near it."""
    mover = "black" if board.next_hand_color > 0 else "white"
    return (f"discs in absolute colour, {mover} to move at $t_0$; outlined cell = a\n"
            "branch square, its marker in the corner so a disc underneath stays\n"
            "visible; dashed ring = a disc the played move flips")
