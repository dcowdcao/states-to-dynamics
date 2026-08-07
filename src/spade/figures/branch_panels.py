"""The two panels the violation-divergence figures draw a branch set in.

`fork_panel` is one representation space: the branches as paths, with move index
on the third axis. `rule_fork_panel` is the same panel with the six illegal
categories collapsed onto the three rules they violate -- one mean per rule,
over its members drawn faint. `board_panel` is the position they fork on, with
each branch move on its square. All take their colour, marker and label from
`branch_style`, so a branch reads the same in either. Layout -- how many panels,
where the legend goes -- belongs to the caller; these draw one axes each.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, MaxNLocator

from spade.figures.branch_style import MEMBER, branch_style, legend_proxy
from spade.figures.othello_board import draw_board, square_xy
from spade.figures.paper_style import DISC_EDGE, DISC_STYLE, INK, MUTED
from spade.othello.taxonomy import LEGAL
from spade.violation_divergence.rule_groups import rule_rows

ELEV, AZIM = 18, -58  # fixed so the fork panel is reproducible
CORNER = 0.30         # marker offset from the cell centre, which may hold a disc
GHOST_R = 0.12        # would-be disc, in cells: half a real disc, in the far corner
FLIP_RING = 0.38      # ring radius around a flipped disc, in cells
PLANE_TICKS = 3       # intervals; symmetric about zero, so the reference always gets a tick


def axis_labels(method: str, var: np.ndarray) -> tuple[str, str]:
    if method == "isomap":
        return "Isomap-1 (arb. units)", "Isomap-2 (arb. units)"
    return f"PC1 ({var[0]:.0%} var)", f"PC2 ({var[1]:.0%} var)"


def plane_displacement(coords: np.ndarray, pos: np.ndarray, t0: int,
                       ref: int) -> np.ndarray:
    """[n, T, 2] -> [n]: mean in-plane distance from the reference over r >= 0.

    What the eye is trying to judge off a fork panel, as a number. It measures
    the plane the panel actually draws, so it inherits that plane's limits: the
    plane is refit per space and holds only part of the displacement, so this
    compares branches WITHIN a panel and never across panels.
    """
    post = pos >= t0
    return np.linalg.norm(coords[:, post] - coords[ref, post], axis=-1).mean(axis=1)


def _path(ax, seg: np.ndarray, z: np.ndarray, st: dict, label: str | None) -> None:
    """One branch or one rule mean: the line, its sparse markers, its end marker."""
    ax.plot(seg[:, 0], seg[:, 1], z, ls=st["ls"], color=st["color"], lw=st["lw"],
            marker=st["marker"], ms=3.5, markevery=3, markerfacecolor=st["mfc"],
            markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"],
            label=label)
    ax.plot([seg[-1, 0]], [seg[-1, 1]], [z[-1]], marker=st["marker"], ms=6,
            color=st["color"], markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
            markeredgewidth=0.7, zorder=st["zorder"] + 1)


def _fork_axes(ax, coords: np.ndarray, pos: np.ndarray, t0: int, legend: bool,
               zlabel: bool) -> None:
    """The furniture every fork panel shares: prefix line, branch point, axes.

    The plane axes carry three sparse ticks symmetric about zero. The plane is
    refit per panel, so they are a within-panel scale bar only -- the caller
    must say the cells are not comparable (FIGURE_GUIDELINES 4.6, 4.8)."""
    lo, hi = coords.min(), coords.max()
    ax.plot([0, 0], [0, 0], [pos[0], t0], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label="shared prefix (branches identical)" if legend else None)
    ax.plot([0], [0], [t0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(pos[0], pos[-1])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(PLANE_TICKS, symmetric=True))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="x", labelsize=7, pad=-3)
    ax.tick_params(axis="y", labelsize=7, pad=-3)
    if zlabel:
        ax.set_zlabel("move index $t$", labelpad=-4)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def fork_panel(ax, coords: np.ndarray, rows: list[dict], pos: np.ndarray, t0: int,
               legend: bool, zlabel: bool = True) -> None:
    """coords [n_branches, n_pos, 2]; move index is the vertical axis."""
    k = int(np.flatnonzero(pos == t0)[0])
    _fork_axes(ax, coords, pos, t0, legend, zlabel)
    for i, r in enumerate(rows):
        st = branch_style(r["category"], r["is_control"])
        _path(ax, coords[i, k:], pos[k:], st, st["label"] if legend else None)


def rule_fork_panel(ax, coords: np.ndarray, rows: list[dict], pos: np.ndarray,
                    t0: int, legend: bool, zlabel: bool = True) -> None:
    """One path per rule -- the mean of its members' displacements -- over its
    members drawn faint, so the reader sees the spread the mean is taken over.

    The mean is taken in the plane, after projection. PCA here is an uncentred
    linear map, so that is exactly the projection of the mean displacement;
    Isomap is not linear, and there the mean is a mean of embedded points."""
    k = int(np.flatnonzero(pos == t0)[0])
    _fork_axes(ax, coords, pos, t0, legend, zlabel)
    for i, r in enumerate(rows):
        if r["category"] != LEGAL:
            continue
        st = branch_style(r["category"], r["is_control"])
        _path(ax, coords[i, k:], pos[k:], st, st["label"] if legend else None)
    for members in rule_rows(rows).values():
        st = branch_style(rows[members[0]]["category"], 0, by_rule=True)
        for i in members:
            ax.plot(coords[i, k:, 0], coords[i, k:, 1], pos[k:], color=st["color"],
                    **MEMBER)
        label = f"{st['label']} (mean of {len(members)})" if legend else None
        _path(ax, coords[members, k:].mean(axis=0), pos[k:], st, label)
    if legend:
        legend_proxy(ax)


def mover_name(board) -> str:
    return "black" if board.next_hand_color > 0 else "white"


def board_panel(ax, board, rows: list[dict], flipped: list[int],
                by_rule: bool = False) -> None:
    """The shared prefix's last position, with every branch move on its square.

    Every branch square also carries the disc the move would place, in the
    mover's colour: half a real disc's size, dashed, translucent, and pinned to
    the corner opposite the category marker. Two of the six illegal categories
    play onto a square that already holds a real disc -- that is what makes them
    illegal, and the disc's colour is the only evidence for which of the two the
    branch is -- so the would-be disc is never drawn over the centre.
    """
    draw_board(ax, board.state.flatten())
    ghost = DISC_STYLE[mover_name(board)]["color"]
    for square in flipped:
        x, y = square_xy(square)
        ax.add_patch(plt.Circle((x, y), FLIP_RING, fill=False, lw=1.1, zorder=5,
                                ls=(0, (3, 2)), edgecolor=INK))
    for r in rows:
        st = branch_style(r["category"], r["is_control"], by_rule)
        x, y = square_xy(r["square"])
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, zorder=4,
                                   lw=st["lw"], ls=st["ls"], edgecolor=st["color"]))
        ax.add_patch(plt.Circle((x + CORNER, y + CORNER), GHOST_R, zorder=5,
                                facecolor=to_rgba(ghost, 0.45), lw=0.7,
                                ls=(0, (2, 1.5)), edgecolor=DISC_EDGE))
        ax.plot(x - CORNER, y - CORNER, marker=st["marker"], ms=8, lw=0,
                color=st["color"], markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
                markeredgewidth=1.0, zorder=6, label=st["label"])


def board_note(board, by_rule: bool = False) -> str:
    """The board panel's grammar, for the caller to place near it."""
    grouped = ("; a rule's member categories share one colour\nand marker here, and "
               "are told apart by their square" if by_rule else "")
    return (f"discs in absolute colour, {mover_name(board)} to move at $t_0$; outlined "
            "cell = a branch\nsquare, its marker in the corner so a disc underneath "
            "stays visible; small\ndashed disc in the opposite corner = the disc that "
            "branch would place, never\na disc on the board; dashed ring = a disc the "
            f"played move flips{grouped}")
