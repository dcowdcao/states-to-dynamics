"""The panels the violation-divergence figures draw a branch set in.

`fork_panel` is one representation space: the branches as paths, with move index
on the third axis. `rule_fork_panel` is the same panel with the six illegal
categories collapsed onto the three rules they violate -- one mean per rule,
over its members drawn faint. `rule_fork_population_panel` is that panel over a
population of source games: the mean field alone, with no member traces.
`rule_ratio_panel` is the 2-D event-locked form: one per-position
quantity, already divided by the reference branch's own, one curve per rule, and
`rule_ratio_population_panel` is that panel over a population: no member traces,
a bootstrap band instead, and both legal controls drawn, since neither is the
divisor there. `board_panel` is the position they fork on, with each branch move
on its square.
All take their colour, marker and label from `branch_style`, so a branch reads
the same in either. Layout -- how many panels, where the legend goes -- belongs
to the caller; these draw one axes each.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter, MaxNLocator

from spade.figures.branch_style import MEMBER, branch_style, ci_band_proxy, legend_proxy
from spade.figures.othello_board import draw_board, square_xy
from spade.figures.paper_style import (
    DISC_EDGE, DISC_STYLE, INK, MUTED, PRE_EVENT_FACE)
from spade.othello.taxonomy import LEGAL
from spade.violation_divergence.rule_groups import rule_rows

ELEV, AZIM = 18, -58  # fixed so the fork panel is reproducible
CORNER = 0.30         # marker offset from the cell centre, which may hold a disc
GHOST_R = 0.12        # would-be disc, in cells: half a real disc, in the far corner
FLIP_RING = 0.38      # ring radius around a flipped disc, in cells
PLANE_TICKS = 3       # intervals; symmetric about zero, so the reference always gets a tick
# Four curves of one hue family in a ratio panel: markers every third sampled
# position, staggered, are what tells them apart with the colour gone.
MARK_STEP = 3


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


def rule_fork_population_panel(ax, coords: np.ndarray, rows: list[dict],
                               pos: np.ndarray, t0: int, legend: bool,
                               zlabel: bool = True) -> None:
    """`rule_fork_panel` over a population: one path per rule through the mean
    field of `coords` [n_branches, T, 2].

    No faint member traces: a member here is itself a mean over games, so it
    would read as game-to-game spread rather than as the within-rule spread the
    single-game panel's traces carry. The panel shows shape only -- the caller's
    2-D companion panel carries magnitude and its interval (FIGURE_GUIDELINES 4.8).
    """
    k = int(np.flatnonzero(pos == t0)[0])
    drawn = []
    for i, r in enumerate(rows):
        if r["category"] != LEGAL:
            continue
        st = branch_style(r["category"], r["is_control"])
        _path(ax, coords[i, k:], pos[k:], st, st["label"] if legend else None)
        drawn.append(coords[i, k:])
    for members in rule_rows(rows).values():
        st = branch_style(rows[members[0]]["category"], 0, by_rule=True)
        label = f"{st['label']} (mean of {len(members)})" if legend else None
        mean = coords[members, k:].mean(axis=0)
        _path(ax, mean, pos[k:], st, label)
        drawn.append(mean)
    # The illegal member rows are not drawn here, only their rule mean, so the
    # limits are taken over what is on the canvas and not over `coords`.
    _fork_axes(ax, np.concatenate(drawn), pos, t0, legend, zlabel)


def _ratio_line(ax, x: np.ndarray, y: np.ndarray, st: dict, offset: int,
                label: str | None) -> None:
    ax.plot(x, y, ls=st["ls"], color=st["color"], lw=st["lw"], marker=st["marker"],
            ms=4, markevery=(offset, MARK_STEP), markerfacecolor=st["mfc"],
            markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"],
            label=label)


def _ratio_axes(ax, pos: np.ndarray, t0: int, legend: bool) -> None:
    """The furniture every ratio panel shares: the shaded shared prefix, the
    branch rule at t0, and the reference at 1.

    Event-locked at t0 (FIGURE_GUIDELINES 4.2). The positions left of the branch
    -- where every branch is token-identical, so the ratio is 1 by construction
    -- are shaded and drawn rather than cut away, with a legend entry saying
    which. The caller asserts the 1."""
    free = pos < t0
    ax.set_xlim(pos[0] - 1, pos[-1] + 1)
    ax.axvspan(pos[0] - 1, (pos[free][-1] + pos[~free][0]) / 2, color=PRE_EVENT_FACE,
               lw=0, zorder=0,
               label="shared prefix: branches identical, ratio = 1 by construction"
                     if legend else None)
    ax.axvline(t0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    st = branch_style(LEGAL, 0)
    # Flat and thinner than REFERENCE's weight: it spans every panel, and a 2.4 pt
    # black rule through the middle of one would read as the loudest series there.
    ax.axhline(1, color=st["color"], lw=1.3, zorder=1.5,
               label=f"{st['label']}, the line at 1" if legend else None)


def ratio_series(rows: list[dict]) -> list[list[int]]:
    """Member row indices per series of a population ratio panel: one per rule,
    then each legal control. A rule is one series and not a mean drawn over its
    members, so the caller can bootstrap exactly what is drawn."""
    return list(rule_rows(rows).values()) + [[i] for i, r in enumerate(rows)
                                             if r["is_control"]]


def rule_ratio_panel(ax, ratio: np.ndarray, rows: list[dict], pos: np.ndarray,
                     t0: int, legend: bool) -> None:
    """One curve per rule of a per-position quantity already divided by the
    reference branch's own at the same position, so the reference *is* the line
    at 1 and the legal control's curve is the empirical null.

    `ratio` is [n_branches, len(pos)] and `pos` the board positions its columns
    sit at.
    """
    _ratio_axes(ax, pos, t0, legend)
    for i, r in enumerate(rows):
        if r["is_control"]:
            st = branch_style(r["category"], r["is_control"])
            _ratio_line(ax, pos, ratio[i], st, 0, st["label"] if legend else None)
    for j, members in enumerate(rule_rows(rows).values()):
        st = branch_style(rows[members[0]]["category"], 0, by_rule=True)
        for i in members:
            ax.plot(pos, ratio[i], color=st["color"], **MEMBER)
        _ratio_line(ax, pos, ratio[members].mean(axis=0), st, (j + 1) % MARK_STEP,
                    f"{st['label']} (mean of {len(members)})" if legend else None)
    if legend:
        legend_proxy(ax)


def rule_ratio_population_panel(ax, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                                rows: list[dict], pos: np.ndarray, t0: int,
                                legend: bool) -> None:
    """`rule_ratio_panel` over a population of source games: the mean over games
    of each series' per-game ratio, inside the band the caller bootstrapped.

    `mean`, `lo` and `hi` are [len(ratio_series(rows)), len(pos)], in that
    function's order. No faint member traces: a member here is itself a mean over
    games and would read as game-to-game spread, which is what the band carries.
    Both legal controls are drawn, because neither is the divisor in this panel --
    they are two draws of one null, and the gap between them is the size of a
    difference that means nothing.
    """
    _ratio_axes(ax, pos, t0, legend)
    for k, members in enumerate(ratio_series(rows)):
        r = rows[members[0]]
        st = branch_style(r["category"], r["is_control"], by_rule=True)
        ax.fill_between(pos, lo[k], hi[k], color=st["color"], alpha=0.16, lw=0,
                        zorder=st["zorder"] - 0.5)
        label = (st["label"] if r["is_control"]
                 else f"{st['label']} (mean of {len(members)})")
        _ratio_line(ax, pos, mean[k], st, k % MARK_STEP, label if legend else None)
    if legend:
        ci_band_proxy(ax)


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
