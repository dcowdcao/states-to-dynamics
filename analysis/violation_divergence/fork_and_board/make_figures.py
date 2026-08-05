"""What the branches are, and where they go in every representation space.

Two halves of one question, so one figure. (b) is the board the game forks on,
with each branch move on the square it plays. (a) is where those same branches
travel: one 3-D fork per representation space, raw layer 5 first so the eye
calibrates on it. Both halves draw from spade.figures.branch_style, so a colour
and a marker mean the same thing in either half and one legend serves both; the
free space beside the square board carries that legend and the rule key rather
than sitting empty.

The six illegal categories are grouped into the three constraint rules they
violate: (a) draws one mean per rule over its members drawn faint, and (b)
colours a branch square by its rule, so the series and the R1/R2/R3 key match
exactly. Group sizes are 2, 3 and 1, so the legend carries n and the legend
title carries the within- over between-rule spread ratio.

Planes are refit per space, so cells are comparable in the *shape* of the fork
and never in position. One game (n=1): an illustration of the design, not a
claim; the study's numbers are in the parent README.

Two planes, same panels. `displacement` embeds D(t) = z_b(t) - z_ref(t), the
better-validated default. `r21` embeds the curvature displacement
Δφ(t) = φ(z_b(t)) - φ(z_ref(t)) inside ReStraV's 21-D descriptor space. φ is
nonlinear, so it is applied to each member category separately and the rule mean
is taken afterwards, in the plane; the projection is linear, so mean-then-project
and project-then-mean agree, while phi-of-a-mean would be a different quantity.

The figure carries no prose block. Those facts, the selection of the game above
all, are the figure's caption and live in README.md (FIGURE_GUIDELINES 2, 5);
the rule key stays inside the figure, because a reader must not need another
document to know what R2 is.

Usage: python analysis/violation_divergence/fork_and_board/make_figures.py
       [--game 1235] [--plane displacement|r21] [--embedding pca|isomap]
       [--camera-ready]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.branch_panels import (
    axis_labels, board_note, board_panel, mover_name, rule_fork_panel)
from spade.figures.branch_style import below, ordered_legend, rule_member_key
from spade.figures.paper_style import MUTED, apply_style, save_figure
from spade.othello.boards import flip_set, replay
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.curvature_plane import W, fit_plane, windowed
from spade.violation_divergence.fork_plane import (
    EMBEDDINGS, positions, space_features, spaces)
from spade.violation_divergence.rule_groups import spread_ratio

FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../game_trajectory's grid
BOARD_H = 3.9              # the board row; the board is square, so this is its width too
PLANE_NAME = {"displacement": "Displacement", "r21": "R$^{21}$ Curvature Space"}
METHOD_NAME = {"pca": "PCA", "isomap": "Isomap"}


def project(plane: str, method: str, F: np.ndarray, ref: int, k: int):
    """(coords [n, T', 2], the two axis labels, positions dropped from the front).

    `k` indexes t0 in the sampled positions. The R^21 plane applies phi to each
    branch on its own and differences inside R^21: phi(D) is undefined over the
    shared prefix, where D is exactly zero, and phi of a mean of trajectories
    would be the curvature of a path no branch takes.
    """
    if plane == "displacement":
        n, T, D = F.shape
        coords, var, _ = EMBEDDINGS[method]((F - F[ref]).reshape(-1, D))
        return coords.reshape(n, T, 2), axis_labels(method, var), 0
    phi = windowed(F, W)
    dphi = (phi - phi[ref])[:, W - 1:]
    branch_free = k - (W - 1)
    assert branch_free > 0, \
        f"only {k} sampled positions before t0: no branch-free w={W} window"
    assert np.nanmax(np.abs(dphi[:, :branch_free])) == 0, \
        "the shared prefix does not collapse onto the origin in R^21"
    flat = dphi.reshape(-1, dphi.shape[-1])
    p = fit_plane(flat, centred=False)
    return (p.project(flat).reshape(dphi.shape[:2] + (2,)),
            axis_labels(method, p.var), W - 1)


def spread_field(plane: str, Z: np.ndarray, ref: int, post: np.ndarray) -> np.ndarray:
    """What the within/between-rule spread ratio is measured on, in raw layer 5 at
    full width and never in a 2-D projection: the same quantity the panels embed.

    Δφ is column-scaled first — its 21 components mix degrees with
    activation-scale lengths, so a raw Euclidean distance is a distance in θ."""
    if plane == "displacement":
        return (Z - Z[ref])[:, post]
    phi = windowed(Z, W)
    dphi = phi - phi[ref]
    sd = np.nanstd(dphi.reshape(-1, dphi.shape[-1]), axis=0)
    return (dphi / np.where(sd > 0, sd, 1))[:, post]


def figure(g: Game, Z: np.ndarray, pos: np.ndarray, plane: str, method: str,
           device: str, camera_ready: bool) -> Path:
    cells = spaces()
    ref = reference_row(g.rows)
    k = int(np.flatnonzero(pos == g.t0)[0])
    nrow = int(np.ceil(len(cells) / NCOL))

    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + BOARD_H))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BOARD_H],
                          left=0.03, right=0.97, top=0.93, bottom=0.02, hspace=0.05)
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.06, hspace=0.16)
    # The board cell is as wide as it is tall, so the square board fills it and the
    # band beside it gets the rest rather than a strip of dead space.
    lower = gs[1].subgridspec(1, 2, wspace=0.02,
                              width_ratios=[BOARD_H, CELL_W * NCOL * 0.94 - BOARD_H])

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("(a) where the branches go: the fork in each representation space, "
                   "one path per rule",
                   loc="left", fontsize=11, pad=18)

    for j, (name, entry) in enumerate(cells):
        F = space_features(Z, entry, device)
        coords, (xlab, ylab), dropped = project(plane, method, F, ref, k)
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL], projection="3d")
        rule_fork_panel(ax, coords, g.rows, pos[dropped:], g.t0,
                        legend=(j == 0), zlabel=(j % NCOL == 0))
        ax.set_xlabel(xlab, labelpad=-2, fontsize=8)
        ax.set_ylabel(ylab, labelpad=-2, fontsize=8)
        ax.set_title(name, loc="left", fontsize=8.5)
        if j == 0:
            keyed = ax

    board = replay(g.tokens[:g.t0])
    flipped = flip_set(board.state, g.rows[ref]["square"], board.next_hand_color)
    ax_board = fig.add_subplot(lower[0])
    board_panel(ax_board, board, g.rows, flipped, by_rule=True)
    ax_board.set_title(f"(b) what they are: the position at $t_0$ "
                       f"({mover_name(board).capitalize()} to move),\nand the move each "
                       f"branch plays there", loc="left", fontsize=11)

    band = fig.add_subplot(lower[1])
    band.axis("off")
    # Legend and note are centred in the band, not pinned to its top: the band is
    # as tall as the square board beside it and the block is shorter than that.
    # The note hangs off the legend's own bottom edge, which moves with how many
    # entries wrapped; the rule key goes under the whole figure, which is the only
    # place with the width its three lines need.
    leg = ordered_legend(band, keyed, 2, loc="center left", bbox_to_anchor=(0.0, 0.62))
    band.text(0.0, below(band, leg, 0.06), board_note(board, by_rule=True),
              ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    rule_member_key(fig, x=0.005, y=-0.012)

    if not camera_ready:
        fig.suptitle(f"Fork Trajectory ({PLANE_NAME[plane]} {METHOD_NAME[method]}) in "
                     f"Each Representation Space", fontsize=12)
    return save_figure(fig, FIG_DIR / f"game_{g.game}_fork_and_board_{plane}_{method}",
                       camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--plane", default="displacement", choices=list(PLANE_NAME))
    parser.add_argument("--embedding", default="pca", choices=list(METHOD_NAME))
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    assert args.plane == "displacement" or args.embedding == "pca", \
        "--plane r21 fits its own plane inside R^21; --embedding is the displacement one"

    apply_style(args.camera_ready)
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)

    assert args.game in complete_games(meta), \
        f"game {args.game} does not carry every branch type"
    g = load_game(meta, args.game)
    # One control, not two: control 2 is the null *for* control 1 and belongs
    # where control 1 is measured, not in an illustration of the construction.
    keep = [i for i, r in enumerate(g.rows) if r["is_control"] != 2]
    g = replace(g, idx=g.idx[keep], rows=[g.rows[i] for i in keep])

    # The R^21 window reads 14 plies back, so it needs prefix context the study
    # window does not carry; the extra offsets are computed and never drawn.
    pos = positions(g.t0, W - 1 if args.plane == "r21" else 0)
    Z = acts[g.idx][:, pos, :].float().numpy()
    # The ratio is the figure's caption, not the figure's: README.md quotes it.
    field = spread_field(args.plane, Z, reference_row(g.rows), pos >= g.t0)
    print(f"within/between rule spread ratio, game {g.game}, {args.plane}: "
          f"{spread_ratio(field, g.rows):.3f}")
    out = figure(g, Z, pos, args.plane, args.embedding, device, args.camera_ready)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
