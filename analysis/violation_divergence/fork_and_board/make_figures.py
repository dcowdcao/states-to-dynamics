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

`--variant rollout` draws the same eight branches out of the rollout branch set,
where every branch -- the reference included -- samples its own continuation from
OthelloGPT rather than replaying the source game's. A branch is K rows there, so
every quantity is reduced over K explicitly: paired on the sample index, because
a rollout's reference is the one drawn from the same uniform stream, and after
phi rather than before, because phi is nonlinear and the curvature of an averaged
path is a path no rollout takes. That set saturates, so it is drawn to
ROLLOUT_RMAX and not to the study window's r = 24; README.md carries the numbers.

The figure carries no prose block. Those facts, the selection of the game above
all, are the figure's caption and live in README.md (FIGURE_GUIDELINES 2, 5);
the rule key and, on the rollout variant, what was averaged and where the drawn
window stops stay inside the figure, because a reader must not need another
document to know what R2 is or what a path is a mean of.

Usage: python analysis/violation_divergence/fork_and_board/make_figures.py
       [--game 1235] [--plane displacement|r21] [--embedding pca|isomap]
       [--variant spliced|rollout] [--camera-ready]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
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
ROW_GAP = 0.7              # clear air between the grid's axis labels and (b)'s title
# R^21 is 11 step-length components and 10 turning-angle ones, so it carries
# displacement magnitude as well as curvature; the name says both.
PLANE_NAME = {"displacement": "Displacement",
              "r21": "R$^{21}$ Curvature & Displacement Space"}
METHOD_NAME = {"pca": "PCA", "isomap": "Isomap"}
VARIANTS = {"spliced": "", "rollout": "rollout"}
# Past r = +8 the rollout set's illegal and control δ̂ agree to within 0.5% and
# cross over by r = +13, so the extra offsets draw separation that is not there.
ROLLOUT_RMAX = 8


def paired_mean(X: np.ndarray, ref: int, K: int) -> np.ndarray:
    """[n*K, ...] -> [n, ...]: each row's difference from the reference drawn on
    the same rollout, averaged over the K rollouts of its branch.

    Rows are K-contiguous per branch, and a rollout's reference is the one sharing
    its uniform stream, so the difference is paired before it is averaged. K = 1
    is the spliced set, where this is exactly X - X[ref]."""
    Y = X.reshape(-1, K, *X.shape[1:])
    return (Y - Y[ref]).mean(axis=1)


def project(plane: str, method: str, F: np.ndarray, ref: int, k: int, K: int):
    """(coords [n, T', 2], the two axis labels, positions dropped from the front).

    `k` indexes t0 in the sampled positions. The R^21 plane applies phi to each
    branch on its own and differences inside R^21: phi(D) is undefined over the
    shared prefix, where D is exactly zero, and phi of a mean of trajectories
    would be the curvature of a path no branch takes. That is also why the K
    rollouts are averaged after phi and never before it.
    """
    if plane == "displacement":
        D = paired_mean(F, ref, K)
        n, T, dim = D.shape
        coords, var, _ = EMBEDDINGS[method](D.reshape(-1, dim))
        return coords.reshape(n, T, 2), axis_labels(method, var), 0
    dphi = paired_mean(windowed(F, W), ref, K)[:, W - 1:]
    branch_free = k - (W - 1)
    assert branch_free > 0, \
        f"only {k} sampled positions before t0: no branch-free w={W} window"
    assert np.nanmax(np.abs(dphi[:, :branch_free])) == 0, \
        "the shared prefix does not collapse onto the origin in R^21"
    flat = dphi.reshape(-1, dphi.shape[-1])
    p = fit_plane(flat, centred=False)
    return (p.project(flat).reshape(dphi.shape[:2] + (2,)),
            axis_labels(method, p.var), W - 1)


def spread_field(plane: str, Z: np.ndarray, ref: int, post: np.ndarray,
                 K: int) -> np.ndarray:
    """What the within/between-rule spread ratio is measured on, in raw layer 5 at
    full width and never in a 2-D projection: the same quantity the panels embed.

    Δφ is column-scaled first — its 21 components mix degrees with
    activation-scale lengths, so a raw Euclidean distance is a distance in θ."""
    if plane == "displacement":
        return paired_mean(Z, ref, K)[:, post]
    dphi = paired_mean(windowed(Z, W), ref, K)
    sd = np.nanstd(dphi.reshape(-1, dphi.shape[-1]), axis=0)
    return (dphi / np.where(sd > 0, sd, 1))[:, post]


def variant_note(plane: str, K: int) -> str:
    """The rollout set's own caveat, in the figure because it changes how every
    panel reads: what a path is a mean of, and where the drawn window stops.

    Wrapped no wider than the board note beside it, or the tight bbox grows the
    canvas and the two variants stop being the same size."""
    where = ("after φ, which is nonlinear" if plane == "r21"
             else "in the plane, whose projection is linear")
    return (f"rollout variant: every branch, the reference included, samples its own "
            f"continuation from OthelloGPT.\nEach path is the mean over the branch's "
            f"K = {K} rollouts, paired on the sample index and\ntaken {where}.\nDrawn to "
            f"r = +{ROLLOUT_RMAX}: further out illegal and control δ̂ agree to within "
            f"0.5% and the fork\ncarries no separation (README).")


def figure(g: Game, Z: np.ndarray, pos: np.ndarray, plane: str, method: str,
           variant: str, K: int, device: str, camera_ready: bool) -> Path:
    cells = spaces()
    ref = reference_row(g.rows)
    k = int(np.flatnonzero(pos == g.t0)[0])
    nrow = int(np.ceil(len(cells) / NCOL))

    # ROW_GAP buys clear air between the grid's bottom axis labels and (b)'s
    # title; it is added to the canvas rather than taken out of the panels.
    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + BOARD_H + ROW_GAP))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BOARD_H],
                          left=0.03, right=0.97, top=0.93, bottom=0.02,
                          hspace=ROW_GAP / BOARD_H)
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
        coords, (xlab, ylab), dropped = project(plane, method, F, ref, k, K)
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
    note = board_note(board, by_rule=True)
    if variant:
        note += "\n\n" + variant_note(plane, K)
    band.text(0.0, below(band, leg, 0.06), note,
              ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    rule_member_key(fig, x=0.005, y=-0.012)

    if not camera_ready:
        fig.suptitle(f"Fork Trajectory ({PLANE_NAME[plane]} {METHOD_NAME[method]}) in "
                     f"Each Representation Space"
                     + (f" — {variant} branches" if variant else ""), fontsize=12)
    suffix = f"_{variant}" if variant else ""
    return save_figure(
        fig, FIG_DIR / f"game_{g.game}_fork_and_board_{plane}_{method}{suffix}",
        camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--plane", default="displacement", choices=list(PLANE_NAME))
    parser.add_argument("--embedding", default="pca", choices=list(METHOD_NAME))
    parser.add_argument("--variant", default="spliced", choices=list(VARIANTS),
                        help="rollout: every branch samples its own continuation, "
                             "K rows per branch, averaged over K")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    assert args.plane == "displacement" or args.embedding == "pca", \
        "--plane r21 fits its own plane inside R^21; --embedding is the displacement one"

    apply_style(args.camera_ready)
    device = select_device()
    variant = VARIANTS[args.variant]
    base = variant_dir("divergence", variant=variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # mmap: the rollout tensor is 4.2 GB and eight branches of one game are read
    # out of it. The spliced tensor reads the same values either way.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1

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
    if variant:
        pos = pos[pos - g.t0 <= ROLLOUT_RMAX]
    Z = acts[g.idx][:, pos, :].float().numpy()
    # Z keeps all K rows of every branch, and each reduction below is over them.
    # g carries one row per branch from here on: the metadata is the branch's.
    g = replace(g, idx=g.idx[::K], rows=g.rows[::K])
    # The ratio is the figure's caption, not the figure's: README.md quotes it.
    field = spread_field(args.plane, Z, reference_row(g.rows), pos >= g.t0, K)
    print(f"within/between rule spread ratio, game {g.game}, {args.plane}, "
          f"{args.variant}: {spread_ratio(field, g.rows):.3f}")
    out = figure(g, Z, pos, args.plane, args.embedding, variant, K, device,
                 args.camera_ready)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
