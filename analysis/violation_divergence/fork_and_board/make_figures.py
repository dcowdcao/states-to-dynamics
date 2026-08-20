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

(c) is the number (a) asks the reader to eyeball: how far each rule runs from
the reference, averaged over the drawn offsets r >= 0. It is drawn twice per
rule -- in the plane the cell above it draws, and in that cell's full space --
and both are divided by the same cell's legal control. The division is what
makes nine cells readable side by side at all: a refit plane's units are the
space's own, and those differ by ~800x across the grid, so the raw distances
are not a quantity nine panels share. The pair is what keeps the plane honest,
since a plane holds only part of the displacement (27%+10% here) and holds
different parts of different branches: in raw layer 5 the plane says R3 runs
4.1x the control while the full 512-d space says 1.5x.

Three planes, same panels. `displacement` embeds D(t) = z_b(t) - z_ref(t), the
better-validated default. `r21` embeds the curvature displacement
Δφ(t) = φ(z_b(t)) - φ(z_ref(t)) inside ReStraV's 21-D descriptor space, and `d7`
embeds that same Δφ cut to the 7 raw step-length components d1..d7 -- the half
`../r21_shared_basis` finds disagreeing with the turning angles, drawn as a fork
rather than as one scalar run ratio. φ is nonlinear, so it is applied to each
member category separately and the rule mean is taken afterwards, in the plane;
the projection is linear, so mean-then-project and project-then-mean agree, while
phi-of-a-mean would be a different quantity.

`--variant rollout` draws the same eight branches out of the rollout branch set,
where every branch -- the reference included -- samples its own continuation from
OthelloGPT rather than replaying the source game's. A branch is K rows there, so
every quantity is reduced over K explicitly: paired on the sample index, because
a rollout's reference is the one drawn from the same uniform stream, and after
phi rather than before, because phi is nonlinear and the curvature of an averaged
path is a path no rollout takes. That set saturates, so it is drawn to
ROLLOUT_RMAX and not to the study window's r = 24; README.md carries the numbers.

The figure carries no prose. Its caption -- how the rollout paths were reduced,
where the window stops, what (c) is not -- lives in README.md (FIGURE_GUIDELINES
2, 5). What stays inside is what a reader cannot look up while looking at it: the
board's reading key, the R1/R2/R3 key, (c)'s estimator on (c)'s own axis, and one
line saying the game was selected, without which the panels read as a result.

Usage: python analysis/violation_divergence/fork_and_board/make_figures.py
       [--game 1235] [--plane displacement|r21|d7] [--embedding pca|isomap]
       [--variant spliced|rollout] [--camera-ready]
"""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.figures.branch_panels import (
    axis_labels, board_note, board_panel, mover_name, plane_displacement, rule_fork_panel)
from spade.figures.branch_style import (
    below, branch_style, ordered_legend, rule_member_key)
from spade.figures.paper_style import GRID, INK, MUTED, NULL, apply_style, save_figure
from spade.othello.boards import flip_set, replay
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.curvature_plane import (
    PLANE_COLUMNS, W, fit_plane, windowed)
from spade.violation_divergence.fork_plane import (
    EMBEDDINGS, positions, space_features, spaces)
from spade.violation_divergence.population import paired_mean
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../game_trajectory's grid
BOARD_H = 3.9              # the board row; the board is square, so this is its width too
METRIC_H = 2.8             # the (c) row; bars need far less height than the board
BAR_W = 0.26               # three rule bars inside a space's unit-wide group
ROW_GAP = 0.7              # clear air between the grid's axis labels and (b)'s title
# R^21 is 11 step-length components and 10 turning-angle ones, so it carries
# displacement magnitude as well as curvature; the name says both, and `d7`'s says
# which half of it is left.
PLANE_NAME = {"displacement": "Displacement",
              "r21": "R$^{21}$ Curvature & Displacement Space",
              "d7": "R$^{21}$ Raw Step-Length Block $d_1$–$d_7$"}
METHOD_NAME = {"pca": "PCA", "isomap": "Isomap"}
VARIANTS = {"spliced": "", "rollout": "rollout"}
# Past r = +8 the rollout set's illegal and control δ̂ agree to within 0.5% and
# cross over by r = +13, so the extra offsets draw separation that is not there.
ROLLOUT_RMAX = 8
# The default game is the argmax of a differentiation criterion over 1550 games, so
# its panels are cleaner than a typical fork's and a reader who does not know that
# reads a selected game as a result. One line; the rest is in README.md.
SELECTION_NOTE = (f"game {FIGURE_GAME} is selected, not typical: the argmax over 1550 "
                  f"branch-complete games of its illegal branches' mutual separation "
                  f"(README)")


@dataclass(frozen=True)
class Projection:
    """One cell: the plane it draws, and the space that plane was fit in.

    `coords` and `space` carry the same rows over the same positions, so a
    distance can be measured in either and the two answers compared."""
    coords: np.ndarray          # [n, T', 2], what the fork panel draws
    space: np.ndarray           # [n, T', dim], the quantity the plane projects
    labels: tuple[str, str]     # the two plane axis labels
    dropped: int                # sampled positions dropped from the front


def project(plane: str, method: str, F: np.ndarray, ref: int, k: int,
            K: int) -> Projection:
    """`k` indexes t0 in the sampled positions. A descriptor plane applies phi to
    each branch on its own and differences inside the descriptor: phi(D) is
    undefined over the shared prefix, where D is exactly zero, and phi of a mean
    of trajectories would be the curvature of a path no branch takes. That is also
    why the K rollouts are averaged after phi and never before it. `d7` keeps the
    same Δφ and only the raw step-length columns of it.

    `space` is the projected-from quantity itself, column-scaled exactly as
    `fit_plane` scales it, so a distance in it is a distance in the space the
    plane was fit in and not in raw descriptor units.
    """
    if plane == "displacement":
        D = paired_mean(F, ref, K)
        n, T, dim = D.shape
        coords, var, _ = EMBEDDINGS[method](D.reshape(-1, dim))
        return Projection(coords.reshape(n, T, 2), D, axis_labels(method, var), 0)
    dphi = paired_mean(windowed(F, W), ref, K)[:, W - 1:, PLANE_COLUMNS[plane]]
    branch_free = k - (W - 1)
    assert branch_free > 0, \
        f"only {k} sampled positions before t0: no branch-free w={W} window"
    assert np.nanmax(np.abs(dphi[:, :branch_free])) == 0, \
        f"the shared prefix does not collapse onto the origin in {plane}"
    flat = dphi.reshape(-1, dphi.shape[-1])
    p = fit_plane(flat, centred=False)
    return Projection(p.project(flat).reshape(dphi.shape[:2] + (2,)), dphi / p.scale,
                      axis_labels(method, p.var), W - 1)


def control_ratio(X: np.ndarray, pos: np.ndarray, t0: int, rows: list[dict]) -> np.ndarray:
    """[n]: each branch's mean distance from the reference over the drawn r >= 0,
    as a multiple of the legal control's.

    The distance itself is `plane_displacement`, and in a plane it is not a
    quantity nine cells share: the plane is refit per space and the spaces
    differ ~800x in scale, so the raw numbers only ever compare branches inside
    one cell. Dividing by that cell's own control leaves a dimensionless
    within-cell contrast -- the study's own illegal/control form -- which is
    what may be set side by side. The reference is 0 by construction.
    """
    d = plane_displacement(X, pos, t0, reference_row(rows))
    return d / d[next(i for i, r in enumerate(rows) if r["is_control"] == 1)]


def spread_field(plane: str, Z: np.ndarray, ref: int, post: np.ndarray,
                 K: int) -> np.ndarray:
    """What the within/between-rule spread ratio is measured on, in raw layer 5 at
    full width and never in a 2-D projection: the same quantity the panels embed.

    Δφ is column-scaled first — R^21's components mix degrees with
    activation-scale lengths, so a raw Euclidean distance is a distance in θ, and
    the raw step lengths d1..d7 are unequal in variance among themselves."""
    if plane == "displacement":
        return paired_mean(Z, ref, K)[:, post]
    dphi = paired_mean(windowed(Z, W), ref, K)[..., PLANE_COLUMNS[plane]]
    sd = np.nanstd(dphi.reshape(-1, dphi.shape[-1]), axis=0)
    return (dphi / np.where(sd > 0, sd, 1))[:, post]


def metric_panel(ax, cells: list[tuple[str, np.ndarray, np.ndarray]],
                 rows: list[dict], offsets: np.ndarray) -> None:
    """(c): the number every cell of (a) asks the reader to eyeball.

    Bar = the distance in the plane that cell draws; open marker = the same
    distance in that cell's full space. The gap between the two is the part of
    the displacement the plane does not hold, which is otherwise invisible --
    a branch that leaves the plane reads as a branch that stayed near the
    reference. Both are ratios to the cell's own legal control, the only form
    in which nine refit planes may be set side by side (`control_ratio`).
    """
    groups = rule_rows(rows)
    ax.axhline(1, color=NULL["color"], ls=NULL["ls"], lw=1.1, zorder=1)
    for j, (_, plane, space) in enumerate(cells):
        ax.axvline(j - 0.5, color=GRID, lw=0.8, zorder=0)
        for m, members in enumerate(groups.values()):
            st = branch_style(rows[members[0]]["category"], 0, by_rule=True)
            x = j + (m - 1) * BAR_W
            ax.bar(x, plane[members].mean(), BAR_W * 0.88, color=st["color"],
                   alpha=0.85, edgecolor=INK, lw=0.4, zorder=2)
            ax.plot([x] * len(members), plane[members], lw=0, marker="o", ms=2.8,
                    color=INK, alpha=0.6, zorder=4)
            ax.plot(x, space[members].mean(), lw=0, marker=st["marker"], ms=7,
                    markerfacecolor="white", markeredgecolor=st["color"],
                    markeredgewidth=1.3, zorder=5)

    top = max(v.max() for _, p, s in cells for v in (p, s))
    ax.set_ylim(0, top * 1.34)
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_xticks(range(len(cells)), [name for name, _, _ in cells], fontsize=9)
    ax.set_ylabel("distance from the reference,\n$\\div$ the cell's own legal control",
                  fontsize=9)
    # The estimator rides on the axis, not in a note: it is what the bars are a
    # mean of, and the drawn window changes with the variant.
    ax.set_xlabel(f"mean over the {len(offsets)} drawn offsets r = "
                  f"{', '.join(str(r) for r in offsets[:2])}, …, {offsets[-1]}; "
                  f"one game (n = 1)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(False, axis="x")
    marks = [(Patch(facecolor=MUTED, alpha=0.85, edgecolor=INK, lw=0.4),
              "in the plane that cell draws (2-D)"),
             (plt.Line2D([], [], lw=0, marker="o", ms=7, markerfacecolor="white",
                         markeredgecolor=INK, markeredgewidth=1.3),
              "in that cell's full space (marker = rule)"),
             (plt.Line2D([], [], lw=0, marker="o", ms=2.8, color=INK, alpha=0.6),
              "one category inside the rule"),
             (plt.Line2D([], [], color=NULL["color"], ls=NULL["ls"], lw=1.1),
              "the cell's legal control (null) = 1")]
    ax.legend(*zip(*marks), ncol=2, fontsize=9, loc="upper left")


def figure(g: Game, Z: np.ndarray, pos: np.ndarray, plane: str, method: str,
           variant: str, K: int, device: str, camera_ready: bool) -> Path:
    cells = spaces()
    ref = reference_row(g.rows)
    k = int(np.flatnonzero(pos == g.t0)[0])
    nrow = int(np.ceil(len(cells) / NCOL))

    # ROW_GAP buys clear air between the grid's bottom axis labels and (b)'s
    # title; it is added to the canvas rather than taken out of the panels, once
    # per seam, so the grid and the board keep the size they had before (c).
    fig = plt.figure(figsize=(CELL_W * NCOL,
                              CELL_H * nrow + BOARD_H + METRIC_H + 2 * ROW_GAP))
    gs = fig.add_gridspec(3, 1, height_ratios=[CELL_H * nrow, BOARD_H, METRIC_H],
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

    measured = []
    for j, (name, entry) in enumerate(cells):
        F = space_features(Z, entry, device)
        p = project(plane, method, F, ref, k, K)
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL], projection="3d")
        rule_fork_panel(ax, p.coords, g.rows, pos[p.dropped:], g.t0,
                        legend=(j == 0), zlabel=(j % NCOL == 0))
        ax.set_xlabel(p.labels[0], labelpad=-2, fontsize=8)
        ax.set_ylabel(p.labels[1], labelpad=-2, fontsize=8)
        ax.set_title(name, loc="left", fontsize=8.5)
        # The cell's own name minus its hyperparameters: (c) has nine ticks to
        # fit and the settings are already on the cell above each of them.
        measured.append((name.split(" (")[0],
                         control_ratio(p.coords, pos[p.dropped:], g.t0, g.rows),
                         control_ratio(p.space, pos[p.dropped:], g.t0, g.rows)))
        if j == 0:
            keyed = ax

    # (c) as text, so the README can quote it without reading it off the bars.
    print("(c) mean distance from the reference over the drawn r >= 0, "
          "/ the cell's legal control — plane | full space")
    for name, dp, ds in measured:
        print(f"  {name:<14} " + "  ".join(
            f"{rule.split()[0]} {dp[m].mean():5.2f}|{ds[m].mean():5.2f}"
            for rule, m in rule_rows(g.rows).items()))

    board = replay(g.tokens[:g.t0])
    flipped = flip_set(board.state, g.rows[ref]["square"], board.next_hand_color)
    ax_board = fig.add_subplot(lower[0])
    board_panel(ax_board, board, g.rows, flipped, by_rule=True)
    ax_board.set_title(f"(b) what they are: the position at $t_0$ "
                       f"({mover_name(board).capitalize()} to move),\nand the move each "
                       f"branch plays there", loc="left", fontsize=11)

    band = fig.add_subplot(lower[1])
    band.axis("off")
    # Legend and board key are centred in the band, not pinned to its top: the band
    # is as tall as the square board beside it and the block is shorter than that.
    # The key hangs off the legend's own bottom edge, which moves with how many
    # entries wrapped.
    leg = ordered_legend(band, keyed, 2, loc="center left", bbox_to_anchor=(0.0, 0.62))
    band.text(0.0, below(band, leg, 0.06), board_note(board, by_rule=True),
              ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.5)

    metric = fig.add_subplot(gs[2])
    metric_panel(metric, measured, g.rows, pos[pos >= g.t0] - g.t0)
    metric.set_title("(c) how far they run: each rule's distance from the reference, "
                     "in the plane its cell draws (bar)\nand in that cell's full space "
                     "(open marker), as a multiple of the same cell's legal control",
                     loc="left", fontsize=11)

    # The muted band under the figure, in reading order: the selection disclosure,
    # then the key naming what R1/R2/R3 average. Each hangs off the last, since
    # both heights depend on where the line above them wrapped.
    y = below(fig, metric.xaxis.label, 0.010)
    if g.game == FIGURE_GAME:
        y = below(fig, fig.text(0.005, y, SELECTION_NOTE, ha="left", va="top",
                                fontsize=7.5, color=MUTED), 0.004)
    rule_member_key(fig, x=0.005, y=y)

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
        "a descriptor plane fits its own; --embedding is the displacement one"

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

    # A descriptor window reads 14 plies back, so it needs prefix context the study
    # window does not carry; the extra offsets are computed and never drawn.
    pos = positions(g.t0, 0 if args.plane == "displacement" else W - 1)
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
