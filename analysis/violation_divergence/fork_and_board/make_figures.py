"""Where the branches go in every representation space.

A 3x3 grid of forks, one per space, raw layer 5 first so the eye calibrates on
it. Each cell carries one label centred under it -- its letter and the space it
draws, formatted as ../divergence_by_space formats its own so the two grids of an
appendix spread read alike -- enumerated in the figure's caption
(FIGURE_GUIDELINES 5). One legend under the grid serves all nine: they draw the
same series out of spade.figures.branch_style, so a colour and a marker mean the
same thing in every cell.

The six illegal categories are grouped into the three constraint rules they
violate -- one mean per rule over its members drawn faint, group sizes 2, 3 and
1, which the legend carries as n. What the branches *are* (the board at t0, each
branch move on the square it plays) is `../game_trajectory`'s panel and the
parent README's key.

Planes are refit per space, so cells are comparable in the *shape* of the fork
and never in position. One game (n=1), and a selected one: an illustration of
the design, not a claim. How far each rule actually runs from the reference,
over a population and with an interval, is `../divergence_by_space` in the same
plane and `../fork_population` for the same shape -- the quantitative companion
FIGURE_GUIDELINES 4.8 asks a 3-D projection to be read beside. `control_ratio`
still prints this game's own run ratios to stdout so README.md can quote them.

Three planes, same panels. `displacement` embeds D(t) = z_b(t) - z_ref(t), the
better-validated default. `r21` embeds the curvature displacement
Δφ(t) = φ(z_b(t)) - φ(z_ref(t)) inside ReStraV's 21-D descriptor space, and `d7`
embeds that same Δφ cut to the 7 raw step-length components d1..d7 -- the half
`../r21_shared_basis` finds disagreeing with the turning angles, drawn as a fork
rather than as one scalar run ratio. φ is nonlinear, so it is applied to each
member category separately and the rule mean is taken afterwards, in the plane;
the projection is linear, so mean-then-project and project-then-mean agree, while
phi-of-a-mean would be a different quantity.

`--l0-matched` swaps the seed-42 roster for the eight l0_operating_point picks,
one checkpoint per SAE type at the shared target L0. The seed-42 cells sit
wherever each type's own defaults left it on the sparsity axis, so a difference
between two of them confounds the architecture with its L0; this roster holds L0
fixed. Its files carry an _l0_matched tag and the canonical names are unchanged.
A cell is titled with its architecture and nothing else, so **the two grids are
identical on their faces** and which roster drew one is carried by the filename
and by README.md alone (FIGURE_GUIDELINES 2, 5).

`--variant rollout` draws the same eight branches out of the rollout branch set,
where every branch -- the reference included -- samples its own continuation from
OthelloGPT rather than replaying the source game's. A branch is K rows there, so
every quantity is reduced over K explicitly: paired on the sample index, because
a rollout's reference is the one drawn from the same uniform stream, and after
phi rather than before, because phi is nonlinear and the curvature of an averaged
path is a path no rollout takes. That set saturates, so it is drawn to
ROLLOUT_RMAX and not to the study window's r = 24; README.md carries the numbers.

The figure carries no prose at all -- no title, no key, no selection note. How
the rollout paths were reduced, where the window stops, what R1/R2/R3 average,
and that the game is an argmax over 1550 and not a typical fork are the caption's
job and live in README.md's `## Figure notes` (FIGURE_GUIDELINES 5). Lifted into
a talk without that entry the panels read as a result, which is the cost that
rule names and accepts.

It reads the game's activations from game_acts.py's slice, not the full tensor.

Usage: python analysis/violation_divergence/fork_and_board/make_figures.py
       [--game 1235] [--plane displacement|r21|d7] [--embedding pca|isomap]
       [--variant spliced|rollout] [--l0-matched] [--camera-ready]
"""

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.figures.branch_panels import axis_labels, plane_displacement, rule_fork_panel
from spade.figures.branch_style import ordered_legend
from spade.figures.paper_style import apply_style, save_figure
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.curvature_plane import (
    PLANE_COLUMNS, W, fit_plane, windowed)
from spade.violation_divergence.fork_plane import (
    EMBEDDINGS, positions, space_features, spaces, tag)
from spade.violation_divergence.population import paired_mean
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game_acts  # noqa: E402  (sibling module, not an installed package)

FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../game_trajectory's grid
LEGEND_H = 1.3             # the strip under the grid: the bottom row's labels,
                           # then eight legend entries over two rows
# A 3-D cell's bbox runs well below its drawn cube, so a letter placed inside it
# lands beside the PC1 label. This drops it into the row gap instead, measured
# down from the cell's bottom edge; `hspace` below is what buys that gap.
SUBCAP_Y = 0.016
PLANES = ("displacement", "r21", "d7")
# Past r = +8 the rollout set's illegal and control δ̂ agree to within 0.5% and
# cross over by r = +13, so the extra offsets draw separation that is not there.
ROLLOUT_RMAX = 8

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


def figure(g: Game, Z: np.ndarray, pos: np.ndarray, plane: str, method: str,
           variant: str, K: int, device: str, l0_matched: bool,
           camera_ready: bool) -> Path:
    cells = spaces(l0_matched)
    ref = reference_row(g.rows)
    k = int(np.flatnonzero(pos == g.t0)[0])
    nrow = int(np.ceil(len(cells) / NCOL))

    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + LEGEND_H))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, LEGEND_H],
                          left=0.03, right=0.97, top=0.96, bottom=0.01, hspace=0.02)
    # hspace is the strip each row's label is centred in, so it is set by SUBCAP_Y
    # and not by how far apart the cubes should sit. Nothing is drawn above a cell
    # any more, so it buys the label and nothing else.
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.06, hspace=0.14)

    measured = []
    for j, (name, entry) in enumerate(cells):
        F = space_features(Z, entry, device)
        p = project(plane, method, F, ref, k, K)
        cell = grid[j // NCOL, j % NCOL]
        ax = fig.add_subplot(cell, projection="3d")
        rule_fork_panel(ax, p.coords, g.rows, pos[p.dropped:], g.t0,
                        legend=(j == 0), zlabel=(j % NCOL == 0))
        ax.set_xlabel(p.labels[0], labelpad=-2, fontsize=8)
        ax.set_ylabel(p.labels[1], labelpad=-2, fontsize=8)
        # The panel letter and the space it draws are one label centred under the
        # cell, formatted as ../divergence_by_space formats its own. The
        # architecture alone names the space: its hyperparameters -- the seed-42
        # defaults, or the L0 the matched roster seats it at -- are what the
        # filename and README.md carry, so the two grids read cell for cell.
        box = cell.get_position(fig)
        fig.text(0.5 * (box.x0 + box.x1), box.y0 - SUBCAP_Y,
                 f"({ascii_lowercase[j]}) {name.split(' (')[0]}",
                 ha="center", va="top", fontsize=8)
        measured.append((name.split(" (")[0],
                         control_ratio(p.coords, pos[p.dropped:], g.t0, g.rows),
                         control_ratio(p.space, pos[p.dropped:], g.t0, g.rows)))
        if j == 0:
            keyed = ax

    # The run ratios as text and not as a panel: ../divergence_by_space draws them
    # over a population and with an interval, and README.md quotes these.
    print("mean distance from the reference over the drawn r >= 0, "
          "/ the cell's legal control — plane | full space")
    for name, dp, ds in measured:
        print(f"  {name:<14} " + "  ".join(
            f"{rule.split()[0]} {dp[m].mean():5.2f}|{ds[m].mean():5.2f}"
            for rule, m in rule_rows(g.rows).items()))

    band = fig.add_subplot(gs[1])
    band.axis("off")
    ordered_legend(band, keyed, 4, loc="center", bbox_to_anchor=(0.5, 0.5))
    return save_figure(
        fig, FIG_DIR / f"game_{g.game}_fork_and_board_{plane}_{method}"
                       f"{tag(variant, l0_matched)}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--plane", default="displacement", choices=PLANES)
    parser.add_argument("--embedding", default="pca", choices=list(EMBEDDINGS))
    parser.add_argument("--variant", default="spliced", choices=list(game_acts.VARIANTS),
                        help="rollout: every branch samples its own continuation, "
                             "K rows per branch, averaged over K")
    parser.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                        help="the eight l0_operating_point picks instead of the seed-42 "
                             "roster: one per SAE type, all at the shared target L0")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    assert args.plane == "displacement" or args.embedding == "pca", \
        "a descriptor plane fits its own; --embedding is the displacement one"

    apply_style(args.camera_ready)
    device = select_device()
    variant = game_acts.VARIANTS[args.variant]
    base = variant_dir("divergence", variant=variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1

    assert args.game in complete_games(meta), \
        f"game {args.game} does not carry every branch type"
    g = load_game(meta, args.game)
    acts = game_acts.load(base, g)
    # One control, not two: control 2 is the null *for* control 1 and belongs
    # where control 1 is measured, not in an illustration of the construction.
    keep = [i for i, r in enumerate(g.rows) if r["is_control"] != 2]
    g = replace(g, idx=g.idx[keep], rows=[g.rows[i] for i in keep])

    # A descriptor window reads 14 plies back, so it needs prefix context the study
    # window does not carry; the extra offsets are computed and never drawn.
    pos = positions(g.t0, 0 if args.plane == "displacement" else W - 1)
    if variant:
        pos = pos[pos - g.t0 <= ROLLOUT_RMAX]
    Z = acts[keep][:, pos, :].float().numpy()
    # Z keeps all K rows of every branch, and each reduction below is over them.
    # g carries one row per branch from here on: the metadata is the branch's.
    g = replace(g, idx=g.idx[::K], rows=g.rows[::K])
    # The ratio is the figure's caption, not the figure's: README.md quotes it.
    field = spread_field(args.plane, Z, reference_row(g.rows), pos >= g.t0, K)
    print(f"within/between rule spread ratio, game {g.game}, {args.plane}, "
          f"{args.variant}: {spread_ratio(field, g.rows):.3f}")
    out = figure(g, Z, pos, args.plane, args.embedding, variant, K, device,
                 args.l0_matched, args.camera_ready)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
