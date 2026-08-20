"""One game's layer-5 trajectory, forking at t0 into every branch.

The aggregate figures answer "how far apart, on average". This one shows the
object those averages are taken over: a single game as a path, shared up to t0
because the branches are token-identical there, then one path per branch.

Which positions are sampled and what plane they are embedded in belong to
spade.violation_divergence.fork_plane; move index rides on the third axis, so
the plane embeds the displacement D(t) = z_branch(t) - z_ref(t) rather than raw
activations. Both a linear and a nonlinear plane ship, because the branches need
not separate linearly.

(c) is (a) as a number and (b) as a number, on one axis: each branch's mean
distance from the reference over r >= 0, in the plane (a) draws and in the full
512-d space (b) measures, both divided by the legal control's. The plane is
PC1+PC2 = 37% of the displacement's variance and holds a different share of each
branch -- 30% of the control's length here, 81% of opponent_only's -- so the two
marks on a row are not one quantity twice, and the pair is what stops the
projection being read as a magnitude.

`--variant rollout` draws the same nine branches out of the rollout branch set,
where every branch -- the reference included -- samples its own continuation from
OthelloGPT rather than replaying the source game's. A branch is K rows there, and
the two panels reduce over K differently on purpose: (a) averages the
displacement, because the plane projection is linear, and (b) averages delta-hat
itself, because the norm is not -- the norm of a mean displacement cancels
sampling noise and is a smaller, different quantity. That set saturates, so it is
drawn to ROLLOUT_RMAX and not to the study window's r = 24; README.md has the
numbers.

The figure carries no prose. Its caption -- the fixed camera, how the rollout
variant reduces over K, where its window stops, and that game 1235 is selected
rather than typical -- lives in README.md (FIGURE_GUIDELINES 2, 5). What stays
inside is (c)'s estimator on (c)'s own axis.

Usage: python analysis/violation_divergence/game_trajectory/make_figures.py
       [--game 1235] [--n_games 1] [--embedding pca|isomap]
       [--variant spliced|rollout] [--camera-ready]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from spade.common.study import variant_dir
from spade.figures.branch_panels import axis_labels, fork_panel, plane_displacement
from spade.figures.branch_style import (
    branch_style, legend_order, ordered_legend)
from spade.figures.paper_style import GRID, INK, MUTED, NULL, apply_style, save_figure
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.fork_plane import EMBEDDINGS, positions

FIG_DIR = Path(__file__).resolve().parent / "figures"
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


def delta_hat(Z: np.ndarray, ref: int, K: int) -> np.ndarray:
    """Mean over each branch's K rollouts of δ̂ = ‖Δ‖ / ‖z_ref‖.

    Averaged after the norm, not before: ‖mean_k Δ_k‖ cancels the sampling noise
    the K rollouts differ by and measures a smaller, different quantity. K = 1 is
    the spliced set, where this is the branch's own δ̂."""
    Y = Z.reshape(-1, K, *Z.shape[1:])
    d = np.linalg.norm(Y - Y[ref], axis=-1) / np.linalg.norm(Y[ref], axis=-1)
    return d.mean(axis=1)


def metric_panel(ax, coords: np.ndarray, dh: np.ndarray, rows: list[dict],
                 pos: np.ndarray, t0: int, ref: int) -> None:
    """(c): (a) as a number beside (b) as a number, on one axis.

    Bar = the branch's mean distance from the reference in the plane (a) draws;
    open marker = the mean of (b)'s δ̂ over the same offsets. Dividing both by
    the legal control's puts a plane distance in arb. units and a full-space
    ratio on one scale, and the gap between a bar and its marker is the part of
    the displacement the plane does not hold -- a branch that leaves the plane
    reads on (a) as a branch that stayed near the reference.
    """
    ctrl = next(i for i, r in enumerate(rows) if r["is_control"] == 1)
    plane = plane_displacement(coords, pos, t0, ref)
    space = dh[:, pos >= t0].mean(axis=1)
    plane, space = plane / plane[ctrl], space / space[ctrl]

    shown = [i for i in range(len(rows)) if i not in (ref, ctrl)]
    styles = [branch_style(rows[i]["category"], rows[i]["is_control"]) for i in shown]
    order = legend_order([st["label"] for st in styles])  # nulls last, as in the legend
    # The control line stops with the bars: it is a scale for them, and run to
    # the axes floor it strikes through the legend parked there.
    ax.plot([1, 1], [-0.75, len(shown) - 0.25], color=NULL["color"], ls=NULL["ls"],
            lw=1.1, zorder=1)
    # Branch names ride above their own bar rather than on the y axis: as tick
    # labels they reach left out of this narrow panel and into (b).
    for k, o in enumerate(order):
        i, st = shown[o], styles[o]
        ax.barh(k, plane[i], 0.64, color=st["color"], alpha=0.85, edgecolor=INK,
                lw=0.4, zorder=2)
        ax.plot(space[i], k, lw=0, marker=st["marker"], ms=7, markerfacecolor="white",
                markeredgecolor=st["color"], markeredgewidth=1.3, zorder=4)
        ax.text(0.012, k - 0.40, styles[o]["label"].split(":")[0], fontsize=9,
                ha="left", va="bottom", color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_ylim(len(order) + 1.6, -0.9)
    ax.set_xlim(0, max(plane.max(), space.max()) * 1.06)
    r = pos[pos >= t0] - t0
    ax.set_xlabel(f"distance from the reference, $\\div$ the legal control's\n"
                  f"mean over the {len(r)} drawn offsets r = {r[0]}, {r[1]}, …, {r[-1]}; "
                  f"one game (n = 1)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(False, axis="y")
    marks = [(Patch(facecolor=MUTED, alpha=0.85, edgecolor=INK, lw=0.4),
              "in (a)'s plane (2-D)"),
             (plt.Line2D([], [], lw=0, marker="o", ms=7, markerfacecolor="white",
                         markeredgecolor=INK, markeredgewidth=1.3),
              "in the full 512-d space — the mean of (b)'s $\\hat{\\delta}$"),
             (plt.Line2D([], [], color=NULL["color"], ls=NULL["ls"], lw=1.1),
              "the legal control (null) = 1; the reference = 0")]
    ax.legend(*zip(*marks), fontsize=9, loc="lower left",
              bbox_to_anchor=(0.0, -0.02), borderaxespad=0)
    # (c) as text, so the README can quote it without reading it off the bars.
    print("(c) distance from the reference over the drawn r >= 0, / the legal "
          "control's — plane | full space")
    for o in order:
        print(f"  {styles[o]['label'].split(':')[0]:<20} "
              f"{plane[shown[o]]:5.2f}|{space[shown[o]]:5.2f}")


def marginal_panel(ax, dh: np.ndarray, rows: list[dict], pos: np.ndarray,
                   t0: int) -> None:
    """The magnitude a projection cannot show: relative displacement vs offset."""
    r = pos - t0
    ax.axvspan(r[0] - 0.5, -0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        ax.plot(r, dh[i], ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.5, markevery=3, markerfacecolor=st["mfc"],
                markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"])
    ax.set_xlabel("position relative to the branch, $t-t_0$")
    ax.set_ylabel(r"$\hat{\delta}=\|\Delta\|\,/\,\|z_{\mathrm{ref}}\|$")


def fork_figure(g: Game, Z: np.ndarray, pos: np.ndarray, method: str, variant: str,
                K: int, camera_ready: bool) -> Path:
    ref = reference_row(g.rows)
    D = paired_mean(Z, ref, K)
    n, T, dim = D.shape
    coords, var, _ = EMBEDDINGS[method](D.reshape(-1, dim))
    coords = coords.reshape(n, T, 2)
    dh = delta_hat(Z, ref, K)
    xlab, ylab = axis_labels(method, var)

    # (c) is a third column rather than a second row: the canvas grows by its
    # width, so (a) and (b) keep the size and aspect they had without it.
    fig = plt.figure(figsize=(12.6 * (2.35 + 0.72) / 2.35, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.72], wspace=0.24)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax3.set_box_aspect((1, 1, 0.85))
    fork_panel(ax3, coords, g.rows, pos, g.t0, legend=True)
    ax3.set_xlabel(xlab, labelpad=-2)
    ax3.set_ylabel(ylab, labelpad=-2)
    ax3.set_title("(a) displacement from the reference, by move", loc="left", fontsize=10)

    ax2 = fig.add_subplot(gs[1])
    marginal_panel(ax2, dh, g.rows, pos, g.t0)
    ax2.set_title("(b) the magnitude the projection cannot show", loc="left", fontsize=10)

    axc = fig.add_subplot(gs[2])
    metric_panel(axc, coords, dh, g.rows, pos, g.t0, ref)
    axc.set_title("(c) how far each branch runs,\nin the plane and in the space",
                  loc="left", fontsize=10)

    ordered_legend(fig, ax3, 4, loc="lower center", bbox_to_anchor=(0.5, -0.10))
    if not camera_ready:
        fig.suptitle(f"Game {g.game}: one prefix, {n} futures, forking at $t_0$={g.t0} "
                     f"(layer 5, positions {pos[0]}–{pos[-1]})"
                     + (f", {variant} branches" if variant else ""), fontsize=11)
    fig.tight_layout()
    suffix = f"_{variant}" if variant else ""
    return save_figure(fig, FIG_DIR / f"game_{g.game}_fork_{method}{suffix}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--n_games", type=int, default=1)
    parser.add_argument("--embedding", default="pca", choices=["pca", "isomap"])
    parser.add_argument("--variant", default="spliced", choices=list(VARIANTS),
                        help="rollout: every branch samples its own continuation, "
                             "K rows per branch, averaged over K")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    variant = VARIANTS[args.variant]
    base = variant_dir("divergence", variant=variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # mmap: the rollout tensor is 4.2 GB and nine branches of one game are read
    # out of it. The spliced tensor reads the same values either way.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    K = int(meta["rollout"].max()) + 1 if "rollout" in meta else 1

    complete = complete_games(meta)
    assert args.game in complete, f"game {args.game} does not carry every branch type"

    for game in [args.game] + [g for g in complete if g != args.game][:args.n_games - 1]:
        g = load_game(meta, game)
        pos = positions(g.t0)
        if variant:
            pos = pos[pos - g.t0 <= ROLLOUT_RMAX]
        Z = acts[g.idx][:, pos, :].float().numpy()
        # Z keeps all K rows of every branch, and both panels reduce over them.
        # g carries one row per branch from here on: the metadata is the branch's.
        g = replace(g, idx=g.idx[::K], rows=g.rows[::K])
        print(f"wrote {fork_figure(g, Z, pos, args.embedding, variant, K, args.camera_ready)}")


if __name__ == "__main__":
    main()
