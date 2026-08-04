"""One game's layer-5 trajectory, forking at t0 into every branch.

The aggregate figures answer "how far apart, on average". This one shows the
object those averages are taken over: a single game as a path, shared up to t0
because the branches are token-identical there, then one path per branch.

Which positions are sampled and what plane they are embedded in belong to
spade.violation_divergence.fork_plane; move index rides on the third axis, so
the plane embeds the displacement D(t) = z_branch(t) - z_ref(t) rather than raw
activations. Both a linear and a nonlinear plane ship, because the branches need
not separate linearly.

Usage: python analysis/violation_divergence/game_trajectory/make_figures.py
       [--game 1235] [--n_games 1] [--embedding pca|isomap] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.study import study_dir
from spade.figures.branch_panels import AZIM, ELEV, axis_labels, fork_panel
from spade.figures.branch_style import branch_style, ordered_legend
from spade.figures.paper_style import (
    GRID, MUTED, apply_style, rule_key, save_figure)
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row, selection_note)
from spade.violation_divergence.fork_plane import EMBEDDINGS, STRIDE, positions

FIG_DIR = Path(__file__).resolve().parent / "figures"


def delta_hat(Z: np.ndarray, ref: int) -> np.ndarray:
    d = np.linalg.norm(Z - Z[ref], axis=-1)
    return d / np.linalg.norm(Z[ref], axis=-1)


def marginal_panel(ax, Z: np.ndarray, rows: list[dict], pos: np.ndarray,
                   t0: int, ref: int) -> None:
    """The magnitude a projection cannot show: relative displacement vs offset."""
    dh, r = delta_hat(Z, ref), pos - t0
    ax.axvspan(r[0] - 0.5, -0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        ax.plot(r, dh[i], ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.5, markevery=3, markerfacecolor=st["mfc"],
                markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"])
    ax.text(r[0] + 0.4, ax.get_ylim()[1] * 0.93, "0 by construction\n(shared prefix)",
            fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("position relative to the branch, $t-t_0$")
    ax.set_ylabel(r"$\hat{\delta}=\|\Delta\|\,/\,\|z_{\mathrm{ref}}\|$")


def fork_figure(g: Game, Z: np.ndarray, pos: np.ndarray, method: str,
                camera_ready: bool) -> Path:
    ref = reference_row(g.rows)
    n, T, D = Z.shape
    coords, var, _ = EMBEDDINGS[method]((Z - Z[ref]).reshape(-1, D))
    xlab, ylab = axis_labels(method, var)

    fig = plt.figure(figsize=(12.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.24)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax3.set_box_aspect((1, 1, 0.85))
    fork_panel(ax3, coords.reshape(n, T, 2), g.rows, pos, g.t0, legend=True)
    ax3.set_xlabel(xlab, labelpad=-2)
    ax3.set_ylabel(ylab, labelpad=-2)
    ax3.set_title("(a) displacement from the reference, by move", loc="left", fontsize=10)

    ax2 = fig.add_subplot(gs[1])
    marginal_panel(ax2, Z, g.rows, pos, g.t0, ref)
    ax2.set_title("(b) the magnitude the projection cannot show", loc="left", fontsize=10)

    ordered_legend(fig, ax3, 4,
                   f"{method.upper()} plane, a learned basis refit for this game; view "
                   f"elev={ELEV}, azim={AZIM}; every {STRIDE}nd ply. One game (n=1): an "
                   f"illustration of the design, not a claim — the study's numbers, with "
                   f"their CIs over 13,531 branches, are in the parent README.\n"
                   f"{selection_note(g.game)}",
                   loc="lower center", bbox_to_anchor=(0.5, -0.28))
    if not camera_ready:
        fig.suptitle(f"Game {g.game}: one prefix, {n} futures, forking at $t_0$={g.t0} "
                     f"(layer 5, positions {pos[0]}–{pos[-1]})", fontsize=11)
    fig.tight_layout()
    rule_key(fig, ["R1", "R2", "R3"], y=-0.325)
    return save_figure(fig, FIG_DIR / f"game_{g.game}_fork_{method}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--n_games", type=int, default=1)
    parser.add_argument("--embedding", default="pca", choices=["pca", "isomap"])
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)

    complete = complete_games(meta)
    assert args.game in complete, f"game {args.game} does not carry every branch type"

    for game in [args.game] + [g for g in complete if g != args.game][:args.n_games - 1]:
        g = load_game(meta, game)
        pos = positions(g.t0)
        Z = acts[g.idx][:, pos, :].float().numpy()
        print(f"wrote {fork_figure(g, Z, pos, args.embedding, args.camera_ready)}")


if __name__ == "__main__":
    main()
