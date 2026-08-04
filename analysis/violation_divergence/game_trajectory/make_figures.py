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

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.branch_panels import AZIM, ELEV, axis_labels, fork_panel
from spade.figures.branch_style import branch_style, ordered_legend
from spade.figures.paper_style import (
    GRID, MUTED, apply_style, rule_key, sae_label, save_figure)
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.branches import (
    Game, complete_games, load_game, reference_row)
from spade.violation_divergence.fork_plane import (
    EMBEDDINGS, STRIDE, positions, space_features)

FIG_DIR = Path(__file__).resolve().parent / "figures"
# Most mutually separated illegal categories among games at or above the median
# illegal-vs-control gap; picked by analysis/violation_divergence/game_selection.
DEFAULT_GAME = 1235


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
                marker=st["marker"], ms=3.5, markevery=3, markeredgecolor="white",
                markeredgewidth=0.3, zorder=st["zorder"])
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
    ax3.set_xlabel(xlab, labelpad=-8)
    ax3.set_ylabel(ylab, labelpad=-8)
    ax3.set_title("(a) displacement from the reference, by move", loc="left", fontsize=10)

    ax2 = fig.add_subplot(gs[1])
    marginal_panel(ax2, Z, g.rows, pos, g.t0, ref)
    ax2.set_title("(b) the magnitude the projection cannot show", loc="left", fontsize=10)

    ordered_legend(fig, ax3, 4,
                   f"{method.upper()} plane, a learned basis refit for this game; view "
                   f"elev={ELEV}, azim={AZIM}; every {STRIDE}nd ply. One game (n=1): an "
                   f"illustration of the design, not a claim — the study's numbers, with "
                   f"their CIs over 13,531 branches, are in the parent README.",
                   loc="lower center", bbox_to_anchor=(0.5, -0.19))
    if not camera_ready:
        fig.suptitle(f"Game {g.game}: one prefix, {n} futures, forking at $t_0$={g.t0} "
                     f"(layer 5, positions {pos[0]}–{pos[-1]})", fontsize=11)
    fig.tight_layout()
    rule_key(fig, ["R1", "R2", "R3"], y=-0.235)
    return save_figure(fig, FIG_DIR / f"game_{g.game}_fork_{method}", camera_ready)


def space_grid(g: Game, Z: np.ndarray, pos: np.ndarray, method: str, device: str,
               camera_ready: bool) -> Path:
    """One cell per representation space, one game. Planes are refit per space,
    so positions are not comparable across cells -- only the shape of the fork is."""
    spaces = [("raw layer 5 (reference space)", None)] + \
        [(sae_label(e.config.replace("Trainer-othello", "")), e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    ref = reference_row(g.rows)
    ncol = 3
    nrow = int(np.ceil(len(spaces) / ncol))
    fig = plt.figure(figsize=(3.7 * ncol, 3.5 * nrow))
    for j, (name, entry) in enumerate(spaces):
        F = space_features(Z, entry, device)
        n, T, D = F.shape
        coords, var, _ = EMBEDDINGS[method]((F - F[ref]).reshape(-1, D))
        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        fork_panel(ax, coords.reshape(n, T, 2), g.rows, pos, g.t0,
                   legend=(j == 0), zlabel=(j % ncol == 0))
        xlab, ylab = axis_labels(method, var)
        ax.set_xlabel(xlab, labelpad=-8, fontsize=8)
        ax.set_ylabel(ylab, labelpad=-8, fontsize=8)
        ax.set_title(name, loc="left", fontsize=8.5)
    ordered_legend(fig, fig.axes[0], 4,
                   f"game {g.game}, $t_0$={g.t0}; {method.upper()} plane refit per space "
                   f"(arb. units, not comparable across cells); view elev={ELEV}, azim={AZIM}. "
                   f"One game (n=1): shape only — the parent README carries the numbers.",
                   loc="lower center", bbox_to_anchor=(0.5, -0.06))
    if not camera_ready:
        fig.suptitle(f"The fork in every representation space, game {g.game} "
                     f"({method.upper()})", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.06, wspace=0.06, hspace=0.16)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.085)
    return save_figure(fig, FIG_DIR / f"space_fork_grid_{method}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=DEFAULT_GAME)
    parser.add_argument("--n_games", type=int, default=1)
    parser.add_argument("--embedding", default="pca", choices=["pca", "isomap"])
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)

    complete = complete_games(meta)
    assert args.game in complete, f"game {args.game} does not carry every branch type"

    for i, game in enumerate([args.game]
                             + [g for g in complete if g != args.game][:args.n_games - 1]):
        g = load_game(meta, game)
        pos = positions(g.t0)
        Z = acts[g.idx][:, pos, :].float().numpy()
        print(f"wrote {fork_figure(g, Z, pos, args.embedding, args.camera_ready)}")
        if i == 0:
            print(f"wrote {space_grid(g, Z, pos, args.embedding, device, args.camera_ready)}")


if __name__ == "__main__":
    main()
