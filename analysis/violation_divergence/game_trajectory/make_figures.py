"""One game's layer-5 trajectory, forking at t0 into every branch.

The aggregate figures answer "how far apart, on average". This one shows the
object those averages are taken over: a single game as a path, shared up to t0
because the branches are token-identical there, then one path per branch.

Which positions are sampled and what plane they are embedded in belong to
spade.violation_divergence.fork_plane; move index rides on the third axis, so
the plane embeds the displacement D(t) = z_branch(t) - z_ref(t) rather than raw
activations. Both a linear and a nonlinear plane ship, because the branches need
not separate linearly.

`--variant rollout` draws the same nine branches out of the rollout branch set,
where every branch -- the reference included -- samples its own continuation from
OthelloGPT rather than replaying the source game's. A branch is K rows there, and
the two panels reduce over K differently on purpose: (a) averages the
displacement, because the plane projection is linear, and (b) averages delta-hat
itself, because the norm is not -- the norm of a mean displacement cancels
sampling noise and is a smaller, different quantity. That set saturates, so it is
drawn to ROLLOUT_RMAX and not to the study window's r = 24; README.md has the
numbers.

The figure carries no prose block: how the game was selected, n=1, and the fixed
camera are its caption and live in README.md (FIGURE_GUIDELINES 2, 5). What the
rollout variant averaged, and where its window stops, are in the figure, because
they change what every curve on it means.

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

from spade.common.study import variant_dir
from spade.figures.branch_panels import axis_labels, fork_panel
from spade.figures.branch_style import (
    below, branch_style, ordered_legend, rule_member_key)
from spade.figures.paper_style import GRID, MUTED, apply_style, save_figure
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


def variant_note(K: int) -> str:
    """The rollout set's own caveat, in the panel it qualifies. Wrapped narrow
    enough to sit inside the pre-branch band, where δ̂ is 0 and nothing else is."""
    return (f"rollout branches: every\nbranch, the reference too,\nsamples its own\n"
            f"continuation, K = {K} times.\nEach path is a mean over\nthose K. "
            f"(a) averages Δ;\n(b) averages δ̂ itself —\nthe norm is not linear.\n"
            f"No spread is drawn over\nthose K.\n"
            f"Drawn to r = +{ROLLOUT_RMAX}: past it\nillegal "
            f"and control δ̂\nagree within 0.5%.")


def marginal_panel(ax, Z: np.ndarray, rows: list[dict], pos: np.ndarray,
                   t0: int, ref: int, K: int, variant: str) -> None:
    """The magnitude a projection cannot show: relative displacement vs offset."""
    dh = delta_hat(Z, ref, K)
    r = pos - t0
    ax.axvspan(r[0] - 0.5, -0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        ax.plot(r, dh[i], ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.5, markevery=3, markerfacecolor=st["mfc"],
                markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"])
    top = ax.get_ylim()[1]
    ax.text(r[0] + 0.4, top * 0.93, "0 by construction\n(shared prefix)",
            fontsize=7, color=MUTED, va="top")
    if variant:
        ax.text(r[0] + 0.4, top * 0.78, variant_note(K),
                fontsize=7, color=MUTED, va="top", linespacing=1.5)
    ax.set_xlabel("position relative to the branch, $t-t_0$")
    ax.set_ylabel(r"$\hat{\delta}=\|\Delta\|\,/\,\|z_{\mathrm{ref}}\|$")


def fork_figure(g: Game, Z: np.ndarray, pos: np.ndarray, method: str, variant: str,
                K: int, camera_ready: bool) -> Path:
    ref = reference_row(g.rows)
    D = paired_mean(Z, ref, K)
    n, T, dim = D.shape
    coords, var, _ = EMBEDDINGS[method](D.reshape(-1, dim))
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
    marginal_panel(ax2, Z, g.rows, pos, g.t0, ref, K, variant)
    ax2.set_title("(b) the magnitude the projection cannot show", loc="left", fontsize=10)

    leg = ordered_legend(fig, ax3, 4, loc="lower center", bbox_to_anchor=(0.5, -0.10))
    if not camera_ready:
        fig.suptitle(f"Game {g.game}: one prefix, {n} futures, forking at $t_0$={g.t0} "
                     f"(layer 5, positions {pos[0]}–{pos[-1]})"
                     + (f", {variant} branches" if variant else ""), fontsize=11)
    fig.tight_layout()
    rule_member_key(fig, y=below(fig, leg))
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
