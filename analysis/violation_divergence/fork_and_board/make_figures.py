"""What the nine branches are, and where they go in every representation space.

Two halves of one question, so one figure. (b) is the board the game forks on,
with each branch move on the square it plays. (a) is where those same nine
branches travel: one 3-D fork per representation space, raw layer 5 first so the
eye calibrates on it. Both halves draw from spade.figures.branch_style, so a
colour and a marker mean the same branch in either half and one legend serves
both; the free space beside the square board carries that legend and the rule
key rather than sitting empty.

Planes are refit per space, so cells are comparable in the *shape* of the fork
and never in position. One game (n=1): an illustration of the design, not a
claim; the study's numbers are in the parent README.

Usage: python analysis/violation_divergence/fork_and_board/make_figures.py
       [--game 1235] [--embedding pca|isomap] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.branch_panels import (
    AZIM, ELEV, axis_labels, board_note, board_panel, fork_panel)
from spade.figures.branch_style import ordered_legend
from spade.figures.paper_style import (
    MUTED, apply_style, rule_key, sae_label, save_figure)
from spade.othello.boards import flip_set, replay
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row, selection_note)
from spade.violation_divergence.fork_plane import (
    EMBEDDINGS, STRIDE, positions, space_features)

FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../game_trajectory's grid
BOARD_H = 5.4              # the board row, tall enough for a square board


def spaces() -> list[tuple[str, object]]:
    """Raw layer 5 leads, then one SAE family per cell (FIGURE_GUIDELINES 4.6)."""
    return [("raw layer 5 (reference space)", None)] + \
        [(sae_label(e.config.replace("Trainer-othello", "")), e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]


def legend_title(g: Game, method: str) -> str:
    n_legal = g.rows[0]["n_legal"]
    return (f"the same nine branches in both panels, game {g.game}, $t_0$={g.t0}\n"
            f"(a) {method.upper()} plane refit per space — arb. units, not comparable "
            f"across cells;\n"
            f"      view elev={ELEV}, azim={AZIM}; every {STRIDE}nd ply\n"
            f"(b) {n_legal} moves are legal at $t_0$; the played move and two of the "
            f"other {n_legal - 1} are branched\n"
            f"one game (n=1): an illustration of the design, not a claim — the study's\n"
            f"numbers, with their CIs over 13,531 branches, are in the parent README\n"
            f"{selection_note(g.game)}")


def figure(g: Game, Z: np.ndarray, pos: np.ndarray, method: str, device: str,
           camera_ready: bool) -> Path:
    cells = spaces()
    ref = reference_row(g.rows)
    nrow = int(np.ceil(len(cells) / NCOL))

    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + BOARD_H))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BOARD_H],
                          left=0.03, right=0.97, top=0.93, bottom=0.02, hspace=0.05)
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.06, hspace=0.16)
    lower = gs[1].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.02)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("(a) where the nine branches go: the fork in each representation space",
                   loc="left", fontsize=11, pad=18)

    for j, (name, entry) in enumerate(cells):
        F = space_features(Z, entry, device)
        n, T, D = F.shape
        coords, var, _ = EMBEDDINGS[method]((F - F[ref]).reshape(-1, D))
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL], projection="3d")
        fork_panel(ax, coords.reshape(n, T, 2), g.rows, pos, g.t0,
                   legend=(j == 0), zlabel=(j % NCOL == 0))
        xlab, ylab = axis_labels(method, var)
        ax.set_xlabel(xlab, labelpad=-8, fontsize=8)
        ax.set_ylabel(ylab, labelpad=-8, fontsize=8)
        ax.set_title(name, loc="left", fontsize=8.5)
        if j == 0:
            keyed = ax

    board = replay(g.tokens[:g.t0])
    flipped = flip_set(board.state, g.rows[ref]["square"], board.next_hand_color)
    ax_board = fig.add_subplot(lower[0])
    board_panel(ax_board, board, g.rows, flipped)
    ax_board.set_title("(b) what they are: the position at $t_0$,\nand the move each "
                       "branch plays there", loc="left", fontsize=11)

    band = fig.add_subplot(lower[1])
    band.axis("off")
    ordered_legend(band, keyed, 2, legend_title(g, method),
                   loc="upper left", bbox_to_anchor=(0.0, 1.0))
    band.text(0.0, 0.45, board_note(board), ha="left", va="top", fontsize=8,
              color=MUTED, linespacing=1.5)
    rule_key(band, ["R1", "R2", "R3"], x=0.0, y=0.18)

    if not camera_ready:
        fig.suptitle(f"Game {g.game}: nine futures of one position, in nine "
                     f"representation spaces ({method.upper()})", fontsize=12)
    return save_figure(fig, FIG_DIR / f"game_{g.game}_fork_and_board_{method}",
                       camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--embedding", default="pca", choices=["pca", "isomap"])
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)

    assert args.game in complete_games(meta), \
        f"game {args.game} does not carry every branch type"
    g = load_game(meta, args.game)
    pos = positions(g.t0)
    Z = acts[g.idx][:, pos, :].float().numpy()
    print(f"wrote {figure(g, Z, pos, args.embedding, device, args.camera_ready)}")


if __name__ == "__main__":
    main()
