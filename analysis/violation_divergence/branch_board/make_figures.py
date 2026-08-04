"""The board the game forks on, with all nine branch moves drawn on it.

The fork figure (analysis/violation_divergence/game_trajectory) draws nine
trajectories in a projected space and never says what the nine branches *are*.
This is the missing half: the shared prefix's last position, with every branch
move marked on its own square in the fork figure's colours, markers and labels,
so the two read side by side.

Superseded by analysis/violation_divergence/fork_and_board, which draws this
board as panel (b) under the same nine branches in every representation space.

One game (n=1): an illustration of the design, not a claim.

Usage: python analysis/violation_divergence/branch_board/make_figures.py
       [--game 1235] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from spade.common.study import study_dir
from spade.figures.branch_panels import board_note, board_panel
from spade.figures.branch_style import ordered_legend
from spade.figures.paper_style import MUTED, apply_style, rule_key, save_figure
from spade.othello.boards import flip_set, replay
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import Game, load_game, reference_row

FIG_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_GAME = 1235  # the game the fork figure draws by default


def figure(g: Game, camera_ready: bool) -> Path:
    board = replay(g.tokens[:g.t0])
    reference = g.rows[reference_row(g.rows)]
    flipped = flip_set(board.state, reference["square"], board.next_hand_color)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    board_panel(ax, board, g.rows, flipped)
    if not camera_ready:
        ax.set_title(f"Game {g.game}: the position at $t_0$={g.t0} and the "
                     f"{len(g.rows)} moves\nthe branches play there", fontsize=10)
    fig.text(0.5, -0.02, board_note(board), ha="center", va="top", fontsize=8,
             color=MUTED)
    n_legal = g.rows[0]["n_legal"]
    ordered_legend(fig, ax, 2,
                   f"one move per branch, styled as in the fork figure.\n"
                   f"{n_legal} moves are legal at $t_0$; the played move and two of the "
                   f"other {n_legal - 1} are branched.\nOne game (n=1): an illustration "
                   f"of the design, not a claim.",
                   loc="lower center", bbox_to_anchor=(0.5, -0.40))
    rule_key(fig, ["R1", "R2", "R3"], y=-0.43)
    return save_figure(fig, FIG_DIR / f"game_{g.game}_branch_board", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=DEFAULT_GAME)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    meta = torch.load(study_dir("divergence") / "branches" / "branches.pt",
                      weights_only=True)
    g = load_game(meta, args.game)
    assert len(g.rows) == len(CATEGORIES) + 2, \
        f"game {args.game} carries {len(g.rows)} branches, not every branch type"
    print(f"wrote {figure(g, args.camera_ready)}")


if __name__ == "__main__":
    main()
