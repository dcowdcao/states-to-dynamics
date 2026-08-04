"""The board the game forks on, with all nine branch moves drawn on it.

The fork figure (analysis/violation_divergence/game_trajectory) draws nine
trajectories in a projected space and never says what the nine branches *are*.
This is the missing half: the shared prefix's last position, with every branch
move marked on its own square in the fork figure's colours, markers and labels,
so the two read side by side.

Markers ride in the top-left corner of their square rather than at its centre.
Two of the six illegal categories -- occupied_own and occupied_opp -- play onto
a square that already holds a disc, and a centred marker would sit on top of it;
one placement rule for all nine keeps the grammar uniform and leaves every disc
fully visible. The outlined cell, not the marker's position, says which square a
branch plays, and it carries the branch's line style, so the two legal controls
are dashed exactly as they are in the fork figure.

One game (n=1): an illustration of the design, not a claim.

Usage: python analysis/violation_divergence/branch_board/make_figures.py
       [--game 0] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from spade.common.study import study_dir
from spade.figures.branch_style import branch_style, ordered_legend
from spade.figures.othello_board import draw_board, square_xy
from spade.figures.paper_style import (
    INK, MUTED, apply_style, rule_key, save_figure)
from spade.othello.boards import flip_set, replay
from spade.othello.taxonomy import CATEGORIES, LEGAL

from circuits.othello_engine_utils import to_string

FIG_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_GAME = 0        # the game the fork figure draws by default
CORNER = 0.30           # marker offset from the cell centre, which may hold a disc
FLIP_RING = 0.38        # ring radius around a flipped disc, in cells


def branches(meta: dict, game: int) -> tuple[list[dict], int]:
    """Every branch row of one game, plus t0."""
    idx = (meta["game_id"] == game).nonzero().flatten().tolist()
    assert idx, f"game {game} has no branches"
    rows = [{"category": int(meta["category"][i]),
             "is_control": int(meta["is_control"][i]),
             "square": to_string(int(meta["branch_token"][i])),
             "n_legal": int(meta["n_legal"][i])} for i in idx]
    return rows, int(meta["t0"][idx[0]])


def board_panel(ax, board, rows: list[dict], flipped: list[int]) -> None:
    draw_board(ax, board.state.flatten())
    for square in flipped:
        x, y = square_xy(square)
        ax.add_patch(plt.Circle((x, y), FLIP_RING, fill=False, lw=1.1, zorder=5,
                                ls=(0, (3, 2)), edgecolor=INK))
    for r in rows:
        st = branch_style(r["category"], r["is_control"])
        x, y = square_xy(r["square"])
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, zorder=4,
                                   lw=st["lw"], ls=st["ls"], edgecolor=st["color"]))
        ax.plot(x - CORNER, y - CORNER, marker=st["marker"], ms=8, lw=0,
                color=st["color"], markeredgecolor="white", markeredgewidth=0.8,
                zorder=6, label=st["label"])
    mover = "black" if board.next_hand_color > 0 else "white"
    ax.set_xlabel(
        f"discs in absolute colour, {mover} to move at $t_0$\n"
        "outlined cell = a branch square, its marker in the corner so a disc\n"
        "underneath stays visible; dashed ring = a disc the played move flips",
        fontsize=8, color=MUTED)


def figure(game: int, rows: list[dict], t0: int, tokens: list[int],
           camera_ready: bool) -> Path:
    board = replay(tokens[:t0])
    reference = next(r for r in rows if r["category"] == LEGAL and not r["is_control"])
    flipped = flip_set(board.state, reference["square"], board.next_hand_color)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    board_panel(ax, board, rows, flipped)
    if not camera_ready:
        ax.set_title(f"Game {game}: the position at $t_0$={t0} and the {len(rows)} "
                     "moves\nthe branches play there", fontsize=10)
    n_legal = rows[0]["n_legal"]
    ordered_legend(fig, ax, 2, -0.30,
                   f"one move per branch, styled as in the fork figure.\n"
                   f"{n_legal} moves are legal at $t_0$; the played move and two of the "
                   f"other {n_legal - 1} are branched.\nOne game (n=1): an illustration "
                   f"of the design, not a claim.")
    rule_key(fig, ["R1", "R2", "R3"], y=-0.33)
    return save_figure(fig, FIG_DIR / f"game_{game}_branch_board", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=DEFAULT_GAME)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    meta = torch.load(study_dir("divergence") / "branches" / "branches.pt",
                      weights_only=True)
    rows, t0 = branches(meta, args.game)
    assert len(rows) == len(CATEGORIES) + 2, \
        f"game {args.game} carries {len(rows)} branches, not every branch type"
    tokens = meta["tokens"][(meta["game_id"] == args.game).nonzero()[0, 0]].long().tolist()
    print(f"wrote {figure(args.game, rows, t0, tokens, args.camera_ready)}")


if __name__ == "__main__":
    main()
