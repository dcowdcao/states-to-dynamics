"""The three constraint rules of docs/othello_rules.md, drawn on one board.

Two panels off ONE position of one corpus game, so a reader compares a legal move and the
three ways a move can be illegal without the board changing under them:

    (a)  the move the game actually played, and the brackets that make it legal -- every
         disc it flips, and the mover's own disc that closes each line
    (b)  one illegal candidate per constraint rule at that same position

The panels sit in one figure because the comparison IS the figure: (b) is only readable as
"the same board, one square moved", and two files make the reader hold a board in memory
across them. Both keep the side to move in their own axes, because R3 is *legality is judged
for the mover*: its square would bracket perfectly well for the opponent, and a panel read on
its own is unreadable rather than merely terse without it.

R2 is drawn as `empty_no_bracket` rather than the shallower `empty_detached`. The taxonomy
orders R2's members by the board computation a violation takes to detect, and only the
deepest one is visible as a rule: there IS a contiguous line of opponent discs out of the
square, it simply is not closed by one of the mover's own. `empty_detached` touches no
opponent disc at all and would show the reader an empty square with nothing to see.

Usage: python -m spade.figures.figure_othello_rules [--camera_ready]
"""

import argparse
from collections import Counter
from pathlib import Path
from textwrap import fill

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from spade.common.paths import repo_root
from spade.figures.othello_board import draw_board, mark_square, square_xy
from spade.figures.paper_style import INK, MUTED, RULE_STYLE, apply_style, rule_key, save_figure
from spade.othello.boards import rays, replay
from spade.othello.games import collect_legal, stream_games
from spade.othello.taxonomy import CATEGORIES, LEGAL, classify

from circuits.othello_engine_utils import to_string

FIGURES = repo_root() / "figures" / "othello_rules"
SEED = 42
FIGURE_SIZE = (8.0, 4.4)
NOTE_WRAP = 62     # characters that fit under one panel at 7.5 pt
FLIP_RING = 0.36   # outside the disc, so the ring reads on a black one and a white one alike

TITLE = "Move legality at one position: a legal move, and one illegal move per rule"

# Scanned t in [14, 26] over the first 800 games of the seed-42 stream for a position where
# all three rules have a candidate, the played move flips several discs in more than one
# direction, and the drawn squares stay >= 3 cells apart. 606/16 was the pick; the three
# illegal squares are named rather than re-derived, and `position` asserts their categories
# so a corpus change fails loudly instead of drawing a wrong board.
FIGURE_GAME, FIGURE_T = 606, 16
ILLEGAL = {"R1 placement": 9, "R2 bracketing": 23, "R3 turn": 53}
EXPECTED = {9: "occupied_opp", 23: "empty_no_bracket", 53: "opponent_only"}


def name(square: int) -> str:
    return f"{'abcdefgh'[square % 8]}{square // 8 + 1}"


def position(dataset_name: str) -> tuple:
    """The one board both panels draw, re-derived from the corpus and checked."""
    game = collect_legal(stream_games(dataset_name, SEED), FIGURE_GAME + 1,
                         Counter())[FIGURE_GAME]
    board = replay(game[:FIGURE_T])
    played = to_string(game[FIGURE_T])
    assert classify(board, played) == LEGAL, f"{name(played)} is not the legal move"
    for square, category in EXPECTED.items():
        got = CATEGORIES[classify(board, square)]
        assert got == category, f"{name(square)} is {got}, not {category}"
    return board, int(board.next_hand_color), played


def ray_line(ax, square: int, through: int, color: str, ls="-") -> None:
    """A thin line from a candidate square out to the square that decides its bracket."""
    (x0, y0), (x1, y1) = square_xy(square), square_xy(through)
    ax.plot([x0, x1], [y0, y1], color=color, ls=ls, lw=1.3, zorder=2, solid_capstyle="round")


def mover_note(ax, mover: int) -> None:
    """Whose turn it is, in the axes rather than in a title: R3 is unreadable without it and
    a camera-ready figure drops its title (FIGURE_GUIDELINES 5). Rank 8 is empty here."""
    ax.text(0.03, 0.04, f"{'black' if mover > 0 else 'white'} to move", transform=ax.transAxes,
            fontsize=8.5, color=INK, ha="left", va="bottom", zorder=6)


def panel_note(ax, text: str) -> None:
    """What the overlay means, under its own panel and wrapped to the panel's width."""
    ax.text(0, -0.17, fill(text, NOTE_WRAP), transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=MUTED, linespacing=1.5)


def legal_panel(ax, board, mover: int, played: int) -> None:
    draw_board(ax, board.state.flatten())

    brackets = [r for r in rays(board.state, played, mover) if r.closed]
    flipped = [sq for r in brackets for sq in r.run]
    for ray in brackets:
        ray_line(ax, played, ray.terminator, MUTED)
    for square in flipped:
        x, y = square_xy(square)
        ax.add_patch(Circle((x, y), FLIP_RING, fill=False, ls=(0, (3, 2)), lw=1.2,
                            edgecolor=INK, zorder=5))
    # One marked square, so nothing to tell it apart from: the outline and the name carry it,
    # and a marker here would read as a fifth disc.
    mark_square(ax, played, RULE_STYLE["legal"], label=name(played), marker=False)

    mover_note(ax, mover)
    ax.set_title("(a) the legal move", loc="left")
    panel_note(ax, f"{name(played)} is legal: it brackets in {len(brackets)} directions. Lines "
                   f"run to the mover's own disc that closes each bracket; the {len(flipped)} "
                   f"ringed discs are the ones it flips.")


def illegal_panel(ax, board, mover: int) -> None:
    draw_board(ax, board.state.flatten())

    # R2: the one direction out of the square that holds opponent discs, drawn to the empty
    # square that breaks it. R3: the bracket that does close -- for the opponent.
    broken = next(r for r in rays(board.state, ILLEGAL["R2 bracketing"], mover)
                  if r.run and r.blocker is not None)
    ray_line(ax, ILLEGAL["R2 bracketing"], broken.blocker,
             RULE_STYLE["R2 bracketing"]["color"], ls=(0, (3, 2)))
    bx, by = square_xy(broken.blocker)
    ax.plot(bx, by, marker="x", ms=7, mew=2.0, color=RULE_STYLE["R2 bracketing"]["color"],
            zorder=6)

    theirs = next(r for r in rays(board.state, ILLEGAL["R3 turn"], -mover) if r.closed)
    ray_line(ax, ILLEGAL["R3 turn"], theirs.terminator, RULE_STYLE["R3 turn"]["color"])

    # No legend: each square is named on the board and the rule key spells the names out, so
    # a legend would be a third copy of one fact (FIGURE_GUIDELINES 5).
    for rule, square in ILLEGAL.items():
        mark_square(ax, square, RULE_STYLE[rule], label=rule.split()[0],
                    below=square == ILLEGAL["R3 turn"])

    mover_note(ax, mover)
    ax.set_title("(b) one illegal move per rule", loc="left")
    panel_note(ax, f"R1's square is already occupied. R2's dashed line is the opponent run out "
                   f"of {name(ILLEGAL['R2 bracketing'])}; the x is the empty square where it "
                   f"fails to close. R3's line is the bracket "
                   f"{name(ILLEGAL['R3 turn'])} does make -- for the opponent, who is not to "
                   f"move.")


def rules_figure(board, mover: int, played: int, camera_ready: bool) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    legal_panel(left, board, mover, played)
    illegal_panel(right, board, mover)
    if not camera_ready:
        fig.suptitle(TITLE)
    fig.tight_layout()
    # Under the whole figure, below both panel notes: the key covers both panels, and the
    # notes hang off the axes, so its offset clears four wrapped lines of them.
    rule_key(fig, ["R1", "R2", "R3"], x=left.get_position().x0, y=-0.125)
    return save_figure(fig, FIGURES / "rules_legal_and_illegal", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    board, mover, played = position(args.dataset_name)
    print(rules_figure(board, mover, played, args.camera_ready))


if __name__ == "__main__":
    main()
