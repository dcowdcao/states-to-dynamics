"""What the detector's event IS: five plies of one game, with the fork drawn on the board.

The trace and the AUROC panels of this study assert that a step-length drop peaks at $t_0$.
Neither says what $t_0$ is, and a reader cannot check a localization claim against a number.
This figure is the position itself -- the two plies before the fork, the fork, and the two
after -- so the peak has a board, a side to move and a square name attached to it. Nothing
here is a measurement; it is the event the other three figures score.

The corpus is the spliced branch tree: a branch differs from the reference in exactly one
token, at $t_0$, and replays the source game verbatim after it. So the five boards are one
game -- the reference row, the game as actually played -- and every move marked outside $t_0$
is the legal move that game played there, drawn with the brackets that make it legal. The
$t_0$ panel marks the three illegal moves the tree forks into instead, one per constraint
rule, and is the only panel with more than one mark; it carries a full frame so the event is
findable at a glance rather than by reading the panel labels.

Every board is drawn the way `figure_othello_rules` draws its one board: the candidate disc
centred on each marked square, a tick on the legal move and a cross on each illegal one, the
rule code alone beside the square, and the panel's label under the panel rather than over it.
The one departure is the side to move, which stays on the x label -- these panels are a fifth
as wide and their rank-7 and rank-8 cells hold discs, so the in-axes note that figure puts on
its empty rank 8 has nowhere to sit.

Sizing: five 8x8 boards need ~1.9 in each before the discs, the algebraic ticks and a
square's name stop being legible, which puts the figure at 12 in -- past NeurIPS full width
(5.5 in), and FIGURE_GUIDELINES 1.3 forbids buying that back by shrinking in LaTeX. This is a
README/appendix figure at its drawn size; the trace figure is the one that fits a column.

R2's branch at this $t_0$ is `empty_detached`, the shallowest of R2's three members: a2
touches no disc at all, so there is no line that fails to close and nothing to draw along --
the square is marked and named and that is all there is to see. R1's square is occupied by the
mover's own disc and R3's line brackets for the opponent, so both keep figure_othello_rules'
geometry.

Under each panel is what the detector of this study says about that ply, and whether it is
right. Off $t_0$ the row scored is the reference game itself, whose move there is the legal
one it played; at $t_0$ it is the three branch rows, one per rule, all three illegal. On this
game the detector is right at $t_0$ and at $t_0-2$ and false-alarms at the other three, which
is the same thing README's "tau flags the legal control too" says at population scale.

The game, its $t_0$ and the three branch rows are `select_game.py`'s pick, read from
figure_game.json; every square's category is re-derived with `taxonomy.classify`, so a corpus
change fails loudly instead of drawing a board the caption no longer describes.

Usage: PYTHONPATH=src:analysis/violation_divergence/illegal_move_detector python \
    analysis/violation_divergence/illegal_move_detector/figure_boards.py [--camera-ready]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

import detector
from detector import FIGURES, load
from spade.common.study import study_dir
from spade.figures.othello_board import (candidate, draw_board, mark_square, square_xy,
                                          verdict)
from spade.figures.paper_style import INK, RULE_STYLE, VERDICT_STYLE, apply_style, save_figure
from spade.othello.boards import rays, replay
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_OF, classify

from circuits.othello_engine_utils import to_string

HERE = Path(__file__).resolve().parent
OFFSETS = (-2, -1, 0, 1, 2)
LETTERS = "abcde"
FIGURE_SIZE = (12.0, 3.3)
# A fifth of this canvas is about half figure_othello_rules' panel, so the disc shrinks with
# it: a disc that keeps ms=13 here fills three quarters of its cell and swallows the outlines.
DISC_SIZE = 8.0
LABEL_STRIP = 0.19      # figure fraction reserved under the panels for their two label lines
# Where a rule's code goes, in cells from the square's centre, tried in order until one lands
# inside the board: the square itself is spoken for by the candidate disc and the verdict
# cross, and a2 and a8 sit on the board's edge where a centred label hangs off it. Keying this on
# the rule instead of on the square is what broke when `select_game.py` learned to prefer the
# legible member of each rule: R2 moved to rank 8 and its below-the-square label fell off the
# edge and was clipped away silently.
LABEL_AT = ((-0.5, -0.55, "left", "bottom"),    # above, left-aligned
            (-0.46, 0.55, "left", "top"),       # below, left-aligned
            (0.6, 0.0, "left", "center"),       # to the right
            (-0.6, 0.0, "right", "center"))     # to the left


def name(square: int) -> str:
    return f"{'abcdefgh'[square % 8]}{square // 8 + 1}"


def ray_line(ax, square: int, through: int, color: str) -> None:
    """A thin line from a candidate square out to the square that decides its bracket."""
    (x0, y0), (x1, y1) = square_xy(square), square_xy(through)
    ax.plot([x0, x1], [y0, y1], color=color, lw=1.2, zorder=2, solid_capstyle="round")


def game(spec: dict) -> tuple[list[int], dict[str, int]]:
    """The reference game's tokens, and the illegal square each rule's branch plays at t0.

    figure_game.json indexes detector.load(); `row` maps those onto the branch tree, where
    the tokens live.
    """
    data = load()
    branches = torch.load(study_dir("divergence") / "branches" / "branches.pt",
                          weights_only=True)
    for r in [spec["reference_row"], *spec["rows"].values()]:
        assert data["game_id"][r] == spec["game_id"] and data["t0"][r] == spec["t0"], \
            f"row {r} is not game {spec['game_id']} at t0 {spec['t0']}: stale figure_game.json"
    tokens = branches["tokens"][data["row"][spec["reference_row"]]].tolist()
    return tokens, {code: to_string(int(branches["tokens"][data["row"][r], spec["t0"]]))
                    for code, r in spec["rows"].items()}


def rule_squares(board, squares: dict[str, int]) -> dict[str, int]:
    """The three branch squares keyed by the rule the board says they break."""
    out = {}
    for code, square in squares.items():
        category = CATEGORIES[classify(board, square)]
        rule = RULE_OF[category]
        assert rule.startswith(code), f"{name(square)} is {category}, which breaks {rule}"
        out[rule] = square
    return out


def side_to_move(ax, board) -> None:
    """Whose turn it is, on the axis rather than in a title: R3 is legality judged for the
    mover, and a camera-ready figure has no title to carry it (FIGURE_GUIDELINES 4.9, 5)."""
    ax.set_xlabel(f"file — {'black' if board.next_hand_color > 0 else 'white'} to move")


def played_panel(ax, board, played: int) -> None:
    """One ply as the game played it: the move, and every bracket that makes it legal."""
    assert classify(board, played) == LEGAL, f"{name(played)} is not legal for the mover"
    draw_board(ax, board.state.flatten(), DISC_SIZE)
    mover = int(board.next_hand_color)
    for ray in rays(board.state, played, mover):
        if ray.closed:
            ray_line(ax, played, ray.terminator, RULE_STYLE["legal"]["color"])
    # One marked square: the outline and the name carry it, and a marker would read as a disc.
    mark_square(ax, played, RULE_STYLE["legal"], label=name(played), marker=False)
    candidate(ax, played, mover, DISC_SIZE)
    verdict(ax, played, legal=True, disc_size=DISC_SIZE)


def event_panel(ax, board, rules: dict[str, int]) -> None:
    """t0: the same board, and the three illegal moves the branch tree forks into."""
    draw_board(ax, board.state.flatten(), DISC_SIZE)

    # R2 here is `empty_detached`: it touches no disc at all, so there is no line to draw
    # along and its square is marked and named and nothing else.
    # R3: the bracket that does close -- for the opponent.
    theirs = next(r for r in rays(board.state, rules["R3 turn"], -int(board.next_hand_color))
                  if r.closed)
    ray_line(ax, rules["R3 turn"], theirs.terminator, RULE_STYLE["R3 turn"]["color"])

    # No legend: each square carries its rule's colour, its rule's marker, and its rule's code
    # with the move that branch played, placed in whichever empty cells sit beside it.
    mover = int(board.next_hand_color)
    for rule, square in rules.items():
        mark_square(ax, square, RULE_STYLE[rule])
        candidate(ax, square, mover, DISC_SIZE)
        verdict(ax, square, legal=False, disc_size=DISC_SIZE)
        x, y = square_xy(square)
        dx, dy, ha, va = next((p for p in LABEL_AT
                               if -0.5 <= x + p[0] <= 7.5 and -0.5 <= y + p[1] <= 7.5),
                              LABEL_AT[0])
        ax.annotate(f"{rule.split()[0]} {name(square)}", (x + dx, y + dy), ha=ha, va=va,
                    fontsize=9, fontweight="bold", zorder=6,
                    color=RULE_STYLE[rule]["color"])


def frame(ax) -> None:
    """The event panel's own border, so t0 is findable without reading five titles."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.2)


def verdicts(spec: dict) -> dict[int, tuple[str, bool]]:
    """What the detector says at each drawn ply, and whether it is right.

    Off t0 the reference row is scored: the move drawn there is the legal one the game played,
    so a score above tau is a false alarm. At t0 the three branch rows are, all three illegal,
    and the panel reports how many of them clear tau.
    """
    data = load()
    cand = detector.band()
    s = detector.score(data, cand)
    tau = detector.operating_point(s, cand, data, detector.arms(data))
    pos = cand + detector.CAND_OFFSET

    def at(row: int, t: int) -> float:
        j = int(np.searchsorted(pos, t))
        assert pos[j] == t, f"move {t} is outside the scanned band"
        return float(s[row, j])

    out = {}
    for offset in OFFSETS:
        t = spec["t0"] + offset
        if offset:
            flagged = at(spec["reference_row"], t) > tau
            out[offset] = (f"detector: {'illegal' if flagged else 'legal'}"
                           f"{', false alarm' if flagged else ', correct'}", not flagged)
        else:
            hits = sum(at(row, t) > tau for row in spec["rows"].values())
            out[offset] = (f"detector: illegal, correct {hits}/3", hits == len(spec["rows"]))
    return out


def panel_label(fig, ax, text: str, bold: bool, call: tuple[str, bool]) -> None:
    """The panel's letter and ply under the board, and the detector's call under that.

    The call is coloured by whether it is right, in the same green/red the board's verdict
    marks spend on legal/illegal -- there the mark states the move, here the colour states
    the detector.
    """
    x = ax.get_position().x0 + ax.get_position().width / 2
    fig.text(x, 0.085, text, ha="center", va="bottom", color=INK,
             fontsize=plt.rcParams["axes.titlesize"], fontweight="bold" if bold else "normal")
    fig.text(x, 0.02, call[0], ha="center", va="bottom", fontsize=8.0,
             color=VERDICT_STYLE["legal" if call[1] else "illegal"]["color"])


def boards_figure(tokens: list[int], squares: dict[str, int], spec: dict,
                  camera_ready: bool) -> Path:
    fig, axes = plt.subplots(1, len(OFFSETS), figsize=FIGURE_SIZE)
    for ax, offset in zip(axes, OFFSETS):
        board = replay(tokens[:spec["t0"] + offset])
        if offset:
            played_panel(ax, board, to_string(tokens[spec["t0"] + offset]))
        else:
            event_panel(ax, board, rule_squares(board, squares))
            frame(ax)
        side_to_move(ax, board)
        ax.set_ylabel("rank" if offset == OFFSETS[0] else "")
    fig.tight_layout(rect=(0, LABEL_STRIP, 1, 1))
    calls = verdicts(spec)
    for ax, offset, letter in zip(axes, OFFSETS, LETTERS):
        tag = "$t_0$" if offset == 0 else f"$t_0{offset:+d}$"
        panel_label(fig, ax, f"({letter}) {tag}, move {spec['t0'] + offset}", offset == 0,
                    calls[offset])
    return save_figure(fig, FIGURES / "board_states", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    spec = json.loads((HERE / "figure_game.json").read_text())
    tokens, squares = game(spec)
    print(boards_figure(tokens, squares, spec, args.camera_ready))


if __name__ == "__main__":
    main()
