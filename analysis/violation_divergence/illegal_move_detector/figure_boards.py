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
illegal move the branch tree plays there instead; it carries a full frame so the event is
findable at a glance rather than by reading the panel labels.

**One board, one move.** A position has one move played in it, so the $t_0$ panel draws one --
the branch whose arm the trace figure's peak belongs to, of the three `select_game.py` picked.
It is labelled with the rule it breaks, in that rule's colour, so the board says which of R1,
R2 and R3 the event is without the reader going to the trace for it.

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

The geometry drawn beside the move is whatever its category has to show, as in
figure_othello_rules: an `opponent_only` square gets the bracket that closes for the opponent,
an `empty_no_bracket` or `empty_own_only` square the one line of opponent discs out of it and
the square that breaks it, and an occupied or `empty_detached` square nothing -- the disc on it
is the evidence, or there is no line to draw at all.

Under each panel is what the detector of this study says about that ply, and whether it is
right. Off $t_0$ the row scored is the reference game itself, whose move there is the legal
one it played, so anything over tau is a false alarm; at $t_0$ it is the drawn branch, which is
illegal. `select_game.py` requires at least four of the five to come out right. The current
pick comes out at four of five, false-alarming once, and one panel is one game either way --
37% of legal positions sit above tau.

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
# Where a rule's label goes, in cells from the square's centre, tried in order until the drawn
# text lands inside the board: the square itself is spoken for by the candidate disc and the
# verdict cross, and an edge square has no room on one side. The fit is measured, not predicted
# from the anchor -- `R2 h4` anchored inside the board still hangs off it, which is how the
# label was clipped away silently when the selected game moved to the h file.
LABEL_AT = ((-0.5, -0.55, "left", "bottom"),    # above, left-aligned
            (-0.46, 0.55, "left", "top"),       # below, left-aligned
            (0.5, -0.55, "right", "bottom"),    # above, right-aligned, for a right-edge square
            (0.46, 0.55, "right", "top"),       # below, right-aligned
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
    # One marked square and no name on it: the outline, the candidate disc and the tick say
    # everything a legal ply needs to, and a marker here would read as a disc. The t0 panel is
    # the only one that names its square, because it is the only one making a claim about which.
    mark_square(ax, played, RULE_STYLE["legal"], marker=False)
    candidate(ax, played, mover, DISC_SIZE)
    verdict(ax, played, legal=True, disc_size=DISC_SIZE)


def event_panel(ax, board, square: int, rule: str) -> None:
    """t0: the same board, and the one illegal move the drawn branch plays there."""
    draw_board(ax, board.state.flatten(), DISC_SIZE)
    mover = int(board.next_hand_color)
    color = RULE_STYLE[rule]["color"]
    category = CATEGORIES[classify(board, square)]

    if category == "opponent_only":
        theirs = next(r for r in rays(board.state, square, -mover) if r.closed)
        ray_line(ax, square, theirs.terminator, color)
    elif category in ("empty_no_bracket", "empty_own_only"):
        broken = next(r for r in rays(board.state, square, mover)
                      if r.run and r.blocker is not None)
        ray_line(ax, square, broken.blocker, color)
        bx, by = square_xy(broken.blocker)
        ax.plot(bx, by, marker="x", ms=5, mew=1.5, color=color, zorder=6)

    mark_square(ax, square, RULE_STYLE[rule])
    candidate(ax, square, mover, DISC_SIZE)
    verdict(ax, square, legal=False, disc_size=DISC_SIZE)


def label_move(fig, ax, square: int, rule: str, text: str) -> None:
    """Name the marked square, in the first LABEL_AT slot whose drawn text fits on the board.

    Run after the layout is final, because whether a label fits is a question about the drawn
    text and the axes it lands in, and both move under `tight_layout`.
    """
    renderer = fig.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    x, y = square_xy(square)
    for dx, dy, ha, va in LABEL_AT:
        drawn = ax.annotate(text, (x + dx, y + dy), ha=ha, va=va, fontsize=9,
                            fontweight="bold", zorder=6, color=RULE_STYLE[rule]["color"])
        box = drawn.get_window_extent(renderer)
        if panel.contains(*box.p0) and panel.contains(*box.p1):
            return
        drawn.remove()
    raise AssertionError(f"no room on this board for the {name(square)} label")


def frame(ax) -> None:
    """The event panel's own border, so t0 is findable without reading five titles."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.2)


def event(spec: dict) -> tuple[str, dict[int, tuple[str, bool]]]:
    """The rule whose branch the t0 board draws, and the detector's call at each drawn ply.

    The drawn branch is the highest-scoring of the three at t0 -- the arm the trace figure's
    "peak at t0" points at, so the pair names one move between them. Off t0 the reference row
    is scored: the move drawn there is the legal one the game played, so a score above tau is
    a false alarm.
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

    drawn = max(spec["rows"], key=lambda code: at(spec["rows"][code], spec["t0"]))
    out = {}
    for offset in OFFSETS:
        t = spec["t0"] + offset
        if offset:
            flagged = at(spec["reference_row"], t) > tau
            out[offset] = (f"detector: {'illegal' if flagged else 'legal'}"
                           f"{', false alarm' if flagged else ', correct'}", not flagged)
        else:
            flagged = at(spec["rows"][drawn], t) > tau
            out[offset] = (f"detector: {'illegal, correct' if flagged else 'legal, missed'}",
                           flagged)
    return drawn, out


def panel_label(fig, ax, text: str, bold: bool, call: tuple[str, bool],
                y: float = 0.085, call_y: float = 0.02) -> None:
    """The panel's letter and ply under the board, and the detector's call under that.

    The call is coloured by whether it is right, in the same green/red the board's verdict
    marks spend on legal/illegal -- there the mark states the move, here the colour states
    the detector.
    """
    x = ax.get_position().x0 + ax.get_position().width / 2
    fig.text(x, y, text, ha="center", va="bottom", color=INK,
             fontsize=plt.rcParams["axes.titlesize"], fontweight="bold" if bold else "normal")
    fig.text(x, call_y, call[0], ha="center", va="bottom", fontsize=8.0,
             color=VERDICT_STYLE["legal" if call[1] else "illegal"]["color"])


def boards_row(fig, axes, tokens: list[int], squares: dict[str, int], spec: dict,
               letters: bool = True) -> tuple[str, dict]:
    """The five boards onto `axes`, so a figure that stacks this row draws it and not a copy.

    `letters` off for a figure that letters its own panels: two levels of (a)-(e) on one canvas
    is one level too many, and the ply and the detector's call are what the row is for.
    """
    drawn, calls = event(spec)
    board_at_t0 = replay(tokens[:spec["t0"]])
    rule = next(r for r in rule_squares(board_at_t0, squares) if r.startswith(drawn))
    played = squares[drawn]
    for ax, offset in zip(axes, OFFSETS):
        board = replay(tokens[:spec["t0"] + offset])
        if offset:
            played_panel(ax, board, to_string(tokens[spec["t0"] + offset]))
        else:
            event_panel(ax, board, played, rule)
            frame(ax)
        side_to_move(ax, board)
        ax.set_ylabel("rank" if offset == OFFSETS[0] else "")
    return drawn, {"calls": calls, "played": played, "rule": rule}


def boards_figure(tokens: list[int], squares: dict[str, int], spec: dict,
                  camera_ready: bool) -> Path:
    fig, axes = plt.subplots(1, len(OFFSETS), figsize=FIGURE_SIZE)
    drawn, made = boards_row(fig, axes, tokens, squares, spec)
    fig.tight_layout(rect=(0, LABEL_STRIP, 1, 1))
    fig.canvas.draw()
    label_move(fig, axes[OFFSETS.index(0)], made["played"], made["rule"], drawn)
    for ax, offset, letter in zip(axes, OFFSETS, LETTERS):
        tag = "$t_0$" if offset == 0 else f"$t_0{offset:+d}$"
        panel_label(fig, ax, f"({letter}) {tag}, move {spec['t0'] + offset}", offset == 0,
                    made["calls"][offset])
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
