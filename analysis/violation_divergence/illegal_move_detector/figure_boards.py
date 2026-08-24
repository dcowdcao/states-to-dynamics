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
findable at a glance rather than by reading the panel titles.

Sizing: five 8x8 boards need ~1.9 in each before the discs, the algebraic ticks and a
square's name stop being legible, which puts the figure at 12 in -- past NeurIPS full width
(5.5 in), and FIGURE_GUIDELINES 1.3 forbids buying that back by shrinking in LaTeX. This is a
README/appendix figure at its drawn size; the trace figure is the one that fits a column.

R2's branch at this $t_0$ is `empty_detached`, the shallowest of R2's three members: a2
touches no disc at all, so there is no line that fails to close and nothing to draw along.
What the panel shades instead is the neighbourhood any line would have to start in, empty in
every direction. R1's square is occupied by the mover's own disc and R3's line brackets for
the opponent, so both keep figure_othello_rules' geometry.

The game, its $t_0$ and the three branch rows are `select_game.py`'s pick, read from
figure_game.json; every square's category is re-derived with `taxonomy.classify`, so a corpus
change fails loudly instead of drawing a board the caption no longer describes.

Usage: PYTHONPATH=src:analysis/violation_divergence/illegal_move_detector python \
    analysis/violation_divergence/illegal_move_detector/figure_boards.py [--camera-ready]
"""

import argparse
import json
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from detector import FIGURES, load
from spade.common.study import study_dir
from spade.figures.othello_board import draw_board, mark_square, square_xy
from spade.figures.paper_style import INK, MUTED, RULE_STYLE, apply_style, save_figure
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
NEIGHBOUR_TINT = 0.16   # R2's empty neighbourhood, shaded under the grid rather than outlined
# Where each rule's square carries its name, in cells from the square's centre. a2 and a8 sit
# on the board's edge, where a centred name hangs off it, and c3's own disc is the evidence
# for R1, so nothing may sit on the square itself.
# Where a rule's name goes, tried in order until one lands inside the board. Keying this on
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
    for ray in rays(board.state, played, int(board.next_hand_color)):
        if ray.closed:
            ray_line(ax, played, ray.terminator, MUTED)
    # One marked square: the outline and the name carry it, and a marker would read as a disc.
    mark_square(ax, played, RULE_STYLE["legal"], label=name(played), marker=False)


def event_panel(ax, board, rules: dict[str, int]) -> None:
    """t0: the same board, and the three illegal moves the branch tree forks into."""
    draw_board(ax, board.state.flatten(), DISC_SIZE)

    # R2 here is `empty_detached`, so the geometry is an absence and there is no line to draw
    # along: the cells a bracket would have to start in are shaded instead, under the grid,
    # and they are empty. An outline around them would have to run along the board's edge.
    x, y = square_xy(rules["R2 bracketing"])
    lo_x, lo_y = max(x - 1, 0), max(y - 1, 0)
    ax.add_patch(Rectangle((lo_x - 0.5, lo_y - 0.5), min(x + 1, 7) - lo_x + 1,
                           min(y + 1, 7) - lo_y + 1, lw=0, zorder=0,
                           facecolor=to_rgba(RULE_STYLE["R2 bracketing"]["color"],
                                             NEIGHBOUR_TINT)))

    # R3: the bracket that does close -- for the opponent.
    theirs = next(r for r in rays(board.state, rules["R3 turn"], -int(board.next_hand_color))
                  if r.closed)
    ray_line(ax, rules["R3 turn"], theirs.terminator, RULE_STYLE["R3 turn"]["color"])

    # No legend: each square carries its rule's colour, its rule's marker, and its own name,
    # placed in whichever empty cells sit beside it.
    for rule, square in rules.items():
        mark_square(ax, square, RULE_STYLE[rule])
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


def boards_figure(tokens: list[int], squares: dict[str, int], spec: dict,
                  camera_ready: bool) -> Path:
    fig, axes = plt.subplots(1, len(OFFSETS), figsize=FIGURE_SIZE)
    for ax, offset, letter in zip(axes, OFFSETS, LETTERS):
        t = spec["t0"] + offset
        board = replay(tokens[:t])
        if offset:
            played_panel(ax, board, to_string(tokens[t]))
        else:
            event_panel(ax, board, rule_squares(board, squares))
            frame(ax)
        tag = "$t_0$" if offset == 0 else f"$t_0{offset:+d}$"
        ax.set_title(f"({letter}) {tag}, move {t}", loc="left",
                     fontweight="bold" if offset == 0 else "normal")
        side_to_move(ax, board)
        if offset != OFFSETS[0]:
            ax.set_ylabel("")
    fig.tight_layout()
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
