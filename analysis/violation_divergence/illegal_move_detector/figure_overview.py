"""The study's four panels on one canvas: what the event is, and what the detector does with it.

(a) the five plies around the spliced move, (b) the score across that game, (c) AUROC by rule
over the 500-game evaluation set, (d) the score distributions that AUROC summarizes, on the
same 500 games. (a)-(b) are one game and illustrative; (c)-(d) are the population.

Every panel is drawn by its own figure's `panel` -- `figure_boards.boards_row`,
`figure_trace.panel`, `figure_auroc.panel`, `figure_histograms.panel` -- so nothing here can
drift from the standalone figures, and the standalone ones re-render byte-identical after this
module was split out of them. The only thing this file adds is the layout and the four letters.

(a)'s five boards keep their ply and their detector call and lose their own (a)-(e): two
levels of letters on one canvas is one level too many.

Usage: PYTHONPATH=src:analysis/violation_divergence/illegal_move_detector python \
    analysis/violation_divergence/illegal_move_detector/figure_overview.py [--camera-ready]
"""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import detector
from detector import CAND_OFFSET, FIGURES
from figure_auroc import panel as auroc_panel, table as auroc_table
from figure_boards import (OFFSETS, boards_row, game as boards_game, label_move,
                           panel_label)
from figure_histograms import panel as density_panel, subset
from figure_trace import panel as trace_panel
from spade.figures.paper_style import apply_style, letters_topleft, save_figure

HERE = Path(__file__).resolve().parent
FIGURE_SIZE = (12.0, 7.5)
ROWS = [3.3, 4.0]        # the boards are shallower than a plot with an axis label under it
MARGINS = {"left": 0.055, "right": 0.988, "top": 0.945, "bottom": 0.075}
PLY, CALL = 0.030, 0.052   # drops below the boards, in figure fractions


def under(fig, axes, pad: float = 0.006) -> float:
    """The lowest figure-fraction edge of what `axes` actually occupy, less `pad`.

    Measured rather than assumed: the boards carry an x label and the plots carry a taller
    one, so a fixed offset from the axes lands in two different places.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in axes]
    return min(b.y0 for b in boxes) - pad


def overview(spec: dict, camera_ready: bool) -> Path:
    tokens, squares = boards_game(spec)
    data = detector.load()
    cand = detector.band()
    s = detector.score(data, cand)
    sel = detector.arms(data)
    tau = detector.operating_point(s, cand, data, sel)
    rows = subset(data, sel)
    bars = auroc_table()

    # One grid of 15 columns rather than two nested ones: 5 boards at 3 columns each over 3
    # plots at 5, which `tight_layout` can lay out and a subgridspec cannot.
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = GridSpec(2, 15, figure=fig, height_ratios=ROWS, hspace=0.44, wspace=1.25)
    boards = [fig.add_subplot(grid[0, 3 * i:3 * i + 3]) for i in range(len(OFFSETS))]
    plots = [fig.add_subplot(grid[1, 5 * i:5 * i + 5]) for i in range(3)]

    drawn, made = boards_row(fig, boards, tokens, squares, spec, letters=False)
    trace_panel(plots[0], spec, cand + CAND_OFFSET, s, tau)
    auroc_panel(plots[1], bars)
    density_panel(plots[2], {k: detector.at_t0(s, cand, data["t0"], r) for k, r in rows.items()},
                  tau)

    # Margins rather than `tight_layout`: the boards hold a fixed aspect, which that refuses
    # to lay out, and the letters are placed off measured extents anyway.
    grid.update(**MARGINS)
    fig.canvas.draw()
    label_move(fig, boards[OFFSETS.index(0)], made["played"], made["rule"], drawn)

    base = under(fig, boards)
    for ax, offset in zip(boards, OFFSETS):
        tag = "$t_0$" if offset == 0 else f"$t_0{offset:+d}$"
        panel_label(fig, ax, f"{tag}, move {spec['t0'] + offset}", offset == 0,
                    made["calls"][offset], y=base - PLY, call_y=base - CALL)
    # Two rows at different heights, so each letters on its own baseline: A over the leftmost
    # board, which is the top-left of the whole row, then B C D over the three panels.
    letters_topleft(fig, [boards[0]])
    letters_topleft(fig, plots, start=1)
    return save_figure(fig, FIGURES / "detector_overview", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    print(overview(json.loads((HERE / "figure_game.json").read_text()), args.camera_ready))


if __name__ == "__main__":
    main()
