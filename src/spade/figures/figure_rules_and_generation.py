"""The rules on one board, beside what the model does with them.

(a) is `figure_othello_rules`' board and (b) the generation-legality study's accumulated
by-rule mass, each drawn by the function that owns it. This module is layout: it places the
two panels and their letters and sets nothing else.

The letters go at each panel's top-left, bolded, as everywhere else in the paper; (a)'s square labels sit above
its top rank and a title lands on them.

Usage: python -m spade.figures.figure_rules_and_generation [--camera_ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spade.common.paths import repo_root
from spade.figures.figure_othello_rules import position, rules_panel
from spade.figures.generation_panels import cumulative_panel
from spade.figures.paper_style import apply_style, letters_topleft, save_figure

FIGURES = repo_root() / "figures" / "rules_and_generation"
ROLLOUT = (repo_root() / "analysis" / "generation_legality" / "results"
           / "rollout_by_move.csv")
FIGURE_SIZE = (10.4, 4.5)
WIDTHS = [1, 1.45]   # A holds a square board, B a wide move axis; the board sets the height


def combined_figure(board, mover: int, played: int, rollout: pd.DataFrame,
                    camera_ready: bool) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=FIGURE_SIZE,
                                      gridspec_kw={"width_ratios": WIDTHS})
    rules_panel(left, board, mover, played)
    cumulative_panel(right, rollout)
    fig.tight_layout()
    letters_topleft(fig, (left, right))
    return save_figure(fig, FIGURES / "rules_and_generation", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    board, mover, played = position(args.dataset_name)
    print(combined_figure(board, mover, played, pd.read_csv(ROLLOUT), args.camera_ready))


if __name__ == "__main__":
    main()
