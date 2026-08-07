"""The legality-probe figure. Drawing only.

One bar per representation space: how well does each support the 4-way rule probe?

Everything is read from `results/probe_fit.csv`; nothing here computes a statistic.
`results/readout.csv` and `space_quality.csv` are the study's other outputs and are cited
by RED_TEAM.md, but no figure reads them.

Read the CSV with keep_default_na=False: `class` is an empty string on the macro rows, and
NaN there silently drops every one of them.

The figure carries its own refutation, because the ranking it draws does not mean what a
reader will assume. Legality is a deterministic function of the board, so the bar is mostly
board-reading -- decoding the board and applying the rules engine beats the legality probe
in all 11 spaces, and board macro-F1 tracks legality at pearson 0.998. The board marker on
each bar is what makes that visible. The two untrained controls are the second guard: a
dense random projection preserves nearly all of the input and outranks a trained
dictionary, so a high bar mostly means the map did not lose information. The comparison
that survives is sparsity-matched -- TopK 0.923 against random TopK 0.774, both at L0 80.

Usage: python analysis/legality_probe/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spade.figures.paper_style import (
    CONTROL_SPACES, INK, MUTED, REFERENCE_SPACE, SPACE_STYLE, SPACES, apply_style,
    save_figure)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def styled(space: str) -> dict:
    return SPACE_STYLE[space]


def read(results: Path) -> pd.DataFrame:
    fit = pd.read_csv(results / "probe_fit.csv", keep_default_na=False)
    assert set(fit.space) == set(SPACES), \
        f"probe_fit.csv and paper_style.SPACES disagree: {set(fit.space) ^ set(SPACES)}"
    return fit


def macro(fit: pd.DataFrame, target: str) -> pd.DataFrame:
    return fit[(fit.metric == "macro_f1") & (fit["class"] == "")
               & (fit.target == target)].set_index("space")


def which_space(fit: pd.DataFrame, camera_ready: bool) -> None:
    legality, board = macro(fit, "legality"), macro(fit, "board")
    floor = macro(fit, "legality_shuffled").value.mean()
    order = legality.drop(index=CONTROL_SPACES).value.sort_values(ascending=False)
    spaces = list(order.index) + CONTROL_SPACES

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    y = np.arange(len(spaces))
    for row, space in zip(y, spaces):
        r, style = legality.loc[space], styled(space)
        ax.barh(row, r.value, height=0.6, color=style["color"],
                alpha=1.0 if space == REFERENCE_SPACE else 0.85,
                edgecolor=INK, lw=0.8 if space == REFERENCE_SPACE else 0.0, zorder=3)
        ax.errorbar(r.value, row, xerr=[[r.value - r.lo], [r.hi - r.value]], lw=0,
                    elinewidth=0.9, capsize=1.8, ecolor=INK, zorder=6)
        ax.plot(board.value[space], row, marker="|", ms=8, mew=1.4, color=INK, zorder=5)

    ax.axvline(floor, color=MUTED, ls=(0, (4, 2)), lw=1.0, zorder=2)
    # paper_style wraps the two controls onto two lines; at this row pitch they collide
    ax.set_yticks(y, [styled(s)["short"].replace("\n", " ") for s in spaces])
    ax.set_ylim(len(spaces) - 0.5, -0.5)
    ax.axhline(0.5, color=MUTED, lw=0.5)
    ax.axhline(len(spaces) - len(CONTROL_SPACES) - 0.5, color=MUTED, lw=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("rule-probe macro-F1 (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="|", ms=8, mew=1.4, ls="none", color=INK,
            label="same space, board probe")
    ax.plot([], [], color=MUTED, ls=(0, (4, 2)), lw=1.0, label=f"shuffled labels ({floor:.2f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False)
    if not camera_ready:
        fig.suptitle("Which space supports the rule probe best?", y=0.99)
    fig.tight_layout()
    save_figure(fig, FIGURES / "f0_which_space", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results",
                    help="directory holding probe_fit.csv")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    which_space(read(args.results), args.camera_ready)
    print(f"1 figure in {FIGURES}")


if __name__ == "__main__":
    main()
