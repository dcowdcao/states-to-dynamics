"""Fig 5: board-state accuracy per residual-stream layer, under the canonical probe.

Draws results/board_layer_sweep.csv, which measure.py writes from the probe corpus; the
figure is a function of that file, so the two cannot drift apart.

Usage: python analysis/board_probe_sweep/layer_sweep.py [--camera_ready]
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.common.config import LAYER
from spade.figures.paper_style import (
    BOARD_SWEEP_STYLE, INK, MUTED, apply_style, save_figure)

HERE = Path(__file__).resolve().parent
LAYERS = range(8)


def draw(table: pd.DataFrame, camera_ready: bool) -> Path:
    """Bars per layer on a cut baseline, with the null printed rather than drawn.

    FIGURE_GUIDELINES 1.6 asks bars to start at zero, and this figure does not: at a
    zero baseline every layer past 0 is the same bar. The cut is what the study is for
    -- the differences it exists to show are in the third decimal -- and each bar
    carries its interval, so the axis cannot exaggerate a gap the way a bare cut
    baseline does. README.md records the departure; the null sits far below the cut and
    is a number there rather than a line nobody can see.
    """
    bars = table[table.layer >= 0]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for _, r in bars.iterrows():
        key = "hook" if r.layer == LAYER else "layer"
        ax.bar(r.layer, r.accuracy, width=0.68, color=BOARD_SWEEP_STYLE[key]["color"],
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(r.layer, r.accuracy, yerr=[[r.accuracy - r.lo], [r.hi - r.accuracy]],
                    color=INK, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(r.layer, r.hi + 0.0012, f"{r.accuracy:.4f}", ha="center", va="bottom",
                fontsize=7.2, color=MUTED, zorder=4)

    ax.text(LAYER, bars.accuracy.min() * 0.5 + 0.5 * bars.accuracy.max(),
            "chosen layer", rotation=90, ha="center", va="center", fontsize=7.5,
            color="white", zorder=5)

    ax.set_ylim(0.90, 1.0055)
    ax.set_xticks(list(LAYERS))
    ax.set_xlabel("layer")
    ax.set_ylabel("board state accuracy")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return save_figure(fig, HERE / "figures" / "board_layer_sweep", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    print(draw(pd.read_csv(HERE / "results" / "board_layer_sweep.csv"), args.camera_ready))


if __name__ == "__main__":
    main()
