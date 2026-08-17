"""The layer-sweep figure: where in OthelloGPT's residual stream the board is linearly read.

Drawing only; everything comes from results/layer_sweep.csv.

Two panels because two things are being claimed and they need different scales. (a) is
anchored at the input, so the reader can see the board state is *computed* rather than
inherited: the embedding space is its own tick, and the per-square majority-class null is
drawn. (b) resolves the layer-to-layer differences the original bar chart exaggerated by
truncating its baseline to 0.86 -- points with intervals, so a truncated axis is honest,
and the five seeds are drawn behind their mean rather than summarised away.

Usage: python analysis/linear_probe/make_figures.py [--camera-ready]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spade.figures.paper_style import (
    INK, MUTED, NULL, PROBE_STYLE, REFERENCE, apply_style, save_figure)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
EMBED = "embed"         # the space column train_linear_probe.py writes for resid_pre block 0
EMBED_X = -1.0          # the input space, left of the blocks and ruled off from them
Y_LABEL = "board accuracy, held out (higher is better)"
X_LABEL = "residual space (resid_post; layer 0 is the first block)"


def read(results: Path) -> tuple[pd.DataFrame, float, dict]:
    """Per-space fits, the majority-class null, and the run's own record of its n."""
    sweep = pd.read_csv(results / "layer_sweep.csv")
    null = sweep[sweep.space == "majority"].eval_acc.item()
    probes = sweep[sweep.space != "majority"].copy()
    probes["x"] = [EMBED_X if space == EMBED else float(space) for space in probes.space]
    return probes, null, json.loads((results / "probe_metadata.json").read_text())


def sweep_panel(ax, probes: pd.DataFrame, null: float | None) -> None:
    stats = probes.groupby("x").eval_acc.agg(["mean", "std", "count"]).sort_index()
    style = PROBE_STYLE["linear"]
    ax.plot(probes.x, probes.eval_acc, ls="none", marker=style["marker"], ms=2.6,
            color=style["color"], alpha=0.35, zorder=4, label="per-seed fit")
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], color=style["color"],
                marker=style["marker"], ms=5, lw=REFERENCE["lw"], elinewidth=1.0,
                capsize=2.4, zorder=REFERENCE["zorder"], label=style["label"])
    if null is not None:
        ax.axhline(null, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"],
                   zorder=NULL["zorder"],
                   label=f"per-square majority class ({null:.3f}, null)")
    ticks = sorted(stats.index)
    ax.set_xticks(ticks, ["embed" if x == EMBED_X else f"{x:.0f}" for x in ticks])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)


def layer_sweep(sweep: pd.DataFrame, null: float, meta: dict, camera_ready: bool) -> Path:
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(9, 3.7))

    sweep_panel(full, sweep, null)
    # FIGURE_GUIDELINES 4.1: the input space is a tick of its own, ruled off from the network.
    full.axvline(EMBED_X + 0.5, color=MUTED, lw=0.5, zorder=1)
    full.set_ylim(0, 1.0)
    embed = sweep[sweep.space == EMBED].eval_acc.mean()
    full.annotate(f"embedding: {embed:.3f}", (EMBED_X, embed), textcoords="offset points",
                  xytext=(10, 7), fontsize=8, color=INK)
    full.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))

    blocks = sweep[sweep.space != EMBED]
    sweep_panel(zoom, blocks, None)
    stats = blocks.groupby("x").eval_acc.agg(["mean", "std"])
    best = stats["mean"].idxmax()
    zoom.set_ylim(stats["mean"].min() - 0.012, 1.0)
    zoom.set_xlabel(X_LABEL.replace("residual space", "layer"))
    zoom.annotate(f"best: layer {best:.0f}, {stats['mean'].max():.4f}",
                  (best, stats["mean"].max()), textcoords="offset points", xytext=(7, -13),
                  fontsize=8, color=INK, ha="left")
    zoom.set_ylabel("same, zoomed")

    for i, ax in enumerate((full, zoom)):
        ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
                color=INK, fontsize=9)
    fig.text(0.005, -0.02,
             f"Bars: ±1 SD over {len(meta['sweep_seeds'])} probe-initialisation seeds "
             f"(largest {stats['std'].max():.4f}), drawn behind their mean. The games, the "
             f"{meta['n_train']}/{meta['n_eval']} split and the activations are fixed,\nso "
             f"the bars are optimisation noise and not sampling noise over games. Accuracy "
             f"is over {meta['n_eval']} held-out positions × 64 squares, 3 classes "
             f"{{empty, theirs, mine}}.\n(b) drops the embedding space and truncates the "
             f"y-axis; the marks carry intervals, so a truncated axis cannot exaggerate a "
             f"difference the way a cut bar baseline does.",
             ha="left", va="top", fontsize=7.5, color=MUTED, linespacing=1.5)
    if not camera_ready:
        fig.suptitle("Where the board state is linearly readable", y=1.0)
    fig.tight_layout()
    return save_figure(fig, FIGURES / "layer_sweep_accuracy", camera_ready)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--camera-ready", action="store_true", help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    print(layer_sweep(*read(args.results), args.camera_ready))


if __name__ == "__main__":
    main()
