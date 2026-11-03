#!/usr/bin/env python3
"""Figure: CosineSpaDE metrics vs gamma across the log sweep.

The Karvonen 2x2 plots every metric against L0, which hides the sweep's actual
knob. This plots L0 / coverage / reconstruction F1 / variance explained directly
against gamma on a log axis with +-1 std seed error bars, and marks the gamma
where the sparsity regularizer reaches parity with the MSE term (measured by
spade/training/sweep_loss_balance.py) — below it gamma is inert, which is what
the first [0.05, 5] sweep sat entirely inside.

Usage:
    python -m spade.figures.figure_cosine_gamma_response
"""

import matplotlib.pyplot as plt
import pandas as pd

from spade.common.paths import repo_root
from spade.figures.karvonen_plot import SAE_CONFIG

OUT_NAME = "Figure_cosine_gamma_response.png"
PARITY_GAMMA = 16.0  # gamma*reg == MSE, from sweep_loss_balance.py on the trained sweep
BASELINE = "TopKTrainer-othello"

PANELS = [
    ("l0", r"$L_0$", True),
    ("coverage", "Coverage", False),
    ("reconstruction_f1", r"Reconstruction $F_1$", False),
    ("variance_explained", "Variance Explained", False),
]


def main():
    root = repo_root()
    df = pd.read_csv(root / "results/multiseed/aggregate_cosine_sweep.csv")
    cos = df[df["gamma"].notna()].sort_values("gamma")
    base = pd.read_csv(root / "results/multiseed/aggregate_baseline.csv")
    topk = base[base["config"] == BASELINE].iloc[0]

    color = SAE_CONFIG["cosine_spade"]["color"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for ax, (metric, label, log_y) in zip(axes.flatten(), PANELS):
        ax.errorbar(cos["gamma"], cos[metric], yerr=cos[f"{metric}_std"].fillna(0),
                    marker="v", markersize=7, color=color, ecolor="dimgray",
                    elinewidth=1.0, capsize=3, linewidth=1.5, zorder=5)
        ax.axvline(PARITY_GAMMA, color="gray", linestyle=":", linewidth=1.2, zorder=1)
        ax.axhline(topk[metric], color=SAE_CONFIG["topk_relu"]["color"],
                   linestyle="--", linewidth=1.2, zorder=1,
                   label=f"TopK baseline ({topk[metric]:.3g})")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs $\\gamma$", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    axes[0, 0].annotate(r"MSE / reg parity", (PARITY_GAMMA, 0.04), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=8, color="gray", ha="right", va="bottom")

    fig.suptitle("CosineSpaDE: metric response to the sparsity weight $\\gamma$ "
                 "(error bars: ±1 std across 5 seeds)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_dir = root / "figures/cosine_spade_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / OUT_NAME, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_dir / OUT_NAME}")


if __name__ == "__main__":
    main()
