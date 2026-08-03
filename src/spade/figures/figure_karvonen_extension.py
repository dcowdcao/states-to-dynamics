#!/usr/bin/env python3
"""Figure 4: Karvonen Figure 4 extension with SpaDE.

Renders the 2x2 grid (loss recovered / coverage / reconstruction F1 vs L0) from
a single run's all_results.csv, with SpaDE highlighted. The plotting itself
lives in karvonen_plot; this entrypoint just resolves paths and loads the CSV.

Usage:
    python -m spade.figures.figure_karvonen_extension \
        --results_csv results/main/all_results.csv --output_dir figures
"""

import argparse
import os

import pandas as pd

from spade.common.paths import repo_root
from spade.figures.karvonen_plot import render_grid

import matplotlib.pyplot as plt

OUT_NAME = "Figure_4_karvonen_extension.png"
SUPTITLE = "Karvonen Figure 4 Extension: SAE Comparison with SpaDE"


def save_placeholder(output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax in axes.flatten():
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    fig.suptitle("Figure 4: Karvonen Extension (placeholder)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(output_dir, OUT_NAME)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved placeholder: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure 4: Karvonen extension")
    parser.add_argument("--results_csv", type=str,
                        default="results/multiseed/aggregate_baseline.csv",
                        help="Multi-seed aggregate (error bars) or a single-run "
                             "all_results.csv (plain points)")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    for attr in ("results_csv", "output_dir"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(str(repo_root()), val))

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.results_csv):
        print(f"Results CSV not found: {args.results_csv}")
        save_placeholder(args.output_dir)
        return

    df = pd.read_csv(args.results_csv)
    render_grid(df, os.path.join(args.output_dir, OUT_NAME), SUPTITLE)


if __name__ == "__main__":
    main()
