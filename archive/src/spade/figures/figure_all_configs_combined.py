#!/usr/bin/env python3
"""Figure: Karvonen Figure 4 extension with ALL SpaDE configs combined.

Renders the shared 2x2 grid over every figure config (8 baselines + 6 SpaDE
sweep points) from the multi-seed aggregate, with ±1-std seed error bars.
Build the aggregate first: python -m spade.multiseed.aggregate_seeds

Usage:
    python -m spade.figures.figure_all_configs_combined
"""

import pandas as pd

from spade.common.paths import repo_root
from spade.figures.karvonen_plot import TRAINER_CLASS_MAP, render_grid

OUT_NAME = "Figure_4_all_configs_combined.png"
SUPTITLE = "Karvonen Figure 4 Extension: All SAE Configs (error bars: ±1 std across seeds)"

# The chosen-best SpaDE comes from the superseded 8-config sweep; its model is
# lost, so it joins as a single-seed point (no error bars) from the old CSV.
BEST_SPADE_CONFIG = "SpaDE-lam1e-03-gam1e-04"
BEST_SPADE = {BEST_SPADE_CONFIG: r"Best SpaDE ($\lambda$=1e-3, $\gamma$=1e-4)"}
OLD_SWEEP_CSV = "results/spade_sweep_8config/all_results.csv"


def main():
    agg_path = repo_root() / "results/multiseed/aggregate.csv"
    if not agg_path.exists():
        raise FileNotFoundError(
            f"{agg_path} missing — run python -m spade.multiseed.aggregate_seeds first")
    df = pd.read_csv(agg_path)
    print(f"  Loaded {len(df)} aggregated configs "
          f"(seeds per config: {sorted(df['n_seeds'].unique())})")

    old = pd.read_csv(repo_root() / OLD_SWEEP_CSV)
    best = old[old["wandb_name"].str.startswith(BEST_SPADE_CONFIG)].iloc[0]
    df = pd.concat([df, pd.DataFrame([{
        "config": BEST_SPADE_CONFIG,
        "sae_type": best["sae_type"],
        "n_seeds": 1,
        **{m: best[m] for m in ("l0", "frac_recovered", "coverage",
                                "reconstruction_f1", "variance_explained")},
    }])], ignore_index=True)
    print(f"  Appended {BEST_SPADE_CONFIG} (single seed, from {OLD_SWEEP_CSV})")

    # Plot SpaDE points last so they sit on top.
    df["_is_spade"] = df["sae_type"].apply(
        lambda x: 1 if TRAINER_CLASS_MAP.get(x, x) == "sparsemax_dist" else 0)
    df = df.sort_values("_is_spade").reset_index(drop=True)

    out_dir = repo_root() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_grid(df, str(out_dir / OUT_NAME), SUPTITLE, annotate=BEST_SPADE)


if __name__ == "__main__":
    main()
