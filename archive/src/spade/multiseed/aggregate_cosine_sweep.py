"""Aggregate the CosineSpaDE gamma sweep across seeds into mean/std rows.

Reads all 5 seeds (42-46) from every per-seed
$SPADE_MULTISEED_DIR/seed_<seed>/results/karvonen_cosine_spade_sweep*/ segment
(the original [0.05, 5] grid plus the high-gamma extension, which had to train
into its own directory because trainer indices restart at 0), then concatenates
the gamma-config mean/std rows with the baseline+distance-SpaDE aggregate
(results/multiseed/aggregate.csv) so the Karvonen figure renders every config
with error bars. Reuses config_key / load_seed_rows / METRICS from
aggregate_seeds.

Output (results/multiseed/):
    cosine_sweep_all_seeds.csv    every (gamma-config, seed) row
    aggregate_cosine_sweep.csv    baseline+distance aggregate + one row per gamma

Usage:
    python -m spade.multiseed.aggregate_cosine_sweep
"""

import pandas as pd

from spade.common.paths import repo_root
from spade.multiseed.aggregate_seeds import MULTISEED_DIR, METRICS, load_seed_rows

SEEDS = [42, 43, 44, 45, 46]
COSINE_PREFIX = "CosineSpaDE-gam"


def main():
    root = repo_root()
    frames = []
    for seed in SEEDS:
        csvs = sorted((MULTISEED_DIR / f"seed_{seed}/results")
                      .glob("karvonen_cosine_spade_sweep*/all_results.csv"))
        if not csvs:
            print(f"  (seed {seed}: no cosine-sweep results)")
        frames.extend(load_seed_rows(p, seed) for p in csvs)

    if not frames:
        raise FileNotFoundError("no cosine-sweep results found for any seed")

    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["config"].str.startswith(COSINE_PREFIX)]
    rows = rows.drop_duplicates(subset=["config", "seed"], keep="first")
    rows["gamma"] = rows["config"].str[len(COSINE_PREFIX):].astype(float)

    out_dir = root / "results/multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "cosine_sweep_all_seeds.csv", index=False)

    grouped = rows.groupby("config")
    agg = grouped[METRICS].mean()
    for m in METRICS:
        agg[f"{m}_std"] = grouped[m].std(ddof=1)
    agg["n_seeds"] = grouped.size()
    agg["sae_type"] = grouped["sae_type"].first()
    agg["gamma"] = grouped["gamma"].first()
    agg = agg.reset_index().sort_values("gamma").reset_index(drop=True)

    base_agg_path = out_dir / "aggregate.csv"
    if base_agg_path.exists():
        combined = pd.concat([pd.read_csv(base_agg_path), agg], ignore_index=True)
    else:
        print(f"  (no baseline aggregate at {base_agg_path} — cosine-only figure)")
        combined = agg
    combined.to_csv(out_dir / "aggregate_cosine_sweep.csv", index=False)

    print(f"  {len(rows)} cosine rows across seeds {sorted(rows['seed'].unique())}")
    print(agg[["gamma", "n_seeds", "l0", "l0_std", "coverage", "coverage_std",
               "reconstruction_f1", "variance_explained"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"Saved: {out_dir}/aggregate_cosine_sweep.csv")


if __name__ == "__main__":
    main()
