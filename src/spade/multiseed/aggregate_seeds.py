"""Aggregate per-seed Karvonen eval results into mean/std rows for error-bar plots.

Seed 42 comes from the repo's existing result CSVs; seeds 43+ from the
multi-seed runs under $SPADE_MULTISEED_DIR (see scripts/slurm/multiseed/).

Outputs (results/multiseed/):
    all_seeds.csv           every (config, seed) row kept for aggregation
    aggregate.csv           one row per config: metric means + <metric>_std
    aggregate_baseline.csv  the 8 baseline SAE rows only

Usage:
    python -m spade.multiseed.aggregate_seeds
"""

import re
from pathlib import Path

import pandas as pd

from spade.common.paths import repo_root
from spade.common.study import study_dir

MULTISEED_DIR = study_dir("multiseed")
NEW_SEEDS = [43, 44, 45, 46]

# Entries 5 and 6 carry the seed-42 point for five of the six sweep configs; they
# are the only record of it, not an archival duplicate.
SEED42_CSVS = [
    "results/main/all_results.csv",
    "results/batch_topk/all_results.csv",
    "results/cosine_spade/all_results.csv",
    "results/spade_sweep/all_results.csv",
    "results/lambda_sweep/all_results.csv",
    "results/gamma_sweep_lam1e-07/all_results.csv",
]

BASELINE_CONFIGS = [
    "PAnnealTrainer-othello",
    "StandardTrainer-othello",
    "GatedSAETrainer-othello",
    "TopKTrainer-othello",
    "GatedAnnealTrainer-othello",
    "SpaDETrainer-othello",
    "BatchTopKTrainer-othello",
    "CosineSpaDETrainer-othello",
]
SWEEP_CONFIGS = [
    "SpaDE-lam8e-08-gam1e-03",
    "SpaDE-lam1e-07-gam1e-03",
    "SpaDE-lam1e-07-gam5e-03",
    "SpaDE-lam1e-07-gam8e-03",
    "SpaDE-lam1e-07-gam1e-02",
    "SpaDE-lam1e-07-gam3e-02",
]
FIGURE_CONFIGS = BASELINE_CONFIGS + SWEEP_CONFIGS

METRICS = ["l0", "frac_recovered", "coverage", "reconstruction_f1",
           "variance_explained"]


def config_key(wandb_name: str) -> str:
    return re.sub(r"_trainer_\d+$", "", wandb_name)


def load_seed_rows(csv_path: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["config"] = df["wandb_name"].map(config_key)
    df["seed"] = seed
    return df


def main():
    root = repo_root()
    frames = [load_seed_rows(root / rel, 42)
        for rel in SEED42_CSVS
        if (root / rel).exists()]

    for seed in NEW_SEEDS:
        csv_path = MULTISEED_DIR / f"seed_{seed}/results/karvonen/all_results.csv"
        if csv_path.exists():
            frames.append(load_seed_rows(csv_path, seed))
        else:
            print(f"  (seed {seed}: no results yet at {csv_path})")

    rows = pd.concat(frames, ignore_index=True)

    dropped = sorted(set(rows["config"]) - set(FIGURE_CONFIGS))
    if dropped:
        print(f"  Dropping non-figure configs: {dropped}")
    rows = rows[rows["config"].isin(FIGURE_CONFIGS)]
    rows = rows.drop_duplicates(subset=["config", "seed"], keep="first")

    out_dir = root / "results/multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "all_seeds.csv", index=False)

    grouped = rows.groupby("config")
    agg = grouped[METRICS].mean()
    for m in METRICS:
        agg[f"{m}_std"] = grouped[m].std(ddof=1)
    agg["n_seeds"] = grouped.size()
    agg["sae_type"] = grouped["sae_type"].first()
    agg = agg.reset_index()
    agg = agg.set_index("config").loc[[c for c in FIGURE_CONFIGS
        if c in agg["config"].values]].reset_index()

    agg.to_csv(out_dir / "aggregate.csv", index=False)
    baseline = agg[agg["config"].isin(BASELINE_CONFIGS)]
    baseline.to_csv(out_dir / "aggregate_baseline.csv", index=False)

    print(f"  {len(rows)} rows across seeds {sorted(rows['seed'].unique())}")
    print(agg[["config", "n_seeds", "l0", "l0_std"]].to_string(index=False))
    print(f"Saved: {out_dir}/all_seeds.csv, aggregate.csv, aggregate_baseline.csv")


if __name__ == "__main__":
    main()
