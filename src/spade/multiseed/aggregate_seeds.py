"""Aggregate per-seed eval CSVs into mean/std rows for error-bar figures.

Each seed is a full independent training run landing in its own tree; aggregation is a
post-hoc CSV merge. Seed 42 comes from the in-repo results CSVs; seeds 43-46 from
`$SPADE_MULTISEED_DIR/seed_<n>/results/main/all_results.csv`. Defines the canonical config
lists reused by the legality SAE registry.

Entrypoint: `python -m spade.multiseed.aggregate_seeds`  ->  results/multiseed/{all_seeds,aggregate,aggregate_baseline}.csv
"""

# BASELINE_CONFIGS = [...]   # 8 baseline+BatchTopK types
# SWEEP_CONFIGS = [...]      # 6 SpaDE λ/γ figure points
# FIGURE_CONFIGS = BASELINE_CONFIGS + SWEEP_CONFIGS
# METRICS = [...]

# def config_key(wandb_name): ...          # strip trailing _trainer_N to normalize
# def load_seed_rows(seed): ...
# def main(): ...   # concat seeds, dedup on (config, seed), emit mean + <metric>_std + n_seeds
