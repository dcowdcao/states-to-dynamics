#!/usr/bin/env bash
# SpaDE λ/γ sweep: train the grid -> eval all -> merge with baselines -> regenerate figures.
# Also the home for the CosineSpaDE γ sweep and BatchTopK add-on. Skeleton.
set -euo pipefail
cd "$HOME/spade_sae"

# TRAIN=$(sbatch --parsable submit_spade_sweep_train.sbatch)   # --spade_pairs "λ:γ ..."
# sbatch --dependency=afterok:$TRAIN submit_spade_sweep_eval.sbatch  # -> results/spade_sweep/combined_results.csv
