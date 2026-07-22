#!/usr/bin/env bash
# Seed error-bar study. Array 43-46 == seed: one train job (gpu:4, all 14 figure configs via
# --spade_pairs) then a per-seed eval array (gpu:1, chained afterok). Seed 42 is NOT retrained
# — merged at aggregation. Big artifacts on $SPADE_MULTISEED_DIR (netscratch). Skeleton.
set -euo pipefail
cd "$HOME/spade_sae"

# TRAIN=$(sbatch --parsable --array=43-46 submit_train.sbatch)
# for SEED in 43 44 45 46; do
#   SEED=$SEED sbatch --dependency=afterok:${TRAIN}_${SEED} submit_eval.sbatch
# done
# # then, after all finish:
# python -m spade.multiseed.aggregate_seeds && python -m spade.figures.figure_all_configs_combined
