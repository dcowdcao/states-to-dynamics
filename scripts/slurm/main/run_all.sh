#!/usr/bin/env bash
# Full chained pipeline on FASRC: train + extract -> eval -> figures.  Skeleton.
set -euo pipefail
cd "$HOME/spade_sae"

# TRAIN=$(sbatch --parsable submit_train.sbatch)
# EXTRACT=$(sbatch --parsable submit_extract.sbatch)
# EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN:$EXTRACT submit_eval.sbatch)
# echo "submitted train=$TRAIN extract=$EXTRACT eval=$EVAL"
