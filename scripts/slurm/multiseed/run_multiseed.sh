#!/bin/bash
# ==========================================================================
# Train the multi-seed study: seeds 43-46 x 14 SAE configs, one array task per seed
# (4 GPUs each). Seed 42 is NOT retrained. The paper evaluates the checkpoints with
# karvonen_repro/submit_eval.sbatch; the old per-seed eval arrays are archived.
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/multiseed/run_multiseed.sh
#
# Storage root: study_dir("multiseed") ($SPADE_MULTISEED_DIR, else $SPADE_DATA_ROOT/spade_multiseed).
# ==========================================================================

set -euo pipefail
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

BASE="$(python -m spade.common.study multiseed)"
LOGS="${BASE}/logs"
SEEDS=(43 44 45 46)

cd "${SPADE_REPO:?}"

mkdir -p "${LOGS}/train"

TRAIN_JOB=$(sbatch --parsable \
    --output="${LOGS}/train/seed_%a_%j.out" \
    --error="${LOGS}/train/seed_%a_%j.err" \
    scripts/slurm/multiseed/submit_train.sbatch)
echo "Training array: ${TRAIN_JOB} (tasks = seeds ${SEEDS[*]})"
echo "Monitor:  squeue -u \$USER"
