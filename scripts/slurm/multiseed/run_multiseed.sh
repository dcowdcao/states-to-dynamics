#!/bin/bash
# ==========================================================================
# Submit the full multi-seed study: seeds 43-46 x 14 SAE configs.
#
#   train array (1 task/seed, 4 GPUs each)
#     -> per-seed eval array (14 tasks, 1 GPU each), chained afterok
#
# Seed 42 is NOT retrained: its existing results are merged at aggregation
# time (spade.multiseed.aggregate_seeds).
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/multiseed/run_multiseed.sh
#
# Storage root (override with SPADE_MULTISEED_DIR):
#   /n/netscratch/ba_lab/Lab/$USER/spade_multiseed
#   NOTE: netscratch has a 90-day retention policy.
# ==========================================================================

set -euo pipefail

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
LOGS="${BASE}/logs"
SEEDS=(43 44 45 46)

cd "${SPADE_REPO:?}"

if [ ! -f "${BASE}/data/activations/test_activations.pt" ]; then
    echo "ERROR: eval data missing at ${BASE}/data/activations/ (stage it first)"
    exit 1
fi

mkdir -p "${LOGS}/train" "${LOGS}/eval"

TRAIN_JOB=$(sbatch --parsable \
    --output="${LOGS}/train/seed_%a_%j.out" \
    --error="${LOGS}/train/seed_%a_%j.err" \
    scripts/slurm/multiseed/submit_train.sbatch)
echo "Training array: ${TRAIN_JOB} (tasks = seeds ${SEEDS[*]})"

for SEED in "${SEEDS[@]}"; do
    EVAL_JOB=$(sbatch --parsable \
        --dependency="afterok:${TRAIN_JOB}_${SEED}" \
        --export=ALL,SEED="${SEED}" \
        --output="${LOGS}/eval/seed_${SEED}_trainer_%a_%j.out" \
        --error="${LOGS}/eval/seed_${SEED}_trainer_%a_%j.err" \
        scripts/slurm/multiseed/submit_eval.sbatch)
    echo "Eval array for seed ${SEED}: ${EVAL_JOB} (after ${TRAIN_JOB}_${SEED})"
done

echo ""
echo "Monitor:  squeue -u \$USER"
echo "When all evals finish, aggregate + plot:"
echo "  python -m spade.multiseed.aggregate_seeds && python -m spade.figures.figure_karvonen_extension --results_csv results/multiseed/aggregate_baseline.csv && python -m spade.figures.figure_all_configs_combined"
