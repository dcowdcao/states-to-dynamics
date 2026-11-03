#!/bin/bash
# ==========================================================================
# Multi-seed CosineSpaDE gamma sweep: seeds 42-46 x one gamma grid, for error
# bars on the Karvonen plot.
#
#   train array (1 task/seed, 3 GPUs each)
#     -> per-seed eval array (1 task/gamma, 1 GPU each), chained afterok
#
# Each sweep segment gets its own directory because trainer indices restart at
# 0; spade.multiseed.aggregate_cosine_sweep globs them all back together.
# Segments run so far:
#   cosine_spade_sweep            0.05,0.1,0.2,0.5,1,1.5,2,3,5   (MSE-dominated)
#   cosine_spade_sweep_highgamma  the default grid below         (spans collapse)
#
# The [0.05, 5] grid turned out to sit entirely below the point where the
# sparsity regularizer matters: measured MSE ~10.7 vs reg ~0.78 means parity
# needs gamma ~16 (spade.training.sweep_loss_balance). The default grid here is
# the 1-2-5 log continuation past that point.
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/multiseed/run_cosine_sweep_multiseed.sh
#   SPADE_REPO=<repo root> bash scripts/slurm/multiseed/run_cosine_sweep_multiseed.sh "1,10,100" my_segment
#
# Storage root (override with SPADE_MULTISEED_DIR):
#   /n/netscratch/ba_lab/Lab/$USER/spade_multiseed   (90-day retention)
# ==========================================================================

set -euo pipefail
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
LOGS="${BASE}/logs"
SEEDS=(42 43 44 45 46)

GAMMAS="${1:-10,20,50,100,200,500,1000,2000,5000,10000}"
SWEEP="${2:-cosine_spade_sweep_highgamma}"
N_GAMMAS=$(awk -F',' '{print NF}' <<< "${GAMMAS}")

cd "${SPADE_REPO:?}"

if [ ! -f "${BASE}/data/activations/test_activations.pt" ]; then
    echo "ERROR: eval data missing at ${BASE}/data/activations/ (stage it first)"
    exit 1
fi

for SEED in "${SEEDS[@]}"; do
    if [ -d "${BASE}/seed_${SEED}/autoencoders_${SWEEP}" ]; then
        echo "ERROR: ${BASE}/seed_${SEED}/autoencoders_${SWEEP} exists — pick a new"
        echo "       segment name, or the new grid will overwrite trainer_0..N there."
        exit 1
    fi
done

mkdir -p "${LOGS}/cos_train" "${LOGS}/cos_eval"
echo "Segment: ${SWEEP}"
echo "Gammas:  ${GAMMAS}  (${N_GAMMAS} configs)"

# Exported into this shell rather than passed via --export=VAR=...: sbatch's
# --export list is itself comma-delimited, so a comma-separated gamma grid gets
# split into bogus variable names and only the first value survives.
export COSINE_GAMMAS="${GAMMAS}"
export COSINE_SWEEP="${SWEEP}"

TRAIN_JOB=$(sbatch --parsable \
    --export=ALL \
    --output="${LOGS}/cos_train/${SWEEP}_seed_%a_%j.out" \
    --error="${LOGS}/cos_train/${SWEEP}_seed_%a_%j.err" \
    scripts/slurm/multiseed/submit_cosine_sweep_train.sbatch)
echo "Training array: ${TRAIN_JOB} (tasks = seeds ${SEEDS[*]})"

for SEED in "${SEEDS[@]}"; do
    export SEED
    EVAL_JOB=$(sbatch --parsable \
        --dependency="afterok:${TRAIN_JOB}_${SEED}" \
        --array="0-$((N_GAMMAS - 1))" \
        --export=ALL \
        --output="${LOGS}/cos_eval/${SWEEP}_seed_${SEED}_trainer_%a_%j.out" \
        --error="${LOGS}/cos_eval/${SWEEP}_seed_${SEED}_trainer_%a_%j.err" \
        scripts/slurm/multiseed/submit_cosine_sweep_eval.sbatch)
    echo "Eval array for seed ${SEED}: ${EVAL_JOB} (after ${TRAIN_JOB}_${SEED})"
done

echo ""
echo "Monitor:  squeue -u \$USER"
echo "When all evals finish, aggregate + plot (login node, no GPU):"
echo "  python -m spade.multiseed.aggregate_cosine_sweep"
echo "  python -m spade.figures.figure_karvonen_extension --results_csv results/multiseed/aggregate_cosine_sweep.csv --output_dir figures/cosine_spade_sweep"
echo "  python -m spade.figures.figure_cosine_gamma_response"
