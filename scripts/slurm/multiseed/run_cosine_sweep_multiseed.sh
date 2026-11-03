#!/bin/bash
# ==========================================================================
# Multi-seed CosineSpaDE gamma sweep: seeds 42-46 x one gamma grid, for error
# bars on the Karvonen plot. One training array task per seed (3 GPUs each); the
# paper evaluates the checkpoints with karvonen_repro/submit_eval.sbatch, and the
# old per-seed eval arrays are archived.
#
# Each sweep segment gets its own directory because trainer indices restart at 0.
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
# Storage root: study_dir("multiseed") ($SPADE_MULTISEED_DIR, else $SPADE_DATA_ROOT/spade_multiseed).
# ==========================================================================

set -euo pipefail
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

BASE="$(python -m spade.common.study multiseed)"
LOGS="${BASE}/logs"
SEEDS=(42 43 44 45 46)

GAMMAS="${1:-10,20,50,100,200,500,1000,2000,5000,10000}"
SWEEP="${2:-cosine_spade_sweep_highgamma}"
N_GAMMAS=$(awk -F',' '{print NF}' <<< "${GAMMAS}")

cd "${SPADE_REPO:?}"

for SEED in "${SEEDS[@]}"; do
    if [ -d "${BASE}/seed_${SEED}/autoencoders_${SWEEP}" ]; then
        echo "ERROR: ${BASE}/seed_${SEED}/autoencoders_${SWEEP} exists — pick a new"
        echo "       segment name, or the new grid will overwrite trainer_0..N there."
        exit 1
    fi
done

mkdir -p "${LOGS}/cos_train"
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
echo "Monitor:  squeue -u \$USER"
