#!/bin/bash
# ==========================================================================
# Retrain + evaluate ONLY the SpaDE SAE with updated hyperparameters.
#
# Reuses existing extraction data — no need to re-extract.
# Saves to separate directories so existing results are preserved:
#   checkpoints/spade_v2/   (checkpoints)
#   results/spade_v2/       (eval results)
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/sweeps/run_spade_only.sh
# ==========================================================================

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

cd "${SPADE_REPO:?}"

LOG_DIR="${SPADE_REPO}/logs"

mkdir -p "${LOG_DIR}"/{training,karvonen}
mkdir -p checkpoints/spade_v2 results/spade_v2

echo "=============================================="
echo " SpaDE-Only Retrain + Eval"
echo " Account: ${SBATCH_ACCOUNT}, partition ${SBATCH_PARTITION}"
echo " lambda_init=1e-3, gamma_reg=1e-2"
echo "=============================================="
echo ""

echo "=== Step 1: Retrain SpaDE (~1.5 hrs) ==="
TRAIN_JOB=$(sbatch --parsable \
    --output="${LOG_DIR}/training/spade_retrain_%j.out" \
    --error="${LOG_DIR}/training/spade_retrain_%j.err" \
    scripts/slurm/sweeps/submit_spade_only_train.sbatch)
echo "  Submitted training job: $TRAIN_JOB"

echo "=== Step 2: Evaluate SpaDE (after training, ~5 min) ==="
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB} \
    --output="${LOG_DIR}/karvonen/spade_eval_%j.out" \
    --error="${LOG_DIR}/karvonen/spade_eval_%j.err" \
    scripts/slurm/sweeps/submit_spade_only_eval.sbatch)
echo "  Submitted evaluation job: $EVAL_JOB"

echo ""
echo "=============================================="
echo " Jobs submitted!"
echo "=============================================="
echo ""
echo "Job IDs:"
echo "  Training:    $TRAIN_JOB"
echo "  Evaluation:  $EVAL_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $TRAIN_JOB,$EVAL_JOB --format=JobID,JobName%25,Elapsed,State,MaxRSS"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/training/spade_retrain_${TRAIN_JOB}.out"
echo "  tail -f ${LOG_DIR}/training/spade_retrain_${TRAIN_JOB}.err"
echo "  tail -f ${LOG_DIR}/karvonen/spade_eval_${EVAL_JOB}.out"
echo ""
echo "Results will be at:"
echo "  ${SPADE_REPO}/results/spade_v2/"
