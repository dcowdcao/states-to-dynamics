#!/bin/bash
# ==========================================================================
# SpaDE Lambda x Gamma Sweep — Train 8 configs + Evaluate + Figures
#
# Lambda x Gamma grid (2 x 4 = 8 configs):
#   lambda_init: [1e-3, 1e-4]
#   gamma_reg:   [1e-4, 1e-3, 1e-2, 1e-1]
#
# Reuses existing extraction data. Saves to separate directories:
#   checkpoints/spade_sweep/   (checkpoints)
#   results/spade_sweep/       (eval results + combined CSV)
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/sweeps/run_spade_sweep.sh
# ==========================================================================

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

cd "${SPADE_REPO:?}"

LOG_DIR="${SPADE_REPO}/logs"

mkdir -p "${LOG_DIR}"/{training,karvonen}
mkdir -p checkpoints/spade_sweep results/spade_sweep

echo "=============================================="
echo " SpaDE Lambda x Gamma Sweep"
echo " Account: ${SBATCH_ACCOUNT}, partition ${SBATCH_PARTITION}"
echo "=============================================="
echo " Lambda x Gamma grid (2 x 4 = 8 configs):"
echo "   lambda_init: [1e-3, 1e-4]"
echo "   gamma_reg:   [1e-4, 1e-3, 1e-2, 1e-1]"
echo "=============================================="
echo ""

echo "=== Step 1: Train 8 SpaDE configs (~2-4 hrs) ==="
TRAIN_JOB=$(sbatch --parsable \
    --output="${LOG_DIR}/training/spade_sweep_%j.out" \
    --error="${LOG_DIR}/training/spade_sweep_%j.err" \
    scripts/slurm/sweeps/submit_spade_sweep_train.sbatch)
echo "  Submitted training job: $TRAIN_JOB"

echo "=== Step 2: Evaluate + Figures (after training, ~30-60 min) ==="
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB} \
    --output="${LOG_DIR}/karvonen/spade_sweep_eval_%j.out" \
    --error="${LOG_DIR}/karvonen/spade_sweep_eval_%j.err" \
    scripts/slurm/sweeps/submit_spade_sweep_eval.sbatch)
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
echo "  tail -f ${LOG_DIR}/training/spade_sweep_${TRAIN_JOB}.out"
echo "  tail -f ${LOG_DIR}/training/spade_sweep_${TRAIN_JOB}.err"
echo "  tail -f ${LOG_DIR}/karvonen/spade_sweep_eval_${EVAL_JOB}.out"
echo ""
echo "Results will be at:"
echo "  ${SPADE_REPO}/results/spade_sweep/combined_results.csv"
echo "  ${SPADE_REPO}/figures/Figure_4_karvonen_extension.png"
echo "  ${SPADE_REPO}/figures/diagnostic_norm_histograms.png"
