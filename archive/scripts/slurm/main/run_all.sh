#!/bin/bash
# ==========================================================================
# Master script to run the full OthelloGPT SAE pipeline on SLURM.
#
# BEFORE RUNNING — you must:
#   1. Set scripts/slurm/cluster.env for your cluster and build the env it activates
#   2. Export SPADE_REPO to this checkout
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/main/run_all.sh
#
# Pipeline (with SLURM dependency chaining):
#   Step 1: Train all 8 SAEs               (1 job, ~8-16 hrs on H100)
#   Step 2: Extract evaluation data        (1 job, runs in parallel with training, ~1-2 hrs)
#   Step 3: Evaluation + Figures           (1 job, after both 1 & 2 complete, ~4-8 hrs)
#   Step 4: SpaDE 8-config sweep train     (1 job, after Step 3,    ~2-4 hrs)
#   Step 5: SpaDE sweep eval + figures     (1 job, after Step 4,    ~30-60 min)
#   Step 6: Per-game MSE diagnostic        (1 job, after Step 5,    ~10-20 min)
#
# Total estimated wall time: ~22-32 hours (Steps 1 & 2 overlap)
# ==========================================================================

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

cd "${SPADE_REPO:?}"

LOG_DIR="${SPADE_REPO}/logs"

mkdir -p "${LOG_DIR}"/{training,extraction,karvonen,figures,diagnostics}
mkdir -p checkpoints/main data/activations results/main figures

echo "=============================================="
echo " OthelloGPT SAE Pipeline"
echo " Account: ${SBATCH_ACCOUNT}, partition ${SBATCH_PARTITION}"
echo " Experiment dir: ${SPADE_REPO}"
echo "=============================================="
echo ""

echo "=== Step 1: Train 8 SAEs (Karvonen-matched, ~8-16 hrs) ==="
TRAIN_JOB=$(sbatch --parsable \
    --output="${LOG_DIR}/training/train_%j.out" \
    --error="${LOG_DIR}/training/train_%j.err" \
    scripts/slurm/main/submit_train.sbatch)
echo "  Submitted training job: $TRAIN_JOB"

echo "=== Step 2: Extract evaluation data (parallel with training, ~1-2 hrs) ==="
EXTRACT_JOB=$(sbatch --parsable \
    --output="${LOG_DIR}/extraction/extract_%j.out" \
    --error="${LOG_DIR}/extraction/extract_%j.err" \
    scripts/slurm/main/submit_extract.sbatch)
echo "  Submitted extraction job: $EXTRACT_JOB"

echo "=== Step 3: Evaluation + Figures (depends on Steps 1 & 2) ==="
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB}:${EXTRACT_JOB} \
    --output="${LOG_DIR}/karvonen/karvonen_%j.out" \
    --error="${LOG_DIR}/karvonen/karvonen_%j.err" \
    scripts/slurm/main/submit_eval.sbatch)
echo "  Submitted evaluation job: $EVAL_JOB"

echo "=== Step 4: SpaDE 8-config sweep training (after Step 3) ==="
SWEEP_TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:${EVAL_JOB} \
    --output="${LOG_DIR}/training/spade_sweep_%j.out" \
    --error="${LOG_DIR}/training/spade_sweep_%j.err" \
    scripts/slurm/sweeps/submit_spade_sweep_train.sbatch)
echo "  Submitted SpaDE sweep training job: $SWEEP_TRAIN_JOB"

echo "=== Step 5: SpaDE sweep eval + figures (after Step 4) ==="
SWEEP_EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${SWEEP_TRAIN_JOB} \
    --output="${LOG_DIR}/karvonen/spade_sweep_eval_%j.out" \
    --error="${LOG_DIR}/karvonen/spade_sweep_eval_%j.err" \
    scripts/slurm/sweeps/submit_spade_sweep_eval.sbatch)
echo "  Submitted SpaDE sweep eval job: $SWEEP_EVAL_JOB"

echo "=== Step 6: Per-game MSE diagnostic on trainer_3 (after Step 5) ==="
DIAG_JOB=$(sbatch --parsable \
    --dependency=afterok:${SWEEP_EVAL_JOB} \
    --output="${LOG_DIR}/diagnostics/perf_game_diag_%j.out" \
    --error="${LOG_DIR}/diagnostics/perf_game_diag_%j.err" \
    scripts/slurm/diagnostics/submit_perf_game_diag.sbatch)
echo "  Submitted per-game diagnostic job: $DIAG_JOB"

echo ""
echo "=============================================="
echo " All 6 jobs submitted!"
echo "=============================================="
echo ""
echo "Job IDs:"
echo "  Training:           $TRAIN_JOB"
echo "  Extraction:         $EXTRACT_JOB"
echo "  Evaluation:         $EVAL_JOB"
echo "  SpaDE sweep train:  $SWEEP_TRAIN_JOB"
echo "  SpaDE sweep eval:   $SWEEP_EVAL_JOB"
echo "  Per-game diag:      $DIAG_JOB"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $TRAIN_JOB,$EXTRACT_JOB,$EVAL_JOB --format=JobID,JobName%25,Elapsed,State,MaxRSS"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/training/train_${TRAIN_JOB}.out"
echo "  tail -f ${LOG_DIR}/extraction/extract_${EXTRACT_JOB}.out"
echo "  tail -f ${LOG_DIR}/karvonen/karvonen_${EVAL_JOB}.out"
echo ""
echo "After completion, results will be at:"
echo "  ${SPADE_REPO}/results/main/all_results.csv"
echo "  ${SPADE_REPO}/figures/Figure_4_karvonen_extension.png"
echo "  ${SPADE_REPO}/figures/diagnostic_norm_histograms.png"
