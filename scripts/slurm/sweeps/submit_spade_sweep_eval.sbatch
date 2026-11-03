#!/bin/bash
#SBATCH --job-name=spade_sw_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/karvonen/spade_sweep_eval_%j.out
#SBATCH --error=logs/karvonen/spade_sweep_eval_%j.err
# ==========================================================================
# Evaluate all 8 SpaDE sweep SAEs using Karvonen's metrics.
# Merges with baseline results and generates comparison figures.
#
# Lambda x Gamma grid (2 x 4 = 8 configs):
#   lambda_init: [1e-3, 1e-4]
#   gamma_reg:   [1e-4, 1e-3, 1e-2, 1e-1]
#
# Expected runtime on H100: ~30-60 minutes
# ==========================================================================

set -e

# ---- Environment setup ----
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Evaluating SpaDE sweep SAEs on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

TOTAL_SAES=$(ls -d checkpoints/spade_sweep/trainer_*/ 2>/dev/null | wc -l)
CURRENT=0

for CKPT_DIR in checkpoints/spade_sweep/trainer_*/; do
    CURRENT=$((CURRENT + 1))
    SAE_NAME=$(basename "$CKPT_DIR")
    echo ""
    echo "=========================================="
    echo "  SAE ${CURRENT}/${TOTAL_SAES}: ${SAE_NAME}"
    echo "=========================================="
    python -m spade.evaluation.run_eval \
        --checkpoint_dir "$CKPT_DIR" \
        --test_data_dir data/activations \
        --output_dir results/spade_sweep \
        --n_samples 59000 \
        --device auto
done

# ---- Merge with baseline results and generate figure ----
echo ""
echo "--- Merging results with baseline SAEs ---"
python scripts/slurm/sweeps/merge_baseline_results.py --results_dir results/spade_sweep

echo "--- Generating Figure 1: Karvonen extension (SAE comparison) ---"
python -m spade.figures.figure_karvonen_extension \
    --results_csv results/spade_sweep/combined_results.csv \
    --output_dir figures

echo "--- Generating Figure 2: Diagnostic norm histograms ---"
python -m spade.diagnostics.norm_diagnostics \
    --output_dir figures

echo ""
echo "=== SpaDE sweep evaluation + figures complete ==="
echo "Results: results/spade_sweep/"
echo "Figure 1: figures/Figure_4_karvonen_extension.png"
echo "Figure 2: figures/diagnostic_norm_histograms.png"
