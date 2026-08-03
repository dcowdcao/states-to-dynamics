#!/bin/bash
#SBATCH --job-name=othello_karvonen_eval
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/karvonen/karvonen_%j.out
#SBATCH --error=logs/karvonen/karvonen_%j.err
# ==========================================================================
# Evaluate all trained SAEs using Karvonen's metrics:
#   1. Loss Recovered (Fidelity) — LM intervention with nnsight
#   2. L0 (sparsity, f != 0)
#   3. Variance Explained (R^2)
#   4. Coverage (mean best-F1 across board labels, train/test protocol)
#   5. Reconstruction F1 (board reconstruction, train/test protocol)
#
# Karvonen train/test protocol: feature-label assignments are learned on
# training data, then evaluated on held-out test data.
#
# Resource notes:
#   - Loads BOTH train and test activations/board states into memory
#   - F1 tensor: [11, 8192, 8, 8, 3] = ~5.5GB per computation
#   - Each SAE eval loads OthelloGPT for loss_recovered (~30-60 min per SAE)
#   - Sequential evaluation over every trainer_* checkpoint
#   - Expected runtime on H100: ~4-8 hours
#   - Memory set to 96GB to accommodate train+test data + F1 tensors
# ==========================================================================

set -e

# ---- Environment setup ----
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Karvonen Evaluation on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Count total SAEs for progress tracking
TOTAL_SAES=$(ls -d checkpoints/main/trainer_*/ 2>/dev/null | wc -l)
CURRENT=0

for CKPT_DIR in checkpoints/main/trainer_*/; do
    CURRENT=$((CURRENT + 1))
    SAE_NAME=$(basename "$CKPT_DIR")
    echo ""
    echo "=========================================="
    echo "  SAE ${CURRENT}/${TOTAL_SAES}: ${SAE_NAME}"
    echo "=========================================="
    python -m spade.evaluation.run_eval \
        --checkpoint_dir "$CKPT_DIR" \
        --test_data_dir data/activations \
        --output_dir results/main \
        --n_samples 59000 \
        --device auto
done

# Generate figures (no GPU needed, just matplotlib)
echo ""
echo "--- Generating Figure 1: Karvonen extension (SAE comparison) ---"
python -m spade.figures.figure_karvonen_extension \
    --results_csv results/main/all_results.csv \
    --output_dir figures

echo "--- Generating Figure 2: Diagnostic norm histograms ---"
python -m spade.diagnostics.norm_diagnostics \
    --output_dir figures

echo ""
echo "=== All evaluations + figures complete ==="
echo "Results: results/main/all_results.csv"
echo "Figure 1: figures/Figure_4_karvonen_extension.png"
echo "Figure 2: figures/diagnostic_norm_histograms.png"
