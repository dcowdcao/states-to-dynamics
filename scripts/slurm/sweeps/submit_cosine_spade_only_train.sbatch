#!/bin/bash
#SBATCH --job-name=cosine_spade_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/training/cosine_spade_train_%j.out
#SBATCH --error=logs/training/cosine_spade_train_%j.err
# ==========================================================================
# Train ONLY the CosineSpaDE SAE — 9th SAE type for the Karvonen extension.
#
# CosineSpaDE = SpaDE variant where the encoder scores atoms with
# cosine(x, Ae) instead of -lambda * ||x - Ae||^2, then sparsemax. No lambda
# (cosine is bounded and scale-invariant); gamma_reg is the only sparsity
# hyperparameter. Regularizer is cosine-misalignment-weighted L1 on f.
#
# Hyperparams:
#   gamma_reg = 1.0  (default; cosine misalignment is O(1) per atom vs.
#                    SpaDE's distance reg O(d_model) ~ 500, so this gamma
#                    is ~500x the distance-SpaDE default of 1e-2)
#   lr        = 3e-4
#
# Saves to a separate directory (checkpoints/cosine_spade) so existing
# checkpoints from previous runs are preserved.
#
# Expected runtime on H100: ~1.5 hours (300M tokens, 36621 steps).
# ==========================================================================

set -e

# ---- Environment setup ----
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

export WANDB_MODE=online

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Training CosineSpaDE SAE on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Hyperparams: gamma_reg=1.0, lr=3e-4 (no lambda)"

python -m spade.training.train_cosine_spade \
    --save_dir checkpoints/cosine_spade \
    --device cuda:0

echo "=== CosineSpaDE training complete ==="
