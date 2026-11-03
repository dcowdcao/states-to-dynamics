#!/bin/bash
#SBATCH --job-name=batch_topk_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/training/batch_topk_train_%j.out
#SBATCH --error=logs/training/batch_topk_train_%j.err
# ==========================================================================
# Train ONLY the BatchTopK SAE — 8th SAE type for the Karvonen extension.
#
# Hyperparams (matching TopKTrainer for direct comparison):
#   k = 80
#   auxk_alpha = 1/32
#   threshold_beta = 0.999
#   threshold_start_step = 1000
#   lr auto = 2e-4 / sqrt(8192/16384) ≈ 2.83e-4
#
# Saves to a separate directory (checkpoints/batch_topk) so existing
# checkpoints from the full multi-type run are preserved.
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

echo "=== Training BatchTopK SAE on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Hyperparams: k=80, auxk_alpha=1/32, threshold_beta=0.999"

python -m spade.training.train_batch_topk \
    --save_dir checkpoints/batch_topk \
    --device cuda:0

echo "=== BatchTopK training complete ==="
