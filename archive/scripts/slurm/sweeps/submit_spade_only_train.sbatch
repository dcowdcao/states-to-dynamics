#!/bin/bash
#SBATCH --job-name=spade_retrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/training/spade_retrain_%j.out
#SBATCH --error=logs/training/spade_retrain_%j.err
# ==========================================================================
# Retrain ONLY the SpaDE SAE with updated hyperparameters.
#
# Hyperparams:
#   lambda_init = 1e-3  ≈ 1/(2*512), theoretical default
#   gamma_reg   = 1e-2
#
# Saves to a separate directory (checkpoints/spade_v2) so existing
# checkpoints from the full multi-type run are preserved.
#
# Expected runtime on H100: ~1.5 hours
# ==========================================================================

set -e

# ---- Environment setup ----
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

export WANDB_MODE=online

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Retraining SpaDE SAE on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Hyperparams: lambda_init=1e-3, gamma_reg=1e-2"

python -m spade.training.train_spade \
    --save_dir checkpoints/spade_v2 \
    --device cuda:0

echo "=== SpaDE retraining complete ==="
