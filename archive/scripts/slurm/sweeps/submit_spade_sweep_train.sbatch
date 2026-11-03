#!/bin/bash
#SBATCH --job-name=spade_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/training/spade_sweep_%j.out
#SBATCH --error=logs/training/spade_sweep_%j.err
# ==========================================================================
# Train 8 SpaDE SAEs with a hyperparameter sweep over lambda and gamma.
#
# Lambda x Gamma grid (2 x 4 = 8 configs):
#   lambda_init: [1e-3, 1e-4]
#   gamma_reg:   [1e-4, 1e-3, 1e-2, 1e-1]
#
# All 8 train simultaneously on the same activation stream (nearly
# same wall time as training 1).
#
# Expected runtime on H100: ~2-4 hours
# ==========================================================================

set -e

# ---- Environment setup ----
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

export WANDB_MODE=online

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== SpaDE Hyperparameter Sweep on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""
echo "Lambda x Gamma grid (2 x 4 = 8 configs):"
echo "  lambda_init: [1e-3, 1e-4]"
echo "  gamma_reg:   [1e-4, 1e-3, 1e-2, 1e-1]"

python -m spade.training.sweep \
    --save_dir checkpoints/spade_sweep \
    --spade_sweep \
    --spade_lambdas "1e-3,1e-4" \
    --spade_gammas "1e-4,1e-3,1e-2,1e-1" \
    --device cuda:0

echo "=== SpaDE sweep training complete ==="
