#!/bin/bash
#SBATCH --job-name=perf_game_diag
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/diagnostics/perf_game_diag_%j.out
#SBATCH --error=logs/diagnostics/perf_game_diag_%j.err
# ==========================================================================
# Per-game activation-norm + SpaDE-MSE diagnostic.
#
# Usage: sbatch submit_perf_game_diag.sbatch [sae_path]
#
# The default checkpoint is the canonical SpaDE point (lambda_init=1e-3,
# gamma_reg=1e-1), which lands at checkpoints/spade_sweep/trainer_3 after a
# default --spade_sweep run (lambdas=1e-3,1e-4 x gammas=1e-4,1e-3,1e-2,1e-1,
# loop order for-lam-for-gam). That index is only correct for that exact grid
# and loop order, so pass the path when running against any other sweep.
#
# Produces 4 PNGs in figures/activation_norms/:
#   plot1_norm_trajectories.png
#   plot2_mean_sq_norm_vs_position.png
#   plot3_mse_trajectories.png
#   plot4_mse_vs_norm_scatter.png
# ==========================================================================

set -e

module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

SAE_PATH="${1:-checkpoints/spade_sweep/trainer_3}"

cd "${SPADE_REPO:?}"
mkdir -p logs/diagnostics figures/activation_norms

echo "=== Per-game diagnostic on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "SAE: ${SAE_PATH}"

python -m spade.diagnostics.activation_norms_per_game \
    --sae_path "${SAE_PATH}" \
    --n_traj 100 --n_pop 2000 --batch_size 64 \
    --device cuda

echo "=== Diagnostic complete — outputs in figures/activation_norms/ ==="
