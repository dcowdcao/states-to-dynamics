#!/bin/bash
#SBATCH --job-name=cos_sweep_train
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --array=42-46
# ==========================================================================
# One array task per seed (task id == seed). Trains a CosineSpaDE gamma grid in
# a single trainSAE() call, round-robin device-parallel across the 3 H100s, so
# wall time is ~2.5h per config on the busiest GPU (ceil(n_configs / 3) of them).
#
# COSINE_GAMMAS and COSINE_SWEEP are required, not defaulted: trainer indices
# restart at 0 in every sweep dir, so reusing a dir with a different grid would
# silently overwrite existing checkpoints and desync them from their results.
# Submit via run_cosine_sweep_multiseed.sh (sets log paths and dependencies).
# ==========================================================================

set -e

source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
# HF datasets cache lives on NFS; hard flocks fail under concurrent array tasks
export HF_USE_SOFTFILELOCK=1

BASE="$(python -m spade.common.study multiseed)"
SEED="${SLURM_ARRAY_TASK_ID}"

: "${COSINE_GAMMAS:?COSINE_GAMMAS must be set (run_cosine_sweep_multiseed.sh exports it)}"
: "${COSINE_SWEEP:?COSINE_SWEEP must be set (run_cosine_sweep_multiseed.sh exports it)}"

cd "${SPADE_REPO:?}"
echo "=== CosineSpaDE ${COSINE_SWEEP}: seed ${SEED} on $(hostname) ==="
echo "Gamma grid: ${COSINE_GAMMAS}"
nvidia-smi --query-gpu=index,name --format=csv,noheader

python -m spade.training.sweep \
    --save_dir "${BASE}/seed_${SEED}/autoencoders_${COSINE_SWEEP}" \
    --device cuda:0 \
    --seed "${SEED}" \
    --cosine_spade_sweep \
    --cosine_spade_gammas "${COSINE_GAMMAS}"

echo "=== Seed ${SEED} ${COSINE_SWEEP} training complete ==="
