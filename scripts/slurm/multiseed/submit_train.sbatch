#!/bin/bash
#SBATCH --job-name=sae_multiseed_train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=43-46
# ==========================================================================
# One array task per seed (task id == seed). Trains all 14 figure configs
# (8 baseline SAEs + 6 SpaDE sweep points) in a single trainSAE() call,
# device-parallel across the 4 H100s.
# Submit via run_multiseed.sh (sets log paths and dependencies).
# ==========================================================================

set -e

source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
# HF datasets cache lives on NFS; hard flocks fail under concurrent array tasks
export HF_USE_SOFTFILELOCK=1

BASE="$(python -m spade.common.study multiseed)"
SEED="${SLURM_ARRAY_TASK_ID}"

cd "${SPADE_REPO:?}"
echo "=== Multi-seed training: seed ${SEED} on $(hostname) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

python -m spade.training.sweep \
    --save_dir "${BASE}/seed_${SEED}/autoencoders" \
    --device cuda:0 \
    --seed "${SEED}" \
    --spade_pairs "8e-8:1e-3,1e-7:1e-3,1e-7:5e-3,1e-7:8e-3,1e-7:1e-2,1e-7:3e-2"

echo "=== Seed ${SEED} training complete ==="
