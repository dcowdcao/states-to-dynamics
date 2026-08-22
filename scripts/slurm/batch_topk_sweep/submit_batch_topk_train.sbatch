#!/bin/bash
#SBATCH --job-name=btopk_sweep_train
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
# ==========================================================================
# One array task per seed (task id == seed). Trains a BatchTopK k grid in a single
# trainSAE() pass, round-robin device-parallel across the 4 H100s. Wall time tracks
# trainers-per-GPU, not the total: 10 on 4 GPUs is 3 per GPU, ~6 h.
#
# BATCH_TOPK_KS and BATCH_TOPK_SEGMENT are required, not defaulted: trainer indices
# restart at 0 in every sweep dir, so reusing a dir with a different grid would
# silently overwrite existing checkpoints and desync them from their results.
# Submit via run_batch_topk_sweep.sh (sets log paths, the array range and the eval
# dependency).
# ==========================================================================

set -e

module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
# HF datasets cache lives on NFS; hard flocks fail under concurrent array tasks
export HF_USE_SOFTFILELOCK=1

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
SEED="${SLURM_ARRAY_TASK_ID}"

: "${BATCH_TOPK_KS:?BATCH_TOPK_KS must be set (run_batch_topk_sweep.sh exports it)}"
: "${BATCH_TOPK_SEGMENT:?BATCH_TOPK_SEGMENT must be set (run_batch_topk_sweep.sh exports it)}"

cd "${SPADE_REPO:?}"
# The editable spade install points at the main checkout, so a worktree's own src
# has to come first or the sweep resolves to the wrong tree.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

echo "=== BatchTopK ${BATCH_TOPK_SEGMENT}: seed ${SEED} on $(hostname) ==="
echo "k grid: ${BATCH_TOPK_KS}"
nvidia-smi --query-gpu=index,name --format=csv,noheader

python -m spade.training.sweep \
    --save_dir "${BASE}/seed_${SEED}/autoencoders_${BATCH_TOPK_SEGMENT}" \
    --device cuda:0 \
    --seed "${SEED}" \
    --batch_topk_sweep \
    --batch_topk_ks "${BATCH_TOPK_KS}"

echo "=== Seed ${SEED} ${BATCH_TOPK_SEGMENT} training complete ==="
