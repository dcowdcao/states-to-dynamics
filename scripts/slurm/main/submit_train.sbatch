#!/bin/bash
#SBATCH --job-name=othello_sae_train
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/training/train_%j.out
#SBATCH --error=logs/training/train_%j.err
# ==========================================================================
# Train all 8 SAE types (Standard, PAnneal, Gated, GatedAnneal, TopK,
# BatchTopK, SpaDE, CosineSpaDE) through Karvonen's trainSAE() loop.
#
# Partition: kempner_h100 (H100 GPUs, 3-day max, 96 cores / 1.5TB per node)
# Account:   kempner_ba_lab
#
# Resource notes:
#   - Peak GPU memory ~24GB (model + activation buffer + SAEs at 8192 width)
#   - 300M tokens / 8192 batch = 36,621 steps
#   - Expected runtime on H100: ~8-16 hours
#   - kempner_h100 allows up to 48 CPUs/GPU and 512GB/GPU
#
# Alternative: use kempner_requeue for half fairshare cost (preemptible):
#   #SBATCH --partition=kempner_requeue
#   #SBATCH --requeue
# ==========================================================================

set -e

# ---- Environment setup ----
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

export WANDB_MODE=online

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Training 8 SAEs on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

for SAE_TYPE in standard panneal gated gated_anneal topk batch_topk spade cosine_spade; do
    echo "--- ${SAE_TYPE} ---"
    python -m "spade.training.train_${SAE_TYPE}" \
        --save_dir checkpoints/main \
        --device cuda:0
done

echo "=== Training complete ==="
