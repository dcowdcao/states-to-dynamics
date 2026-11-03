#!/bin/bash
#SBATCH --job-name=bracket_sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err
# One TPR fit per space is seconds; the cost is encoding 26k layer-5 slices
# through nine encoders (raw + eight SAEs), so all spaces run in one job.

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket_flip

python -m spade.bracket_flip.sae_arm
