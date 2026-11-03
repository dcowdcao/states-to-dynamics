#!/bin/bash
#SBATCH --job-name=bracket_sae_pca
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/bracket/%x_%j.out
#SBATCH --error=logs/bracket/%x_%j.err

set -e
module load python
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket
python analysis/bracket_flip/sae_pca_figure.py
