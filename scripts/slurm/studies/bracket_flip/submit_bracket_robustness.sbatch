#!/bin/bash
#SBATCH --job-name=bracket_robustness
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err
# Every probe fit of the seed / weight-decay / rank sweep in one job: 68 fits of
# 56,582 parameters on cached activations, which is minutes on an H100.

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket_flip

python -m spade.bracket_flip.robustness
