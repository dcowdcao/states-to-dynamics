#!/bin/bash
#SBATCH --job-name=l0_matched_profile
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/straightness/%x_%j.out
#SBATCH --error=logs/straightness/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"
mkdir -p logs/straightness

# The profile the change-point studies read, and the all-classes twin quantity_ablation
# scores. Both skip any pick whose file is already on disk.
python analysis/straightness/l0_matched_profile.py "$@"
