#!/bin/bash
#SBATCH --job-name=legality_frame
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err
set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
cd "${SPADE_REPO:?}"
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH}"
python analysis/legality_probe/redteam/frame_probe.py "$@"
