#!/bin/bash
# Encode the probe fit set and the violation-divergence branch window through the 9
# spaces. Skip-if-exists per space, so a rerun only fills what is missing; pass
# --spaces to redo one. Run after the violation-divergence acts exist.
#SBATCH --job-name=legality_spaces
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/legality_probe

python -m spade.legality_probe.spaces "$@"
