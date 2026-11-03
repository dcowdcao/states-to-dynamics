#!/bin/bash
# Cache the probe fit set in each of the 9 spaces + 2 nulls, and stream the branch window
# through each for L0 and prefix invariance. Skip-if-exists per space, so a rerun only
# fills what is missing; pass --spaces to redo one. Needs the violation-divergence acts.
#SBATCH --job-name=legality_spaces
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/legality_probe

python -m spade.legality_probe.spaces "$@"
