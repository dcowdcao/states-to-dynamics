#!/bin/bash
#SBATCH --job-name=legality_seeds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err
set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
cd "${SPADE_REPO:?}"
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH}"
python analysis/legality_probe/redteam/seed_probe.py "$@"
