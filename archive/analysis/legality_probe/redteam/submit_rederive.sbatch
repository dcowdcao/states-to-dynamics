#!/bin/bash
# Red-team re-derivation of the legality-probe study, one GPU.
#SBATCH --job-name=legality_redteam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/legality_probe
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH}"

python analysis/legality_probe/redteam/rederive.py "$@"
