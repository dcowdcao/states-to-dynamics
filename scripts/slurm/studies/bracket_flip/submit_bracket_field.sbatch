#!/bin/bash
#SBATCH --job-name=bracket_field
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err
set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1
cd "${SPADE_REPO:?}"
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"
mkdir -p logs/bracket_flip
python -m spade.bracket_flip.events --variant field --n_events 4000
python -m spade.bracket_flip.extract_acts --variant field
python -m spade.bracket_flip.probe_space
