#!/bin/bash
#SBATCH --job-name=bracket_flip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket_flip

BRACKET_DIR="${SPADE_BRACKET_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_bracket}"

[ -f "$BRACKET_DIR/events/events.csv" ] || python -m spade.bracket_flip.events
python -m spade.bracket_flip.extract_acts
