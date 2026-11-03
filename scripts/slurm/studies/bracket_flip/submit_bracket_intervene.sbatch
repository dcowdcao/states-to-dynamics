#!/bin/bash
#SBATCH --job-name=bracket_intervene
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

[ -f "$BRACKET_DIR/events/events.csv" ] || {
  echo "no mined events at $BRACKET_DIR: run scripts/slurm/studies/bracket_flip/submit_bracket_flip.sbatch first" >&2
  exit 1
}

# CSVs only. Fig 2(b) is drawn by analysis/bracket_flip/make_figures.py, off the GPU.
python -m spade.bracket_flip.tpr_intervene "$@"
