#!/bin/bash
#SBATCH --job-name=bracket_readout
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err
# The GPU is for the arms probe_space fits itself (absolute basis, layer -1); the
# mover-relative blocks are loaded from spade.probes.load and cost nothing.

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket_flip

BRACKET_DIR="$(python -m spade.common.study bracket)"
[ -f "$BRACKET_DIR/events/events.csv" ] || {
  echo "no mined events at $BRACKET_DIR: run submit_bracket_flip.sbatch first" >&2
  exit 1
}

python -m spade.bracket_flip.probe_space      # M1/M2 read-out tables
