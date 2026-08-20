#!/bin/bash
#SBATCH --job-name=bracket_readout
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err
# The GPU is for the arms probe_space fits itself (absolute basis, layer -1); the
# mover-relative blocks are loaded from spade.probes.load and cost nothing.

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket_flip

BRACKET_DIR="${SPADE_BRACKET_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_bracket}"
[ -f "$BRACKET_DIR/events/events.csv" ] || {
  echo "no mined events at $BRACKET_DIR: run submit_bracket_flip.sbatch first" >&2
  exit 1
}

python -m spade.bracket_flip.probe_space      # M1/M2 read-out tables
python -m spade.bracket_flip.geometry_tables  # M3/M4 activation-geometry tables
