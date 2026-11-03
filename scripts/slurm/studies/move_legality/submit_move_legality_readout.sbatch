#!/bin/bash
# Fit both probes in each of the 11 spaces and score every position of the 34,546-branch
# violation-divergence tree through them. Resumable per space (a space with a
# scores_<space>.json is skipped), so a rerun fills only what is missing and --spaces
# redoes one. Memory holds the branch activations ([34546, 59, 512] fp16, 2 GB) plus one
# space's encoded fit set ([~185k, 8192] fp16, 3 GB) at a time.
#SBATCH --job-name=move_legality_readout
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/move_legality/%x_%j.out
#SBATCH --error=logs/move_legality/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/move_legality

export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.move_legality.readout "$@"
