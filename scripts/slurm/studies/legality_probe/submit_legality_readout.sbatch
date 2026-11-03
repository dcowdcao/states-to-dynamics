#!/bin/bash
# Read each space's legality probe out over the branch window. Needs the cached encodings
# from submit_legality_spaces.sbatch and the fitted probes from submit_legality_fit.sbatch;
# both carry the fit set's fingerprint and both are checked before anything is decoded.
#SBATCH --job-name=legality_readout
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/legality_probe

python -m spade.legality_probe.readout "$@"
