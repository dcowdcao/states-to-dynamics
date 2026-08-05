#!/bin/bash
# Fit the 4 probe arms in each space and read the legality probe out over the branch window.
# Needs the cached encodings from submit_legality_spaces.sbatch; the fingerprint in each
# row_<space>.json is checked against the current fit set before anything is fitted.
#SBATCH --job-name=legality_readout
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/legality_probe

python -m spade.legality_probe.readout "$@"
