#!/bin/bash
# Fit and persist the 4 probe arms in each space. Needs the cached encodings from
# submit_legality_spaces.sbatch; the fingerprint in each row_<space>.json is checked
# against the current fit set before anything is fitted. Run before the read-out, which
# loads these artifacts rather than refitting.
#SBATCH --job-name=legality_fit
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/legality_probe/%x_%j.out
#SBATCH --error=logs/legality_probe/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/legality_probe

python -m spade.legality_probe.fit_probes "$@"
