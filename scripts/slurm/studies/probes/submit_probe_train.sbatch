#!/bin/bash
# One probe per layer, trained to a validation plateau; spade.probes.train raises if it
# reaches MAX_EPOCHS. Usage: sbatch submit_probe_train.sbatch [board|tpr]
#SBATCH --job-name=probe_train
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/probes/%x_%j.out
#SBATCH --error=logs/probes/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/probes
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.probes.train --arm "${1:-board}" --layers 0-7
