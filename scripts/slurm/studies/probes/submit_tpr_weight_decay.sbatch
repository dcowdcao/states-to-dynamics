#!/bin/bash
# The TPR arm's weight-decay sweep on validation over the full corpus at layer 5; its
# argmax is what spade.probes.tpr.WEIGHT_DECAY records.
#SBATCH --job-name=tpr_weight_decay
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

python -m spade.probes.tpr_weight_decay --layer 5
