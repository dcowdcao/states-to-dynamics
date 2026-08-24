#!/bin/bash
#SBATCH --job-name=detector_sae_geom
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err
set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
cd "${SPADE_REPO:?}"
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"
mkdir -p logs/violation_divergence
# One npz per space beside raw's; raw is skipped if already written.
python analysis/violation_divergence/illegal_move_detector/extract_geometry.py "$@"
