#!/bin/bash
#SBATCH --job-name=branch_norms
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

# Same encode as branch_geometry, keeping only the per-position norm. Writes
# <space>_norm.npz beside <space>.npz and never rewrites the latter.
python analysis/violation_divergence/curvature_discriminant/branch_norms.py "$@"
