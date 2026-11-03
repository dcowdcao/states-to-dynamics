#!/bin/bash
# The same append on the resampled tree: roll out only the source branches that
# extend_branches added, then re-extract the rollout activations. Both halves are
# hours once the append is thousands of games: ~70k new rows to sample, and a K = 4
# activation pass over ~140k rows that extract_acts rewrites whole. The memory is for
# that pass -- it holds nine [n_rows, 59, 512] fp16 tensors, ~75 GB at 140k rows.
#SBATCH --job-name=extend_rollout_branches
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.violation_divergence.extend_rollout_branches
python -m spade.violation_divergence.extract_acts --variant rollout
