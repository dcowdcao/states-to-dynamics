#!/bin/bash
# The same append on the resampled tree: roll out only the source branches that
# extend_branches added, then re-extract the rollout activations. Sampling a handful
# of new branches is minutes; the hour is the K = 4 activation pass over all 70k
# rows, which extract_acts rewrites whole.
#SBATCH --job-name=extend_rollout_branches
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.violation_divergence.extend_rollout_branches
python -m spade.violation_divergence.extract_acts --variant rollout
