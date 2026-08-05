#!/bin/bash
#SBATCH --job-name=rollout_branches
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

DIR="${SPADE_DIVERGENCE_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_divergence}"

# Resamples the suffix of the spliced branches; branches.pt is its input and is
# never rewritten. ~36 sequential steps per t0 group against extract_acts' single
# batched pass, times K=4 rollouts, and with no KV cache each step re-runs the
# whole prefix (see the module docstring) -- ~5 PFLOP total, so under 15 min of
# H100 at batch 1024. The hour is model download plus headroom.
[ -f "$DIR/branches/branches.pt" ] || python -m spade.violation_divergence.branches
[ -f "$DIR/rollout/branches/branches.pt" ] || python -m spade.violation_divergence.rollout_branches

# Tokens alone draw nothing. K=4 rollouts per row, so this pass is 4x the spliced
# tree's; the wall clock above covers it.
[ -f "$DIR/rollout/acts/acts.pt" ] || python -m spade.violation_divergence.extract_acts --variant rollout
