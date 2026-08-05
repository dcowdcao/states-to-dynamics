#!/bin/bash
#SBATCH --job-name=rollout_geometry
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
mkdir -p logs/violation_divergence

DIR="${SPADE_DIVERGENCE_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_divergence}"

# The rollout tree's per-branch geometry: 20700 rows x 9 spaces. This is the step
# that gets OOM-killed on a login node roughly 12 minutes in, silently and with no
# traceback, which is why it has its own job rather than a bare invocation.
[ -f "$DIR/rollout/acts/resid_post_5.pt" ] || {
    echo "rollout activations missing: run submit_rollout_branches.sbatch, then extract_acts --variant rollout" >&2
    exit 1
}
python analysis/violation_divergence/curvature_discriminant/branch_geometry.py --variant rollout
