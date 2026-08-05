#!/bin/bash
# Per-space divergence, one H100 array task per shard. Spaces are independent
# and the driver is skip-if-exists, so tasks need no coordination and a rerun
# only recomputes what is missing. Run after submit_violation_divergence.sbatch
# has written $DIR/acts/.
#SBATCH --job-name=violation_divergence_spaces
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/violation_divergence/%x_%A_%a.out
#SBATCH --error=logs/violation_divergence/%x_%A_%a.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

N_SHARDS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# SPADE_VARIANT=rollout runs the resampled tree instead of the spliced one. Written
# as an if, not `[ -n x ] && ...`, which returns 1 when unset and set -e would kill.
VARIANT_ARG=""
if [ -n "${SPADE_VARIANT:-}" ]; then
    VARIANT_ARG="--variant ${SPADE_VARIANT}"
fi

python -m spade.violation_divergence.compute_divergence \
    --shard "$SLURM_ARRAY_TASK_ID" --n_shards "$N_SHARDS" $VARIANT_ARG
