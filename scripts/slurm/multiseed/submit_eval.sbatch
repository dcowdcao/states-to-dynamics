#!/bin/bash
#SBATCH --job-name=sae_multiseed_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-13
# ==========================================================================
# One array task per trainer checkpoint (task id == trainer index 0-13) for
# a single seed, passed as SEED in the environment. Matches the original
# eval protocol: --n_samples 59000, full fidelity. The per-seed
# all_results.csv append is flock-protected, so concurrent tasks are safe.
# Submit via run_multiseed.sh (one eval array per seed, chained after its
# training task).
# ==========================================================================

set -e

source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
# HF datasets cache lives on NFS; hard flocks fail under concurrent array tasks
export HF_USE_SOFTFILELOCK=1

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
TRAINER="trainer_${SLURM_ARRAY_TASK_ID}"

: "${SEED:?SEED must be set (run_multiseed.sh exports it per eval array)}"

cd "${SPADE_REPO:?}"
echo "=== Multi-seed eval: seed ${SEED} ${TRAINER} on $(hostname) ==="

python -m spade.evaluation.run_eval \
    --checkpoint_dir "${BASE}/seed_${SEED}/autoencoders/${TRAINER}" \
    --test_data_dir "${BASE}/data/activations" \
    --output_dir "${BASE}/seed_${SEED}/results/karvonen" \
    --n_samples 59000 \
    --device auto

echo "=== Seed ${SEED} ${TRAINER} eval complete ==="
