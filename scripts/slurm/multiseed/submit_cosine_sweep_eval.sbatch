#!/bin/bash
#SBATCH --job-name=cos_sweep_eval
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
# ==========================================================================
# One array task per CosineSpaDE gamma checkpoint (task id == trainer index)
# for a single seed, passed as SEED in the environment; the array range is set
# from the grid size by run_cosine_sweep_multiseed.sh, which also exports
# COSINE_SWEEP to pick the sweep segment. Karvonen metrics at --n_samples
# 59000, full fidelity. The per-seed all_results.csv append is flock-protected,
# so concurrent tasks are safe.
# ==========================================================================

set -e

module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
# The datasets builder hard-flocks its cache, and flock on NFS dies with
# "Errno 116 Stale file handle" when concurrent array tasks hit it (~1 task per
# array). HF_USE_SOFTFILELOCK is not honored by this datasets/filelock version,
# so give each task a node-local datasets cache instead — streaming=True means
# only metadata lands there. HF_HOME stays on NFS so the LM is not refetched.
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
TRAINER="trainer_${SLURM_ARRAY_TASK_ID}"

: "${SEED:?SEED must be set (run_cosine_sweep_multiseed.sh exports it per eval array)}"
: "${COSINE_SWEEP:?COSINE_SWEEP must be set (run_cosine_sweep_multiseed.sh exports it)}"

cd "${SPADE_REPO:?}"
echo "=== Cosine ${COSINE_SWEEP} eval: seed ${SEED} ${TRAINER} on $(hostname) ==="

python -m spade.evaluation.run_eval \
    --checkpoint_dir "${BASE}/seed_${SEED}/autoencoders_${COSINE_SWEEP}/${TRAINER}" \
    --test_data_dir "${BASE}/data/activations" \
    --output_dir "${BASE}/seed_${SEED}/results/karvonen_${COSINE_SWEEP}" \
    --n_samples 59000 \
    --device auto

echo "=== Seed ${SEED} ${TRAINER} eval complete ==="
