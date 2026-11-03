#!/bin/bash
#SBATCH --job-name=violation_divergence
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
# HF_USE_SOFTFILELOCK is not honored by this datasets/filelock version and the
# builder's hard flock dies on NFS; give the job a node-local datasets cache and
# leave HF_HOME on NFS so the LM is not refetched (PROCESS.md).
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

DIR="$(python -m spade.common.study divergence)"

[ -f "$DIR/branches/branches.pt" ] || bash scripts/build_branch_tree.sh
[ -f "$DIR/acts/logits.pt" ] || python -m spade.violation_divergence.extract_acts
