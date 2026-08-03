#!/bin/bash
#SBATCH --job-name=violation_divergence
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
# HF_USE_SOFTFILELOCK is not honored by this datasets/filelock version and the
# builder's hard flock dies on NFS; give the job a node-local datasets cache and
# leave HF_HOME on NFS so the LM is not refetched (PROCESS.md).
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

DIR="${SPADE_DIVERGENCE_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_divergence}"

[ -f "$DIR/branches/branches.pt" ] || python -m spade.violation_divergence.branches
[ -f "$DIR/acts/logits.pt" ] || python -m spade.violation_divergence.extract_acts
