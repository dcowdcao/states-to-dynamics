#!/bin/bash
# The one corpus every probe trains on: OthelloGPT resid_post at all 8 layers over 20,000
# adamkarvonen games, board labels from engine replay. All layers in one forward pass,
# because the forward is the cost. -> $SPADE_PROBES_DIR
#SBATCH --job-name=probe_corpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/probes/%x_%j.out
#SBATCH --error=logs/probes/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

cd "${SPADE_REPO:?}"
mkdir -p logs/probes

# `pip install -e` pinned `spade` to whichever checkout ran `make setup`, so cd alone
# does not decide which tree a worktree's job imports from.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.probes.build_corpus
