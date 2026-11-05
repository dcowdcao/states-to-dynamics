#!/bin/bash
# Build the move-legality probe corpus and extract its layer-5 activations. The corpus
# walk is CPU (a board replay per position of every accepted game, plus the exact-token
# disjointness check against the violation-divergence tree); the extraction is one batched
# GPU pass over ~185k spliced sequences. Skip-if-exists is not implemented -- a rerun
# rewrites both, and renumbers game_id, so do not rerun over a tree a probe was fit on.
#SBATCH --job-name=move_legality_fit_set
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/move_legality/%x_%j.out
#SBATCH --error=logs/move_legality/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1
# The builder's hard flock dies on NFS; give the job a node-local datasets cache and
# leave HF_HOME on NFS so the LM is not refetched.
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

cd "${SPADE_REPO:?}"
mkdir -p logs/move_legality

# `pip install -e` pinned `spade` to whichever checkout ran `make setup`, so cd alone
# does not decide which tree a worktree's job imports from.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.move_legality.fit_set "$@"
python -m spade.move_legality.extract_acts
