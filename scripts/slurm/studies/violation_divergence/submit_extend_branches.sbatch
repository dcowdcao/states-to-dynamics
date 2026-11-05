#!/bin/bash
# Append source games forking at t0=20 to the spliced tree, then re-extract its
# activations. The append replays every game already on disk from the stream before
# it draws anything, so this is a long CPU walk (a few thousand streamed games, then
# a board replay per game in verify) followed by the usual batched GPU pass;
# extract_acts has no append mode and rewrites $DIR/acts/ whole, which is why it is
# not guarded by a [ -f ] here the way the first-run script guards it.
#SBATCH --job-name=extend_branches
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1
# HF_USE_SOFTFILELOCK is not honored by this datasets/filelock version and the
# builder's hard flock dies on NFS; give the job a node-local datasets cache and
# leave HF_HOME on NFS so the LM is not refetched.
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

# `pip install -e` pinned `spade` to whichever checkout ran `make setup`, so cd
# alone does not decide which tree a worktree's job imports from.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

# t0 = 20 is rule_curvature.FIGURE_T0, the one position the population panel draws.
python -m spade.violation_divergence.extend_branches --t0 20
python -m spade.violation_divergence.extract_acts
