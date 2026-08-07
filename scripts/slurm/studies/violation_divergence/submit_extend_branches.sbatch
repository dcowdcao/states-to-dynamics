#!/bin/bash
# Append rule-complete source games to the spliced tree, then re-extract its
# activations. The append replays all 2000 existing games from the stream before it
# draws anything, so this is a ~10 min CPU walk followed by the usual batched GPU
# pass; extract_acts has no append mode and rewrites $DIR/acts/ whole, which is why
# it is not guarded by a [ -f ] here the way the first-run script guards it.
#SBATCH --job-name=extend_branches
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
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

# `pip install -e` pinned `spade` to whichever checkout ran `make setup`, so cd
# alone does not decide which tree a worktree's job imports from.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python -m spade.violation_divergence.extend_branches
python -m spade.violation_divergence.extract_acts
