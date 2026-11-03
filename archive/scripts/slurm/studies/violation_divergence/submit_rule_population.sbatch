#!/bin/bash
#SBATCH --job-name=rule_population
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/violation_divergence/%x_%A_%a.out
#SBATCH --error=logs/violation_divergence/%x_%A_%a.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
# The env's editable install points at whichever clone was installed, so a
# worktree run needs its own src first; in the main clone this is the same path.
export PYTHONPATH="${SPADE_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/violation_divergence

# The rule population in all nine spaces, per tree: the per-rule turning-angle
# curves panel (a) of the rule_curvature figure draws, and the per-game 21-D
# ReStraV descriptors the straightness feature_pca figure draws the rules from.
# The population is the rule-complete games forking at one t0 (--t0, default 20),
# 2000 of them since extend_branches drew that many at t0=20; the descriptors are
# every rule-complete game. Either product encodes ~2000 games x ~9 branches x K
# rollouts per space -- a login node with one core takes ~3 h and is capped at 8 GB;
# an H100 does the spliced tree in minutes and the rollout tree in about half an hour.
# Resumable per product and per t0, both of which are in the npz name: a space is
# re-encoded only for the npz it is missing.
# One array task per tree: they read different tensors and share nothing, so
# running them in sequence only doubled the wall clock.
TREE=(branches rollout)
python -m spade.violation_divergence.rule_curvature --tree "${TREE[$SLURM_ARRAY_TASK_ID]}" "$@"
