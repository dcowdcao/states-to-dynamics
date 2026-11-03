#!/bin/bash
# Pick the game for the one-game/three-rules/nine-spaces figure, then rescore the
# shortlist in the R^21 curvature field. Eight SAE encoders over a ~430-game pool
# is the cost. The scripts are memory-bounded (one game, one space at a time), but
# a login node caps a user at 8 GB and holds no GPU, so this is a job.
#SBATCH --job-name=three_rules_selection
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/violation_divergence

# `pip install -e` pinned `spade` to whichever checkout ran `make setup`, so cd
# alone does not decide which tree a worktree's job imports from.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

SELECT=analysis/violation_divergence/game_selection/three_rules
python "$SELECT/score_three_rules.py"
python "$SELECT/check_r21_rules.py"
