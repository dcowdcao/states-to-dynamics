#!/bin/bash
#SBATCH --job-name=rule_population
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/violation_divergence/%x_%j.out
#SBATCH --error=logs/violation_divergence/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
# The env's editable install points at whichever clone was installed, so a
# worktree run needs its own src first; in the main clone this is the same path.
export PYTHONPATH="${SPADE_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/violation_divergence

# Panel (a) of the rule_curvature figure: the three rules' turning-angle curves
# over every source game, in all nine spaces, per tree. 2000 games x ~9 branches
# x K rollouts encoded per space -- a login node with one core takes ~3 h and is
# capped at 8 GB. Resumable: a space whose npz exists is skipped.
python -m spade.violation_divergence.rule_curvature --tree branches
python -m spade.violation_divergence.rule_curvature --tree rollout
