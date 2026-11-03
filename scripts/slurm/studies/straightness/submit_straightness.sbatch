#!/bin/bash
#SBATCH --job-name=straightness
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/straightness/%x_%j.out
#SBATCH --error=logs/straightness/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/straightness

DIR="${SPADE_STRAIGHTNESS_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_straightness}"

[ -f "$DIR/games/games.pt" ] || python -m spade.straightness.build_games
[ -f "$DIR/traj/resid_post_7.pt" ] || python -m spade.straightness.extract_traj
python -m spade.straightness.compute_geometry
python -m spade.straightness.classify
python analysis/straightness/make_figures.py
