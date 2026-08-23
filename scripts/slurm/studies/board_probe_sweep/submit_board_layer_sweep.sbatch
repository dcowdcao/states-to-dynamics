#!/bin/bash
# Read the eight canonical board probes out over the corpus's test games. The GPU is for
# that pass; the bootstrap and the drawing are CPU and cost seconds beside it.
#SBATCH --job-name=board_layer_sweep
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/board_probe_sweep/%x_%j.out
#SBATCH --error=logs/board_probe_sweep/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"
mkdir -p logs/board_probe_sweep

# `pip install -e` pinned `spade` to whichever checkout ran `make setup`, so cd alone
# does not decide which tree a worktree's job imports from.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

python analysis/board_probe_sweep/layer_sweep.py
python analysis/board_probe_sweep/layer_sweep.py --camera_ready
