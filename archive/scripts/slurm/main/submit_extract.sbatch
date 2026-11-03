#!/bin/bash
#SBATCH --job-name=othello_extract
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/extraction/extract_%j.out
#SBATCH --error=logs/extraction/extract_%j.err
# ==========================================================================
# Extract evaluation data: activations + board states from OthelloGPT.
#
# Streams games from adamkarvonen/othello_45MB_games (Karvonen's eval dataset),
# runs them through OthelloGPT, saves train/test splits of:
#   - Layer-5 residual stream activations [N, 512]
#   - Ground-truth board states [N, 8, 8, 3] (mine/yours encoding)
#   - Blank-mask board states [N, 8, 8, 3]
#
# Resource notes:
#   - Peak system memory ~48GB (full dataset activations accumulate in RAM)
#   - 100K games x 59 turns x 512 dims = ~12GB activation tensor + overhead
#   - Expected runtime on H100: ~1-2 hours
# ==========================================================================

set -e

# ---- Environment setup ----
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Extracting evaluation data on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m spade.extraction.extract_activations \
    --output_dir data/activations \
    --device auto

echo "=== Extraction complete ==="
