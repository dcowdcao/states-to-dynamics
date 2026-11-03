#!/bin/bash
#SBATCH --job-name=manifold_capture
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/manifold_capture/%x_%j.out
#SBATCH --error=logs/manifold_capture/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/manifold_capture

DIR="$(python -m spade.common.study manifold)"

[ -f "$DIR/corpus/corpus.pt" ] || python -m spade.manifold_capture.build_corpus
[ -f "$DIR/acts/acts.pt" ] || python -m spade.manifold_capture.extract_acts
python -m spade.manifold_capture.manifold_views
python -m spade.manifold_capture.subspace_capture
python -m spade.manifold_capture.tuning_curves
python -m spade.manifold_capture.receptive_fields
python -m spade.manifold_capture.feature_geometry
python -m spade.manifold_capture.capture_regimes
python -m spade.manifold_capture.landmarks
python analysis/manifold_capture/make_figures.py
