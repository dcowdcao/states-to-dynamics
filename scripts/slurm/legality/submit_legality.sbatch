#!/usr/bin/env bash
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# Legality study: build pairs -> extract acts -> compute metrics.  Skeleton.
cd "$HOME/spade_sae"
# python -m spade.legality.build_pairs
# python -m spade.legality.extract_acts
# python -m spade.legality.compute_metrics
