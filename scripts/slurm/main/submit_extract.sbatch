#!/usr/bin/env bash
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# Extract eval activations + board labels.  Skeleton.
cd "$HOME/spade_sae"
# python -m spade.extraction.extract_activations --output_dir data/activations --device auto
