#!/usr/bin/env bash
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# Eval every checkpoint, then build figures.  Skeleton.
cd "$HOME/spade_sae"
# HF_USE_SOFTFILELOCK=1  # concurrent array tasks vs NFS datasets cache flock
# for ckpt in checkpoints/main/trainer_*; do
#   python -m spade.evaluation.run_eval --checkpoint_dir "$ckpt" \
#     --test_data_dir data/activations --output_dir results/main --device auto
# done
# python -m spade.figures.figure_karvonen_extension --results_csv results/main/all_results.csv --output_dir figures
