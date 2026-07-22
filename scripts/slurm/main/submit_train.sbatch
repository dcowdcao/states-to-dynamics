#!/usr/bin/env bash
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
# Train all 8 SAE types.  Skeleton.
cd "$HOME/spade_sae"
# module load python cuda/12.4.1-fasrc01; source activate spade_env
# for t in standard panneal gated gated_anneal topk batch_topk spade cosine_spade; do
#   python -m spade.training.train_$t --save_dir checkpoints/main --device cuda:0
# done
