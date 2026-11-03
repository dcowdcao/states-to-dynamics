#!/bin/bash
#SBATCH --job-name=spade_revival
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/revival/%x_%j.out
#SBATCH --error=logs/revival/%x_%j.err
# Revive the lost SpaDE lam=1e-3 gam=1e-4 selection (seed 42). Fig 6 evaluates it with
# karvonen_repro/submit_eval.sbatch; the old eval and the geometry/rules figures are archived.

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.sh"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/revival

CKPT=checkpoints/spade_revival

if [ ! -f "$CKPT/trainer_0/ae.pt" ]; then
    python -m spade.training.sweep \
        --save_dir "$CKPT" \
        --spade_sweep --spade_lambdas "1e-3" --spade_gammas "1e-4" \
        --device cuda:0 --no_wandb
fi
