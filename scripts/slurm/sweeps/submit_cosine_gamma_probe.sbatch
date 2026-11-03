#!/bin/bash
#SBATCH --job-name=cos_gamma_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/training/cos_gamma_probe_%j.out
#SBATCH --error=logs/training/cos_gamma_probe_%j.err
# ==========================================================================
# Vet a candidate gamma range before spending GPU-days on the multi-seed sweep.
#
# Short (--smoke_test, 100-step) CosineSpaDE runs across a wide log grid, then
# sweep_loss_balance reads L0 / MSE / regularizer share off the checkpoints.
#
# What it can and cannot answer: it confirms the regularizer stays simplex-bounded
# (~0.55) across every decade of gamma, so raising gamma only ever buys loss
# SHARE, never a bigger penalty. It cannot locate the L0 response — at 100 steps
# L0 sits at its init value (~31) for every gamma, because gamma does not shrink
# L0, it throttles how far L0 GROWS toward ~179 over the full 36k steps. Pick the
# grid from sweep_loss_balance on fully trained checkpoints instead.
#
# Usage:
#   cd "$SPADE_REPO" && sbatch scripts/slurm/sweeps/submit_cosine_gamma_probe.sbatch
# ==========================================================================

set -e

source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
OUT="${BASE}/cosine_gamma_probe"
GAMMAS="0.1,1,5,10,20,50,100,300,1000,3000"

cd "${SPADE_REPO:?}"
echo "=== CosineSpaDE gamma range probe on $(hostname) ==="
echo "Gamma grid: ${GAMMAS}"

python -m spade.training.sweep \
    --save_dir "${OUT}" \
    --device cuda:0 \
    --seed 42 \
    --smoke_test \
    --no_wandb \
    --cosine_spade_sweep \
    --cosine_spade_gammas "${GAMMAS}"

python -m spade.training.sweep_loss_balance --sweep_dir "${OUT}"
