#!/bin/bash
#SBATCH --job-name=rules_causal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/rules/%x_%j.out
#SBATCH --error=logs/rules/%x_%j.err

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/rules

RULES_DIR="${SPADE_RULES_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_rules}"

[ -f "$RULES_DIR/causal/forfeit_pairs.pt" ] || python -m spade.rules.causal.mine_forfeits
python -m spade.rules.causal.unflip_edit
python -m spade.rules.causal.pair_patch
python -m spade.rules.causal.forfeit_patch
python analysis/rules_causal/make_figures.py
