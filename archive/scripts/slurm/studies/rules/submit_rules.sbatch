#!/bin/bash
#SBATCH --job-name=rules
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/rules/%x_%j.out
#SBATCH --error=logs/rules/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/rules

RULES_DIR="${SPADE_RULES_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_rules}"

[ -f "$RULES_DIR/pairs/items.pt" ] || python -m spade.rules.build_pairs
[ -f "$RULES_DIR/acts/acts_final.pt" ] || python -m spade.rules.extract_acts
[ -f "$RULES_DIR/pairs/flip_labels.pt" ] || python -m spade.rules.flip_labels
python -m spade.rules.embed
python analysis/rules_manifolds/make_figures.py
