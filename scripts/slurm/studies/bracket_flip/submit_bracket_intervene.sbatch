#!/bin/bash
#SBATCH --job-name=bracket_intervene
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/bracket_flip/%x_%j.out
#SBATCH --error=logs/bracket_flip/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/bracket_flip

# analysis/linear_probe/activations/ is gitignored, so a worktree has to be told
# which checkout carries it (PROBE_REPORT.md 6.9).
export SPADE_LINEAR_PROBE_ACTIVATIONS="${SPADE_LINEAR_PROBE_ACTIVATIONS:-${SPADE_REPO}/analysis/linear_probe/activations}"
BRACKET_DIR="${SPADE_BRACKET_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_bracket}"

[ -f "$BRACKET_DIR/events/events.csv" ] || {
  echo "no mined events at $BRACKET_DIR: run scripts/slurm/studies/bracket_flip/submit_bracket_flip.sbatch first" >&2
  exit 1
}

python -m spade.bracket_flip.tpr_intervene "$@"
python analysis/bracket_flip/intervention_figures.py
python analysis/bracket_flip/intervention_figures.py --camera-ready
