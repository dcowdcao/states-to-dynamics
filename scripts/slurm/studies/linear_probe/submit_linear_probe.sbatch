#!/bin/bash
# CPU, not kempner_h100: 45 quick probes plus one full probe is seven minutes on 16 cores, and
# the login node gives one. A GPU rerun is fine but will differ in the last 1e-4.
#SBATCH --job-name=linear_probe
#SBATCH --account=ba_lab
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/linear_probe/%x_%j.out
#SBATCH --error=logs/linear_probe/%x_%j.err

set -e
module load python
source activate spade_env
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SPADE_REPO:?}"
mkdir -p logs/linear_probe

# The editable install points at one checkout and these scripts run by path out of $SPADE_REPO,
# so the package and the sibling imports have to come from this tree or a worktree silently
# uses the other one.
export PYTHONPATH="$SPADE_REPO/src:$SPADE_REPO/analysis/linear_probe"

python analysis/linear_probe/train_linear_probe.py --device cpu
python analysis/linear_probe/parity_fix_effect.py --device cpu
python analysis/linear_probe/make_figures.py
python analysis/linear_probe/make_figures.py --camera-ready
