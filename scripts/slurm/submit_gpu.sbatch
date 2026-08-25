#!/bin/bash
#SBATCH --job-name=spade_gpu
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/gpu/%x_%j.out
#SBATCH --error=logs/gpu/%x_%j.err

# One analysis command on one H100. The login node has a single core, so anything that
# encodes through a dictionary belongs here; slurm is also the only way to run several at
# once. Slurm starts the job in the submitting directory, so a worktree runs its own tree.
#
#   mkdir -p logs/gpu   # once: SBATCH --output is opened before this script runs
#   sbatch --job-name=r21_spliced /n/home10/dcao/spade_sae/scripts/slurm/submit_gpu.sbatch \
#       python analysis/violation_divergence/divergence_by_space/measure.py --plane r21 ...
set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
"$@"
