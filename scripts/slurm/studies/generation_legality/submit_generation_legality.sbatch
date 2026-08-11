#!/bin/bash
#SBATCH --job-name=generation_legality
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/generation_legality/%x_%j.out
#SBATCH --error=logs/generation_legality/%x_%j.err

set -e
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
# Node-local datasets cache: the builder's hard flock dies on NFS (PROCESS.md).
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID}"

cd "${SPADE_REPO:?}"
mkdir -p logs/generation_legality

DIR="${SPADE_GENERATION_DIR:-/n/netscratch/ba_lab/Lab/$USER/spade_generation}"

[ -f "$DIR/positions/cats.pt" ]      || python -m spade.generation_legality.positions
[ -f "$DIR/scan/positions.csv" ]     || python -m spade.generation_legality.scan
[ -f "$DIR/rollout/rollouts.csv" ]   || python -m spade.generation_legality.rollout
python -m spade.generation_legality.aggregate
