#!/bin/bash
#SBATCH --job-name=karvonen_repro
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/karvonen_repro/repro_%j.out
#SBATCH --error=logs/karvonen_repro/repro_%j.err
# ==========================================================================
# Karvonen's OWN evaluation (circuits/full_pipeline.analyze_sae_groups), run on
# whatever checkpoints KARVONEN_CKPT_GLOB names. Not the study's reimplementation.
#
# KARVONEN_MODE=repro reproduces his published table and is the only mode whose
# SAE-quality columns may be compared to it (PROTOCOL.md section 2: those columns
# are one batch of 25 activations). Use measurement for numbers we report.
#
# Resumable: a checkpoint whose per_checkpoint/<group>/<trainer>/f1_results.csv
# exists is skipped, so requeueing the same command with the same KARVONEN_OUT
# picks up where a timeout stopped. One job per KARVONEN_OUT at a time.
#
#   mkdir -p logs/karvonen_repro          # once per checkout; SLURM will not
#   SPADE_REPO=$PWD \
#   KARVONEN_CKPT_GLOB='/n/netscratch/.../othello-trained_model-layer_5-*/trainer*' \
#   KARVONEN_OUT=/n/netscratch/ba_lab/Lab/$USER/karvonen_repro \
#     sbatch --export=ALL karvonen_repro/submit_eval.sbatch
#
# For the reproduction run, take the checkpoint list from validation_set.csv:
#   KARVONEN_RELEASED_ROOT=/n/netscratch/ba_lab/Lab/$USER/karvonen_released \
#   KARVONEN_OUT=... SPADE_REPO=$PWD sbatch --export=ALL karvonen_repro/submit_eval.sbatch
#
# ~1000 SAEs at his n_inputs=1000 will not fit in one 24h job; submit in slices of
# the glob, or requeue against the same KARVONEN_OUT until the log stops working.
# ==========================================================================

set -euo pipefail

module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

cd "${SPADE_REPO:?}"

# The editable spade install points at the main checkout, so a worktree's own src
# has to come first or load_sae resolves to the wrong tree.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

: "${KARVONEN_OUT:?set to the output directory}"
KARVONEN_MODE="${KARVONEN_MODE:-repro}"
KARVONEN_N_INPUTS="${KARVONEN_N_INPUTS:-1000}"

echo "=== Karvonen repro harness on $(hostname), mode ${KARVONEN_MODE} ==="
nvidia-smi --query-gpu=name --format=csv,noheader

if [ -n "${KARVONEN_RELEASED_ROOT:-}" ]; then
    mapfile -t CHECKPOINTS < <(python -m karvonen_repro.validation_checkpoints \
        --released-root "${KARVONEN_RELEASED_ROOT}")
else
    # Unquoted on purpose: the glob is expanded here, into one argument per checkpoint.
    CHECKPOINTS=(${KARVONEN_CKPT_GLOB:?set this or KARVONEN_RELEASED_ROOT})
fi
echo "${#CHECKPOINTS[@]} checkpoints -> ${KARVONEN_OUT}"

python -m karvonen_repro.evaluate \
    --output-dir "${KARVONEN_OUT}" \
    --mode "${KARVONEN_MODE}" \
    --n-inputs "${KARVONEN_N_INPUTS}" \
    "${CHECKPOINTS[@]}"

echo "=== done: ${KARVONEN_OUT}/f1_results.csv ==="
echo "verdict: python karvonen_repro/compare_to_published.py --harness ${KARVONEN_OUT}/f1_results.csv"
