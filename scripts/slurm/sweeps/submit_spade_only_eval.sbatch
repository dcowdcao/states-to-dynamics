#!/bin/bash
#SBATCH --job-name=spade_eval
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/karvonen/spade_eval_%j.out
#SBATCH --error=logs/karvonen/spade_eval_%j.err
# ==========================================================================
# Evaluate ONLY the retrained SpaDE SAE using Karvonen's metrics.
#
# Uses the existing extraction data (data/activations/) — no need to
# re-extract since the evaluation data is model-dependent, not SAE-dependent.
#
# Evaluation uses 59,000 samples (1,000 games x 59 turns) to match
# Karvonen's exact protocol.
#
# Expected runtime on H100: ~5 minutes
# ==========================================================================

set -e

# ---- Environment setup ----
module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Evaluating retrained SpaDE SAE on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for CKPT_DIR in checkpoints/spade_v2/trainer_*/; do
    SAE_NAME=$(basename "$CKPT_DIR")
    echo ""
    echo "=========================================="
    echo "  Evaluating: ${SAE_NAME}"
    echo "=========================================="
    python -m spade.evaluation.run_eval \
        --checkpoint_dir "$CKPT_DIR" \
        --test_data_dir data/activations \
        --output_dir results/spade_v2 \
        --n_samples 59000 \
        --device auto
done

# ---- Merge with baseline results and generate figure ----
echo ""
echo "--- Merging results with baseline SAEs ---"
python scripts/slurm/sweeps/merge_baseline_results.py --results_dir results/spade_v2

echo "--- Generating comparison figure ---"
python -m spade.figures.figure_karvonen_extension \
    --results_csv results/spade_v2/combined_results.csv \
    --output_dir figures

echo ""
echo "=== SpaDE evaluation + figure complete ==="
echo "Results: results/spade_v2/"
echo "Figure:  figures/Figure_4_karvonen_extension.png"
