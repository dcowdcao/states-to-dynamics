#!/bin/bash
#SBATCH --job-name=cosine_spade_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/karvonen/cosine_spade_eval_%j.out
#SBATCH --error=logs/karvonen/cosine_spade_eval_%j.err
# ==========================================================================
# Evaluate the CosineSpaDE SAE using Karvonen's metrics.
#
# Reuses the existing extraction data in data/activations/ — extraction is
# model-dependent (OthelloGPT layer 5 resid_post), not SAE-dependent.
#
# Evaluation uses 59,000 samples (1,000 games x 59 turns) to match
# Karvonen's exact protocol.
#
# Expected runtime on H100: ~5 minutes
# ==========================================================================

set -e

# ---- Environment setup ----
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1

# ---- HuggingFace authentication (required to avoid 429 rate-limit on shared cluster IP) ----
# Eval calls load_dataset('taufeeque/othellogpt') inside compute_frac_recovered.
# Run `huggingface-cli login` on the login node once before submitting.
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    echo "HF token loaded from ~/.cache/huggingface/token (length=${#HF_TOKEN})"
else
    echo "WARNING: ~/.cache/huggingface/token not found. Run 'huggingface-cli login' on the login node."
    echo "Without authentication, the HF datasets API may rate-limit (HTTP 429) this cluster IP."
fi

# ---- Working directory ----
cd "${SPADE_REPO:?}"

echo "=== Evaluating CosineSpaDE SAE on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for CKPT_DIR in checkpoints/cosine_spade/trainer_*/; do
    SAE_NAME=$(basename "$CKPT_DIR")
    echo ""
    echo "=========================================="
    echo "  Evaluating: ${SAE_NAME}"
    echo "=========================================="
    python -m spade.evaluation.run_eval \
        --checkpoint_dir "$CKPT_DIR" \
        --test_data_dir data/activations \
        --output_dir results/cosine_spade \
        --n_samples 59000 \
        --device auto
done

echo ""
echo "=== CosineSpaDE evaluation complete ==="
echo "Results: results/cosine_spade/"
