#!/bin/bash
#SBATCH --job-name=l0_grid_train
#SBATCH --account=kempner_ba_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
# ==========================================================================
# Train one architecture's 20-config L0 grid in a single trainSAE() pass,
# round-robin device-parallel across the 4 H100s. Wall time tracks
# trainers-per-GPU, not the total: 20 on 4 GPUs is 5 per GPU, ~10 h.
#
# L0_ARCH, L0_GRID and L0_SEGMENT are required, not defaulted: trainer indices
# restart at 0 in every sweep dir, so reusing a dir with a different grid would
# silently overwrite existing checkpoints and desync them from their results.
# Submit via run_l0_grid.sh, which owns the grids and the eval dependency.
# ==========================================================================

set -e

module load python
module load cuda/12.4.1-fasrc01
source activate spade_env
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
# HF datasets cache lives on NFS; hard flocks fail under concurrent array tasks
export HF_USE_SOFTFILELOCK=1

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"

: "${L0_ARCH:?spade | cosine_spade | batch_topk}"
: "${L0_GRID:?comma-separated grid (gamma or k)}"
: "${L0_SEGMENT:?sweep directory name}"
SEED="${L0_SEED:-42}"

cd "${SPADE_REPO:?}"
# The editable spade install points at the main checkout, so a worktree's own src
# has to come first or the sweep resolves to the wrong tree.
export PYTHONPATH="${SPADE_REPO}/src:${PYTHONPATH:-}"

case "${L0_ARCH}" in
  # lambda is pinned: gamma is the sparsity dial (L0 = 4.381 * gamma^-0.752,
  # R^2 0.9992), and four decades of lambda_init move L0 by under 2x.
  spade)        FLAGS=(--spade_sweep --spade_lambdas 1e-3 --spade_gammas "${L0_GRID}") ;;
  cosine_spade) FLAGS=(--cosine_spade_sweep --cosine_spade_gammas "${L0_GRID}") ;;
  batch_topk)   FLAGS=(--batch_topk_sweep --batch_topk_ks "${L0_GRID}") ;;
  # Karvonen's five, swept over their own sparsity knob so they can reach the shared L0.
  # spade.training.sweep.BASELINES_SWEPT maps each name to its knob.
  standard|gated|panneal|gated_anneal|topk)
                FLAGS=(--baseline_sweep "${L0_ARCH}" --baseline_values "${L0_GRID}") ;;
  *) echo "unknown L0_ARCH ${L0_ARCH}" >&2; exit 1 ;;
esac

echo "=== ${L0_ARCH} ${L0_SEGMENT}: seed ${SEED} on $(hostname) ==="
echo "grid: ${L0_GRID}"
nvidia-smi --query-gpu=index,name --format=csv,noheader

python -m spade.training.sweep \
    --save_dir "${BASE}/seed_${SEED}/autoencoders_${L0_SEGMENT}" \
    --device cuda:0 \
    --seed "${SEED}" \
    "${FLAGS[@]}"

echo "=== ${L0_ARCH} ${L0_SEGMENT} seed ${SEED} training complete ==="
