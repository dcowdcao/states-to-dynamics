#!/bin/bash
#SBATCH --job-name=spade_revival
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/revival/%x_%j.out
#SBATCH --error=logs/revival/%x_%j.err
# Revive the lost SpaDE lam=1e-3 gam=1e-4 selection (seed 42) and push it
# through every SAE-comparison analysis: Karvonen eval, the geometry studies,
# and the rules manifolds.

set -e
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"
export PYTHONUNBUFFERED=1
export HF_USE_SOFTFILELOCK=1

cd "${SPADE_REPO:?}"
mkdir -p logs/revival

CKPT=checkpoints/spade_revival
MULTISEED_DIR="$(python -m spade.common.study multiseed)"

if [ ! -f "$CKPT/trainer_0/ae.pt" ]; then
    python -m spade.training.sweep \
        --save_dir "$CKPT" \
        --spade_sweep --spade_lambdas "1e-3" --spade_gammas "1e-4" \
        --device cuda:0 --no_wandb
fi

python -m spade.evaluation.run_eval \
    --checkpoint_dir "$CKPT/trainer_0" \
    --test_data_dir "$MULTISEED_DIR/data/activations" \
    --output_dir results/spade_revival \
    --n_samples 59000 --device auto
python -m spade.evaluation.merge_revival_results
python -m spade.figures.figure_karvonen_extension \
    --results_csv results/spade_revival/combined_results.csv \
    --output_dir figures/spade_revival

python analysis/geometry/l0_per_move/make_l0_per_move.py
python analysis/geometry/mse_per_token_heatmaps/make_mse_per_token_plots.py
python analysis/geometry/embedding_pca_umap/make_embedding_pca_umap.py
python analysis/geometry/feature_locality_heatmaps/make_heatmaps.py
python analysis/geometry/feature_locality_distributions/make_distributions.py
python analysis/geometry/per_square_best_feature_heatmaps/make_per_square_heatmaps.py
python analysis/geometry/sae_directions_pca/make_sae_pca_plot.py
python analysis/geometry/per_move_board_reconstructions/make_per_move_reconstructions.py
python analysis/geometry/per_move_accuracy_multigame/make_per_recon_trajectories.py

python -m spade.rules.embed
python analysis/rules_manifolds/make_figures.py
