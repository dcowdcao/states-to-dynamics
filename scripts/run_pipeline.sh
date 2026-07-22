#!/usr/bin/env bash
# Local end-to-end driver.  Usage: bash scripts/run_pipeline.sh {all|extract|train|eval|figures}
# Skeleton — port the real commands from the original shell/run_pipeline.sh.
set -euo pipefail

stage="${1:-all}"

# extract)  python -m spade.extraction.extract_activations --output_dir data/activations --device auto
# train)    for t in standard panneal gated gated_anneal topk batch_topk spade cosine_spade; do
#             python -m spade.training.train_$t --save_dir checkpoints/main --device cuda:0
#           done
# eval)     for ckpt in checkpoints/main/trainer_*; do
#             python -m spade.evaluation.run_eval --checkpoint_dir "$ckpt" \
#               --test_data_dir data/activations --output_dir results/main --device auto
#           done
# figures)  python -m spade.figures.figure_karvonen_extension --results_csv results/main/all_results.csv --output_dir figures
#           python -m spade.diagnostics.norm_diagnostics --output_dir figures

echo "run_pipeline.sh: stage=$stage (skeleton)"
