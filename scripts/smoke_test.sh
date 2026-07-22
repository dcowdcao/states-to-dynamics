#!/usr/bin/env bash
# Fast end-to-end sanity check on tiny data (all 4 stages, ~10-15 min), then assert artifacts.
# Auto-detect device (mps/cuda/cpu); WANDB_MODE=offline. Skeleton.
set -euo pipefail

# WANDB_MODE=offline bash scripts/run_pipeline.sh all   # with --smoke_test / small --n_games
# assert: >=8 ae.pt, standardization_params.json, >=8 CSV rows, a figure PNG
echo "smoke_test.sh (skeleton)"
