#!/bin/bash
# ===========================================================================
# Master pipeline script for the OthelloGPT SAE experiment.
#
# Trains the 8 SAE types through Karvonen's trainSAE() loop, one entrypoint
# per type. Evaluation computes Loss Recovered (Fidelity) via LM intervention.
#
# Usage: SPADE_REPO=<repo root> bash scripts/run_pipeline.sh [stage]
#
# Stages:
#   all       Run entire pipeline (default)
#   train     Train all 8 SAEs (streams data from OthelloGPT)
#   extract   Extract evaluation data (activations + board states)
#   eval      Run Karvonen evaluation on trained SAEs
#   figures   Generate comparison figures
# ===========================================================================

set -euo pipefail

cd "${SPADE_REPO:?}"

SAE_TYPES=(standard panneal gated gated_anneal topk batch_topk spade cosine_spade)
SAVE_DIR=checkpoints/main
RESULTS_DIR=results/main

echo "=============================================="
echo " OthelloGPT SAE Pipeline (Karvonen-matched)"
echo " Experiment dir: ${SPADE_REPO}"
echo "=============================================="

# ---------------------------------------------------------------------------
# Stage: train (all 8 SAEs via Karvonen's trainSAE loop)
# ---------------------------------------------------------------------------
stage_train() {
    echo ""
    echo "=== Stage: Train 8 SAEs (Karvonen-matched) ==="
    mkdir -p logs/training "${SAVE_DIR}"

    for sae_type in "${SAE_TYPES[@]}"; do
        python -m "spade.training.train_${sae_type}" \
            --save_dir "${SAVE_DIR}" \
            --device cuda:0 \
            2>&1 | tee "logs/training/train_${sae_type}.log"
    done

    echo "  Training complete. SAEs saved to: ${SAVE_DIR}"
}

# ---------------------------------------------------------------------------
# Stage: extract (evaluation data: activations + board states)
# ---------------------------------------------------------------------------
stage_extract() {
    echo ""
    echo "=== Stage: Extract Evaluation Data ==="
    mkdir -p logs/extraction data/activations

    python -m spade.extraction.extract_activations \
        --output_dir data/activations \
        --device auto \
        2>&1 | tee logs/extraction/extract.log

    echo "  Extraction complete."
}

# ---------------------------------------------------------------------------
# Stage: eval (Karvonen evaluation on all trained SAEs)
# ---------------------------------------------------------------------------
stage_eval() {
    echo ""
    echo "=== Stage: Karvonen Evaluation ==="
    if [ ! -d "${SAVE_DIR}" ] || [ -z "$(ls -A "${SAVE_DIR}" 2>/dev/null)" ]; then
        echo "ERROR: No trained SAEs found at ${SAVE_DIR}"
        echo "  Run the train stage first."
        exit 1
    fi
    mkdir -p logs/karvonen "${RESULTS_DIR}"

    for ckpt_dir in "${SAVE_DIR}"/trainer_*/; do
        sae_name=$(basename "$ckpt_dir")
        echo "  Evaluating: ${sae_name}"
        python -m spade.evaluation.run_eval \
            --checkpoint_dir "$ckpt_dir" \
            --test_data_dir data/activations \
            --output_dir "${RESULTS_DIR}" \
            --device auto \
            2>&1 | tee "logs/karvonen/karvonen_${sae_name}.log"
    done

    echo "  Karvonen evaluation complete."
}

# ---------------------------------------------------------------------------
# Stage: figures
# ---------------------------------------------------------------------------
stage_figures() {
    echo ""
    echo "=== Stage: Generate Figures ==="
    mkdir -p logs/figures figures

    echo "  Figure 1: Karvonen extension (SAE comparison scatter plots)"
    python -m spade.figures.figure_karvonen_extension \
        --results_csv "${RESULTS_DIR}/all_results.csv" \
        --output_dir figures \
        2>&1 | tee logs/figures/figure4_karvonen.log

    echo "  Figure 2: Diagnostic norm histograms (activation & weight norms)"
    python -m spade.diagnostics.norm_diagnostics \
        --output_dir figures \
        2>&1 | tee logs/figures/diagnostic_norms.log

    echo "  Figures complete."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
STAGE="${1:-all}"

case "$STAGE" in
    all)
        stage_train
        stage_extract
        stage_eval
        stage_figures
        ;;
    train)     stage_train ;;
    extract)   stage_extract ;;
    eval)      stage_eval ;;
    figures)   stage_figures ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: all, train, extract, eval, figures"
        exit 1
        ;;
esac

echo ""
echo "Pipeline stage '${STAGE}' complete."
