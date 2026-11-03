#!/bin/bash
# ============================================================================
# Local Smoke Test — Full pipeline with tiny data
# ============================================================================
#
# Runs ALL four pipeline stages with minimal data to verify end-to-end
# correctness before deploying to a GPU cluster:
#
#   Stage 1: Train    — 8 SAEs for 100 steps (~3-4 min)
#   Stage 2: Extract  — 100 games of activations + board states (~1-2 min)
#   Stage 3: Eval     — All 8 SAEs: fidelity + coverage + F1 (~5-8 min)
#   Stage 4: Figures  — 2x2 comparison plot (<10 sec)
#
# Expected total runtime: ~10-15 minutes (depends on device)
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/smoke_test.sh
#
# To clean up:
#   SPADE_REPO=<repo root> bash scripts/clean_smoke_test.sh --force
#
# ============================================================================

set -e

cd "${SPADE_REPO:?}"

SAE_TYPES=(standard panneal gated gated_anneal topk batch_topk spade cosine_spade)
SAVE_DIR=checkpoints/main
DATA_DIR=data/activations
RESULTS_DIR=results/main
FIGURES_DIR=figures
LOGS_DIR=logs

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SMOKE TEST]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
fail() { echo -e "${RED}[FAILED]${NC} $1"; exit 1; }

# Auto-detect device
if python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
elif python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
else
    DEVICE="cpu"
fi
log "Using device: $DEVICE"

# Create directories
mkdir -p "${LOGS_DIR}"/{training,extraction,karvonen,figures}
mkdir -p "${SAVE_DIR}" "${DATA_DIR}" "${RESULTS_DIR}" "${FIGURES_DIR}"

export WANDB_MODE=offline

STAGE_START=$(date +%s)

# ============================================================================
# Stage 1: Train all 8 SAEs (100 steps each)
# ============================================================================
log "=========================================="
log "Stage 1/4: Training 8 SAEs (100 steps)"
log "=========================================="

# One call over the eight configs, so trainer_0..7 land in one directory in the order the
# registry expects. The per-type entrypoints each start at trainer_0 in their own dir.
python -m spade.training.sweep \
    --save_dir "${SAVE_DIR}" \
    --device "$DEVICE" \
    --smoke_test \
    --no_wandb \
    2>&1 | tee "${LOGS_DIR}/training/smoke_train.log"

# Verify: 8 checkpoints + standardization params
N_CKPT=$(find "${SAVE_DIR}" -name 'ae.pt' 2>/dev/null | wc -l | tr -d ' ')
[[ "$N_CKPT" -ge 8 ]] || fail "Expected 8 SAE checkpoints, found $N_CKPT"
[[ -f "${SAVE_DIR}/standardization_params.json" ]] || fail "Missing standardization_params.json"
log "Stage 1 PASSED: $N_CKPT checkpoints + standardization_params.json"

STAGE1_END=$(date +%s)
log "  Time: $(( STAGE1_END - STAGE_START ))s"

# ============================================================================
# Stage 2: Extract evaluation data (100 games)
# ============================================================================
log "=========================================="
log "Stage 2/4: Extracting evaluation data (100 games)"
log "=========================================="

python -m spade.extraction.extract_activations \
    --n_games 100 \
    --batch_size 32 \
    --output_dir "${DATA_DIR}" \
    --device "$DEVICE" \
    2>&1 | tee "${LOGS_DIR}/extraction/smoke_extract.log"

# Verify: required train + test files exist (eval needs both for train/test protocol)
for REQUIRED_FILE in train_activations.pt train_board_states.pt test_activations.pt test_board_states.pt metadata.json; do
    [[ -f "${DATA_DIR}/${REQUIRED_FILE}" ]] || fail "Missing ${REQUIRED_FILE}"
done
log "Stage 2 PASSED: train + test data extracted"

STAGE2_END=$(date +%s)
log "  Time: $(( STAGE2_END - STAGE1_END ))s"

# ============================================================================
# Stage 3: Evaluate all 8 SAEs
# ============================================================================
log "=========================================="
log "Stage 3/4: Evaluating 8 SAEs (fidelity + board-state metrics)"
log "=========================================="

N_EVAL=0
for CKPT_DIR in "${SAVE_DIR}"/trainer_*/; do
    SAE_NAME=$(basename "$CKPT_DIR")
    log "  Evaluating: ${SAE_NAME}"
    python -m spade.evaluation.run_eval \
        --checkpoint_dir "$CKPT_DIR" \
        --test_data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}" \
        --n_fidelity_batches 2 \
        --fidelity_batch_size 8 \
        --device "$DEVICE" \
        2>&1 | tee "${LOGS_DIR}/karvonen/smoke_eval_${SAE_NAME}.log"
    N_EVAL=$((N_EVAL + 1))
done

# Verify: CSV exists with all 8 rows
[[ -f "${RESULTS_DIR}/all_results.csv" ]] || fail "Missing all_results.csv"
N_ROWS=$(tail -n +2 "${RESULTS_DIR}/all_results.csv" | wc -l | tr -d ' ')
[[ "$N_ROWS" -ge 8 ]] || fail "Expected 8 rows in all_results.csv, found $N_ROWS"

# Verify: frac_recovered is populated (not empty) for at least one SAE
HAS_FIDELITY=$(awk -F',' 'NR>1 && $6 != ""' "${RESULTS_DIR}/all_results.csv" | wc -l | tr -d ' ')
[[ "$HAS_FIDELITY" -ge 1 ]] || fail "No frac_recovered values in all_results.csv"

log "Stage 3 PASSED: $N_EVAL SAEs evaluated, $N_ROWS rows in CSV, $HAS_FIDELITY with fidelity"

STAGE3_END=$(date +%s)
log "  Time: $(( STAGE3_END - STAGE2_END ))s"

# ============================================================================
# Stage 4: Generate figures
# ============================================================================
log "=========================================="
log "Stage 4/4: Generating comparison figures"
log "=========================================="

python -m spade.figures.figure_karvonen_extension \
    --results_csv "${RESULTS_DIR}/all_results.csv" \
    --output_dir "${FIGURES_DIR}" \
    2>&1 | tee "${LOGS_DIR}/figures/smoke_figure.log"

# Verify: figure file exists
[[ -f "${FIGURES_DIR}/Figure_4_karvonen_extension.png" ]] || fail "Missing Figure_4_karvonen_extension.png"
log "Stage 4 PASSED: Figure generated"

STAGE4_END=$(date +%s)
log "  Time: $(( STAGE4_END - STAGE3_END ))s"

# ============================================================================
# Summary
# ============================================================================
TOTAL_TIME=$(( STAGE4_END - STAGE_START ))
echo ""
echo "============================================================"
log "ALL 4 STAGES PASSED  (${TOTAL_TIME}s total)"
echo "============================================================"
echo ""
echo "Artifacts:"
echo "  checkpoints/main/    - 8 trained SAE checkpoints"
echo "  data/activations/    - extracted test activations + board states"
echo "  results/main/        - per-SAE metrics + all_results.csv"
echo "  figures/             - Figure_4_karvonen_extension.png"
echo "  logs/                - per-stage logs"
echo ""
echo "To inspect results:"
echo "  cat results/main/all_results.csv"
echo "  open figures/Figure_4_karvonen_extension.png"
echo ""
echo "To clean up:"
echo "  bash scripts/clean_smoke_test.sh --force"
echo ""
