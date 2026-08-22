#!/bin/bash
# ==========================================================================
# BatchTopK k sweep: put the one architecture with no L0 curve onto Karvonen's axis.
#
#   train array (1 task/seed, 4 GPUs, 10 trainers in one trainSAE pass)
#     -> ONE eval job over every seed, chained afterok
#
# The eval is a single job, not an array, because karvonen_repro.evaluate takes one
# process per output dir: his per-group results.csv is a fixed path two tasks would
# race on. 50 checkpoints at ~1 min each fits a 24 h job many times over.
#
# Evaluation is karvonen_repro, never spade.evaluation.run_eval — the latter omits his
# >=0.95-precision / >=10-firing feature gate and lands ~0.2 absolute F1 away. The
# l0_coverage sweep was launched against run_eval once and its eval arrays had to be
# cancelled mid-flight.
#
# Usage — pilot seed 42 first and check L0 tracks k before spending the rest:
#   SPADE_REPO=<repo root> bash scripts/slurm/batch_topk_sweep/run_batch_topk_sweep.sh 42
# then:
#   SPADE_REPO=<repo root> bash scripts/slurm/batch_topk_sweep/run_batch_topk_sweep.sh 43,44,45,46
#
# Storage roots (override with SPADE_MULTISEED_DIR / SPADE_KARVONEN_REPRO_DIR):
#   /n/netscratch/ba_lab/Lab/$USER/spade_multiseed             (90-day retention)
#   /n/netscratch/ba_lab/Lab/$USER/karvonen_repro_batch_topk_sweep
# ==========================================================================

set -euo pipefail

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
REPRO_ROOT="${SPADE_KARVONEN_REPRO_DIR:-/n/netscratch/ba_lab/Lab/${USER}}"
LOGS="${BASE}/logs"

SEED_LIST="${1:-42,43,44,45,46}"
SEGMENT="${2:-batch_topk_sweep}"
# k sets L0 directly for this architecture, so the k grid IS the L0 axis. 1.4x steps
# from 10 to 400 covers Karvonen's whole range (his TopK sweep is k=10..400) and puts
# four points inside the L0 60-160 band the comparison is read in. 280 and 400 sit
# past the expected coverage peak on purpose: without the far side a peak cannot be
# shown to be one.
KS="${BATCH_TOPK_KS:-10,20,40,60,80,110,140,200,280,400}"

IFS=',' read -ra SEEDS <<< "${SEED_LIST}"
N_KS=$(awk -F',' '{print NF}' <<< "${KS}")

cd "${SPADE_REPO:?}"

for SEED in "${SEEDS[@]}"; do
    if [ -d "${BASE}/seed_${SEED}/autoencoders_${SEGMENT}" ]; then
        echo "ERROR: ${BASE}/seed_${SEED}/autoencoders_${SEGMENT} exists — pick a new"
        echo "       segment name, or the new grid will overwrite trainer_0..N there."
        exit 1
    fi
done

mkdir -p "${LOGS}/btopk_train" logs/karvonen_repro
echo "Segment: ${SEGMENT}"
echo "Seeds:   ${SEEDS[*]}"
echo "k grid:  ${KS}  (${N_KS} configs)"

# Exported into this shell rather than passed via --export=VAR=...: sbatch's --export
# list is itself comma-delimited, so a comma-separated grid gets split into bogus
# variable names and only the first value survives.
export BATCH_TOPK_KS="${KS}"
export BATCH_TOPK_SEGMENT="${SEGMENT}"

TRAIN_JOB=$(sbatch --parsable \
    --array="$(IFS=,; echo "${SEEDS[*]}")" \
    --export=ALL \
    --output="${LOGS}/btopk_train/${SEGMENT}_seed_%a_%j.out" \
    --error="${LOGS}/btopk_train/${SEGMENT}_seed_%a_%j.err" \
    scripts/slurm/batch_topk_sweep/submit_batch_topk_train.sbatch)
echo "Training array: ${TRAIN_JOB} (tasks = seeds ${SEEDS[*]})"

# One glob per seed, space-separated. submit_eval.sbatch expands this unquoted, which
# word-splits then globs each word — brace expansion would not fire there, because bash
# expands braces before it expands the variable that holds them.
GLOBS=()
for SEED in "${SEEDS[@]}"; do
    GLOBS+=("${BASE}/seed_${SEED}/autoencoders_${SEGMENT}/trainer_*")
done
export KARVONEN_CKPT_GLOB="${GLOBS[*]}"
export KARVONEN_OUT="${REPRO_ROOT}/karvonen_repro_${SEGMENT}"
export KARVONEN_MODE=repro

EVAL_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    --export=ALL \
    karvonen_repro/submit_eval.sbatch)
echo "Eval job: ${EVAL_JOB} (after the whole array) -> ${KARVONEN_OUT}"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "When the eval finishes:"
echo "  python -m spade.l0_operating_point.load_points"
