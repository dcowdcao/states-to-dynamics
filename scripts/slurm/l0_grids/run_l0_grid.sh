#!/bin/bash
# ==========================================================================
# The 20-config L0 grid for one of our three architectures, trained then
# evaluated through Karvonen's own pipeline.
#
#   train job (4 GPUs, 20 trainers in one trainSAE pass)
#     -> eval job over that grid, chained afterok
#
# n = 20 per architecture matches Karvonen's own sweep size. His spacing is
# np.linspace over a hand-picked range, which works because his knobs have a
# 2-7x dynamic range; ours do not (SpaDE needs 206x, CosineSpaDE 126x to span
# his L0 axis) and linspace there puts 19 of 20 points below L0 86. The grids
# below are therefore log-spaced in the knob for the two penalty architectures,
# which reproduces what his linspace actually delivers -- coverage of the L0
# axis -- rather than the mechanism that delivers it. BatchTopK keeps linear
# spacing because k IS L0; its grid is every other value of his own TopK sweep.
# CHOICES.md records the fits these were inverted from.
#
# Evaluation is karvonen_repro, never spade.evaluation.run_eval, and one job
# rather than an array: his per-group results.csv is a fixed path two processes
# would race on.
#
# Usage:
#   SPADE_REPO=<repo root> bash scripts/slurm/l0_grids/run_l0_grid.sh spade
#   SPADE_REPO=<repo root> bash scripts/slurm/l0_grids/run_l0_grid.sh cosine_spade
#   SPADE_REPO=<repo root> bash scripts/slurm/l0_grids/run_l0_grid.sh batch_topk
#   ... optional second arg is the seed (default 42, matching his single-seed sweep)
# ==========================================================================

set -euo pipefail

ARCH="${1:?spade | cosine_spade | batch_topk}"
SEED="${2:-42}"

BASE="${SPADE_MULTISEED_DIR:-/n/netscratch/ba_lab/Lab/${USER}/spade_multiseed}"
REPRO_ROOT="${SPADE_KARVONEN_REPRO_DIR:-/n/netscratch/ba_lab/Lab/${USER}}"
LOGS="${BASE}/logs"

# L0 = 4.381 * gamma^-0.752 inverted over L0 550..10, 20 points log-spaced.
SPADE_GRID="0.00162,0.00214,0.00283,0.00375,0.00497,0.00658,0.00871,0.0115,0.0153,0.0202,0.0267,0.0354,0.0468,0.062,0.0821,0.109,0.144,0.19,0.252,0.334"
# L0 = 16.15 + 167.10/(1+(gamma/19.55)^1.136) inverted over L0 173..26.
# The floor 16.1 and ceiling 183.2 are the architecture's, not the grid's: a
# bounded cosine score with no inverse temperature cannot leave that range.
COSINE_GRID="1.74,4.49,7.33,10.4,13.7,17.4,21.4,26,31.1,36.9,43.6,51.3,60.3,70.9,83.6,99.1,118,143,175,220"
# Every other value of Karvonen's own TopK grid (his: k=10..400 step 10, n=40).
BATCH_TOPK_GRID="20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400"

case "${ARCH}" in
  spade)        GRID="${SPADE_GRID}" ;;
  cosine_spade) GRID="${COSINE_GRID}" ;;
  batch_topk)   GRID="${BATCH_TOPK_GRID}" ;;
  *) echo "unknown architecture ${ARCH}" >&2; exit 1 ;;
esac

SEGMENT="${ARCH}_l0_grid"
N=$(awk -F',' '{print NF}' <<< "${GRID}")

cd "${SPADE_REPO:?}"

if [ -d "${BASE}/seed_${SEED}/autoencoders_${SEGMENT}" ]; then
    echo "ERROR: ${BASE}/seed_${SEED}/autoencoders_${SEGMENT} exists — move it, or the"
    echo "       new grid overwrites trainer_0..N there and desyncs them from their results."
    exit 1
fi

mkdir -p "${LOGS}/l0_grid" logs/karvonen_repro
echo "${ARCH}: ${N} configs, seed ${SEED}"
echo "grid: ${GRID}"

export L0_ARCH="${ARCH}" L0_GRID="${GRID}" L0_SEGMENT="${SEGMENT}" L0_SEED="${SEED}"

TRAIN_JOB=$(sbatch --parsable \
    --job-name="${ARCH}_train" \
    --export=ALL \
    --output="${LOGS}/l0_grid/${SEGMENT}_seed_${SEED}_%j.out" \
    --error="${LOGS}/l0_grid/${SEGMENT}_seed_${SEED}_%j.err" \
    scripts/slurm/l0_grids/submit_l0_grid_train.sbatch)
echo "train: ${TRAIN_JOB}"

export KARVONEN_CKPT_GLOB="${BASE}/seed_${SEED}/autoencoders_${SEGMENT}/trainer_*"
export KARVONEN_OUT="${REPRO_ROOT}/karvonen_repro_${SEGMENT}"
export KARVONEN_MODE=repro

EVAL_JOB=$(sbatch --parsable \
    --job-name="${ARCH}_eval" \
    --dependency="afterok:${TRAIN_JOB}" \
    --export=ALL \
    karvonen_repro/submit_eval.sbatch)
echo "eval:  ${EVAL_JOB} (afterok ${TRAIN_JOB}) -> ${KARVONEN_OUT}"
