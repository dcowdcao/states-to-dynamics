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
source "${SPADE_REPO:?}/scripts/slurm/cluster.env"

ARCH="${1:?spade | cosine_spade | batch_topk | standard | gated | panneal | gated_anneal | topk}"
SEED="${2:-42}"

BASE="$(python -m spade.common.study multiseed)"
REPRO_ROOT="${SPADE_KARVONEN_REPRO_DIR:-$(python -m spade.common.study)}"
LOGS="${BASE}/logs"

# L0 = 4.381 * gamma^-0.752 inverted over L0 550..10, 20 points log-spaced.
SPADE_GRID="0.00162,0.00214,0.00283,0.00375,0.00497,0.00658,0.00871,0.0115,0.0153,0.0202,0.0267,0.0354,0.0468,0.062,0.0821,0.109,0.144,0.19,0.252,0.334"
# L0 = 16.15 + 167.10/(1+(gamma/19.55)^1.136) inverted over L0 173..26.
# The floor 16.1 and ceiling 183.2 are the architecture's, not the grid's: a
# bounded cosine score with no inverse temperature cannot leave that range.
COSINE_GRID="1.74,4.49,7.33,10.4,13.7,17.4,21.4,26,31.1,36.9,43.6,51.3,60.3,70.9,83.6,99.1,118,143,175,220"
# Every other value of Karvonen's own TopK grid (his: k=10..400 step 10, n=40).
BATCH_TOPK_GRID="20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400"

# Karvonen's five. His penalty scale does NOT transfer -- he trains on raw activations
# where we standardize, so his Gated l1 0.958 gives L0 138.5 and ours at 1.368 gives 50.3 --
# and commit 8b95ec9 rewrote standard/p_anneal's recon loss under our pin. So these ranges
# are anchored on OUR one measured point per type and widened until they must bracket:
#   Standard      l1 0.0587 -> L0 1095   (need sparser: higher l1)
#   Gated         l1 1.368  -> L0 50.3   (need denser: lower l1)
#   PAnneal       sp 0.0421 -> L0 338.6  (need a little sparser)
#   GatedAnneal   sp 0.955  -> L0 31.0   (need much denser: lower sp)
# 20 log-spaced points over 2-3 decades each. This is the pilot AND the grid: the response
# is unmeasured, so the width is the insurance. Check what landed before trusting the
# operating point -- a type whose bracket missed needs its range moved, not interpolated.
STANDARD_GRID="0.05,0.06371,0.08119,0.1035,0.1318,0.168,0.2141,0.2728,0.3476,0.4429,0.5644,0.7192,0.9165,1.168,1.488,1.896,2.416,3.079,3.924,5"
GATED_GRID="0.05,0.06202,0.07694,0.09544,0.1184,0.1469,0.1822,0.226,0.2803,0.3477,0.4314,0.5351,0.6638,0.8234,1.021,1.267,1.572,1.95,2.418,3"
PANNEAL_GRID="0.008,0.01031,0.0133,0.01715,0.02211,0.0285,0.03675,0.04738,0.06109,0.07877,0.1016,0.1309,0.1688,0.2177,0.2807,0.3619,0.4666,0.6016,0.7756,1"
GATED_ANNEAL_GRID="0.004,0.005401,0.007291,0.009844,0.01329,0.01794,0.02423,0.03271,0.04416,0.05963,0.0805,0.1087,0.1467,0.1981,0.2675,0.3611,0.4876,0.6583,0.8888,1.2"
# L0 == k, so no fit needed; same grid as BatchTopK for a like-for-like pair.
TOPK_GRID="20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400"

case "${ARCH}" in
  spade)        GRID="${SPADE_GRID}" ;;
  cosine_spade) GRID="${COSINE_GRID}" ;;
  batch_topk)   GRID="${BATCH_TOPK_GRID}" ;;
  standard)     GRID="${STANDARD_GRID}" ;;
  gated)        GRID="${GATED_GRID}" ;;
  panneal)      GRID="${PANNEAL_GRID}" ;;
  gated_anneal) GRID="${GATED_ANNEAL_GRID}" ;;
  topk)         GRID="${TOPK_GRID}" ;;
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
    --gres=gpu:"${L0_GPUS:-4}" \
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
