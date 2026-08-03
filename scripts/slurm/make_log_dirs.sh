#!/bin/bash
# Create every log directory the #SBATCH --output paths name.
#
# logs/ is gitignored, so a fresh checkout has none of these. SLURM does not
# create the parent of an --output path: it fails the job at launch, and there
# is no log to read because the log is what failed. Run this once per checkout,
# before the first sbatch.
#
# Usage: SPADE_REPO=<repo root> bash scripts/slurm/make_log_dirs.sh

set -euo pipefail

cd "${SPADE_REPO:?}"

mkdir -p logs/{training,extraction,karvonen,figures,diagnostics,revival}
mkdir -p logs/{rules,bracket_flip,bracket,violation_divergence,straightness,manifold_capture}

echo "log dirs ready under ${SPADE_REPO}/logs"
