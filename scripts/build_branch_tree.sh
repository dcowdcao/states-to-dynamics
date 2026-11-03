#!/bin/bash
# The paper's branch tree, $SPADE_DIVERGENCE_DIR/branches/branches.pt: 2003 round-robin
# games (t0 in [10, 34]), then t0 = 20 appends until 2000 rule-complete games fork there.
# 3923 games / 34546 rows. CPU, ~32 min, HF stream at datasets 4.8.4.
set -euo pipefail

TREE=$(python -c "from spade.common.study import study_dir; print(study_dir('divergence') / 'branches' / 'branches.pt')")
[ ! -e "$TREE" ] || { echo "$TREE exists; point SPADE_DIVERGENCE_DIR at an empty dir" >&2; exit 1; }

python -m spade.violation_divergence.branches --n_prefixes 2003
python -m spade.violation_divergence.extend_branches --t0 20 --target 2000
echo "6e5a02536633ac81a89dc5f8410ed2a8083c4c457123f96c566270c53c9a5dc4  $TREE" | sha256sum -c
