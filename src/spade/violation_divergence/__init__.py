"""Violation divergence: one prefix, branched into legal and illegal futures.

Branch a legal game at t0 into several continuations — the reference, an injected illegal
move of each category, and two legal controls — then measure how far each branch travels
from the reference in every representation space, and what the model does next.

  branches.py           step 1: build the branch set from one prefix
  extend_branches.py    append further source games to it, renumbering nothing
  extend_rollout_branches.py  the same append on the resampled tree
  extract_acts.py       step 2: 8-layer activations + logits per branch
  divergence.py         the metric kernels (relative divergence, direction cosine,
                        top-k share, participation ratio, support Jaccard, persistence)
  compute_divergence.py step 3: per-space divergence tensors, sharded
  test_divergence.py    kernel unit tests that need no data

The study's founding premise turned out to be false — an illegal branch diverges *less*
than a legal alternative — and several of its metrics were shown to be max-of-noise.
Read archive/docs/claims_status.md before describing any of it.

Working dir: study_dir("divergence"). Figures: analysis/violation_divergence/.
"""
