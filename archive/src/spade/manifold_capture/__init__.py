"""Manifold capture: do the SAEs capture Othello concept manifolds, tile them, or dilute them?

A replication of the capture/shattering/dilution framework on OthelloGPT concepts.

  build_corpus.py     step 1: natural positions + concept labels
  extract_acts.py     step 2: layer-5 activations
  concepts.py         concept manifolds over the game and rules corpora
  spaces.py           per-space encode/decode adapters
  subspace_capture.py greedy span + code-mask capture
  landmarks.py        landmark capture, the Lemma 1 check
  ising.py            pairwise Ising fit (pseudo-likelihood + EBIC)
  capture_regimes.py  the capture / shattering / dilution table
  feature_geometry.py receptive_fields.py  manifold_views.py  tuning_curves.py

Nothing here was retracted, but two post-hoc fixes are load-bearing: an empty support
originally passed the small-support test vacuously and mislabeled SpaDE as capture, and
the raw/centered capture metrics are concept-independent by construction. Both are
recorded in docs/claims_status.md.

Working dir: study_dir("manifold"). Figures: analysis/manifold_capture/.
"""
