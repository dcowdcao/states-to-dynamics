"""Straightness: does ReStraV's perceptual-straightening result transfer to OthelloGPT?

Five game classes (legal, model rollout, rule violation, random tokens, championship
play), trajectories through every layer, then curvature / step length / norm per position
and a 21-dimensional descriptor classifier.

  build_games.py      step 1: the five game classes
  extract_traj.py     step 2: 8-layer resid_post + token embeddings
  geometry.py         the ReStraV kernels (temporal geometry, 21-D descriptor, moments)
  compute_geometry.py step 3: raw + per-t-centered geometry per space
  classify.py         AUROCs and the logistic descriptor classifier

It transfers with the sign inverted and a depth flip. Read docs/claims_status.md: two
earlier claims about SpaDE destroying the curvature signal are retracted.

Working dir: study_dir("straightness"). Figures: analysis/straightness/.
"""
