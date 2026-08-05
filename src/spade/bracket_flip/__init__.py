"""Bracket-flip: when a disc is flanked and flipped, does its layer-5 representation follow?

Reads out a single square with a bilinear tensor-product-representation probe (external/
tpr_othello) alongside the linear board probe, and edits along the flip direction to test
whether the model uses it.

  events.py          step 1: mine single-flip events + the probe training set
  extract_acts.py    step 2: 8-layer resid_post for before / after / matched control
  tpr_probe.py       the TPR probe wrapper (d_r=52, d_f=2) over the vendored implementation
  probe_space.py     read-out measurements
  act_geometry.py    raw layer-5 flip geometry
  geometry_tables.py the flip-geometry tables, for make_figures.py to draw
  sae_arm.py         the same read-out fit in each SAE latent space
  tpr_intervene.py   the un-flip edit, cross-checked against the linear-probe version
  robustness.py      refit across seed / weight decay / role dimension
  redteam.py         independent re-derivation of the headline numbers

Read docs/claims_status.md before writing anything about this study: its original headline
was retired, and three narrower findings survive in its place.

Working dir: study_dir("bracket"). Figures: analysis/bracket_flip/.
"""
