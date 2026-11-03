"""Bracket-flip: when a disc is flanked and flipped, does its layer-5 representation follow?

Reads out a single square with a bilinear tensor-product-representation probe (external/
tpr_othello) alongside the linear board probe, and edits along the flip direction to test
whether the model uses it.

  events.py          step 1: mine single-flip events + the probe training set
  extract_acts.py    step 2: 8-layer resid_post for before / after / matched control
  probe_space.py     read-out measurements; the canonical probes, plus the arms
                     (absolute basis, layer -1) the canonical corpus cannot supply
  (act_geometry, geometry_tables, sae_arm, tpr_intervene, robustness, redteam: archived
  under archive/src/spade/bracket_flip/; the paper uses none of them)

Read archive/docs/claims_status.md before writing anything about this study: its original headline
was retired, and three narrower findings survive in its place.

Working dir: study_dir("bracket"). Figures: analysis/bracket_flip/.
"""
