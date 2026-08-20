"""Legality probe: is a square's *legality* bound to the square, and does an SAE keep it?

The bilinear tensor-product probe of `spade.probes.tpr`, refit with the filler
alphabet widened from the 3 board classes to 4 legality classes — legal plus the three
constraint rules of `spade.othello.taxonomy.RULE_MEMBERS` — then fit independently in each
of 9 layer-5 representation spaces and read out over the violation-divergence branch set.

  probe.py       the 4-way TPR probe: fit at any input dimension, score, save/load per space
  labels.py      per-square {legal, R1, R2, R3}, in the mover's frame from engine replay
  fit_set.py     the probe corpus, disjoint from the violation-divergence source games
  extract_acts.py  layer-5 resid_post for that corpus
  spaces.py      the 9-space roster, and both activation sets encoded through each
  readout.py     fit per space, read out over the branch window, write the tidy CSVs

Legality is a deterministic function of a board that already decodes at ~0.98, so a high
macro-F1 is not on its own evidence of a legality representation: computing legality for
the wrong player still scores 0.910 against 0.937. The claims rest on between-space
differences and on the r<0 vs r=0 contrast, never on the level.

Working dir: study_dir("legality"). Figures: analysis/legality_probe/.
"""
