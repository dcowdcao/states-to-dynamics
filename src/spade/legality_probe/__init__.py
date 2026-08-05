"""Legality probe: is a square's *legality* linearly bound to the square, and does an SAE keep it?

The bilinear tensor-product probe of `spade.bracket_flip.tpr_probe`, refit with the filler
alphabet widened from the 3 board classes to 4 legality classes — legal plus the three
constraint rules of `spade.othello.taxonomy.RULE_MEMBERS` — then fit independently in each
representation space at layer 5 and read out over the violation-divergence branch set.

  probe.py       the 4-way TPR probe: fit at any input dimension, score, save/load per space
  test_probe.py  self-tests: the 4-class filler is real, and a planted signal is recovered

Working dir: study_dir("legality"). Figures: analysis/legality_probe/.
"""
