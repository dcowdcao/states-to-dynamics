"""Move-legality probe: does a linear read-out of the residual stream know that the move
just played was illegal, and can it say where in the game it happened?

Distinct from `spade.legality_probe`, which asks a bilinear probe about the legality of
every *candidate* square before a move is made. Here there is one label per *played*
token -- the category `spade.othello.taxonomy.classify` gives the move at position t,
read off the residual at that same position, which is the first place in the network
where that move exists. One label per position is what makes the probe applicable to a
whole game sequence one move at a time, and that is what localizes a spliced violation.

  fit_set.py       per (game, position): a legal row and one row per available rule
  extract_acts.py  layer-5 resid_post at each row's read position
  probe.py         L2 logistic regression, 2-class and 4-class, balanced by construction
  spaces.py        the 11-space roster and the fit set encoded through each
  readout.py       score every position of the violation-divergence branch tree, per space

Balance is structural rather than a reweighting: a fit-set item is a (game, position)
pair carrying one row per class, so the classes are equal in every split *and* share
their game and move-index marginals exactly. A probe that scored well by reading move
index would score at chance here.

Working dir: study_dir("move_legality"). Figures: analysis/move_legality/.
"""
