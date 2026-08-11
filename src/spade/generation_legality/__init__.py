"""Does OthelloGPT generate legal moves, and when it doesn't, which rule does it break?

Teacher-forced on real legal prefixes, the model's next-move distribution is split by the
7-way taxonomy of `spade.othello.taxonomy`: how much probability sits on legal squares, on
each of the three constraint rules R1-R3, and on the pass token the corpus never contains.
The same positions also record what the model would actually *play* -- the argmax move and
one temperature-1.0 draw -- so a mass result can be checked against behaviour. A separate
free-running arm samples whole games from scratch and asks when the first illegal move
appears, which is the only arm that carries error compounding.

  positions.py  the game/position set and its per-square taxonomy, disjoint from the
                violation-divergence read-out games
  scan.py       one teacher-forced pass -> per-position probability mass and played moves
  rollout.py    free-running temperature-1.0 games -> position of the first illegal move
  aggregate.py  tidy CSVs: per-move curves, the pooled rule skew, the survival curves

The skew question needs a null, and the availability of a rule swings by an order of
magnitude across a game (R1 is 5/60 squares at ply 1 and ~55/60 at ply 56). Every skew
number here is therefore against a *per-position matched* null -- the illegal mass the
model actually emitted, redistributed uniformly over that position's illegal squares --
never against a pooled square count.

Working dir: study_dir("generation"). Figures and CSVs: analysis/generation_legality/.
"""
