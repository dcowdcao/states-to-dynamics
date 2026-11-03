# reference/

Reading material, not experiment output. Port from the original `reference/research_materials/`:
- `Reference_Papers/` — the Karvonen board-games paper + related SAE writeups.
- `ARENA/` — the ARENA mech-interp curriculum. One file is vendored, verbatim from
  `github.com/callummcdougall/ARENA_3.0`: `part53_othellogpt/board_seqs_id_small.npy`, the
  game sequences `analysis/linear_probe/` fits on. It is committed (6 MB) because it is that
  study's entire input and a fresh clone has to be able to run it. Its companion
  `board_seqs_square_small.npy` is not: it is `to_string()` of the same array. Neither is
  ARENA's `utils.py` — its `OthelloBoardState` is the engine already vendored under
  `external/Karvonen_Research`, whitespace aside, and one engine in the tree is enough.
  Candidate SpaDE applications live in ARENA's `part51_balanced_bracket_classifier`,
  `part52_grokking_and_modular_arithmetic`, `part53_othellogpt`.
- Progress PDFs + the SpaDE research paper.

Also port `analysis/model_data_selection/` (SpaDE_Testing_Plan.md + the SAE reference table)
from the original — planning docs for what to apply SpaDE to next.
