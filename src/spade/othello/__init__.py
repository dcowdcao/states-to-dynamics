"""The Othello game layer: corpora, board mechanics, and the legality taxonomy.

Everything here is shared by more than one study, which is why it is not inside any of
them. In the source repo these modules lived under `rules/`, so every downstream study
imported from a sibling study; that is the coupling this package exists to remove.

  games.py     stream games from HF, collect legal prefixes, locate first illegal position
  boards.py    replay, flip_set, regular_moves, next_mover, rays_touch, tokens_of
  taxonomy.py  the 7-category legality taxonomy, mapped to rules R1-R3

Wraps circuits.othello_engine_utils / othello_utils from external/Karvonen_Research.
"""
