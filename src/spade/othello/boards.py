"""Board mechanics on top of the Karvonen engine.

The helpers the causal and bracket-flip work needs and the engine does not expose:
which discs a move flips, which squares bracket it, and how a move sequence tokenizes.
"""

# def replay(moves): ...          # move sequence -> board states
# def flip_set(board, move): ...  # squares this move flips
# def regular_moves(board): ...   # legal non-pass moves
# def rays_touch(sq, move): ...   # does the flip ray from `move` cross `sq`
# def tokens_of(moves): ...       # move sequence -> model tokens
