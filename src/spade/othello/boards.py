"""Engine helpers shared by the rules causal experiments.

All functions speak vocab tokens (1..60) at the boundary and engine squares
(0..63) internally. "Regular" moves flip for the nominal mover — no engine
forfeit fallback; forfeit handling is explicit at each call site.
"""

import numpy as np

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()

from spade.othello.taxonomy import flips

from circuits.othello_engine_utils import OthelloBoardState, eights, stoi, to_string

VOCAB_SQUARES = sorted(to_string(tok) for tok in range(1, 61))


def replay(tokens: list[int]) -> OthelloBoardState:
    board = OthelloBoardState()
    if tokens:
        board.update(to_string(tokens))
    return board


def flip_set(state: np.ndarray, square: int, color: int) -> list[int]:
    """Squares flipped when `color` plays `square`."""
    r, c = square // 8, square % 8
    flipped = []
    for dr, dc in eights:
        buffer = []
        cr, cc = r + dr, c + dc
        while 0 <= cr <= 7 and 0 <= cc <= 7 and state[cr, cc] != 0:
            if state[cr, cc] == color:
                flipped.extend(buffer)
                break
            buffer.append(8 * cr + cc)
            cr, cc = cr + dr, cc + dc
    return flipped


def regular_moves(state: np.ndarray, color: int) -> set[int]:
    """Empty squares where `color` flips at least one disc."""
    return {sq for sq in VOCAB_SQUARES
        if state[sq // 8, sq % 8] == 0 and flips(state, sq, color)}


def rays_touch(square: int, targets: set[int]) -> bool:
    """Does any target lie on one of `square`'s eight full rays?"""
    r, c = square // 8, square % 8
    for dr, dc in eights:
        cr, cc = r + dr, c + dc
        while 0 <= cr <= 7 and 0 <= cc <= 7:
            if 8 * cr + cc in targets:
                return True
            cr, cc = cr + dr, cc + dc
    return False


def tokens_of(squares) -> list[int]:
    return sorted(stoi[sq] for sq in squares)
