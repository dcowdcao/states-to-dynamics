"""Move-legality taxonomy for the OthelloGPT rules study (constraint rules R1-R3).

Classifies each candidate square, given a board and the player to move, into
one of 7 categories ordered by how much board computation is needed to detect
the violation. `classify() == LEGAL` matches the engine's `get_valid_moves()`
exactly on non-forfeit positions (mover has at least one regular legal move);
callers must skip forfeit positions, where the engine falls back to letting
opponent-flipping squares be played.
"""

import numpy as np

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, eights, stoi_indices

CATEGORIES = [
    "legal",             # 0: flips at least one line for the mover
    "occupied_own",      # 1: square already holds the mover's disc
    "occupied_opp",      # 2: square already holds the opponent's disc
    "empty_detached",    # 3: empty, touches no discs at all
    "empty_own_only",    # 4: empty, touches only the mover's discs
    "empty_no_bracket",  # 5: empty, touches opponent discs, no line closes
    "opponent_only",     # 6: flips only for the opponent (turn confusion)
]
LEGAL = 0


def flips(state: np.ndarray, square: int, color: int) -> bool:
    """Would placing `color` on `square` flip at least one opponent disc?"""
    r, c = square // 8, square % 8
    for dr, dc in eights:
        cr, cc = r + dr, c + dc
        seen_opp = False
        while 0 <= cr <= 7 and 0 <= cc <= 7 and state[cr, cc] != 0:
            if state[cr, cc] == color:
                if seen_opp:
                    return True
                break
            seen_opp = True
            cr, cc = cr + dr, cc + dc
    return False


def classify(board: OthelloBoardState, square: int) -> int:
    state, me = board.state, board.next_hand_color
    r, c = square // 8, square % 8
    if state[r, c] == me:
        return 1
    if state[r, c] == -me:
        return 2
    if flips(state, square, me):
        return LEGAL
    if flips(state, square, -me):
        return 6
    neighbors = [state[r + dr, c + dc] for dr, dc in eights
        if 0 <= r + dr <= 7 and 0 <= c + dc <= 7]
    if all(v == 0 for v in neighbors):
        return 3
    if any(v == -me for v in neighbors):
        return 5
    return 4


def classify_all(board: OthelloBoardState) -> list[int]:
    """Category of every vocab square, in token order (index i = token i+1)."""
    return [classify(board, sq) for sq in stoi_indices]
