"""An independent legality labeller, written from docs/othello_rules.md rather than from
`spade.othello.taxonomy` or `spade.legality_probe.labels`.

Own line scan, own category ladder, own scatter into row-major squares. Nothing in this
file is imported from the modules under test, so an agreement between this and the shipped
labels is evidence and not a tautology.

    0 legal   1 R1 placement   2 R2 bracketing   3 R3 turn
"""

import numpy as np

DIRECTIONS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
LABEL_OF_CATEGORY = np.array([0, 1, 1, 2, 2, 2, 3], dtype=np.int8)
CATEGORY_NAMES = ["legal", "occupied_own", "occupied_opp", "empty_detached",
                  "empty_own_only", "empty_no_bracket", "opponent_only"]


def would_flip(state: np.ndarray, square: int, color: int) -> bool:
    r, c = divmod(square, 8)
    for dr, dc in DIRECTIONS:
        cr, cc, seen = r + dr, c + dc, 0
        while 0 <= cr < 8 and 0 <= cc < 8 and state[cr, cc] != 0:
            if state[cr, cc] == color:
                if seen:
                    return True
                break
            seen += 1
            cr, cc = cr + dr, cc + dc
    return False


def categorise(state: np.ndarray, mover: int) -> np.ndarray:
    """[64] int8 over CATEGORY_NAMES, row-major square = row * 8 + col."""
    out = np.empty(64, dtype=np.int8)
    for square in range(64):
        r, c = divmod(square, 8)
        if state[r, c] == mover:
            out[square] = 1
        elif state[r, c] == -mover:
            out[square] = 2
        elif would_flip(state, square, mover):
            out[square] = 0
        elif would_flip(state, square, -mover):
            out[square] = 6
        else:
            neighbours = [state[r + dr, c + dc] for dr, dc in DIRECTIONS
                if 0 <= r + dr < 8 and 0 <= c + dc < 8]
            out[square] = (3 if all(v == 0 for v in neighbours)
                           else 5 if any(v == -mover for v in neighbours) else 4)
    return out


def legality(state: np.ndarray, mover: int) -> np.ndarray:
    """[64] int8 over {0 legal, 1 R1, 2 R2, 3 R3}."""
    return LABEL_OF_CATEGORY[categorise(state, mover)]


def legal_squares(state: np.ndarray, mover: int) -> set[int]:
    return {s for s in range(64) if state[s // 8, s % 8] == 0 and would_flip(state, s, mover)}


def board_from_disc_labels(labels: np.ndarray) -> np.ndarray:
    """{0 empty, 1 theirs, 2 mine} -> an [8, 8] state with the mover as +1."""
    return np.where(labels == 2, 1, np.where(labels == 1, -1, 0)).reshape(8, 8).astype(np.int8)
