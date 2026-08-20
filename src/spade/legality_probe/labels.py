"""Per-square legality labels for the constraint-rule probe: which rule a move there breaks.

    0 legal    1 R1 placement    2 R2 bracketing    3 R3 turn

Two index conventions meet here and only one is right for the probe.
`taxonomy.classify_all` returns the 60 vocab squares in TOKEN order (entry i is
token i+1); the probe reads 64 squares in ROW-MAJOR board order, square = row*8 + col,
matching the TPR probe's role order. This module is that scatter plus the collapse from
7 categories to 3 rules, and nothing else — the rules themselves stay in
`spade.othello.taxonomy`.

The four starting squares are outside the vocab and can never be played. They get
R1 placement, because that is what `classify` returns for them: they always hold a
disc, so the same rule that makes any occupied square illegal makes them illegal.
Labelling all 64 squares uniformly is what
`build_square_selection_mask(exclude_center_squares=False)` — the choice already
of `spade.probes.tpr` — expects.
"""

import copy
from collections import Counter

import numpy as np

from spade.othello.games import classify_prefix
from spade.othello.taxonomy import CATEGORIES, RULE_MEMBERS, RULE_OF, classify, classify_all

from circuits.othello_engine_utils import OthelloBoardState, stoi_indices, to_string

RULES = list(RULE_MEMBERS)
LABELS = ("legal", *RULES)
LABEL_OF_CATEGORY = [0 if c == CATEGORIES[0] else 1 + RULES.index(RULE_OF[c]) for c in CATEGORIES]
STARTING_SQUARES = sorted(set(range(64)) - set(stoi_indices))

# token -> board; games.py asserts board -> token, so together the round trip is the identity
assert all(to_string(i + 1) == square for i, square in enumerate(stoi_indices))


def board_labels(board: OthelloBoardState, categories: list[int]) -> np.ndarray:
    """[64] int8 over LABELS, row-major. `categories` is `classify_all(board)`, token order."""
    out = np.empty(64, dtype=np.int8)
    for i, square in enumerate(stoi_indices):
        out[square] = LABEL_OF_CATEGORY[categories[i]]
    for square in STARTING_SQUARES:
        out[square] = LABEL_OF_CATEGORY[classify(board, square)]
    return out


def disc_labels(board: OthelloBoardState) -> np.ndarray:
    """[64] int8 over {0 empty, 1 theirs, 2 mine}, row-major: the board probe's
    3-class board target, which the fit set does not store.

    Class 2 is the player TO MOVE, never the one who just moved. The mover comes from
    `next_hand_color`, i.e. from engine replay, because forfeits are unmarked in this
    corpus and t % 2 desynchronises from the mover as soon as one happens.
    """
    state = board.state.reshape(64)
    return np.where(state == 0, 0,
                    np.where(state == board.next_hand_color, 2, 1)).astype(np.int8)


def opponent_labels(board: OthelloBoardState) -> np.ndarray:
    """`board_labels` for the player who just moved: legality computed in the wrong frame."""
    other = copy.copy(board)
    other.next_hand_color = -board.next_hand_color
    return board_labels(other, classify_all(other))


def labels_at_ply(game: list[int], t: int, skips: Counter) -> np.ndarray | None:
    """Labels for the move played at ply `t`, i.e. after `game[:t]`. None if unusable.

    Unusable means short, containing a pass token, or a forfeit at t, where the
    taxonomy's legal/illegal split stops matching the engine. The residual at token
    position p has seen p+1 moves, so its labels are `labels_at_ply(game, p + 1)`.
    """
    prefix = classify_prefix(game, t, skips)
    if prefix is None:
        return None
    return board_labels(*prefix)
