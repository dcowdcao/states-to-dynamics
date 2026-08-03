"""Hand-constructed board tests for the legality taxonomy.

Usage: python -m spade.othello.test_taxonomy
"""

import numpy as np

from spade.othello.taxonomy import CATEGORIES, LEGAL, classify, classify_all

from circuits.othello_engine_utils import OthelloBoardState, stoi_indices


def board_from(pieces: dict[int, int], mover: int) -> OthelloBoardState:
    b = OthelloBoardState()
    b.state = np.zeros((8, 8))  # fresh array; never mutate the aliased initial_state
    for sq, color in pieces.items():
        b.state[sq // 8, sq % 8] = color
    b.next_hand_color = mover
    return b


def sq(r: int, c: int) -> int:
    return 8 * r + c


def test_start_position_matches_engine():
    b = OthelloBoardState()
    cats = classify_all(b)
    legal = {stoi_indices[i] for i, cat in enumerate(cats) if cat == LEGAL}
    assert legal == set(b.get_valid_moves()), (legal, b.get_valid_moves())
    assert classify(b, sq(0, 0)) == CATEGORIES.index("empty_detached")
    # (2,2) sees the white diagonal but the line runs into an empty square
    assert classify(b, sq(2, 2)) == CATEGORIES.index("empty_no_bracket")


def test_occupied_categories():
    b = board_from({sq(0, 1): -1, sq(2, 2): 1}, mover=-1)
    assert classify(b, sq(0, 1)) == CATEGORIES.index("occupied_own")
    assert classify(b, sq(2, 2)) == CATEGORIES.index("occupied_opp")


def test_empty_own_only():
    b = board_from({sq(0, 0): 1}, mover=1)
    assert classify(b, sq(1, 1)) == CATEGORIES.index("empty_own_only")


def test_opponent_only():
    # black(0,0) white(0,1), white to move: (0,2) flips nothing for white
    # but brackets (0,1) for black
    b = board_from({sq(0, 0): 1, sq(0, 1): -1}, mover=-1)
    assert classify(b, sq(0, 2)) == CATEGORIES.index("opponent_only")
    # same square is legal for black
    b.next_hand_color = 1
    assert classify(b, sq(0, 2)) == LEGAL


def test_no_bracket_precedence_over_own_only():
    # (0,2) touches both an opponent disc and an own disc, no line closes:
    # opponent-adjacency wins (it is the "almost legal" case)
    b = board_from({sq(0, 1): -1, sq(1, 2): 1, sq(0, 0): -1}, mover=1)
    assert classify(b, sq(0, 2)) == CATEGORIES.index("empty_no_bracket")


def test_edge_line_does_not_wrap():
    # black(0,6) white(0,7): black placing at (0,5)? line right hits own disc
    # first; the line past (0,7) would need to wrap the edge — illegal
    b = board_from({sq(0, 6): -1, sq(0, 7): -1}, mover=1)
    assert classify(b, sq(0, 5)) == CATEGORIES.index("empty_no_bracket")


def test_played_games_match_engine():
    import random
    rng = random.Random(0)
    for _ in range(20):
        b = OthelloBoardState()
        for _ in range(rng.randrange(5, 55)):
            moves = b.get_valid_moves()
            if not moves:
                break
            b.umpire(rng.choice(moves))
        regular = [m for m in b.get_valid_moves() if b.tentative_move(m) == 1]
        if not regular:
            continue  # forfeit position: taxonomy contract does not cover it
        cats = classify_all(b)
        legal = {stoi_indices[i] for i, cat in enumerate(cats) if cat == LEGAL}
        assert legal == set(regular), (legal, regular)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
