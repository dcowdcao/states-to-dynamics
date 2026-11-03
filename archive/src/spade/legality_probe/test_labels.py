"""Tests for the token-order -> board-order scatter and the 4-way rule collapse.

A transposed or off-by-one board is the failure this file exists to catch, so the
checks are stated against the engine and against row/column-asymmetric positions
rather than against a second copy of the same scatter.

Usage: python -m spade.legality_probe.test_labels
"""

import random
from collections import Counter

import numpy as np

from spade.legality_probe.labels import (
    LABELS, LABEL_OF_CATEGORY, STARTING_SQUARES, board_labels, labels_at_ply)
from spade.othello.taxonomy import CATEGORIES, LEGAL, classify, classify_all

from circuits.othello_engine_utils import OthelloBoardState, stoi, stoi_indices, to_string


def played_boards(n: int = 40) -> list[OthelloBoardState]:
    """Non-forfeit positions from random legal play, spread over the game."""
    rng = random.Random(0)
    out = []
    while len(out) < n:
        b = OthelloBoardState()
        for _ in range(rng.randrange(5, 55)):
            moves = b.get_valid_moves()
            if not moves:
                break
            b.umpire(rng.choice(moves))
        if any(b.tentative_move(m) == 1 for m in b.get_valid_moves()):
            out.append(b)
    return out


def test_starting_squares_are_the_non_vocab_squares():
    assert STARTING_SQUARES == [27, 28, 35, 36]
    assert LABELS == ("legal", "R1 placement", "R2 bracketing", "R3 turn")
    assert LABEL_OF_CATEGORY == [0, 1, 1, 2, 2, 2, 3]


def test_round_trip_token_board_token():
    for i in range(len(stoi_indices)):
        assert stoi[to_string(i + 1)] - 1 == i
    for square in stoi_indices:
        assert to_string(stoi[square]) == square


def test_scatter_matches_per_square_classify():
    """The scatter must equal classifying each board square directly, no reindexing."""
    for b in played_boards():
        labels = board_labels(b, classify_all(b))
        direct = np.array([LABEL_OF_CATEGORY[classify(b, sq)] for sq in range(64)], dtype=np.int8)
        assert np.array_equal(labels, direct)


def test_legal_squares_match_engine():
    for b in played_boards():
        labels = board_labels(b, classify_all(b))
        assert set(np.flatnonzero(labels == 0).tolist()) == set(b.get_valid_moves())


def test_occupied_squares_are_r1():
    """Row-major placement: every disc, wherever it sits, is an R1 violation."""
    for b in played_boards():
        labels = board_labels(b, classify_all(b))
        occupied = b.state.reshape(64) != 0
        assert np.array_equal(labels == 1, occupied)
        assert np.array_equal(labels[STARTING_SQUARES], np.ones(4, dtype=np.int8))


def test_transpose_would_be_caught():
    """On a position with no diagonal symmetry left, transposing changes the labels."""
    rng = random.Random(2)
    b = OthelloBoardState()
    for _ in range(7):
        b.umpire(rng.choice(b.get_valid_moves()))
    assert not np.array_equal(b.state, b.state.T)
    labels = board_labels(b, classify_all(b))
    assert not np.array_equal(labels, labels.reshape(8, 8).T.reshape(64))
    assert set(np.flatnonzero(labels == 0).tolist()) == set(b.get_valid_moves())


def test_labels_at_ply_skips_and_aligns():
    rng = random.Random(1)
    b, game = OthelloBoardState(), []
    while len(game) < 40:
        moves = b.get_valid_moves()
        move = rng.choice(moves)
        game.append(stoi[move])
        b.umpire(move)

    skips = Counter()
    for t in (5, 17, 33):
        labels = labels_at_ply(game, t, skips)
        replayed = OthelloBoardState()
        replayed.update(to_string(game[:t]))
        assert set(np.flatnonzero(labels == 0).tolist()) == set(replayed.get_valid_moves())
        assert labels[to_string(game[t])] == 0  # the move actually played is legal
    assert not skips

    assert labels_at_ply(game[:20] + [0], 20, skips) is None
    assert labels_at_ply(game, 60, skips) is None
    assert skips == Counter({"pass_token": 1, "short_game": 1})


def test_category_counts_are_plausible():
    """Every rule shows up, and R1 dominates; the balance argument rests on this."""
    counts = Counter()
    for b in played_boards():
        counts.update(board_labels(b, classify_all(b)).tolist())
    assert set(counts) == {0, 1, 2, 3}
    assert counts[1] == max(counts.values())
    assert all(c > 0.02 * sum(counts.values()) for c in counts.values())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
