"""A.2: "No game in the corpus is illegal", replayed over all 990,623 games.

Python's engine replays too slowly for the whole corpus, so this replays every game at once
on 10x10 int8 boards whose border stops every ray. A move is legal when it flips for the
mover, or when it flips only for the opponent and the mover has no move anywhere: the
corpus marks no passes, so that is a forfeit and the opponent plays it. Anything else is the
game's first illegal move. The engine's own `umpire` is laxer (it accepts any move that
flips for the opponent), so it cannot be the referee here; it is the check on the replay,
whose final boards must equal the engine's on the first 1,500 games, 1,500 at random and the
first 300 with a forfeit.

Writes results/corpus_legality.json.

Usage: python analysis/appendix_numbers/corpus_legality.py
"""

import json
from pathlib import Path

import numpy as np

import corpus
from spade.othello.boards import replay

from circuits.othello_engine_utils import stoi_indices

OUT = Path(__file__).resolve().parent / "results" / "corpus_legality.json"
SQUARE_OF_TOKEN = np.array([-1] + list(stoi_indices))
CELL_OF_SQUARE = np.array([(s // 8 + 1) * 10 + s % 8 + 1 for s in range(64)])
STEPS = (-11, -10, -9, -1, 1, 9, 10, 11)
BORDER = 9
START = {27: -1, 28: 1, 35: 1, 36: -1}   # the engine's opening position; black (1) moves first
SEED, N_CHECK = 0, 1500


def flanks(board: np.ndarray, rows: np.ndarray, cells: np.ndarray, color: np.ndarray) -> list:
    """Per step: (closed, length) of the run of opponent discs out of each row's cell."""
    out = []
    for step in STEPS:
        going = np.ones(len(rows), bool)
        closed = np.zeros(len(rows), bool)
        length = np.zeros(len(rows), np.int8)
        for distance in range(1, 8):
            value = board[rows, np.clip(cells + distance * step, 0, 99)]
            closed |= going & (value == color) & (length > 0)
            going &= value == -color
            length += going
        out.append((closed, length))
    return out


def flips_any(runs: list) -> np.ndarray:
    return np.logical_or.reduce([closed for closed, _ in runs])


def has_move(board: np.ndarray, rows: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = np.zeros(len(rows), bool)
    for cell in CELL_OF_SQUARE:
        cells = np.full(len(rows), cell)
        out |= (board[rows, cell] == 0) & flips_any(flanks(board, rows, cells, color))
    return out


def main() -> None:
    games = np.concatenate([corpus.tokens("train"), corpus.tokens("test")])
    n = len(games)
    board = np.full((n, 100), BORDER, np.int8)
    board[:, CELL_OF_SQUARE] = 0
    for square, color in START.items():
        board[:, CELL_OF_SQUARE[square]] = color
    mover = np.ones(n, np.int8)
    first_illegal = np.full(n, -1, np.int8)
    forfeits = np.zeros(n, np.int16)

    for ply in range(corpus.PLIES):
        rows = np.flatnonzero(first_illegal < 0)
        cells = CELL_OF_SQUARE[SQUARE_OF_TOKEN[games[rows, ply]]]
        color = mover[rows].copy()
        empty = board[rows, cells] == 0
        runs = flanks(board, rows, cells, color)
        legal = empty & flips_any(runs)
        maybe = np.flatnonzero(empty & ~legal)
        opponent = flanks(board, rows[maybe], cells[maybe], -color[maybe])
        forfeit = flips_any(opponent) & ~has_move(board, rows[maybe], color[maybe])
        at = maybe[forfeit]
        color[at] = -color[at]
        legal[at] = True
        forfeits[rows[at]] += 1
        for (closed, length), (closed_opp, length_opp) in zip(runs, opponent):
            closed[at], length[at] = closed_opp[forfeit], length_opp[forfeit]
        first_illegal[rows[~legal]] = ply
        for step, (closed, length) in zip(STEPS, runs):
            for distance in range(1, 8):
                flip = legal & closed & (length >= distance)
                board[rows[flip], cells[flip] + distance * step] = color[flip]
        board[rows[legal], cells[legal]] = color[legal]
        mover[rows[legal]] = -color[legal]
        print(f"ply {ply}: {len(rows) - legal.sum()} illegal, {len(at)} forfeits", flush=True)

    rng = np.random.default_rng(SEED)
    check = np.unique(np.concatenate([np.arange(N_CHECK), rng.choice(n, N_CHECK, replace=False),
                                      np.flatnonzero(forfeits)[:300]]))
    check = check[first_illegal[check] < 0]
    mismatches = sum(not np.array_equal(replay(games[g].tolist()).state.astype(np.int8),
                                        board[g].reshape(10, 10)[1:9, 1:9])
                     for g in check)
    illegal = np.flatnonzero(first_illegal >= 0)
    out = {"dataset": corpus.DATASET, "revision": corpus.REVISION, "n_games": n,
           "n_illegal_games": len(illegal),
           "first_illegal": [[int(g), int(first_illegal[g])] for g in illegal[:20]],
           "n_forfeit_moves": int(forfeits.sum()),
           "n_games_with_forfeit": int((forfeits > 0).sum()),
           "engine_check_games": len(check), "engine_check_mismatches": int(mismatches)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    assert mismatches == 0, "the vectorized replay disagrees with the engine"


if __name__ == "__main__":
    main()
