"""A.2: drawing N games reads the corpus's first N + 10,000 rows, and a new seed stays there.

Every study draws games through `spade.othello.games.stream_games` -- `shuffle(seed,
buffer_size=10,000)` over a one-shard streaming split -- so the shuffle permutes a reservoir
rather than the corpus. This maps each streamed game back to its parquet row and records the
largest row the first N draws touch (always below N + 10,000), how many of seed 42's first
2,400 games seeds 7 and 1234 share (A.2's "replaces about 80%"), and that seed 42 repeats.
Streams from the HF hub, so it needs the network; the stream order depends on the `datasets`
version (recorded).

Writes results/appendix_numbers/stream_window.json.

Usage: python analysis/appendix_numbers/stream_window.py
"""

import json

import datasets
import numpy as np

import corpus
from spade.common.paths import results_dir
from spade.othello.games import stream_games

OUT = results_dir("appendix_numbers") / "stream_window.json"
BUFFER = 10_000
DRAWS = (400, 800, 2000, 2400, 4000)
OVERLAP_SEEDS, N_OVERLAP = (7, 1234), 2400


def stream_rows(row_of: dict[bytes, int], seed: int, n: int) -> list[int]:
    """The parquet row of each of the first n games `stream_games` yields at `seed`."""
    games = stream_games(corpus.DATASET, seed)
    return [row_of[np.asarray(next(games)["tokens"], np.int8).tobytes()] for _ in range(n)]


def main() -> None:
    train = corpus.tokens("train")
    row_of = {game.tobytes(): row for row, game in enumerate(train)}
    assert len(row_of) == len(train), "duplicate games: a token row does not name a parquet row"
    rows = lambda seed, n: stream_rows(row_of, seed, n)

    seed42 = rows(42, max(DRAWS))
    out = {"dataset": corpus.DATASET, "datasets_version": datasets.__version__,
           "buffer_size": BUFFER,
           "max_row_of_first_n": {n: max(seed42[:n]) for n in DRAWS},
           "seed42_repeats": rows(42, N_OVERLAP) == seed42[:N_OVERLAP],
           "shared_with_seed42": {seed: len(set(rows(seed, N_OVERLAP)) & set(seed42[:N_OVERLAP]))
                                  / N_OVERLAP for seed in OVERLAP_SEEDS}}
    assert all(row < n + BUFFER for n, row in out["max_row_of_first_n"].items()), out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
