"""A.2: the corpus facts the paper states, measured off the parquet.

Split sizes; that every game is 60 plies with no pass token; the first-move counts, the e6
share and their chi^2 (3 df) against the 25% per opening the board's symmetry requires, per
split and pooled; and the uniformity test. For k = 1..5, games are grouped by their k-ply
prefix and each prefix's next moves are tested against uniform over the moves the engine
allows there (chi^2, rejecting at p < 1e-3). A next move the engine does not allow fails the
run. Rejecting prefixes are listed, so the truncated e6 opening can be seen to be them.

Writes results/appendix_numbers/corpus_stats.json.

Usage: python analysis/appendix_numbers/corpus_stats.py
"""

import json

import numpy as np
from scipy.stats import chisquare

import corpus
from spade.common.paths import results_dir
from spade.figures.figure_othello_rules import name
from spade.othello.boards import regular_moves, replay, tokens_of

from circuits.othello_engine_utils import to_string

OUT = results_dir("appendix_numbers") / "corpus_stats.json"
OPENINGS = {"d3": 20, "c4": 27, "f5": 34, "e6": 41}   # vocab tokens of the four first moves
PREFIX_PLIES = range(1, 6)
ALPHA, MIN_N = 1e-3, 400


def first_moves(games: np.ndarray) -> dict:
    counts = {square: int((games[:, 0] == token).sum()) for square, token in OPENINGS.items()}
    expected = len(games) / len(OPENINGS)
    return {"counts": counts, "e6_share": counts["e6"] / len(games),
            "chi2_3df": sum((n - expected) ** 2 / expected for n in counts.values())}


def uniformity(games: np.ndarray, k: int) -> dict:
    prefixes, which = np.unique(games[:, :k], axis=0, return_inverse=True)
    pairs, counts = np.unique(which.ravel() * 61 + games[:, k], return_counts=True)
    rows = []
    for i, prefix in enumerate(prefixes):
        seen = dict(zip((pairs[pairs // 61 == i] % 61).tolist(), counts[pairs // 61 == i].tolist()))
        board = replay(prefix.tolist())
        allowed = sorted(tokens_of(regular_moves(board.state, board.next_hand_color)))
        assert set(seen) <= set(allowed), f"illegal move after {prefix.tolist()}"
        observed = [seen.get(t, 0) for t in allowed]
        p = chisquare(observed).pvalue if len(observed) > 1 else 1.0
        rows.append({"prefix": " ".join(name(to_string(t)) for t in prefix.tolist()),
                     "n": sum(observed), "p": float(p)})
    rejects = [r for r in rows if r["p"] < ALPHA]
    return {"n_prefixes": len(rows), "n_prefixes_n_ge_400": sum(r["n"] >= MIN_N for r in rows),
            "n_reject": len(rejects), "n_reject_n_ge_400": sum(r["n"] >= MIN_N for r in rejects),
            "rejects": rejects}


def main() -> None:
    splits = {split: corpus.tokens(split) for split in ("train", "test")}
    pooled = np.concatenate(list(splits.values()))
    out = {"dataset": corpus.DATASET, "revision": corpus.REVISION,
           "rows": {split: len(games) for split, games in splits.items()} | {"pooled": len(pooled)},
           "plies": corpus.PLIES, "any_pass_token": bool((pooled == 0).any()),
           "first_move": {split: first_moves(games)
                          for split, games in [*splits.items(), ("pooled", pooled)]},
           "uniformity": {}}
    for k in PREFIX_PLIES:
        out["uniformity"][k] = uniformity(pooled, k)
        print(f"k = {k}:", {key: v for key, v in out["uniformity"][k].items() if key != "rejects"},
              flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({key: v for key, v in out.items() if key != "uniformity"}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
