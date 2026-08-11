"""Step 1: the games and positions this study scans, with their per-square taxonomy.

1000 games streamed at seed 42 and excluded by exact token match from the
violation-divergence read-out set, the way `spade.legality_probe.fit_set.branch_tokens`
does. A game is taken whole or not at all, so every accepted game contributes exactly
`len(POSITIONS)` rows, n is constant across the move axis and the bootstrap resamples
source games.

POSITIONS is every t at which the model predicts a move, minus the last two. At t = 56
and t = 57 the mover forfeits in 13.5% and 4.0% of games, where `classify() == LEGAL` stops
matching the engine; whole-or-nothing acceptance over [0, 56) still keeps ~97.5% of games,
over [0, 58) it would keep 82%.

These games are disjoint from the divergence branches and from nothing else: they are the
leading rows of the same seed-42 window every other study reads (docs/game_corpus.md), and
the corpus is OthelloGPT's own training distribution, so "held out" here means held out
from a sibling study, not from the model.

Usage: python -m spade.generation_legality.positions [--n_games 1000]
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch

from spade.common.config import CTX_LEN
from spade.common.study import study_dir
from spade.legality_probe.fit_set import branch_tokens
from spade.othello.games import classify_prefix, collect_legal, stream_games
from spade.othello.taxonomy import CATEGORIES

POSITIONS = range(0, 56)  # logits at t predict the move at ply t+1
N_SQUARES = 60


def window_cats(game: list[int], skips: Counter) -> np.ndarray | None:
    """[len(POSITIONS), 60] int8 categories, or None if the game forfeits in the window."""
    rows = []
    for t in POSITIONS:
        prefix = classify_prefix(game, t + 1, skips)
        if prefix is None:
            return None
        rows.append(prefix[1])
    return np.array(rows, dtype=np.int8)


def collect(games, n_games: int, forbidden: set[tuple[int, ...]], skips: Counter) -> dict:
    tokens, cats = [], []
    while len(tokens) < n_games:
        game = collect_legal(games, 1, skips)[0]
        if tuple(game) in forbidden:
            skips["in_readout_set"] += 1
            continue
        window = window_cats(game, skips)
        if window is None:
            skips["forfeit_in_window"] += 1
            continue
        tokens.append(game)
        cats.append(window)
        if len(tokens) % 100 == 0:
            print(f"collected {len(tokens)}/{n_games}")
    return {
        "tokens": torch.tensor(tokens, dtype=torch.int16),
        "cats": torch.tensor(np.concatenate(cats), dtype=torch.int8),
        "t": torch.tensor(list(POSITIONS) * n_games, dtype=torch.int16),
        "game_id": torch.arange(n_games, dtype=torch.int32)
                        .repeat_interleave(len(POSITIONS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    skips = Counter()
    out = collect(stream_games(args.dataset_name, args.seed), args.n_games,
                  branch_tokens(), skips)

    n_rows = args.n_games * len(POSITIONS)
    assert out["tokens"].shape == (args.n_games, CTX_LEN)
    assert out["cats"].shape == (n_rows, N_SQUARES)
    assert (out["cats"] == 0).sum(1).min() > 0, "a position has no legal square"

    out_dir = study_dir("generation") / "positions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in out.items():
        torch.save(tensor, out_dir / f"{name}.pt")
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "dataset": args.dataset_name,
        "n_games": args.n_games,
        "n_positions": n_rows,
        "position_range": [POSITIONS.start, POSITIONS.stop],
        "row_of": "row (g, t) holds the taxonomy of the move at ply t+1 of game g",
        "categories": CATEGORIES,
        "excluded_by_token_match": str(study_dir("divergence") / "branches" / "branches.pt"),
        "not_disjoint_from": "the model's training corpus, nor any other study's games; "
                             "stream_games reads one window of the corpus in a fixed order "
                             "(docs/game_corpus.md)",
        "skips": dict(skips),
    }, indent=2))
    print(f"wrote {n_rows} positions from {args.n_games} games to {out_dir}\n"
          f"skips: {dict(skips)}")


if __name__ == "__main__":
    main()
