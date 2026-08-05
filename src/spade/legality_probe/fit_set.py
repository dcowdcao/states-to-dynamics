"""Step 1 of the legality-probe study: the positions the probe is fit and scored on.

200 games x token positions [5, 55), i.e. the position range of
analysis/linear_probe/activations, drawn from the same stream as the
violation-divergence branches and disjoint from them by exact token match — a probe
fit on a read-out game would make the shared r < 0 prefix look preserved for free.

A game is taken whole or not at all: one forfeit anywhere in the window drops it
(~1% of games), so every accepted game contributes exactly 50 rows and the 8000/2000
row split lands on a game boundary. No position in a split comes from a game in the
other, and the bootstrap can resample source games.

Usage: python -m spade.legality_probe.fit_set
"""

import argparse
import json
from collections import Counter
from random import Random

import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe.labels import LABELS, labels_at_ply
from spade.othello.games import collect_legal, stream_games  # imports force_ipv4() + karvonen path

from circuits.othello_engine_utils import OthelloBoardState, to_string

POSITIONS = range(5, 55)
TRAIN_GAMES, EVAL_GAMES = 160, 40  # 8000 / 2000 rows at 50 positions per game
N_TRAIN = TRAIN_GAMES * len(POSITIONS)


def branch_tokens() -> set[tuple[int, ...]]:
    """Every token sequence the violation-divergence read-out set contains."""
    branches = torch.load(study_dir("divergence") / "branches" / "branches.pt",
                          weights_only=True)
    return {tuple(row) for row in branches["tokens"].tolist()}


def window_labels(game: list[int], skips: Counter) -> np.ndarray | None:
    """[50, 64] int8 labels over POSITIONS, or None if the game forfeits in the window."""
    rows = []
    for position in POSITIONS:
        labels = labels_at_ply(game, position + 1, skips)
        if labels is None:
            return None
        rows.append(labels)
    return np.stack(rows)


def collect(games, n_games: int, forbidden: set[tuple[int, ...]], skips: Counter) -> dict:
    tokens, labels = [], []
    while len(tokens) < n_games:
        game = collect_legal(games, 1, skips)[0]
        if tuple(game) in forbidden:
            skips["in_readout_set"] += 1
            continue
        window = window_labels(game, skips)
        if window is None:
            skips["forfeit_in_window"] += 1
            continue
        tokens.append(game)
        labels.append(window)
    n_positions = len(POSITIONS)
    return {
        "tokens": torch.tensor(tokens, dtype=torch.int16),
        "labels": torch.tensor(np.concatenate(labels), dtype=torch.int8),
        "position": torch.tensor(list(POSITIONS) * n_games, dtype=torch.int16),
        "game_id": torch.arange(n_games, dtype=torch.int16).repeat_interleave(n_positions),
    }


def verify(fit: dict, n: int = 200) -> None:
    """Re-derive a sample of rows from the saved tokens: legal squares, occupied squares,
    no forfeit position, and — the one check that is not circular in the mover's colour —
    the move the record actually plays next has to come out legal."""
    rng = Random(0)
    for row in rng.sample(range(len(fit["position"])), n):
        game = fit["tokens"][int(fit["game_id"][row])].tolist()
        position = int(fit["position"][row])
        board = OthelloBoardState()
        board.update(to_string(game[: position + 1]))
        regular = [m for m in board.get_valid_moves() if board.tentative_move(m) == 1]
        assert regular, f"row {row}: forfeit position leaked into the fit set"
        labels = fit["labels"][row].numpy()
        assert set(np.flatnonzero(labels == 0).tolist()) == set(regular), f"row {row}: legal set"
        assert (labels[board.state.reshape(64) != 0] == 1).all(), f"row {row}: occupied not R1"
        assert labels[to_string(game[position + 1])] == 0, f"row {row}: played move not legal"


def class_balance(labels: torch.Tensor, position: torch.Tensor) -> pd.DataFrame:
    """Long: one row per (position, label), plus a pooled block with position blank."""
    rows = []
    for group, subset in [("", labels)] + [(p, labels[position == p]) for p in POSITIONS]:
        counts = np.bincount(subset.reshape(-1).numpy(), minlength=len(LABELS))
        rows += [{"position": group, "label": name, "n": int(counts[i]),
                  "fraction": round(float(counts[i] / counts.sum()), 5)}
                 for i, name in enumerate(LABELS)]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    skips = Counter()
    fit = collect(stream_games(args.dataset_name, args.seed),
                  TRAIN_GAMES + EVAL_GAMES, branch_tokens(), skips)

    assert fit["tokens"].shape == (TRAIN_GAMES + EVAL_GAMES, CTX_LEN)
    assert fit["labels"].shape == (len(fit["position"]), 64)
    assert int(fit["game_id"][N_TRAIN - 1]) == TRAIN_GAMES - 1
    assert int(fit["game_id"][N_TRAIN]) == TRAIN_GAMES
    verify(fit)

    out_dir = study_dir("legality") / "fit_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in fit.items():
        torch.save(tensor, out_dir / f"{name}.pt")
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "dataset": args.dataset_name,
        "n_games": TRAIN_GAMES + EVAL_GAMES,
        "n_train": N_TRAIN,
        "n_eval": EVAL_GAMES * len(POSITIONS),
        "position_range": [POSITIONS.start, POSITIONS.stop],
        "labels": list(LABELS),
        "label_of_row": "position p sees moves 0..p; labels are for the move at ply p+1",
        "disjoint_from": str(study_dir("divergence") / "branches" / "branches.pt"),
        "skips": dict(skips),
    }, indent=2))

    balance = class_balance(fit["labels"], fit["position"])
    results_dir = repo_root() / "analysis" / "legality_probe" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    balance.to_csv(results_dir / "class_balance.csv", index=False)

    print(f"wrote {len(fit['position'])} positions from {TRAIN_GAMES + EVAL_GAMES} games "
          f"to {out_dir}\nskips: {dict(skips)}")
    print(balance[balance["position"] == ""].to_string(index=False))
    print(balance[balance["position"].isin(range(5, 55, 8))]
          .pivot(index="position", columns="label", values="fraction").to_string())


if __name__ == "__main__":
    main()
