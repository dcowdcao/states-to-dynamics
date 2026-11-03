"""Where the board state is linearly readable, under the canonical probe.

One bar per residual-stream layer: the accuracy of `probes.load.board(layer)` on 1,000
of the canonical corpus's test games, 3-way {empty, theirs, mine} over all 64 squares.
The saved cards already carry a test accuracy over the whole test split; this
re-measures it to attach an interval and a null, neither of which a single scalar per
layer can supply.

Positions inside a game share a board, so the interval is a cluster bootstrap over
test GAMES, not over rows. The null is the per-square majority class, fit on the train
split and scored on the same test rows as the probes -- a null selected on the split it
is scored on would be a ceiling.

The probes are seed 42 only, so this interval is over held-out data and says nothing
about the spread over fits.

Writes results/fig05_board_layer_sweep/board_layer_sweep.csv, which layer_sweep.py draws.

Usage: python analysis/board_probe_sweep/measure.py [--device cuda]
"""

import argparse

import numpy as np
import pandas as pd
import torch

from spade.common.paths import results_dir
from spade.probes import load as probes
from spade.probes.train import BATCH, activations, corpus, game_split, rows

LAYERS = range(8)
SEED = 42
N_TEST_GAMES = 1000     # of the split's 3000; the head of a permutation, so still uniform
N_BOOT = 1000


def hits_per_game(logits_argmax: torch.Tensor, y: torch.Tensor,
                  game_of_row: np.ndarray, n_games: int) -> np.ndarray:
    """[n_games] correct squares per test game, in game order."""
    correct = (logits_argmax == y.long()).sum(1).cpu().numpy()
    return np.bincount(game_of_row, weights=correct, minlength=n_games)


def predictions(weight: torch.Tensor, acts: torch.Tensor, device: str) -> torch.Tensor:
    out = []
    for i in range(0, len(acts), BATCH):
        block = acts[i:i + BATCH].to(device).float()
        out.append(torch.einsum("nd,dsc->nsc", block, weight).argmax(-1).cpu())
    return torch.cat(out)


def boot_ci(hits: np.ndarray, totals: np.ndarray, rng: np.random.Generator) -> tuple:
    """95% percentile CI of the pooled accuracy, resampling games with replacement."""
    draws = rng.integers(0, len(hits), size=(N_BOOT, len(hits)))
    ratios = hits[draws].sum(1) / totals[draws].sum(1)
    return tuple(np.percentile(ratios, [2.5, 97.5]))


def majority_null(y: torch.Tensor, train: torch.Tensor, test: torch.Tensor,
                  game_of_row: np.ndarray, n_games: int) -> np.ndarray:
    """Per square, the class most common in the train rows, scored on the test rows."""
    guess = torch.stack([(y[train] == c).sum(0) for c in range(3)]).argmax(0)   # [64]
    correct = (y[test] == guess).sum(1).numpy()
    return np.bincount(game_of_row, weights=correct, minlength=n_games)


def measure(device: str) -> pd.DataFrame:
    y, games, n_games = corpus()
    fit_games, _, test_games = game_split(n_games, SEED)
    test_games = test_games[:N_TEST_GAMES]
    train, test = rows(games, fit_games), rows(games, test_games)
    game_of_row, uniq = pd.factorize(games[test].numpy())
    totals = np.bincount(game_of_row, weights=np.full(len(test), y.shape[1]),
                         minlength=len(uniq))
    rng = np.random.default_rng(SEED)

    y_test = y[test]
    out = []
    for layer in LAYERS:
        acts = activations(layer)[test]
        pred = predictions(probes.board(layer, flat=True).to(device), acts, device)
        hits = hits_per_game(pred, y_test, game_of_row, len(uniq))
        lo, hi = boot_ci(hits, totals, rng)
        out.append({"layer": layer, "accuracy": hits.sum() / totals.sum(), "lo": lo, "hi": hi,
                    "card_test_accuracy": probes.card(layer)["test_accuracy"]})
        print(f"layer {layer}: {out[-1]['accuracy']:.5f} [{lo:.5f}, {hi:.5f}]", flush=True)
        del acts

    null = majority_null(y, train, test, game_of_row, len(uniq))
    lo, hi = boot_ci(null, totals, rng)
    out.append({"layer": -1, "accuracy": null.sum() / totals.sum(), "lo": lo, "hi": hi,
                "card_test_accuracy": np.nan})
    table = pd.DataFrame(out)
    table.insert(1, "n_games", len(uniq))
    table.insert(2, "n_rows", len(test))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    csv = results_dir("fig05_board_layer_sweep") / "board_layer_sweep.csv"
    with torch.no_grad():
        table = measure(args.device)
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv, index=False)
    print(table.to_string(index=False))
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
