"""How much the mover-parity fix moves the probe: both labellings, same activations, same seeds.

The committed labels used to take the mover from move-index parity (ARENA's construction),
which desyncs from the true mover at every forfeit. This refits the layer sweep and the
full-depth probe under both labellings so the headline numbers can be quoted both ways,
with the seed spread beside the difference. Reads the cached activations; no model forward.

Writes results/parity_fix_effect.csv (labels, space, epochs, seed, eval_acc).

Usage: python analysis/linear_probe/parity_fix_effect.py
"""

import argparse

import numpy as np
import pandas as pd
import torch as t

from spade.common.config import LAYER
from spade.common.device import select_device
from spade.common.paths import linear_probe_activations

from train_linear_probe import (
    ARENA_GAMES, EMBED, N_TRAIN, NUM_GAMES, PROBE_EPOCHS, RESULTS_DIR, SWEEP_EPOCHS,
    SWEEP_SEEDS, TURN_END, TURN_START, accuracy, fit_probe, index_parity, relative_labels,
    replay, sweep_spaces)

from circuits.othello_engine_utils import to_string


def label_sets() -> dict[str, t.Tensor]:
    """The two mover conventions, as [N, 8, 8] label rows over the kept window."""
    games = t.from_numpy(np.load(ARENA_GAMES)).long()[:NUM_GAMES].apply_(to_string)
    states, to_move = replay(games)
    window = slice(TURN_START, TURN_END)
    parity = index_parity(states.shape[1]).expand_as(to_move)
    return {"replay": relative_labels(states, to_move)[:, window].reshape(-1, 8, 8),
            "index_parity": relative_labels(states, parity)[:, window].reshape(-1, 8, 8)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = select_device(args.device)

    activations = linear_probe_activations()
    layers = t.load(activations / "all_layer_activations.pt", weights_only=True)
    embed = t.load(activations / "embedding_activations.pt", weights_only=True)
    spaces = {EMBED: embed, **{str(layer): layers[layer] for layer in range(len(layers))}}

    frames = []
    for name, labels in label_sets().items():
        print(f"=== {name} labels ===")
        sweep = sweep_spaces(spaces, labels, device)
        frames.append(sweep.assign(labels=name, epochs=SWEEP_EPOCHS))

        acts_train, acts_eval = spaces[str(LAYER)][:N_TRAIN].to(device), \
            spaces[str(LAYER)][N_TRAIN:].to(device)
        train, evaluate = labels[:N_TRAIN].to(device), labels[N_TRAIN:].to(device)
        for seed in SWEEP_SEEDS:
            probe = fit_probe(acts_train, train, PROBE_EPOCHS, seed)
            frames.append(pd.DataFrame([{"space": str(LAYER), "seed": seed,
                                         "eval_acc": accuracy(acts_eval, probe, evaluate),
                                         "labels": name, "epochs": PROBE_EPOCHS}]))
            print(f"  full probe layer {LAYER} seed {seed}: {frames[-1].eval_acc.item():.4f}")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS_DIR / "parity_fix_effect.csv", index=False)

    table = out[out.seed.notna()].pivot_table(index=["epochs", "space"], columns="labels",
                                              values="eval_acc", aggfunc=["mean", "std"])
    delta = table["mean"]["replay"] - table["mean"]["index_parity"]
    print(table.to_string())
    print("\nfix - parity, against the seed SD it has to clear:")
    print(pd.DataFrame({"delta": delta, "sd_replay": table["std"]["replay"]}).to_string())


if __name__ == "__main__":
    main()
