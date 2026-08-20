"""Why does `random_proj`'s shipped legality macro-F1 not replicate?

The encodings match (fp16 rounding, verified separately), so the gap is in the fit. This
runs my loop at six seeds and the shipped `legality_probe.fit` at four, on the space that
disagrees and on one that does not. `spade.legality_probe.probe` is imported here because
reproducing its output is the point of this file, not because anything is verified with it.

My loop is still the shipped recipe as it was when this file was written: 8,000 rows, a
fixed 120 epochs, wd=2.0. The shipped side has since moved to the project protocol -- 6,400
rows, a validation plateau, the space's own validated weight decay -- so the two columns now
differ in the training set and the regularizer as well as in the loop. The comparison to
read is the spread WITHIN a column, not the gap between them.

Usage: python analysis/legality_probe/redteam/seed_probe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rederive import N_TRAIN, OUT, f1_of, fit_probe, fit_targets, predict, tally  # noqa: E402

from spade.common.study import study_dir  # noqa: E402
from spade.legality_probe import probe as shipped  # noqa: E402
from spade.legality_probe.weight_decay import selected  # noqa: E402

SPACES = ("random_proj", "TopKTrainer-othello", "raw")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    legal_labels, _ = fit_targets()
    truth = legal_labels[N_TRAIN:]
    labels = torch.from_numpy(legal_labels).long()
    game = torch.load(study_dir("legality") / "fit_set" / "game_id.pt", weights_only=True).long()
    tr, va, te = shipped.split(game)
    assert (te.numpy() == np.arange(N_TRAIN, len(labels))).all(), \
        "the protocol test split is not the eval rows my loop scores on"

    rows = []
    for space in SPACES:
        code = torch.load(study_dir("legality") / "latents" / f"probe_{space}.pt",
                          map_location="cpu", weights_only=True).float()
        for seed in range(6):
            probe = fit_probe(code[:N_TRAIN].to(device), labels[:N_TRAIN].to(device),
                              4, 3, seed, device)
            preds = predict(probe, code[N_TRAIN:], device)
            rows.append({"space": space, "loop": "redteam", "seed": seed,
                         "macro_f1": float(f1_of(tally(preds, truth, 4).sum(0)).mean())})
        weight_decay = selected(space)
        for seed in (shipped.SEED, 0, 1, 2):
            arm, _ = shipped.fit(code, labels, tr, va, weight_decay, seed=seed, device=device)
            rows.append({"space": space, "loop": "shipped", "seed": seed,
                         "macro_f1": shipped.macro_f1(arm, code[te].float(),
                                                      labels[te]).macro_f1})
        print(pd.DataFrame(rows)[lambda d: d.space == space].to_string(index=False), flush=True)
        del code
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rt_seeds.csv", index=False)
    print(table.pivot_table(index="space", columns="loop", values="macro_f1",
                            aggfunc=["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
