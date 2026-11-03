"""Re-select the TPR arm's weight decay on validation.

wd=2.0 was tuned when the TPR probe saw 8,000 training rows and overfit them
badly. The shared corpus gives it ~812,000, so the old value cannot be carried
over unexamined. Every candidate is fit under the same protocol as the real run
and compared on validation accuracy; `fit` never sees the test split, so nothing
here can leak into the reported number.

One layer only: the sweep is for the regularization strength, not the layer, and
5 is the project's hook point.

Usage: python -m spade.probes.tpr_weight_decay [--layer 5]
       (cluster: scripts/slurm/studies/probes/submit_tpr_weight_decay.sbatch)
"""

import argparse
from pathlib import Path

import pandas as pd
import torch


from spade.probes.study_dir import probes_dir
from spade.probes.tpr import TPR
from spade.probes.train import activations, corpus, fit, game_split, rows

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0)  # 1e-2 won the asked-for grid at its edge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    y, games, n_games = corpus()
    tr, va, _ = (rows(games, g) for g in game_split(n_games, args.seed))
    acts = activations(args.layer)
    a_tr, y_tr, a_va, y_va = acts[tr], y[tr], acts[va], y[va]
    del acts
    print(f"layer {args.layer}: train {len(tr)} val {len(va)} rows", flush=True)

    out = []
    for wd in GRID:
        print(f"weight_decay {wd}", flush=True)
        result = fit(TPR(a_tr.shape[1], args.seed, wd), a_tr, y_tr, a_va, y_va, args.device)
        out.append({"weight_decay": wd, "best_epoch": result.best_epoch,
                    "epochs_run": len(result.curve), "val_accuracy": result.val_accuracy})

    table = pd.DataFrame(out)
    table.to_csv(probes_dir() / "probes" / f"tpr_weight_decay_layer{args.layer}.csv",
                 index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
