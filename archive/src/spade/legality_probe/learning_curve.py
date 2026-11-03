"""Is 200 games enough for the legality probe?

The fit set is 160 games of training rows where the board probe's corpus is 20,000, so
"more data would fix it" is a live alternative to "this space has lost the information".
Fit on 16 / 32 / 64 / 128 training games and score every fit on the SAME 40 held-out games,
with the same 32 validation games, so only the training set moves. A curve still climbing
at 128 says the fit set is the binding constraint; a flat one says extending it buys
nothing and the between-space gaps are about the spaces.

Each space keeps its own `weight_decay.selected` value at every size. That value was
selected at the full size, and a smaller training set wants more regularization, so the
small-n points are if anything pessimistic and the curve reads slightly steeper than it is.
Re-selecting per size would make each point its own best fit but would also make the curve
a curve of two things.

Usage: python -m spade.legality_probe.learning_curve [--spaces raw]
"""

import argparse

import pandas as pd
import torch

from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe import probe as legality_probe
from spade.legality_probe import spaces
from spade.legality_probe.weight_decay import selected

GAMES = (16, 32, 64, 128)  # 128 is the full protocol training pool: 160 fit games less 32 val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    parser.add_argument("--seed", type=int, default=legality_probe.SEED)
    args = parser.parse_args()

    device = select_device()
    fit_set = study_dir("legality") / "fit_set"
    labels = torch.load(fit_set / "labels.pt", weights_only=True).long()
    game = torch.load(fit_set / "game_id.pt", weights_only=True).long()

    chosen = [n for n, _ in spaces.roster()] + list(spaces.NULLS)
    if args.spaces:
        keep = set(args.spaces.split(","))
        chosen = [n for n in chosen if n in keep]

    cache = study_dir("legality") / "probes" / "learning_curve"
    cache.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        if (cache / f"{name}.csv").exists():
            print(f"skip {name} (cached)", flush=True)
            continue
        code = torch.load(study_dir("legality") / "latents" / f"probe_{name}.pt",
                          map_location="cpu", weights_only=True)
        out, weight_decay = [], selected(name)
        for n_games in GAMES:
            tr, va, te = legality_probe.split(game, n_games, seed=args.seed)
            arm, result = legality_probe.fit(code, labels, tr, va, weight_decay,
                                             seed=args.seed, device=device)
            test = legality_probe.macro_f1(arm, code[te].float(), labels[te])
            out.append({"space": name, "weight_decay": weight_decay,
                        "train_games": n_games, "train_rows": len(tr),
                        "best_epoch": result.best_epoch, "epochs_run": len(result.curve),
                        "val_accuracy": result.val_accuracy,
                        "test_macro_f1": test.macro_f1, "test_accuracy": test.accuracy})
            print(f"{name:28} {n_games:4} games  test_macro_f1 {test.macro_f1:.5f} "
                  f"val_acc {result.val_accuracy:.5f} epochs {len(result.curve)}", flush=True)
        pd.DataFrame(out).to_csv(cache / f"{name}.csv", index=False)
        del code

    table = pd.concat([pd.read_csv(f) for f in sorted(cache.glob("*.csv"))])
    path = repo_root() / "analysis" / "legality_probe" / "results" / "learning_curve.csv"
    table.to_csv(path, index=False)
    print(f"\nwrote {path}\n",
          table.pivot(index="space", columns="train_games", values="test_macro_f1").to_string())


if __name__ == "__main__":
    main()
