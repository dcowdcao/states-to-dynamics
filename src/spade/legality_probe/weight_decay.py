"""Re-select the legality probe's weight decay on validation.

wd=2.0 was tuned for a 3-class, d=512 fit on 8,000 rows and was carried over unexamined
into a 4-class fit and into d=8192 SAE codes, where it regularizes sixteen times as many
input weights. Every candidate is fit under the same protocol as the real run and compared
on the validation games; the test games are never loaded here, so nothing in this sweep can
leak into a reported number.

Every space, and one value does not serve them all. The argmax runs 0.1 to 2.0, and the
loss from using another space's value is not small: at 2.0 the two SpaDE codes read
0.44-0.50 validation macro-F1 against 0.93 and 0.88 at their own argmax. AdamW's decoupled
decay puts the weight scale at roughly lr/(lr*wd) whatever the input is, so a fixed wd caps
the logit scale and a code with small norm underfits -- the collapse was a scale artifact,
not lost legality. `selected` is therefore per space.

The `legality` target only: the other three arms are floors and cross-checks, and they
inherit their space's value so a floor is not tuned differently from what it floors.
Resumable per space. `weight_decay/<space>.csv` on netscratch is the artifact `selected`
reads; `analysis/legality_probe/results/weight_decay.csv` is the merged table, tracked
because it is the evidence for the per-space choice.

One seed and one validation split, which is decisive for the 0.48 macro-F1 between SpaDE at
0.1 and at 2.0 and meaningless for the 0.001 between 1.0 and 2.0 on raw. Read the argmax as
a decision only where the grid separates by more than that.

Usage: python -m spade.legality_probe.weight_decay [--spaces raw,TopKTrainer-othello]
"""

import argparse

import pandas as pd
import torch

from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe import probe as legality_probe
from spade.legality_probe import spaces

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 4.0, 8.0)  # 2.0 was the old value and
# sits inside this grid rather than at its edge, which a 1e-4..2.0 grid could not have shown


def selected(space: str) -> float:
    """The space's validation-argmax weight decay, from a completed sweep.

    Per space, not one constant, because the argmax moves by more than the effect the study
    reports: at 2.0 the two SpaDE codes read 0.44-0.50 validation macro-F1 and at their own
    argmax 0.88-0.93, so a shared value measures code scale rather than legality.
    """
    path = study_dir("legality") / "probes" / "weight_decay" / f"{space}.csv"
    assert path.exists(), f"no sweep for {space}: run spade.legality_probe.weight_decay first"
    rows = pd.read_csv(path)
    assert len(rows) == len(GRID), f"{path}: {len(rows)} of {len(GRID)} grid points swept"
    return float(rows.loc[rows.val_accuracy.idxmax(), "weight_decay"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    parser.add_argument("--seed", type=int, default=legality_probe.SEED)
    args = parser.parse_args()

    device = select_device()
    fit_set = study_dir("legality") / "fit_set"
    labels = torch.load(fit_set / "labels.pt", weights_only=True).long()
    game = torch.load(fit_set / "game_id.pt", weights_only=True).long()
    tr, va, _ = legality_probe.split(game, seed=args.seed)
    print(f"train {len(tr)} val {len(va)} rows", flush=True)

    chosen = [n for n, _ in spaces.roster()] + list(spaces.NULLS)
    if args.spaces:
        keep = set(args.spaces.split(","))
        chosen = [n for n in chosen if n in keep]

    cache = study_dir("legality") / "probes" / "weight_decay"
    cache.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        if (cache / f"{name}.csv").exists():
            print(f"skip {name} (cached)", flush=True)
            continue
        code = torch.load(study_dir("legality") / "latents" / f"probe_{name}.pt",
                          map_location="cpu", weights_only=True)
        out = []
        for wd in GRID:
            arm, result = legality_probe.fit(code, labels, tr, va, weight_decay=wd,
                                             seed=args.seed, device=device)
            val = legality_probe.macro_f1(arm, code[va].float(), labels[va])
            out.append({"space": name, "d_space": code.shape[-1], "weight_decay": wd,
                        "best_epoch": result.best_epoch, "epochs_run": len(result.curve),
                        "val_accuracy": result.val_accuracy, "val_macro_f1": val.macro_f1})
            print(f"{name:28} wd {wd:<7} val_acc {result.val_accuracy:.5f} "
                  f"val_macro_f1 {val.macro_f1:.5f} epochs {len(result.curve)}", flush=True)
        pd.DataFrame(out).to_csv(cache / f"{name}.csv", index=False)
        del code

    table = pd.concat([pd.read_csv(f) for f in sorted(cache.glob("*.csv"))])
    path = repo_root() / "analysis" / "legality_probe" / "results" / "weight_decay.csv"
    table.to_csv(path, index=False)
    print(f"\nwrote {path} from {len(list(cache.glob('*.csv')))} spaces")
    for metric in ("val_accuracy", "val_macro_f1"):
        print(f"\n{metric}\n",
              table.pivot(index="space", columns="weight_decay", values=metric).to_string())


if __name__ == "__main__":
    main()
