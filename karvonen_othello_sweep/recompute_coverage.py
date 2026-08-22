"""Recompute board-state coverage from each checkpoint's saved per-feature counts.

The per-checkpoint results.csv omits `best_average_f1`, so coverage for the 40 TopK checkpoints
Karvonen never published exists nowhere on disk as a number. It is recoverable exactly: running
his own `analysis.get_all_f1s` + `f1_analysis.best_f1_average` over each checkpoint's
`indexing_None_n_inputs_1000_results.pkl` reproduces the published column bit for bit.

Writes coverage_recomputed.csv, which build_report.py reads. Slow (~23 GB of pickles), so it is
a separate entrypoint and its output is committed.
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "Karvonen_Research"))

import circuits.analysis as analysis  # noqa: E402
import circuits.chess_utils as chess_utils  # noqa: E402
import circuits.f1_analysis as f1_analysis  # noqa: E402

RESULTS_PKL = "indexing_None_n_inputs_1000_results.pkl"


def coverage(trainer_dir: Path) -> list[dict]:
    with open(trainer_dir / RESULTS_PKL, "rb") as f:
        results = pickle.load(f)
    rows = []
    for func_name, f1_TFRRC in analysis.get_all_f1s(results, "cpu").items():
        average_f1_T = f1_analysis.best_f1_average(f1_TFRRC, chess_utils.config_lookup[func_name])
        best_idx = int(torch.argmax(average_f1_T))
        rows.append({"group_dir": trainer_dir.parent.name, "trainer_dir": trainer_dir.name,
                     "func": func_name, "best_average_f1": float(average_f1_T[best_idx]),
                     "best_average_f1_idx": best_idx, "n_features": f1_TFRRC.shape[1]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent /
                    "coverage_recomputed.csv")
    args = ap.parse_args()

    torch.set_num_threads(1)
    trainer_dirs = sorted(d for d in args.config_root.glob("othello-trained*/*/trainer*")
                          if (d / RESULTS_PKL).exists())
    rows = []
    for i, d in enumerate(trainer_dirs, 1):
        rows.extend(coverage(d))
        if i % 20 == 0:
            print(f"{i}/{len(trainer_dirs)}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(trainer_dirs)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
