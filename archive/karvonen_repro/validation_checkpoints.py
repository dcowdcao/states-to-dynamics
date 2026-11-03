"""Resolve validation_set.csv to checkpoint directories inside a released-SAE download.

    python -m karvonen_repro.validation_checkpoints --released-root DIR

One path per line, for `karvonen_repro.evaluate` to consume. The released tree puts
the top_k sweep in its own 2024-07-10 group; that mapping lives here and nowhere else,
so `evaluate.py` keeps taking plain checkpoint directories.
"""

import argparse
from pathlib import Path

import pandas as pd

MAY = "othello-trained_model-layer_5-2024-05-23/othello-trained_model-layer_5"
GROUP_DIRS = {
    "top_k": "othello-trained_model-layer_5-2024-07-10_topk/othello-trained_model-layer_5-top_k",
    "standard": f"{MAY}-standard",
    "gated": f"{MAY}-gated",
    "gated_anneal": f"{MAY}-gated_anneal",
    "p_anneal": f"{MAY}-p_anneal",
}

VALIDATION_SET = Path(__file__).resolve().parent / "validation_set.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--released-root", required=True, type=Path)
    ap.add_argument("--validation-set", type=Path, default=VALIDATION_SET)
    args = ap.parse_args()

    for row in pd.read_csv(args.validation_set).itertuples():
        path = args.released_root / GROUP_DIRS[row.group] / row.trainer
        assert (path / "ae.pt").is_file(), f"no ae.pt at {path}"
        print(path)


if __name__ == "__main__":
    main()
