"""Resolve spade_set.csv to checkpoint directories in the multi-seed study tree.

    python -m karvonen_repro.spade_checkpoints [--multiseed-root DIR]

One path per line, for `karvonen_repro.evaluate` to consume — the SpaDE/CosineSpaDE
counterpart of `validation_checkpoints.py`. Every row is checked against its own
config.json before it is printed, so a re-indexed sweep fails here rather than silently
evaluating the wrong gamma.

Two asymmetries in the set are real, not oversights. The SpaDE families are n=4 and the
CosineSpaDE families n=5, because seed_42 has no `autoencoders/` directory at all — only
the two cosine sweeps. And trainer_8 is left out of the lambda=1e-7 family even though its
config records lambda_init=8e-08 at the same gamma: its ae.pt is byte-identical to
trainer_9's in all four seeds, so the two are one point, not two.

Nothing under checkpoints/main/ is in the set: trainer_5 there is degenerate with
unrecoverable provenance, and that directory's standardization_params.json is smoke-test
output, which makes every one of its siblings suspect.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = os.environ.get("SPADE_MULTISEED_DIR", "/n/netscratch/ba_lab/Lab/dcao/spade_multiseed")
SPADE_SET = Path(__file__).resolve().parent / "spade_set.csv"


def check(path: Path, row) -> None:
    assert (path / "ae.pt").is_file(), f"no ae.pt at {path}"
    trainer = json.loads((path / "config.json").read_text())["trainer"]
    assert trainer["seed"] == row.seed, f"{path}: config seed {trainer['seed']} != {row.seed}"
    assert trainer["gamma_reg"] == row.gamma_reg, \
        f"{path}: config gamma_reg {trainer['gamma_reg']} != {row.gamma_reg}"
    lambda_init = trainer.get("lambda_init")
    assert (lambda_init is None and pd.isna(row.lambda_init)) or lambda_init == row.lambda_init, \
        f"{path}: config lambda_init {lambda_init} != {row.lambda_init}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multiseed-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--spade-set", type=Path, default=SPADE_SET)
    args = ap.parse_args()

    for row in pd.read_csv(args.spade_set).itertuples():
        path = args.multiseed_root / f"seed_{row.seed}" / row.sweep / row.trainer
        check(path, row)
        print(path)


if __name__ == "__main__":
    main()
