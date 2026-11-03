"""Combine baseline + old-sweep + revived lam1e-03-gam1e-04 Karvonen results
into one CSV for the comparison figure.

Usage: python -m spade.evaluation.merge_revival_results
"""

import sys

import pandas as pd

from spade.common.paths import results_dir

SOURCES = [
    results_dir("main") / "all_results.csv",
    results_dir("spade_sweep") / "all_results.csv",
    results_dir("spade_revival") / "all_results.csv",
]


def main():
    frames = [pd.read_csv(p) for p in SOURCES]
    out = results_dir("spade_revival") / "combined_results.csv"
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset="wandb_name", keep="last")
    combined.to_csv(out, index=False)
    print(f"wrote {out} ({len(combined)} rows from {len(SOURCES)} sources)")


if __name__ == "__main__":
    sys.exit(main())
