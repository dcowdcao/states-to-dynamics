"""Combine a sweep's eval CSV with the non-SpaDE rows of the baseline run.

Extracted verbatim from the `python -c` heredocs that submit_spade_only_eval.sh
and submit_spade_sweep_eval.sh used to carry inline. Belongs beside
spade.evaluation.merge_revival_results once wave 3 lands; it lives here for now
because the port of scripts/ runs ahead of the port of src/.

The baseline filter drops any row whose sae_type contains "SpaDE"
case-insensitively, which also drops CosineSpaDE. That is the original
behaviour and it is preserved deliberately.
"""

import argparse
import os

import pandas as pd

BASELINE_CSV = "results/main/all_results.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True,
                   help="sweep results dir holding all_results.csv; combined_results.csv is written here")
    p.add_argument("--baseline_csv", default=BASELINE_CSV)
    args = p.parse_args()

    dfs = []

    if os.path.exists(args.baseline_csv):
        baseline = pd.read_csv(args.baseline_csv)
        baseline = baseline[~baseline["sae_type"].str.contains("SpaDE", case=False, na=False)]
        dfs.append(baseline)
        print(f"  Loaded {len(baseline)} baseline SAE results")

    sweep_csv = os.path.join(args.results_dir, "all_results.csv")
    if os.path.exists(sweep_csv):
        sweep = pd.read_csv(sweep_csv)
        dfs.append(sweep)
        print(f"  Loaded {len(sweep)} results from {sweep_csv}")

    if not dfs:
        print("  WARNING: No results found to combine")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined_path = os.path.join(args.results_dir, "combined_results.csv")
    combined.to_csv(combined_path, index=False)
    print(f"  Combined CSV: {combined_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
