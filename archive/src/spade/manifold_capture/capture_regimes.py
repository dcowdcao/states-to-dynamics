"""Def. 6 regime table: classify each (space, concept) as capture / shattering /
dilution / mixed from the support size vs. manifold dimension (capture_metrics.csv)
and the signed cohesion of the largest supervised code community (geometry npz).

tau defaults to 0.3 — our choice; the paper leaves tau free.

Usage: python -m spade.manifold_capture.capture_regimes [--tau 0.3]
"""

import argparse
import csv

import numpy as np

from spade.common.paths import results_dir
from spade.manifold_capture.spaces import spaces


def classify(support_size: float, k_m: float, rho: float, tau: float) -> str:
    if support_size == 0:
        # atoms fire (SpaDE: thousands) but none covers >=10% of the manifold —
        # extreme localized tiling, not vacuous capture
        return "shattering"
    small = support_size <= 2 * k_m
    if small and rho >= tau:
        return "capture"
    if not small and rho <= -tau:
        return "shattering"
    if not small and abs(rho) < tau:
        return "dilution"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--concepts", default="", help="comma-separated filter")
    ap.add_argument("--spaces", default="", help="comma-separated filter")
    args = ap.parse_args()

    base = results_dir("manifold_capture")
    metrics_path = base / "capture_metrics.csv"
    assert metrics_path.exists(), \
        f"{metrics_path} missing — run python -m spade.manifold_capture.subspace_capture first"
    rows = list(csv.DictReader(open(metrics_path)))
    if args.spaces:
        rows = [r for r in rows if r["space"] in args.spaces.split(",")]
    if args.concepts:
        rows = [r for r in rows if r["concept"] in args.concepts.split(",")]

    sae_names = {n for n, e in spaces() if e is not None}
    out_rows = []
    for r in rows:
        if r["space"] not in sae_names:
            continue  # raw/random spaces have no code communities
        npz_path = base / f"geometry_{r['concept']}_{r['space']}.npz"
        assert npz_path.exists(), \
            f"{npz_path} missing — run python -m spade.manifold_capture.feature_geometry first"
        g = np.load(npz_path)
        labels = g["communities"]
        counts = np.bincount(labels)
        largest = int(counts.argmax())
        rho = float(g["community_rho"][labels == largest][0])
        out_rows.append({
            "space": r["space"], "concept": r["concept"], "variant": r["variant"],
            "k_M": r["k_M"], "support_size": r["support_size"],
            "rf_spread": r["rf_spread"],
            "largest_community": int(counts.max()), "rho": f"{rho:.4f}",
            "regime": classify(float(r["support_size"]), float(r["k_M"]), rho, args.tau),
            "low_ve": r["low_ve"]})

    out = base / "regimes.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]) if out_rows else
            ["space", "concept", "variant", "k_M", "support_size", "rf_spread",
             "largest_community", "rho", "regime", "low_ve"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out}")


if __name__ == "__main__":
    main()
