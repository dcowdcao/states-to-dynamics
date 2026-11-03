"""A.4: the chance level Fig 2(b)'s probe cosines are read against, isotropic and
covariance-matched.

Random unit directions in 512 dimensions meet at E|cos| = sqrt(2 / (pi * 512)) = 0.0353. The
residual stream is anisotropic, so directions shaped like its activations meet higher: draw
L @ white noise with L the Cholesky root of the move-legality fit set's activation covariance,
and unit-normalize. That is `probe_geometry.robustness.null_threat`'s construction, which
needs the gitignored rules corpus; this needs only the fit set. The number is the population
E|cos| over 5 x 10,000 independent pairs, each null drawn from its own seed-0 generator.
ROBUSTNESS.md's "0.084 to 0.110" is instead a range over `robustness_cells.csv`'s cells.

Writes results/covariance_null.csv.

Usage: python analysis/appendix_numbers/covariance_null.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spade.common.config import D_MODEL, LAYER
from spade.common.study import study_dir

OUT = Path(__file__).resolve().parent / "results" / "covariance_null.csv"
SEED = 0
N_PAIRS, N_REPEATS = 10_000, 5


def covariance(rows: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Feature covariance in float64, without a centered copy of the 176k-row fit set."""
    mean = rows.mean(0, dtype=np.float64)
    out = np.zeros((rows.shape[1], rows.shape[1]))
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk].astype(np.float64) - mean
        out += block.T @ block
    return out / (len(rows) - 1)


def unit_directions(n: int, root: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, D_MODEL)) @ root.T
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def mean_abs_cos(root: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    return float(np.mean([
        np.abs((unit_directions(N_PAIRS, root, rng) * unit_directions(N_PAIRS, root, rng)).sum(1)).mean()
        for _ in range(N_REPEATS)]))


def main() -> None:
    acts = torch.load(study_dir("move_legality") / "acts" / f"resid_post_{LAYER}.pt",
                      map_location="cpu", weights_only=True).numpy()
    root = np.linalg.cholesky(covariance(acts) + 1e-9 * np.eye(D_MODEL))
    table = pd.DataFrame([
        {"null": "isotropic_analytic", "mean_abs_cos": np.sqrt(2 / (np.pi * D_MODEL)), "n_pairs": 0},
        {"null": "isotropic", "mean_abs_cos": mean_abs_cos(np.eye(D_MODEL)),
         "n_pairs": N_PAIRS * N_REPEATS},
        {"null": "covariance_matched", "mean_abs_cos": mean_abs_cos(root),
         "n_pairs": N_PAIRS * N_REPEATS}]).round(5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print(f"wrote {OUT} (covariance from {len(acts)} fit-set rows)")


if __name__ == "__main__":
    main()
