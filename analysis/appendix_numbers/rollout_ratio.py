"""B.2: why Figs 10 and 12 stop at t0 + 8 -- the illegal/control ratio of δ̂ on the rollout tree.

Raw layer 5, `compute_divergence --variant rollout`'s δ̂ = |Δ| / |z_ref|. A branch is the mean
of its K rollouts (rollout row = source_row * K + k), and the ratio at offset r is the mean
over illegal branches over the mean over legal-control branches. B.2's claim ("within 0.5%
past t0 + 8, crossing over by t0 + 13") is about the original 2000-game tree, so both that
population and the whole extended tree are written.

Writes results/appendix_numbers/rollout_ratio.csv.

Usage: python analysis/appendix_numbers/rollout_ratio.py
"""

import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import variant_dir

OUT = results_dir("appendix_numbers") / "rollout_ratio.csv"
NPZ = variant_dir("divergence", variant="rollout") / "divergence" / "resid_post_5.npz"
POPULATIONS = {"game_id < 2000": lambda game: game < 2000,
               "all": lambda game: np.ones(len(game), bool)}
SATURATED_R = 8


def branch_means(delta: np.ndarray, branch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[n_branches, T]: δ̂ averaged over each branch's K rollouts."""
    keys, inverse = np.unique(branch[mask], return_inverse=True)
    sums = np.zeros((len(keys), delta.shape[1]))
    np.add.at(sums, inverse, delta[mask])
    return sums / np.bincount(inverse)[:, None]


def main() -> None:
    z = np.load(NPZ)
    after = z["window"] >= 0                     # δ̂ is zero before t0 by construction
    window, delta = z["window"][after], z["delta"][:, after]
    branch = z["row"] // (int(z["rollout"].max()) + 1)
    illegal, control = z["category"] != 0, z["is_control"] == 1
    tables = []
    for population, keep in POPULATIONS.items():
        member = keep(z["game_id"])
        ill = branch_means(delta, branch, illegal & member).mean(0)
        con = branch_means(delta, branch, control & member).mean(0)
        ratio = ill / con
        tables.append(pd.DataFrame({
            "population": population, "r": window,
            "n_illegal": len(np.unique(branch[illegal & member])),
            "n_control": len(np.unique(branch[control & member])),
            "illegal": ill, "control": con, "ratio": ratio}))
        print(f"{population}: max |ratio - 1| past r = {SATURATED_R}: "
              f"{np.abs(ratio[window > SATURATED_R] - 1).max():.4f}; "
              f"first r > 0 with ratio >= 1: {window[(window > 0) & (ratio >= 1)][0]}")
    table = pd.concat(tables, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.6g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
