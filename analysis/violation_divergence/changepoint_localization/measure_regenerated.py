"""The same change-point read-out, on a corpus whose suffix is REGENERATED after t0.

`measure.py` scores the divergence branch tree, where every arm keeps the source game's
`tokens[t0+1:]` bitwise, so the model carries an impossible history for the rest of the game.
The straightness violation corpus is built the other way: the illegal move is placed with no
flips and the turn passed, then the game continues with a *fresh random legal continuation*.
Whether the read-out survives that change of regime is the question this file answers.

Three things do not transfer, and the schema says so rather than pretending otherwise:

- **There is no matched legal splice.** That corpus carries no legal-alternative arm, so
  `auroc_vs_control` here is the violation against **its own source legal game**, which is
  the `auroc_vs_reference` contrast of the parent file, not its `auroc_vs_control`. The
  figure's axis label is changed to match; these bars are not comparable to the 0.60 ones.
- **The null is legal-vs-legal, across two disjoint halves.** With no second control arm,
  the empirical chance band comes from splitting the held-out odd-index legal pool in two and
  scoring one half against the other, averaged over NULL_SPLITS random splits. It has to be
  disjoint sets: pairing a set against a permutation of *itself* gives exactly 0.5 by
  construction, which looks like a null and measures nothing. And it has to be many splits:
  one split's luck is shared by all nine spaces, because they score the same games.
- **Only the four empty-square categories exist** — the board update after an occupied-square
  violation is ill-defined — so the two occupied categories that drag the parent study's hit
  rate down are absent by construction. Do not read a higher hit rate here as the suffix
  regime being easier.

Writes results/changepoint_regenerated.csv.

Usage: PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/measure_regenerated.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spade.common.study import study_dir
from spade.figures.paper_style import DICTIONARIES, REFERENCE_SPACE
from spade.straightness.build_games import CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    auroc, candidates, ci, drop, located, profile, resamples, scan)

HERE = Path(__file__).resolve().parent
SPACES = [REFERENCE_SPACE] + DICTIONARIES
ARMS = ["illegal", "control", "control2", "reference"]
M, PAIRED = 4, True
M_MAX = 12          # the parent study's grid max, so both figures scan the same 34 plies
NEAR = 1
NULL_HALF = 500     # per side of the legal-vs-legal null; the odd pool holds 1000
NULL_SPLITS = 20    # random disjoint splits, averaged, so no single split sets the band
N_LEGAL = 2000
VIOLATION = CLASSES.index("violation")


def steps(space: str) -> np.ndarray:
    stem = "resid_post_5" if space == REFERENCE_SPACE else f"sae_{space}"
    return np.load(study_dir("straightness") / "geometry" /
                   f"{stem}_raw.npz")["d"].astype(np.float64)


def corpus(cand: np.ndarray) -> dict:
    """Row indices per arm, plus t0. Odd-source violations only: the profile is fit on the
    even-index legal games, so every scored game and every negative sits outside it."""
    games = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    block = np.arange(N_LEGAL) + VIOLATION * N_LEGAL
    src, t0 = games["src_game"][block].numpy(), games["t0"][block].numpy()
    keep = (src % 2 == 1) & (t0 >= cand[0] + 2) & (t0 <= cand[-1] + 2)
    src = src[keep]
    # Two further legal arms, rolled so each item is scored against a *different* odd-index
    # legal game. These widen the false-alarm band only -- the null comes from null_splits(),
    # because a rolled copy of one set is the same multiset and scores exactly 0.5.
    return {"illegal": block[keep], "control": src, "control2": np.roll(src, 1),
            "reference": np.roll(src, 2), "t0": t0[keep]}


NULL_POOL = np.arange(1, N_LEGAL, 2)   # legal games outside the even-index profile fit


def null_splits(seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Disjoint halves of the held-out legal pool, as positions into NULL_POOL."""
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(NULL_POOL)) for _ in range(NULL_SPLITS)]
    return [(p[:NULL_HALF], p[NULL_HALF:2 * NULL_HALF]) for p in perms]


def scores(space: str, rows: dict, cand: np.ndarray):
    d = steps(space)
    mu, sd = profile(d[np.arange(0, N_LEGAL, 2)])
    t0 = rows["t0"]
    s = {arm: scan(drop(d[rows[arm]], mu, sd), cand, M, PAIRED) for arm in ARMS}
    pool_peak = scan(drop(d[NULL_POOL], mu, sd), cand, M, PAIRED).max(1)
    return ({arm: v.max(1) for arm, v in s.items()},
            {arm: located(v, cand) == t0 for arm, v in s.items()},
            {arm: np.abs(located(v, cand) - t0) <= NEAR for arm, v in s.items()},
            pool_peak)


def row(space: str, peak, hit, near, pool_peak, ref_peak, draws, splits, cand) -> dict:
    against = lambda a, b, p=peak: (lambda i: auroc(p[a][i], p[b][i]))
    out = {"space": space, "statistic": "MOSUM" if PAIRED else "forward window", "m": M,
           "n_games": len(peak["illegal"]), "chance_uniform": 1.0 / len(cand)}
    for name, (a, b) in {"vs_control": ("illegal", "control"),
                         "vs_reference": ("illegal", "reference")}.items():
        lo, hi = ci(against(a, b), draws)
        out |= {f"auroc_{name}": against(a, b)(np.arange(out["n_games"])),
                f"auroc_{name}_lo": lo, f"auroc_{name}_hi": hi}
    nulls = [auroc(pool_peak[a], pool_peak[b]) for a, b in splits]
    out |= {"auroc_null": float(np.mean(nulls)),
            "auroc_null_lo": float(np.quantile(nulls, 0.025)),
            "auroc_null_hi": float(np.quantile(nulls, 0.975))}
    delta = lambda i: against("illegal", "control")(i) - against("illegal", "control", ref_peak)(i)
    lo, hi = ci(delta, draws)
    out |= {"auroc_minus_raw": delta(np.arange(out["n_games"])),
            "auroc_minus_raw_lo": lo, "auroc_minus_raw_hi": hi}
    for arm in ARMS:
        out[f"hit_{arm}"] = float(hit[arm].mean())
    out[f"hit_illegal_within_{NEAR}"] = float(near["illegal"].mean())
    out["hit_illegal_lo"], out["hit_illegal_hi"] = ci(
        lambda i: float(hit["illegal"][i].mean()), draws)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cand = candidates(steps(REFERENCE_SPACE).shape[1], M_MAX)
    rows = corpus(cand)
    print(f"scanning move indices {cand[0] + 2}..{cand[-1] + 2} ({len(cand)} candidates), "
          f"{len(rows['t0'])} regenerated-suffix violations")

    splits = null_splits(args.seed)
    scored = {s: scores(s, rows, cand) for s in SPACES}
    draws = resamples(len(rows["t0"]), args.seed)
    ref_peak = scored[REFERENCE_SPACE][0]
    table = pd.DataFrame([row(s, *scored[s], ref_peak, draws, splits, cand)
                          for s in SPACES])

    (HERE / "results").mkdir(exist_ok=True)
    csv = HERE / "results" / "changepoint_regenerated.csv"
    table.to_csv(csv, index=False)
    print(f"wrote {csv}\n")
    print(table[["space", "auroc_vs_control", "auroc_null", "hit_illegal", "hit_control",
                 "chance_uniform"]].to_string(index=False))


if __name__ == "__main__":
    main()
