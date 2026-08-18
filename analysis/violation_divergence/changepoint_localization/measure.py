"""Score every branch of the divergence tree with the reference-free change-point statistic.

The window and the paired/unpaired choice are picked on the *straightness* violation corpus
and then applied here untouched, so the reported numbers are not selection-inflated: the
two corpora share no game, and the divergence tree carries R1 violations and matched legal
controls that the straightness corpus has neither of.

The per-move-index profile that "normal" means is fit on straightness legal games, minus
the ones that ARE divergence games (`meta.held_out_legal`), so no scored trajectory ever
contributed to the baseline it is scored against.

Writes results/changepoint_localization.csv.

Usage: python analysis/violation_divergence/changepoint_localization/measure.py
           [--seed 42]
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
    auroc, boot_ci, candidates, drop, located, profile, scan, shift_chance)

HERE = Path(__file__).resolve().parent
SPACES = [REFERENCE_SPACE] + DICTIONARIES
M_GRID = (4, 6, 8, 10, 12)
N_LEGAL = 2000                       # the straightness legal block, and the violation block
VIOLATION = CLASSES.index("violation")


def npz_stem(space: str) -> str:
    return "resid_post_5" if space == REFERENCE_SPACE else f"sae_{space}"


def steps(directory: Path, space: str, suffix: str = "") -> np.ndarray:
    return np.load(directory / f"{npz_stem(space)}{suffix}.npz")["d"].astype(np.float64)


def selection_corpus(space: str, cand: np.ndarray):
    """(fit d, violation d, matched-legal d, t0) from the straightness study.

    Fit on even-index legal games and score odd-index ones, so the negative arm is outside
    the baseline too. Violations whose t0 falls outside the scanned window are dropped.
    """
    games = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    d = steps(study_dir("straightness") / "geometry", space, "_raw")
    block = np.arange(N_LEGAL) + VIOLATION * N_LEGAL
    src = games["src_game"][block].numpy()
    t0 = games["t0"][block].numpy()
    lo, hi = cand[0] + 2, cand[-1] + 2
    keep = (src % 2 == 1) & (t0 >= lo) & (t0 <= hi)
    return d[np.arange(0, N_LEGAL, 2)], d[block[keep]], d[src[keep]], t0[keep]


def arms(meta: dict, seed: int) -> dict:
    """One row index per game per arm; one illegal branch per game, never a mean over the
    six categories, which would give the illegal arm 1/sqrt(6) of its real spread."""
    gid, cat, ctl = meta["game_id"], meta["category"], meta["is_control"]
    games = np.unique(gid)
    rng = np.random.default_rng(seed)
    pick = lambda mask, g: np.flatnonzero((gid == g) & mask)
    illegal = (cat > 0) & (ctl == 0)
    return {"illegal": np.array([rng.choice(pick(illegal, g)) for g in games]),
            "control": np.array([pick((cat == 0) & (ctl == 1), g)[0] for g in games]),
            "control2": np.array([pick((cat == 0) & (ctl == 2), g)[0] for g in games]),
            "reference": np.array([pick((cat == 0) & (ctl == 0), g)[0] for g in games])}


def sweep(space: str, cand: np.ndarray) -> list[dict]:
    """Every (paired, m) on the selection corpus: AUROC of the violation against its own
    source game, and the localization hit rate."""
    fit, viol, legal, t0 = selection_corpus(space, cand)
    mu, sd = profile(fit)
    rows = []
    for paired in (False, True):
        for m in M_GRID:
            s = {k: scan(drop(x, mu, sd), cand, m, paired) for k, x in
                 (("violation", viol), ("legal", legal))}
            rows.append({"space": space, "paired": paired, "m": m, "n_games": len(t0),
                         "auroc": auroc(s["violation"].max(1), s["legal"].max(1)),
                         "hit_rate": float(np.mean(located(s["violation"], cand) == t0))})
    return rows


def report(space: str, meta: dict, selection: dict, cand: np.ndarray, m: int, paired: bool,
           seed: int) -> dict:
    """The nine-space row: paired detection AUROCs and localization, with CIs over games."""
    legal = steps(study_dir("straightness") / "geometry", space, "_raw")[:N_LEGAL]
    held = set(meta["held_out_legal"].tolist())
    mu, sd = profile(legal[[i for i in range(N_LEGAL) if i not in held]])

    d = steps(study_dir("divergence") / "curvature_discriminant", space)
    s = {arm: scan(drop(d[rows], mu, sd), cand, m, paired) for arm, rows in selection.items()}
    t0 = meta["t0"][selection["reference"]]
    peak = {arm: v.max(1) for arm, v in s.items()}
    hit = {arm: located(v, cand) == t0 for arm, v in s.items()}
    n = len(t0)

    pair = lambda a, b: (lambda i: auroc(peak[a][i], peak[b][i]))
    out = {"space": space, "statistic": "MOSUM" if paired else "forward window", "m": m,
           "n_games": n, "chance_uniform": 1.0 / len(cand),
           "chance_shift": shift_chance(s["illegal"], cand, t0, seed)}
    for name, (a, b) in {"vs_control": ("illegal", "control"),
                         "vs_reference": ("illegal", "reference"),
                         "null": ("control2", "control")}.items():
        lo, hi = boot_ci(pair(a, b), n, seed)
        out |= {f"auroc_{name}": pair(a, b)(np.arange(n)),
                f"auroc_{name}_lo": lo, f"auroc_{name}_hi": hi}
    for arm in ("illegal", "control", "reference"):
        out[f"hit_{arm}"] = float(hit[arm].mean())
    out["hit_illegal_lo"], out["hit_illegal_hi"] = boot_ci(
        lambda i: float(hit["illegal"][i].mean()), n, seed)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    width = steps(study_dir("divergence") / "curvature_discriminant", REFERENCE_SPACE).shape[1]
    cand = candidates(width, max(M_GRID))
    print(f"scanning move indices {cand[0] + 2}..{cand[-1] + 2} "
          f"({len(cand)} candidates, uniform chance {1 / len(cand):.4f})")

    picks = pd.DataFrame([r for space in SPACES for r in sweep(space, cand)])
    best = (picks.groupby(["paired", "m"])[["auroc", "hit_rate"]].mean()
            .sum(axis=1).idxmax())
    paired, m = bool(best[0]), int(best[1])
    print(f"selection corpus ({picks.n_games.iloc[0]} straightness violations): "
          f"{'MOSUM' if paired else 'forward window'}, m = {m}")
    print(picks.groupby(["paired", "m"])[["auroc", "hit_rate"]].mean().to_string())

    meta = dict(np.load(study_dir("divergence") / "curvature_discriminant" / "meta.npz"))
    selection = arms(meta, args.seed)
    table = pd.DataFrame([report(s, meta, selection, cand, m, paired, args.seed)
                          for s in SPACES])
    (HERE / "results").mkdir(exist_ok=True)
    csv = HERE / "results" / "changepoint_localization.csv"
    table.to_csv(csv, index=False)
    picks.to_csv(HERE / "results" / "window_selection.csv", index=False)
    print(f"\nwrote {csv}")
    print(table[["space", "auroc_vs_control", "auroc_vs_reference", "auroc_null",
                 "hit_illegal", "hit_control", "chance_shift"]]
          .rename(columns={"auroc_vs_control": "vs ctl", "auroc_vs_reference": "vs ref"})
          .to_string(index=False))


if __name__ == "__main__":
    main()
