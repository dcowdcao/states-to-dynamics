"""Fake probe_fit.csv / readout.csv that conform to the study contract's schemas.

TEST-ONLY. The numbers are invented; the point is that make_figures.py can be
developed and reviewed before the real fitting and read-out land. Delete this
directory once the real CSVs exist.

Usage: python analysis/legality_probe/smoke/synthetic_results.py [--out DIR]
       python analysis/legality_probe/make_figures.py --results <DIR>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spade.common.config import D_MODEL, DICT_SIZE
from spade.figures.paper_style import NULL_SPACES, REFERENCE_SPACE, SPACES
from spade.othello.taxonomy import CATEGORIES, RULE_MEMBERS, RULE_OF
from spade.violation_divergence.divergence import WINDOW

HERE = Path(__file__).resolve().parent
ROLES = ["reference", "control_legal", "control_legal_2", "illegal"]
CLASSES = ["legal", "R1", "R2", "R3"]
FILLER_DIM, N_GAMES = 3, 2000
# Held-out legality macro-F1 per space: raw best, dictionaries spread, controls near
# the floor a shuffled-label fit gives.
KEEP = dict(zip(SPACES, [0.94, 0.88, 0.86, 0.83, 0.79, 0.90, 0.89, 0.72, 0.70, 0.29, 0.31]))
# Board macro-F1 in the same space, tracking bracket_flip's acc_rel_L5 ordering. The
# study's quantity is KEEP - BOARD, so these deliberately do not rank the same way.
BOARD = dict(zip(SPACES, [0.98, 0.97, 0.98, 0.95, 0.94, 0.97, 0.97, 0.76, 0.73, 0.34, 0.36]))
# How far the class is from the mean one, so per-class F1s spread realistically.
CLASS_TILT = {"legal": 0.05, "R1": 0.04, "R2": -0.02, "R3": -0.14}


def ci(rng, value: float, width: float = 0.02) -> tuple[float, float]:
    half = width * rng.uniform(0.6, 1.4)
    return value - half, value + half


def probe_fit(rng) -> pd.DataFrame:
    rows = []
    for space in SPACES:
        d_space = D_MODEL if space == REFERENCE_SPACE else DICT_SIZE
        # legality_wrong_player scores the same 4 classes for the opponent: it stays
        # high because legality is mostly downstream of board-reading.
        targets = [("legality", KEEP[space], len(CLASSES), FILLER_DIM),
                   ("board", BOARD[space], 3, 2),
                   ("legality_wrong_player", KEEP[space] - 0.03, len(CLASSES), FILLER_DIM)]
        for target, macro, n_classes, filler_dim in targets:
            common = {"space": space, "d_space": d_space, "target": target,
                      "n_classes": n_classes, "filler_dim": filler_dim, "n": N_GAMES}
            cells = [("macro_f1", "", macro), ("accuracy", "", 0.90 + 0.09 * macro)]
            cells += [(metric, code, np.clip(macro + tilt + rng.normal(0, 0.012), 0.02, 0.99))
                      for metric in ("per_class_f1", "per_class_recall")
                      for code, tilt in CLASS_TILT.items()]
            for metric, code, value in cells:
                lo, hi = ci(rng, value)
                rows.append({**common, "metric": metric, "class": code,
                             "value": value, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def window_curve(rng, space: str, floor: float, ceiling: float) -> np.ndarray:
    """Flat on the shared prefix, a step down at r = 0, partial recovery after."""
    r = np.asarray(WINDOW, dtype=float)
    drop = (ceiling - floor) * np.where(r < 0, 0.0, np.exp(-np.maximum(r, 0) / 14) * 0.85 + 0.15)
    return np.clip(ceiling - drop + rng.normal(0, 0.004, len(r)), 0.01, 0.99)


def readout(rng) -> pd.DataFrame:
    rows = []
    raw_pooled = {}
    for space in SPACES:
        ceiling = KEEP[space] - 0.03
        floor = 0.30 if space in NULL_SPACES else 0.32 + 0.42 * KEEP[space]
        for role in ROLES:
            hurt = 1.0 if role == "illegal" else 0.12
            recalls = {}
            for category in CATEGORIES[1:] + ["empty_pooled"]:
                curve = window_curve(rng, space, ceiling - hurt * (ceiling - floor), ceiling)
                recalls[category] = np.clip(curve - 0.10 + rng.normal(0, 0.02), 0.01, 0.99)
                rule = RULE_OF[category] if category in RULE_OF else ""
                cells = [("macro_f1", curve),
                         ("p_pred_legal_at_played_square", 1 - curve),
                         ("p_pred_correct_rule", recalls[category]),
                         ("agreement_with_raw", np.clip(curve + 0.06, 0, 0.99))]
                if space == REFERENCE_SPACE:
                    raw_pooled[(role, category)] = curve
                else:
                    cells.append(("macro_f1_minus_raw", curve - raw_pooled[(role, category)]))
                for metric, values in cells:
                    rows += [{"space": space, "category": category, "rule": rule,
                              "role": role, "r": r, "metric": metric, "mean": v,
                              **dict(zip(("lo", "hi"), ci(rng, v))), "n": N_GAMES}
                             for r, v in zip(WINDOW, values)]
            # The rule-level rows the contract's category column carries as "": the
            # real pipeline bootstraps them, this stand-in averages point estimates.
            for rule, members in RULE_MEMBERS.items():
                mean = np.mean([recalls[c] for c in members], axis=0)
                rows += [{"space": space, "category": "", "rule": rule, "role": role,
                          "r": r, "metric": "p_pred_correct_rule", "mean": v,
                          **dict(zip(("lo", "hi"), ci(rng, v))), "n": N_GAMES}
                         for r, v in zip(WINDOW, mean)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    probe_fit(rng).to_csv(args.out / "probe_fit.csv", index=False, float_format="%.6g")
    readout(rng).to_csv(args.out / "readout.csv", index=False, float_format="%.6g")
    print(f"synthetic probe_fit.csv and readout.csv in {args.out}")


if __name__ == "__main__":
    main()
