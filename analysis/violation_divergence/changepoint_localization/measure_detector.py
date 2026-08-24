"""The thresholded read-out over every space, on the detector study's evaluation set.

`measure.py` scores the 575 games `curvature_discriminant` selected: t0 in [16, 34], and 46%
of them at t0 = 20, because the tree was forked there before `extend_branches` widened it.
`illegal_move_detector` holds 2,415 branch-complete games over the whole fork window and deals
a t0-flat 500 of them -- 20 games at each of 25 positions -- which is the population its own
figures report. Panel (a) of `threshold_detection` is read beside those figures, so it is
scored here, on that corpus and those 500 games, rather than on the other one.

The statistic is unchanged: `measure.threshold_row`, the same score read as a per-position
detector over every candidate ground truth covers. Only the population moves, and
`results/eval_games.csv` -- written by `illegal_move_detector.figure_auroc` -- is what says the
population is the same one, checked here rather than assumed.

`--l0-matched` swaps the seed-42 roster for the eight `l0_operating_point` picks, one
checkpoint per architecture seated at the shared L0, so a difference between two bars is the
architecture and not its sparsity. Population, statistic and operating point are untouched, so
the matched table is comparable with the canonical one row for row -- which it was not while
the matched figure read `threshold_localization_l0_matched.csv`, a table on the other corpus
that no histogram of these 500 games can be drawn beside.

Writes results/threshold_detection[_l0_matched].csv.

Usage: python analysis/violation_divergence/changepoint_localization/measure_detector.py
           [--l0-matched]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spade.figures.paper_style import DICTIONARIES, REFERENCE_SPACE
from spade.l0_operating_point import roster

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "illegal_move_detector"))

import detector  # noqa: E402  (sibling study, not an installed package)
from measure import POOL_ARMS, positions, threshold_row  # noqa: E402


def evaluation(data: dict, sel: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Row indices per arm over the committed 500 games, and their t0."""
    keep = detector.stratified(data, sel)
    rows = {arm: sel[arm][keep] for arm in POOL_ARMS}
    t0 = data["t0"][rows["reference"]]
    detector.assert_eval_set(data["game_id"][rows["reference"]], t0)
    return rows, t0


def line(space: str, cand: np.ndarray, rows: dict, pool) -> dict:
    d = detector.load(space)
    s = detector.score(d, cand)
    return threshold_row(space, {arm: s[r] for arm, r in rows.items()}, pool,
                         detector.M, detector.PAIRED, detector.SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                        help="score the eight l0_operating_point picks instead of the "
                             "canonical seed-42 roster, so a difference between two spaces "
                             "is the architecture and not its sparsity")
    args = parser.parse_args()
    spaces = [REFERENCE_SPACE] + (roster.SPACES if args.l0_matched else DICTIONARIES)
    suffix = "_l0_matched" if args.l0_matched else ""

    data = detector.load()
    cand = detector.band()
    sel = detector.arms(data)
    rows, t0 = evaluation(data, sel)
    pool = positions(data, rows, cand, t0, detector.SEED)
    print(f"scanning move indices {cand[0] + 2}..{cand[-1] + 2} over {len(t0)} games; "
          f"{len(pool.label)} positions, {int(pool.label.sum())} illegal "
          f"({pool.label.mean():.4f} prevalence)")

    table = pd.DataFrame([line(space, cand, rows, pool) for space in spaces])

    out = HERE / "results" / f"threshold_detection{suffix}.csv"
    table.round(5).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(table[["space", "auroc_position", "auroc_position_lo", "auroc_position_hi",
                 "threshold", "balanced_accuracy", "tpr", "tnr"]].round(4)
          .to_string(index=False))


if __name__ == "__main__":
    main()
