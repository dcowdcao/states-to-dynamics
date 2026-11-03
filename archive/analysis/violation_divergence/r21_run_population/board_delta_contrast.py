"""Does the board delta of the branch move grade the run in R^21?

The two R1 categories differ in exactly one thing a board-delta account cares
about: `occupied_own` re-plays a square that is already the mover's and changes
the board not at all, while `occupied_opp` recolors an opponent disc. If what the
fork measures is the board delta rather than the illegality, `occupied_own` is the
more extreme of the two and this contrast is negative. The categorical half of
that account -- "no new disc" against "new disc" -- is perfectly confounded with
R1 against the rest and is not testable in this taxonomy; this graded half is.

Paired within game, because the two categories share the prefix, the reference and
the control divisor, all of which cancel; the interval is a 2000-draw percentile
bootstrap over games, which does not assume the difference is symmetric.

Reads `measure.py`'s npz. `../fork_population` runs the same contrast in the
displacement plane and in the step-length norm; the three numbers belong together
and each README carries its own.

Usage: python analysis/violation_divergence/r21_run_population/board_delta_contrast.py
"""

import numpy as np

from spade.common.study import study_dir

DRAWS, SEED = 2000, 0


def bootstrap(d: np.ndarray) -> tuple[float, float, float]:
    """(mean, lo, hi) of the paired difference, resampling games."""
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(d), size=(DRAWS, len(d)))
    return (float(d.mean()), *np.percentile(d[draws].mean(1), [2.5, 97.5]))


def main():
    z = np.load(study_dir("divergence") / "r21_run_population" / "runs.npz")
    cat = list(z["category"])
    own, opp = cat.index("occupied_own"), cat.index("occupied_opp")
    d = z["member"][:, :, own] - z["member"][:, :, opp]

    print(f"occupied_own - occupied_opp, run from the reference / the legal control's,\n"
          f"full R^21, paired over {d.shape[1]} games, {DRAWS}-draw percentile bootstrap\n"
          "negative = the no-board-change category runs less far, which is what a "
          "board-delta account predicts\n")
    resolved = 0
    for space, row in zip(z["space"], d):
        mean, lo, hi = bootstrap(row)
        mark = "" if lo <= 0 <= hi else "  resolved"
        resolved += bool(mark)
        print(f"  {space:<34} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}")
    print(f"\nresolved in {resolved} of {len(d)} spaces")


if __name__ == "__main__":
    main()
