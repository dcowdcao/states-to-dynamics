"""What the detector's cut is cutting: the score's distribution under legal and illegal moves.

One panel, five overlapping densities of the detector score read at each row's own t0 -- the
value the study's AUROCs rank and the value tau is compared against. Higher means the step at
t0 is shorter than the legal profile expects, so the cut flags everything to the right of tau.

`legal` is the matched legal control arm: a legal move at the same t0 of the same prefix, so
the five series differ only in the move that was played. `illegal` is the pooled arm -- one
branch per game drawn from R1/R2/R3 at seed 42 -- and R1, R2, R3 are that same arm's
components drawn separately, not four independent conditions.

The evaluation set is `detector.stratified`: the same number of games at every t0 in the
scored range, so the curves are flat over move index rather than weighted toward whichever
position the tree happens to hold most of. The five arms are kept aligned game by game
(asserted on game_id and on t0), so all five curves are the same games at the same forks. Per-series n, mean, median and the fraction above tau, the last with a 95%
bootstrap CI over the same 200 games, go to results/score_histograms.csv.

**The curves are counts, not densities.** Each is `straightness_panels.kde_ci`'s Gaussian KDE
(Scott's rule) times n times BIN, so a curve reads as "games per BIN-wide slice of the axis"
and its height is on the same scale the binned version drew. BIN is one shared width over the
pooled range, as the shared bin edges were, or five curves off five bandwidths would not be
comparable. The bootstrap band `kde_ci` also returns is not drawn: five overlapping curves
cannot each carry one and stay readable, the departure from FIGURE_GUIDELINES 1.2 that
`straightness_panels.density_panel` documents for the same reason.

Raw layer 5 only, on the spliced branch tree.

Usage: PYTHONPATH=src:analysis/violation_divergence/illegal_move_detector python
           analysis/violation_divergence/illegal_move_detector/figure_histograms.py
           [--camera-ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import detector
from changepoint import ci, resamples
from spade.figures.paper_style import (
    GENERATION_STYLE, NULL, RULE_STYLE, apply_style, save_figure)
from spade.figures.straightness_panels import kde_ci

FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
SLICES = 22                 # sets the count scale: one slice of the pooled range
GRID = 512                  # points the KDE is evaluated on

SERIES = ["legal", "illegal", "R1", "R2", "R3"]
ARM_OF = {"legal": "control", "illegal": "illegal", "R1": "R1", "R2": "R2", "R3": "R3"}
# Hue: the rules taxonomy for legal and R1-R3; GENERATION_STYLE's aggregate hue for the pooled
# arm, the one hue in the palette that already means "any illegal move, over R1-R3" and the
# only one that collides with neither a rule nor REFERENCE's black.
COLOR = {"legal": RULE_STYLE["legal"]["color"],
         "illegal": GENERATION_STYLE["mass"]["color"],
         "R1": RULE_STYLE["R1 placement"]["color"],
         "R2": RULE_STYLE["R2 bracketing"]["color"],
         "R3": RULE_STYLE["R3 turn"]["color"]}
# Five curves cannot separate by hue in grayscale, so the dash carries them there.
DASH = {"legal": "-", "illegal": (0, (6, 1.6)), "R1": (0, (1, 1.3)),
        "R2": (0, (4, 1.4, 1, 1.4)), "R3": (0, (2.6, 1.3))}
LABEL = {"legal": "legal control (baseline)", "illegal": "illegal",
         "R1": "R1 placement", "R2": "R2 bracketing", "R3": "R3 turn"}
# Five fills at one alpha is mud and unreadable in grayscale (FIGURE_GUIDELINES 4.4), so the
# fill is spent on the contrast the panel is about -- baseline against pooled illegal -- and
# the pooled arm's three components ride on their lines over a light wash.
FILL = {"legal": 0.22, "illegal": 0.30, "R1": 0.0, "R2": 0.0, "R3": 0.0}
LINE = {"legal": 2.0, "illegal": 2.0, "R1": 1.4, "R2": 1.4, "R3": 1.4}


def subset(data: dict, sel: dict) -> dict[str, np.ndarray]:
    """Row indices per series over `detector.stratified`'s games, arms aligned game by game."""
    keep = detector.stratified(data, sel)
    games = data["game_id"][sel["reference"]][keep]
    rows = {k: sel[ARM_OF[k]][keep] for k in SERIES}
    for name, r in rows.items():
        assert (data["game_id"][r] == games).all(), f"{name} is not aligned to the game order"
        assert (data["t0"][r] == data["t0"][rows["legal"]]).all(), f"{name} sits at another t0"
    return rows


def summary(x: dict[str, np.ndarray], tau: float) -> pd.DataFrame:
    draws = resamples(len(next(iter(x.values()))), detector.SEED)
    rows = []
    for name, v in x.items():
        lo, hi = ci(lambda i: float((v[i] > tau).mean()), draws)
        rows.append({"series": name, "n": len(v), "mean": v.mean(), "median": np.median(v),
                     "frac_above_tau": (v > tau).mean(),
                     "frac_above_tau_lo": lo, "frac_above_tau_hi": hi, "tau": tau})
    return pd.DataFrame(rows)


def panel(ax, x: dict[str, np.ndarray], tau: float) -> None:
    """The five densities and the cut, on a caller's axes so another figure can reuse them.

    Split out of `draw` so a panel of this figure elsewhere is the same code rather than a
    copy that drifts: same grid, same bandwidth, same fills, same legend.
    """
    pooled = np.concatenate([x[k] for k in SERIES])
    grid = np.linspace(pooled.min(), pooled.max(), GRID)
    slice_width = (pooled.max() - pooled.min()) / SLICES

    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    top = 0.0
    for i, k in enumerate(SERIES):
        curve = kde_ci(x[k], grid)[0] * len(x[k]) * slice_width
        if FILL[k]:
            ax.fill_between(grid, curve, color=COLOR[k], alpha=FILL[k], lw=0, zorder=2)
        ax.plot(grid, curve, color=COLOR[k], ls=DASH[k], lw=LINE[k], zorder=3 + i,
                label=LABEL[k])
        top = max(top, curve.max())

    ceiling = top * 1.34
    # Stopped below the legend: a full-height line runs through it.
    ax.axvline(tau, ymax=top * 1.06 / ceiling, color=NULL["color"], ls=NULL["ls"], lw=1.1,
               zorder=1, label=f"$\\tau$ = {tau:.3f} (threshold)")

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, ceiling)
    ax.set_xlabel("standardized norm of step length")
    ax.set_ylabel("number of games")
    ax.legend(loc="upper center", ncol=2, fontsize=8.0, labelspacing=0.32, borderpad=0.2,
              columnspacing=1.4, handlelength=2.4, alignment="left")


def draw(x: dict[str, np.ndarray], tau: float, camera_ready: bool, stem: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    panel(ax, x, tau)
    return save_figure(fig, FIGURES / stem, camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    parser.add_argument("--space", default=detector.REFERENCE,
                        help="representation space; needs extract_geometry.py --spaces first")
    args = parser.parse_args()

    data = detector.load(args.space)
    cand = detector.band()
    s = detector.score(data, cand)
    sel = detector.arms(data)
    tau = detector.operating_point(s, cand, data, sel)
    rows = subset(data, sel)
    x = {k: detector.at_t0(s, cand, data["t0"], r) for k, r in rows.items()}

    apply_style(args.camera_ready)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = "score_histograms" + ("" if args.space == detector.REFERENCE else f"_{args.space}")
    table = summary(x, tau).assign(space=args.space)
    table.to_csv(RESULTS / f"{stem}.csv", index=False)
    print(table.to_string(index=False))
    print(draw(x, tau, args.camera_ready, stem))


if __name__ == "__main__":
    main()
