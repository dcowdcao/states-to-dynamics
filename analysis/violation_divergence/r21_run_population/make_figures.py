"""How far an illegal branch runs from the reference, over the population.

`../r21_shared_basis/run_from_reference` asks the question the nine fork grids
raise and none of them can answer -- how far a branch actually runs, and how much
of that run each candidate plane keeps -- for one game, with no interval. This is
the same two panels off the same descriptor and the same planes, refit per game,
with the mean over every branch-complete source game and a 95% CI on it.

(a) puts the four candidate planes against full R^21; (b) splits full R^21 into
its 11 step-length and 10 turning-angle components -- the disagreement one game
shows does not survive the population -- over the six illegal categories the mean
is taken over. Everything is a ratio to the same
game's legal control, because each field is z-scored per space and the refit
planes are fit per space, so nothing absolute carries from one cell to the next.
Isomap is in neither: it is not a linear image of R^21 and its reference is not
at the origin, so a distance in it is not a share of anything.

Reads `measure.py`'s npz. Writes the figure and one tidy CSV of every drawn
point; prose belongs in README.md.

Usage: python analysis/violation_divergence/r21_run_population/make_figures.py
       [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import study_dir
from spade.figures.paper_style import (
    FULL_STYLE, GRID, INK, MUTED, NULL, PLANE_STYLE, QUANTITY_STYLE, apply_style,
    save_figure)
from spade.violation_divergence.fork_r21 import BLOCKS

HERE = Path(__file__).resolve().parent
RUN_AXIS = ("run from the reference ÷ the control's\n(log, ↑ further than another "
            "legal move)")
# A ratio ladder, not a linear one, and wide enough to hold the single-game
# figure's range: whichever rungs the population lands between are the ones drawn.
RUN_TICKS = [0.25, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 4, 8]
NULL_SERIES = "control 2"
MEMBER = {"color": MUTED, "lw": 0.7, "alpha": 0.38,
          "label": "one illegal category (the 6 the mean is over)"}


def mean_ci(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over games (axis 0) and the half-width of its 95% CI. One game is one
    item in every series -- the planes are refit per game -- so no clustering
    correction is needed."""
    return x.mean(0), 1.96 * x.std(0, ddof=1) / np.sqrt(len(x))


def draw(ax, x, mu, half, st: dict, marker_only: bool = False) -> None:
    ax.errorbar(x, mu, yerr=half, color=st["color"], marker=st["marker"],
                ls="none" if marker_only else st.get("ls", "-"), ms=5,
                lw=st.get("lw", 1.6), elinewidth=1.1, capsize=2.5,
                markeredgecolor="white", markeredgewidth=0.5, label=st["label"],
                zorder=st.get("zorder", 4))


def series_mean(z, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(mean, CI half-width) per space for one named series."""
    return mean_ci(z["ratio"][:, :, list(z["series"]).index(name)].T)


def spread_note(z) -> str:
    """What the mean is over: the game-to-game SD of the full-R^21 ratio, whose
    CI is 1/sqrt(n) of it and therefore hairline at this n (1.7)."""
    sd = z["ratio"][:, :, list(z["series"]).index("full")].std(1, ddof=1)
    return (f"game-to-game SD of the full $R^{{21}}$ ratio: {sd.min():.2f}–{sd.max():.2f} "
            f"across the nine cells — the CI is ~1/{np.sqrt(z['ratio'].shape[1]):.0f} of "
            "the spread it summarises.")


def limits(z) -> tuple[float, float]:
    """The drawn range: every series' CI and every member line, with a log margin."""
    mu, half = mean_ci(np.moveaxis(z["ratio"], 1, 0))
    lo = min((mu - half).min(), z["member"].mean(1).min())
    hi = max((mu + half).max(), z["member"].mean(1).max())
    return lo / 1.06, hi * 1.06


def figure(z, camera_ready: bool) -> Path:
    names = list(z["space"])
    x = np.arange(len(names))
    n_games = z["ratio"].shape[1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    for key, st in PLANE_STYLE.items():
        draw(ax, x, *series_mean(z, key), st)
    draw(ax, x, *series_mean(z, "full"), FULL_STYLE)
    ax.set_title("(a) how much of the run each candidate plane keeps", loc="left",
                 fontsize=9.5)

    members = z["member"].mean(1)          # [S, 6]: each category's mean over games
    bx.plot(x, members, **{k: v for k, v in MEMBER.items() if k != "label"}, zorder=2)
    bx.plot([], [], **MEMBER)
    for key, st in (("full", FULL_STYLE), ("d", QUANTITY_STYLE["d"]),
                    ("theta", QUANTITY_STYLE["theta"])):
        label = st["label"] if key == "full" else f"{st['label']} only ({len(BLOCKS[key])} of the 21)"
        draw(bx, x, *series_mean(z, key), st | {"label": label})
    bx.set_title("(b) what the single $R^{21}$ number averages: step length vs turning angle",
                 loc="left", fontsize=9.5)
    # Why every member sits above the mean, which is otherwise read as a plotting
    # error: the mean is taken on the displacement vectors, as the fork panels take
    # it, so the six categories' incoherent parts cancel before the run is measured.
    bx.text(0.02, 0.98, "the mean is over displacement vectors, not over runs: the six\n"
            "categories' incoherent parts cancel, so it sits below every member",
            transform=bx.transAxes, fontsize=7, color=MUTED, style="italic",
            ha="left", va="top")

    for a in (ax, bx):
        a.axhline(1.0, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=1,
                  label="the legal control itself (1 by construction)")
        draw(a, x, *series_mean(z, NULL_SERIES),
             NULL | {"label": "a third legal move ÷ the control, full $R^{21}$ "
                              "(empirical null)"}, marker_only=True)
        a.set_yscale("log")
        a.set_yticks(RUN_TICKS)
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.minorticks_off()
        a.set_ylabel(RUN_AXIS, fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        a.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    # One y range for both panels: (b) is (a)'s full-R^21 curve taken apart, and a
    # per-panel autoscale would put the two on silently different scales.
    ax.set_ylim(*limits(z)), bx.set_ylim(*limits(z))

    both = [h + b for h, b in zip(ax.get_legend_handles_labels(),
                                  bx.get_legend_handles_labels())]
    seen = {lab: h for h, lab in zip(*both)}
    # Nulls last (FIGURE_GUIDELINES 3); the sort is stable, so everything else keeps
    # the order it was drawn in, which is the order the panels read in.
    order = sorted(seen, key=lambda lab: "null" in lab or "control itself" in lab)
    fig.legend([seen[lab] for lab in order], order, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.26))
    if not camera_ready:
        fig.suptitle("How far an illegal branch runs from the reference, and how much "
                     "of it survives the plane", fontsize=10.5)
    fig.tight_layout()
    fig.text(0.5, -0.30, "\n".join([
        f"mean over n = {n_games} branch-complete source games at $t_0$ = 20; bars = 95% "
        "CI of that mean, over games.",
        "Every plane and the z-scoring are refit per game, so one game is one item. "
        "10 anchors $t-t_0$ = 6…24, W = 8.",
        spread_note(z),
        "Ratios, so only the within-space contrast carries across cells; the reference "
        "is 0 in every field by construction."]),
        ha="center", va="top", fontsize=7.5, color=MUTED)
    return save_figure(fig, HERE / "figures" / "run_from_reference_population",
                       camera_ready)


def table(z) -> pd.DataFrame:
    """Every drawn point, plus the median, so the README can quote the skew a mean
    of ratios can hide without anyone re-running the sweep."""
    rows = []
    for si, space in enumerate(z["space"]):
        for pi, name in enumerate(z["series"]):
            r = z["ratio"][si, :, pi]
            mu, half = mean_ci(r)
            rows.append({"space": space, "series": name, "n_games": len(r),
                         "ratio": mu, "ratio_lo": mu - half, "ratio_hi": mu + half,
                         "median": np.median(r)})
        for ci, cat in enumerate(z["category"]):
            r = z["member"][si, :, ci]
            mu, half = mean_ci(r)
            rows.append({"space": space, "series": f"full, {cat}", "n_games": len(r),
                         "ratio": mu, "ratio_lo": mu - half, "ratio_hi": mu + half,
                         "median": np.median(r)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = ap.parse_args()
    apply_style(args.camera_ready)

    z = dict(np.load(study_dir("divergence") / "r21_run_population" / "runs.npz"))
    df = table(z)
    out = results_dir("violation_divergence") / "r21_run_population.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    wide = df[df.series.isin(list(z["series"]))].pivot(
        index="space", columns="series", values="ratio").reindex(list(z["space"]))
    print(wide[list(z["series"])].round(2).to_string())
    print(f"wrote {out}")
    print(f"wrote {figure(z, args.camera_ready)}")


if __name__ == "__main__":
    main()
