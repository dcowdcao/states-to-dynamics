"""Where the branches go over a population of games, not over one selected game.

The population twin of `../fork_and_board`. Same nine representation spaces, same
two planes, same third axis (move index t), but every path is the mean over 100
source games that all fork at t0 = 20, so the vertical axis and the star at t0
mean what they meant there. (a) is that fork; (b) is `../fork_and_board`'s (c),
how far each rule runs, now with an interval over games.

There is no board here. Every game forks at the same move index and each one has
its own position, so the board `../fork_and_board` panels beside its fork is the
one thing the population cannot draw; that figure's (b) is the cross-reference.

Two things the panels disagree about on purpose, which is why both are drawn.
(a) averages displacement *vectors*, the way the fork panels have always taken a
rule mean, so directions that differ between games cancel and the drawn fork is
short; each cell prints the surviving share (`coherence`). (b) averages per-game
*distances*, which cancels nothing. A reader who sees only (a) concludes the
branches barely move, and (b) is the correction.

The uncertainty in (a) is a confidence ellipse of the mean, drawn flat in the
plane at the end of each path: the estimator is a 2-D mean and has no preferred
direction, so a 1-D bar would be a choice of axis and not a measurement. (b)'s is
a percentile bootstrap over source games -- games, not branches, because a game's
branches share a prefix and the control they are divided by.

Usage: python analysis/violation_divergence/fork_population/make_figures.py
       [--plane displacement|r21] [--n_games 100] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from spade.common.paths import results_dir
from spade.common.study import study_dir
from spade.figures.branch_panels import axis_labels, rule_fork_population_panel
from spade.figures.branch_style import below, branch_style, ordered_legend, rule_member_key
from spade.figures.paper_style import GRID, INK, MUTED, NULL, apply_style, save_figure
from spade.violation_divergence.branches import FIGURE_GAME
from spade.violation_divergence.rule_groups import rule_rows

FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../fork_and_board
LEGEND_H = 1.25            # a full-width band: one legend, then the three notes (a) needs
METRIC_H = 3.6             # the (b) row; the ladder needs the legend off its data
# Four rule slots in a space's unit-wide group, not ../fork_and_board's three: at
# its 0.26 the group is wider than the unit and the gap between groups closes below
# the gap inside one.
SLOT_W = 0.21
DOT_DX = 0.055             # the member dots, a hair off the pair of markers they sit under
# A ratio ladder, dense around 1 and sparse above it, in place of matplotlib's log
# decades: the same rungs ../r21_run_population draws this quantity on, extended to
# hold the in-plane series. Whichever ones the population lands between are drawn.
RATIO_TICKS = [0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 4, 6, 8]
LEGEND_HEADROOM = 1.6      # a factor and not a margin: the axis is log
ROW_GAP = 0.7              # clear air between the grid's axis labels and (b)'s title
PLANE_NAME = {"displacement": "Displacement",
              "r21": "R$^{21}$ Curvature & Displacement Space"}
BOOT, BOOT_SEED = 2000, 0
CANCELLATION_NOTE = ("(a) averages displacement vectors, so directions that differ "
                     "between games cancel and each cell prints the surviving share; "
                     "(b) averages per-game distances, which does not cancel")
COHERENCE_DEF = ("coherence, per rule = ‖mean over games of D‖ ÷ mean over games of ‖D‖ "
                 "for the path (a) draws: the share of one game's displacement that "
                 "survives (0 = all of it cancels)")
ELLIPSE_NOTE = ("the ellipses lie flat in the plane, so this camera foreshortens them "
                "and a round one draws as a lens: read them for whether two paths' ends "
                "separate, and (b) for how far anything ran")


def rows_of(z) -> list[dict]:
    """The branch rows `branch_style` and `rule_rows` read, out of the npz columns."""
    return [{"category": int(c), "is_control": int(i)}
            for c, i in zip(z["category"], z["is_control"])]


def cell_series(plane: np.ndarray, space: np.ndarray, rows: list[dict]) -> list[tuple]:
    """One cell's series: (style, in-plane [n_games], full-space [n_games], members).

    A rule's per-game value is the mean over its 2, 3 or 1 member categories, taken
    inside the game. The null is the second legal control, whose members list is
    empty because it is one branch and not a group.
    """
    out = [(branch_style(rows[m[0]]["category"], 0, by_rule=True),
            plane[:, m].mean(axis=1), space[:, m].mean(axis=1), m)
           for m in rule_rows(rows).values()]
    null_row = next(i for i, r in enumerate(rows) if r["is_control"] == 2)
    return out + [(branch_style(0, 2), plane[:, null_row], space[:, null_row], [])]


def boot_ci(x: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of `x` [n_games], over source games.

    `draws` is one index matrix shared by every series and every cell, so the
    intervals are read off the same resamples of the same games."""
    lo, hi = np.percentile(x[draws].mean(axis=1), [2.5, 97.5])
    return lo, hi


def coherence_note(z, si: int, rows: list[dict]) -> str:
    """What one cell's fork does not show: how much of a game's displacement is
    left in the rule path (a) draws, at the position it ends on.

    `coherence_rule` and not `coherence`: the drawn path is the rule mean, which
    cancels across its member categories as well as across games, so the members'
    own coherences average to a different and larger number."""
    per_rule = " · ".join(
        f"{rule.split()[0]} {c:.2f}"
        for rule, c in zip(rule_rows(rows), z["coherence_rule"][si, :, -1]))
    return f"coherence at $t$ = {z['pos'][-1]}: {per_rule}"


def population_note(z, n_games: int, t0: int) -> str:
    """What the population is, and whether the game the single-game figure draws
    is inside it -- `--n_games` may cut the prefix before it."""
    note = (f"the first {n_games} of the {int(z['n_pool'])} branch-complete games "
            f"forking at $t_0$ = {t0}, ascending by game id: no game-level criterion "
            f"enters this figure")
    if FIGURE_GAME in z["game"]:
        note += (f", and game {FIGURE_GAME} — the argmax the single-game figure draws "
                 f"— is in it as one ordinary member of {n_games}")
    return note + " (README)"


def spread_note(z) -> str:
    """The within/between-rule spread ratio, on the canvas (FIGURE_GUIDELINES 1.7).

    The per-game median is the population's; `spread_mean`, the same statistic on
    the mean field, is the spread of three averaged trajectories and a different
    quantity, so the line says which one this is. Space 0, raw layer 5: the space
    `rule_groups` measures the study's own ratio in."""
    g = z["spread_game"][0]
    return (f"within ÷ between-rule spread in {str(z['space'][0]).split(' (')[0]}, per "
            f"game and not the mean field's: median {np.median(g):.2f} "
            f"[IQR {np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}] — near 1, the "
            f"rule grouping is a label more than a finding")


def metric_panel(ax, cells: list[tuple[str, np.ndarray, np.ndarray]], rows: list[dict],
                 draws: np.ndarray, offsets: np.ndarray) -> None:
    """(b): ../fork_and_board's (c), over games, as a ratio ladder.

    Filled marker = the distance in that cell's full space, which is the number
    the study's claim rests on because it is plane-free; open marker = the same
    distance inside the population plane the cell above draws. Both are per-game
    distances divided by that game's own legal control and then averaged over
    games, so nothing cancels the way (a)'s vector mean does. The control is 1
    with no variance -- it is the divisor -- so the empirical null is a *third*
    legal move scored against it, which is what the second control exists for.

    Log y on a ratio ladder, not a linear axis: the two series differ several-fold
    in the displacement plane, so a linear axis is scaled by the in-plane one and
    every full-space marker lands on the line at 1, where it cannot be read. The
    x label carries the measured gap.
    """
    ax.axhline(1, color=NULL["color"], ls=NULL["ls"], lw=1.1, zorder=1)
    lo, hi, gaps = 1.0, 1.0, []
    for j, (_, plane, space) in enumerate(cells):
        ax.axvline(j - 0.5, color=GRID, lw=0.8, zorder=0)
        for m, (st, in_plane, in_space, members) in enumerate(
                cell_series(plane, space, rows)):
            x = j + (m - 1.5) * SLOT_W
            # Same x and same rule marker, so the pair reads as one rule measured
            # twice; fill vs face is what tells the two apart with no colour.
            for values, kw in ((in_space, {"ms": 7, "markerfacecolor": st["color"],
                                           "markeredgecolor": "white",
                                           "elinewidth": 1.2, "capsize": 2.5,
                                           "zorder": 6}),
                               (in_plane, {"ms": 6, "markerfacecolor": "white",
                                           "markeredgecolor": st["color"],
                                           "elinewidth": 0.8, "capsize": 1.8,
                                           "zorder": 5})):
                mu, (l, h) = values.mean(), boot_ci(values, draws)
                ax.errorbar(x, mu, yerr=[[mu - l], [h - mu]], ls="none",
                            marker=st["marker"], color=st["color"],
                            markeredgewidth=1.2, **kw)
                lo, hi = min(lo, l), max(hi, h)
            gaps.append(in_plane.mean() / in_space.mean())
            # The dots are the only within-rule spread on the canvas, so they set
            # the limits alongside the intervals; a bootstrap bound does not bound
            # them. They are the in-plane series': in full space a rule's members
            # sit inside their own mean's marker (the CSV has them).
            dots = plane[:, members].mean(axis=0)
            ax.plot([x + DOT_DX] * len(members), dots, lw=0, marker="o", ms=2.8,
                    color=INK, alpha=0.6, zorder=4)
            lo = min(lo, np.min(dots, initial=lo))
            hi = max(hi, np.max(dots, initial=hi))

    ax.set_yscale("log")
    ax.set_yticks(RATIO_TICKS)          # before the limits: ticks widen the view interval
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_ylim(lo / 1.08, hi * LEGEND_HEADROOM)
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_xticks(range(len(cells)), [name for name, _, _ in cells], fontsize=9)
    ax.set_ylabel("distance from the reference,\n$\\div$ the cell's own legal control\n"
                  "(log; ↑ further than another legal move)", fontsize=9)
    # The estimator rides on the axis, not in a note: a marker is a mean of per-game
    # ratios, which is not the ratio of the means (a) draws. So does what the plane the
    # open marker is measured in was fit on, which is the whole gap between the two.
    ax.set_xlabel(f"per game: the mean over the {len(offsets)} drawn offsets r = "
                  f"{', '.join(str(r) for r in offsets[:2])}, …, {offsets[-1]}, "
                  f"÷ that game's own legal control; markers = the mean of those "
                  f"n = {len(draws[0])} per-game ratios\n(median in the CSV), bars = 95% "
                  f"percentile bootstrap CI over games ({BOOT} resamples, seed "
                  f"{BOOT_SEED}).\nThe open marker's plane is fit on these same "
                  f"{len(draws[0])} games' mean field — the direction the illegal "
                  "branches share — so a distance measured in it is\nalignment-weighted "
                  f"and not plane-free: that is the whole {min(gaps):.1f}–{max(gaps):.1f}× "
                  "gap between the two markers", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(False, axis="x")
    marks = [(plt.Line2D([], [], lw=0, marker="o", ms=7, markerfacecolor=MUTED,
                         markeredgecolor="white", markeredgewidth=1.2),
              "in that cell's full space, plane-free (marker = rule)"),
             (plt.Line2D([], [], lw=0, marker="o", ms=6, markerfacecolor="white",
                         markeredgecolor=MUTED, markeredgewidth=1.2),
              "in the plane its cell draws (2-D, alignment-weighted)"),
             (plt.Line2D([], [], color=INK, lw=1.2),
              "95% CI of the mean over games"),
             (plt.Line2D([], [], lw=0, marker="o", ms=2.8, color=INK, alpha=0.6),
              "one category inside the rule, in the plane"),
             (plt.Line2D([], [], lw=0, marker=NULL["marker"], ms=7,
                         markerfacecolor=NULL["color"], markeredgecolor="white",
                         markeredgewidth=1.2),
              "a third legal move ÷ the control (empirical null)"),
             (plt.Line2D([], [], color=NULL["color"], ls=NULL["ls"], lw=1.1),
              "the cell's legal control (null) = 1")]
    ax.legend(*zip(*marks), ncol=3, fontsize=9, loc="upper left")


def figure(z, draws: np.ndarray, plane: str, camera_ready: bool) -> Path:
    rows = rows_of(z)
    names, pos, t0 = list(z["space"]), z["pos"], int(z["t0"])
    n_games = len(z["game"])
    nrow = int(np.ceil(len(names) / NCOL))
    # (a) drops the second control: it is the null *for* the first, so it belongs
    # in (b), where the first control is what everything is divided by.
    drawn = [i for i, r in enumerate(rows) if r["is_control"] != 2]

    row_h = [CELL_H * nrow, LEGEND_H, METRIC_H]
    fig = plt.figure(figsize=(CELL_W * NCOL, sum(row_h) + 2 * ROW_GAP))
    # hspace is a fraction of the *average* row height, so the gap is one seam
    # wide however the three rows are sized.
    gs = fig.add_gridspec(3, 1, height_ratios=row_h, left=0.03, right=0.97, top=0.93,
                          bottom=0.02, hspace=ROW_GAP / (sum(row_h) / 3))
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.06, hspace=0.16)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("(a) where the branches go: the fork in each representation space, "
                   "one path per rule\n"
                   f"mean over n = {n_games} games, all forking at $t_0$ = {t0}; "
                   "95% CI of that mean at the end of each path",
                   loc="left", fontsize=11, pad=18)

    cells = []
    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL], projection="3d")
        rule_fork_population_panel(ax, z["coords"][j][drawn], z["game_coords"][j][:, drawn],
                                   [rows[i] for i in drawn], pos, t0,
                                   legend=(j == 0), zlabel=(j % NCOL == 0))
        labels = axis_labels("pca", z["var"][j])
        ax.set_xlabel(labels[0], labelpad=-2, fontsize=8)
        ax.set_ylabel(labels[1], labelpad=-2, fontsize=8)
        ax.set_title(name, loc="left", fontsize=8.5)
        ax.text2D(0.0, 1.005, coherence_note(z, j, rows), transform=ax.transAxes,
                  fontsize=7, color=MUTED, ha="left", va="top")
        # The cell's name minus its hyperparameters: (b) has nine ticks to fit and
        # the settings are already on the cell above each of them.
        cells.append((name.split(" (")[0], z["game_plane"][j], z["game_space"][j]))
        if j == 0:
            keyed = ax

    band = fig.add_subplot(gs[1])
    band.axis("off")
    leg = ordered_legend(band, keyed, 4, loc="center", bbox_to_anchor=(0.5, 0.55))
    # Both lines wrap: save_figure crops tight, so a line wider than the canvas
    # widens the canvas past the panel grid instead of being cut.
    y = below(band, leg, 0.10)
    for note in (spread_note(z), COHERENCE_DEF, ELLIPSE_NOTE):
        y = below(band, band.text(0.5, y, note, ha="center", va="top", fontsize=7.5,
                                  color=MUTED, linespacing=1.5, wrap=True), 0.10)

    metric = fig.add_subplot(gs[2])
    metric_panel(metric, cells, rows, draws, pos[pos >= t0] - t0)
    metric.set_title("(b) how far they run: each rule's distance from the reference as a "
                     "multiple of the same cell's legal control —\nin that cell's full "
                     "space (filled marker, the plane-free number) and in the plane its "
                     "cell draws (open marker); log scale",
                     loc="left", fontsize=11)

    y = below(fig, metric.xaxis.label, 0.010)
    for note in (population_note(z, n_games, t0), CANCELLATION_NOTE):
        y = below(fig, fig.text(0.005, y, note, ha="left", va="top", fontsize=7.5,
                                color=MUTED, wrap=True), 0.004)
    rule_member_key(fig, x=0.005, y=y)

    if not camera_ready:
        fig.suptitle(f"Fork Trajectory ({PLANE_NAME[plane]} PCA) in Each Representation "
                     f"Space — {n_games} games at $t_0$ = {t0}", fontsize=12)
    return save_figure(fig, FIG_DIR / f"fork_population_{plane}", camera_ready)


def table(z, draws: np.ndarray) -> pd.DataFrame:
    """Every drawn marker, plus the median, so the README can quote the skew a mean
    of per-game ratios hides without anyone re-running the sweep."""
    rows = rows_of(z)
    # The rules the markers draw, then every branch behind them: the empirical null
    # and the six categories, each under the role name the npz gives it.
    series = list(rule_rows(rows).items())
    series += [(str(z["role"][i]), [i]) for i, r in enumerate(rows)
               if r["category"] != 0 or r["is_control"] == 2]
    out = []
    for si, space in enumerate(z["space"]):
        for name, members in series:
            for where, key in (("plane", "game_plane"), ("full space", "game_space")):
                x = z[key][si][:, members].mean(axis=1)
                lo, hi = boot_ci(x, draws)
                out.append({"space": space, "series": name, "where": where,
                            "n_games": len(x), "n_members": len(members),
                            "ratio": x.mean(), "ratio_lo": lo, "ratio_hi": hi,
                            "median": np.median(x)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plane", default="displacement", choices=list(PLANE_NAME))
    ap.add_argument("--n_games", type=int, default=100, help="names the npz to read")
    ap.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = ap.parse_args()
    apply_style(args.camera_ready)

    npz = study_dir("divergence") / "fork_population" / f"{args.plane}_n{args.n_games}.npz"
    z = dict(np.load(npz))
    assert len(z["game"]) == args.n_games, f"{npz.name} holds {len(z['game'])} games"
    assert np.allclose(z["game_coords"].mean(axis=1), z["coords"], atol=1e-5), \
        "game_coords does not average to the drawn path: the ellipse would not be " \
        "centred on the mean it is a CI of (a per-game refit plane would do this)"
    draws = np.random.default_rng(BOOT_SEED).integers(args.n_games,
                                                      size=(BOOT, args.n_games))

    df = table(z, draws)
    out = results_dir("violation_divergence") / f"fork_population_{args.plane}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    rows = rows_of(z)
    k = int(np.flatnonzero(z["pos"] == int(z["t0"]))[0])
    # The two quantities the figure keeps apart, side by side, so the README can
    # quote the cancellation instead of measuring it off panel (a). The coherence
    # is the drawn rule paths', which is what the panels annotate.
    print("distance of the mean field | mean over games of the per-game distance, "
          "both ÷ the legal control, in the plane — then the three rule paths' "
          "mean coherence")
    for si, space in enumerate(z["space"]):
        coh = z["coherence_rule"][si]
        print(f"  {space.split(' (')[0]:<14} " + "  ".join(
            f"{rule.split()[0]} {z['ratio_plane'][si, m].mean():5.2f}|"
            f"{z['game_plane'][si][:, m].mean():5.2f}"
            for rule, m in rule_rows(rows).items())
            + f"   coherence r = 0 {coh[:, k].mean():.3f}"
            f"  r = +{z['pos'][-1] - int(z['t0'])} {coh[:, -1].mean():.3f}")
    # Raw layer 5, the space rule_groups measures the study's own ratio in.
    g = z["spread_game"][0]
    print(f"within/between rule spread ratio, {args.plane}, {z['space'][0]}: per game "
          f"median {np.median(g):.3f} [IQR {np.percentile(g, 25):.3f}–"
          f"{np.percentile(g, 75):.3f}], above 1 in {(g > 1).mean():.0%}; on the mean "
          f"field {z['spread_mean'][0]:.3f} (a different quantity)")
    print(f"wrote {out}")
    print(f"wrote {figure(z, draws, args.plane, args.camera_ready)}")


if __name__ == "__main__":
    main()
