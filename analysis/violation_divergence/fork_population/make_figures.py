"""Where the branches go over a population of games, not over one selected game.

The population twin of `../fork_and_board`. Same nine representation spaces, same
three planes, same third axis (move index t), but every path is the mean over 100
source games that all fork at t0 = 20, so the vertical axis and the star at t0
mean what they meant there.

There is no board here. Every game forks at the same move index and each one has
its own position, so the board `../fork_and_board` panels beside its fork is the
one thing the population cannot draw; that figure's (b) is the cross-reference.

The figure shows shape and nothing else. A path is a mean over games of the
displacement *vector*, so directions that differ between games cancel and the
drawn fork is short -- each cell prints the surviving share (`coherence`), so that
shortness is not read as "nothing moved". Nothing on the canvas is a magnitude and
nothing on it is an interval. The mean of per-game *distances*, which cancels
nothing, and its bootstrap CI over source games are in the CSV this script writes
and in the README's table: the quantitative companion FIGURE_GUIDELINES 4.8 asks
for is a table here, not a second panel.

Usage: python analysis/violation_divergence/fork_population/make_figures.py
       [--plane displacement|r21|d7] [--n_games 100] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import study_dir
from spade.figures.branch_panels import axis_labels, rule_fork_population_panel
from spade.figures.branch_style import below, ordered_legend, rule_member_key
from spade.figures.paper_style import MUTED, apply_style, save_figure
from spade.violation_divergence.branches import FIGURE_GAME
from spade.violation_divergence.rule_groups import rule_rows

FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../fork_and_board
LEGEND_H = 1.25            # a full-width band: one legend, then the notes under it
ROW_GAP = 0.7              # clear air between the grid's axis labels and the band
TITLE_H = 1.2              # inches over the grid: the two-line title, and the draft suptitle
PLANE_NAME = {"displacement": "Displacement",
              "r21": "R$^{21}$ Curvature & Displacement Space",
              "d7": "R$^{21}$ Raw Step-Length Block $d_1$–$d_7$"}
BOOT, BOOT_SEED = 2000, 0  # the CSV's interval; nothing on this canvas is an estimate
CANCELLATION_NOTE = ("every path is a mean of displacement vectors, so directions that "
                     "differ between games cancel and each cell prints the surviving "
                     "share: this figure carries shape, never magnitude. How far each "
                     "rule runs — a mean of per-game distances, which cancels nothing, "
                     "with a 95% bootstrap CI over games — is in "
                     "results/violation_divergence/fork_population_*.csv, and in the "
                     "README's table")
COHERENCE_DEF = ("coherence, per rule = ‖mean over games of D‖ ÷ mean over games of ‖D‖ "
                 "for the path drawn: the share of one game's displacement that "
                 "survives (0 = all of it cancels)")


def rows_of(z) -> list[dict]:
    """The branch rows `branch_style` and `rule_rows` read, out of the npz columns."""
    return [{"category": int(c), "is_control": int(i)}
            for c, i in zip(z["category"], z["is_control"])]


def boot_ci(x: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of `x` [n_games], over source games.

    `draws` is one index matrix shared by every series and every cell, so the
    intervals are read off the same resamples of the same games."""
    lo, hi = np.percentile(x[draws].mean(axis=1), [2.5, 97.5])
    return lo, hi


def coherence_note(z, si: int, rows: list[dict]) -> str:
    """What one cell's fork does not show: how much of a game's displacement is
    left in the rule path the figure draws, at the position it ends on.

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


def figure(z, plane: str, camera_ready: bool) -> Path:
    rows = rows_of(z)
    names, pos, t0 = list(z["space"]), z["pos"], int(z["t0"])
    n_games = len(z["game"])
    nrow = int(np.ceil(len(names) / NCOL))
    # The second control is the null *for* the first, and nothing here scores one
    # against the other; it is a row of the CSV.
    drawn = [i for i, r in enumerate(rows) if r["is_control"] != 2]

    row_h = [CELL_H * nrow, LEGEND_H]
    height = sum(row_h) + ROW_GAP + TITLE_H
    fig = plt.figure(figsize=(CELL_W * NCOL, height))
    # Both in inches over a fraction-of-canvas API: hspace is a fraction of the
    # *average* row height and top is a fraction of the height, so a row added or
    # cut would otherwise resize the seam and the title's headroom with it.
    gs = fig.add_gridspec(2, 1, height_ratios=row_h, left=0.03, right=0.97,
                          top=1 - TITLE_H / height, bottom=0.02,
                          hspace=ROW_GAP / (sum(row_h) / 2))
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.06, hspace=0.16)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("where the branches go: the fork in each representation space, "
                   "one path per rule\n"
                   f"mean over n = {n_games} games, all forking at $t_0$ = {t0}",
                   loc="left", fontsize=11, pad=18)

    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL], projection="3d")
        rule_fork_population_panel(ax, z["coords"][j][drawn], [rows[i] for i in drawn],
                                   pos, t0, legend=(j == 0), zlabel=(j % NCOL == 0))
        labels = axis_labels("pca", z["var"][j])
        ax.set_xlabel(labels[0], labelpad=-2, fontsize=8)
        ax.set_ylabel(labels[1], labelpad=-2, fontsize=8)
        ax.set_title(name, loc="left", fontsize=8.5)
        ax.text2D(0.0, 1.005, coherence_note(z, j, rows), transform=ax.transAxes,
                  fontsize=7, color=MUTED, ha="left", va="top")
        if j == 0:
            keyed = ax

    band = fig.add_subplot(gs[1])
    band.axis("off")
    leg = ordered_legend(band, keyed, 4, loc="center", bbox_to_anchor=(0.5, 0.55))
    # The lines wrap: save_figure crops tight, so a line wider than the canvas
    # widens the canvas past the panel grid instead of being cut.
    y = below(band, leg, 0.10)
    for note in (spread_note(z), COHERENCE_DEF, population_note(z, n_games, t0),
                 CANCELLATION_NOTE):
        y = below(band, band.text(0.5, y, note, ha="center", va="top", fontsize=7.5,
                                  color=MUTED, linespacing=1.5, wrap=True), 0.10)
    rule_member_key(band, x=0.0, y=y)

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
        "game_coords does not average to the drawn path: the CSV's in-plane series " \
        "would not be measured in the plane drawn (a per-game refit plane would do this)"
    draws = np.random.default_rng(BOOT_SEED).integers(args.n_games,
                                                      size=(BOOT, args.n_games))

    df = table(z, draws)
    out = results_dir("violation_divergence") / f"fork_population_{args.plane}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    rows = rows_of(z)
    k = int(np.flatnonzero(z["pos"] == int(z["t0"]))[0])
    # The two quantities the figure keeps apart, side by side, so the README can
    # quote the cancellation instead of measuring it off the figure. The coherence
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
    print(f"wrote {figure(z, args.plane, args.camera_ready)}")


if __name__ == "__main__":
    main()
