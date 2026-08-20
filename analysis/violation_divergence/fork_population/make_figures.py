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

`--variant rollout` draws the same nine cells over the rollout branch set, where
every branch resamples its continuation from OthelloGPT. That set saturates, so
its window stops at `measure.ROLLOUT_RMAX`; the title says where, and README.md
carries the rest of the caption. Its files carry a `_rollout` tag, so the spliced
figure and CSV are never written over.

Usage: python analysis/violation_divergence/fork_population/make_figures.py
       [--plane displacement|r21|d7] [--n_games 100] [--variant spliced|rollout]
       [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.figures.branch_panels import axis_labels, rule_fork_population_panel
from spade.figures.branch_style import ordered_legend
from spade.figures.paper_style import MUTED, apply_style, save_figure
from spade.violation_divergence.rule_groups import rule_rows

VARIANTS = {"spliced": "", "rollout": "rollout"}
FIG_DIR = Path(__file__).resolve().parent / "figures"
NCOL = 3
CELL_W, CELL_H = 3.7, 3.5  # one fork cell, as in ../fork_and_board
LEGEND_H = 0.75            # a full-width band holding the one legend
ROW_GAP = 0.7              # clear air between the grid's axis labels and the band
TITLE_H = 1.2              # inches over the grid: the two-line title, and the draft suptitle
PLANE_NAME = {"displacement": "Displacement",
              "r21": "R$^{21}$ Curvature & Displacement Space",
              "d7": "R$^{21}$ Raw Step-Length Block $d_1$–$d_7$"}
BOOT, BOOT_SEED = 2000, 0  # the CSV's interval; nothing on the canvas is an estimate


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


def figure(z, plane: str, variant: str, camera_ready: bool) -> Path:
    rows = rows_of(z)
    names, pos, t0 = list(z["space"]), z["pos"], int(z["t0"])
    n_games = len(z["game"])
    # The rollout set saturates and is drawn short, so the window rides on the
    # canvas beside the population it is a mean over (`measure.ROLLOUT_RMAX`).
    window = f"; {variant} branches, drawn to $r$ = +{int(pos[-1]) - t0}" if variant else ""
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
                   f"mean over n = {n_games} games, all forking at $t_0$ = {t0}{window}",
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
    ordered_legend(band, keyed, 4, loc="center", bbox_to_anchor=(0.5, 0.5))

    if not camera_ready:
        fig.suptitle(f"Fork Trajectory ({PLANE_NAME[plane]} PCA) in Each Representation "
                     f"Space — {n_games} games at $t_0$ = {t0}"
                     + (f", {variant} branches" if variant else ""), fontsize=12)
    suffix = f"_{variant}" if variant else ""
    return save_figure(fig, FIG_DIR / f"fork_population_{plane}{suffix}", camera_ready)


def table(z, draws: np.ndarray) -> pd.DataFrame:
    """The figure's quantitative companion (FIGURE_GUIDELINES 4.8), and the record
    of the study's own numbers: one row per (space, series, plane | full space) with
    the mean over games, its bootstrap CI and the median the mean's skew hides."""
    rows = rows_of(z)
    # The three rules, then every branch behind them: the empirical null and the six
    # categories, each under the role name the npz gives it.
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
    ap.add_argument("--variant", default="spliced", choices=list(VARIANTS),
                    help="rollout: the branch set that resamples every continuation")
    ap.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = ap.parse_args()
    apply_style(args.camera_ready)

    variant = VARIANTS[args.variant]
    npz = (variant_dir("divergence", variant=variant) / "fork_population"
           / f"{args.plane}_n{args.n_games}.npz")
    z = dict(np.load(npz))
    assert len(z["game"]) == args.n_games, f"{npz.name} holds {len(z['game'])} games"
    assert np.allclose(z["game_coords"].mean(axis=1), z["coords"], atol=1e-5), \
        "game_coords does not average to the drawn path: the CSV's in-plane series " \
        "would not be measured in the plane drawn (a per-game refit plane would do this)"
    draws = np.random.default_rng(BOOT_SEED).integers(args.n_games,
                                                      size=(BOOT, args.n_games))

    df = table(z, draws)
    suffix = f"_{variant}" if variant else ""
    out = results_dir("violation_divergence") / f"fork_population_{args.plane}{suffix}.csv"
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
    print(f"wrote {figure(z, args.plane, variant, args.camera_ready)}")


if __name__ == "__main__":
    main()
