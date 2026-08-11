"""The two generation-legality figures. Drawing only; every statistic is in results/.

F1 `generation_legality`         — can the model generate a legal move at all?
F2 `generation_legality_by_rule` — when it cannot, which constraint rule goes?

Both read the CSVs `spade.generation_legality.aggregate` writes. Nothing here computes a
statistic, a bootstrap or a bandwidth.

Two conventions a reader has to know, which the panels state for themselves. The rule series
in F2 are **sums** over their taxonomy categories, not means, so the legend says `sum of 3`
(FIGURE_GUIDELINES 1.7). And every skew number is against a per-position matched null: the
model's own illegal mass at each ply, spread uniformly over that ply's illegal squares. A
rule's availability runs from 5/60 squares to ~55/60 across a game, so a pooled square-count
null would mostly draw that drift.

Usage: python analysis/generation_legality/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from spade.figures.paper_style import (
    CATEGORY_STYLE, GENERATION_STYLE, INK, MUTED, NULL, REFERENCE, RULE_STYLE,
    apply_style, rule_key, save_figure)
from spade.othello.taxonomy import RULE_MEMBERS

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RULES = {rule.split()[0]: (rule, members) for rule, members in RULE_MEMBERS.items()}
MARKS = 8          # markers per curve; the move axis has 56 points
NOTE = {"fontsize": 7, "color": MUTED}
LEGEND_BAND = 0.21  # figure fraction reserved under the panels for the legends


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def legend_band(fig, handles: list, column: int, n_columns: int = 3, ncol: int = 1) -> None:
    """One legend per panel, in the band tight_layout reserved under the panels."""
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=ncol,
               bbox_to_anchor=((2 * column + 1) / (2 * n_columns), LEGEND_BAND - 0.005),
               fontsize=7.5, frameon=False)


def read(results: Path) -> dict[str, pd.DataFrame]:
    names = ["by_move", "played_rate", "rule_skew", "survival", "illegal_density", "overall"]
    return {n: pd.read_csv(results / f"{n}.csv") for n in names}


def band(ax, x, frame: pd.DataFrame, style: dict, label: str, value: str = "mean",
         alpha: float = 0.22, **kw):
    """Point curve with its bootstrap band, markers thinned to MARKS."""
    every = max(1, len(x) // MARKS)
    ax.fill_between(x, frame.lo, frame.hi, color=style["color"], alpha=alpha, lw=0,
                    zorder=kw.get("zorder", 3) - 1)
    return ax.plot(x, frame[value], color=style["color"], ls=style.get("ls", "-"),
                   marker=style["marker"], ms=4, markevery=(every // 2, every),
                   label=label, **kw)[0]


def move_series(by_move: pd.DataFrame, metric: str) -> pd.DataFrame:
    return by_move[by_move.metric == metric].sort_values("t")


def log_ticks(ax) -> None:
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))


# --------------------------------------------------------------------------- F1

def illegal_vs_move(ax, by_move: pd.DataFrame, played: pd.DataFrame) -> list:
    mass, null = move_series(by_move, "p_illegal"), move_series(by_move, "uniform_illegal")
    n_games = int(mass.n_games.iloc[0])
    width = int(played.t_hi.iloc[0] - played.t_lo.iloc[0]) + 1
    ax.set_yscale("log")
    ax.set_ylim(min(mass.lo.min(), played.rate[played.rate > 0].min()) / 20, 3.0)
    floor = ax.get_ylim()[0] * 1.6  # where an "no event in this bin" upper limit is drawn

    handles = [band(ax, mass.t, mass, GENERATION_STYLE["mass"],
                    GENERATION_STYLE["mass"]["label"], zorder=5)]
    for arm in ["sampled", "greedy"]:
        rows, style = played[played.arm == arm], GENERATION_STYLE[arm]
        x = ((rows.t_lo + rows.t_hi) / 2).to_numpy()
        rate, lo, hi = rows.rate.to_numpy(), rows.lo.to_numpy(), rows.hi.to_numpy()
        seen = rate > 0
        handles.append(ax.errorbar(
            x[seen], rate[seen], yerr=[rate[seen] - lo[seen], hi[seen] - rate[seen]],
            color=style["color"], marker=style["marker"], ms=5, ls="none", elinewidth=1.0,
            capsize=2, label=style["label"], zorder=6))
        # A bin with no event is an upper limit: open marker on the floor, bar to its 95% hi.
        ax.errorbar(x[~seen], np.full((~seen).sum(), floor),
                    yerr=[np.zeros((~seen).sum()), np.clip(hi[~seen] - floor, 0, None)],
                    color=style["color"], marker=style["marker"], mfc="none", ms=5, ls="none",
                    elinewidth=1.0, capsize=2, zorder=6)
    handles.append(ax.plot(null.t, null["mean"], color=NULL["color"], ls=NULL["ls"],
                           lw=NULL["lw"], marker=NULL["marker"], ms=4, markevery=(3, 9),
                           zorder=2, label=GENERATION_STYLE["uniform"]["label"])[0])

    log_ticks(ax)
    ax.set_xlabel("move index $t$")
    ax.set_ylabel("P(illegal move)   ↓ better")
    ax.text(0.02, 0.60, f"band / bars: 95% bootstrap CI over n = {n_games} games\n"
            f"played rates binned over {width} plies ({n_games * width} positions each)\n"
            "open marker: no event in the bin; bar is its 95% upper bound",
            transform=ax.transAxes, va="bottom", **NOTE)
    return handles


def illegal_density(ax, density: pd.DataFrame, overall: pd.DataFrame) -> list:
    stat = overall.set_index("metric").value
    handles = []
    for name, sign, style in [("observed", 1.0, GENERATION_STYLE["mass"]),
                              ("uniform", -1.0, NULL)]:
        d = density[density.series == name]
        label = ("per-position P(illegal)" if name == "observed"
                 else GENERATION_STYLE["uniform"]["label"])
        ax.fill_between(d.log10_p, sign * d.lo, sign * d.hi, color=style["color"],
                        alpha=0.22, lw=0, zorder=2)
        handles.append(ax.plot(d.log10_p, sign * d.density, color=style["color"],
                               ls=style.get("ls", "-"), lw=style.get("lw", 1.8), zorder=3,
                               label=label)[0])
        # Markers on the body of the curve, never on the flat tails (GUIDELINES 4.4).
        body = np.flatnonzero(d.density.to_numpy() > 0.05 * d.density.max())
        on = body[:: max(1, len(body) // 4)]
        ax.plot(d.log10_p.iloc[on], sign * d.density.iloc[on], ls="none", ms=4,
                marker=style["marker"], color=style["color"], zorder=5)
        key = "p_illegal" if name == "observed" else "uniform_illegal"
        median = np.log10(stat[f"{key}_median"])
        ax.plot(median, sign * np.interp(median, d.log10_p, d.density), marker="v", ms=6,
                color=style["color"], zorder=6)
        ax.axvspan(np.log10(stat[f"{key}_q25"]), np.log10(stat[f"{key}_q75"]),
                   color=style["color"], alpha=0.10, lw=0, zorder=1)

    ax.axhline(0, color=MUTED, lw=0.7, zorder=4)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(v)}}}$"))
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([-1, 0, 1], ["1", "0", "1"])
    ax.set_xlabel("P(illegal move) at one position")
    ax.set_ylabel("↓ null      density (peak 1)      observed ↑")
    ax.grid(axis="y", visible=False)
    observed = density[density.series == "observed"]
    ax.text(0.44, 0.97, "Gaussian KDE on log$_{10}$ P, shared\n"
            f"$h$ = {observed.bandwidth.iloc[0]:.2f} dex (Scott, observed):\n"
            "the null's width is the kernel's,\nits spread is the shaded IQR\n"
            f"median ▾; n = {int(observed.n_values.iloc[0])} positions\n"
            f"band: 95% CI over {int(observed.n_games.iloc[0])} games",
            transform=ax.transAxes, va="top", **NOTE)
    return handles


def survival(ax, table: pd.DataFrame) -> list:
    handles = []
    for arm, dy in [("teacher_forced", -15), ("free_running", 5)]:
        rows, style = table[table.arm == arm].sort_values("plies"), GENERATION_STYLE[arm]
        end = rows.iloc[-1]
        handles.append(band(ax, rows.plies, rows, style,
                            f"{style['label']} (n = {int(rows.n_games.iloc[0])})",
                            value="value", zorder=5 if arm == "free_running" else 4))
        ax.annotate(f"{end.value:.1%}", (end.plies, end.value), textcoords="offset points",
                    xytext=(-8, dy), fontsize=7.5, color=style["color"], ha="right")
    ax.set_xlabel("plies generated")
    ax.set_ylabel("P(at least one illegal move)   ↓ better")
    ax.set_ylim(bottom=0)
    ax.text(0.03, 0.97, "band: 95% bootstrap CI over games", transform=ax.transAxes,
            va="top", **NOTE)
    return handles


def figure_one(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    panels = [illegal_vs_move(axes[0], tables["by_move"], tables["played_rate"]),
              illegal_density(axes[1], tables["illegal_density"], tables["overall"]),
              survival(axes[2], tables["survival"])]
    for i, ax in enumerate(axes):
        panel_letter(ax, i)
    legal = tables["overall"].set_index("metric").value["p_legal"]
    end = tables["survival"].query("arm == 'free_running'").value.iloc[-1]
    if not camera_ready:
        fig.suptitle(f"OthelloGPT puts {legal:.4f} of its next-move mass on legal squares; "
                     f"{end:.1%} of the games it generates for itself still go illegal")
    fig.tight_layout(rect=(0, LEGEND_BAND, 1, 1 if camera_ready else 0.94))
    for i, handles in enumerate(panels):
        legend_band(fig, handles, i)
    return save_figure(fig, FIGURES / "generation_legality", camera_ready)


# --------------------------------------------------------------------------- F2

def rules_vs_move(ax, by_move: pd.DataFrame) -> list:
    legal = move_series(by_move, "p_legal")
    handles = [band(ax, legal.t, legal, REFERENCE, "legal (baseline)", zorder=6,
                    lw=REFERENCE["lw"])]
    for key, (rule, members) in RULES.items():
        for category in members:
            faint = move_series(by_move, f"p_{category}")
            ax.plot(faint.t, faint["mean"], color=CATEGORY_STYLE[category]["color"],
                    lw=0.7, alpha=0.55, zorder=3)
        rows = move_series(by_move, f"p_{key}")
        handles.append(band(ax, rows.t, rows, RULE_STYLE[rule],
                            f"{rule} (sum of {len(members)})", zorder=5))
    ax.set_yscale("log")
    log_ticks(ax)
    ax.set_xlabel("move index $t$")
    ax.set_ylabel("next-move probability mass")
    ax.text(0.03, 0.03, "faint: the 6 taxonomy categories\nband: 95% bootstrap CI over "
            f"n = {int(legal.n_games.iloc[0])} games", transform=ax.transAxes, va="bottom",
            **NOTE)
    return handles


def skew_vs_move(ax, by_move: pd.DataFrame, skew: pd.DataFrame, spread: float) -> list:
    pooled = skew[(skew.arm == "mass") & (skew.level == "rule")].set_index("name")
    handles = []
    for key, (rule, members) in RULES.items():
        for category in members:
            faint = move_series(by_move, f"skew_{category}")
            ax.plot(faint.t, faint["mean"], color=CATEGORY_STYLE[category]["color"],
                    lw=0.7, alpha=0.55, zorder=3)
        rows, r = move_series(by_move, f"skew_{key}"), pooled.loc[key]
        handles.append(band(ax, rows.t, rows, RULE_STYLE[rule],
                            f"{rule} ({len(members)} cat.), pooled "
                            f"{r.ratio:.2f} [{r.lo:.2f}, {r.hi:.2f}]", zorder=5))
    handles.append(ax.axhline(1.0, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=2,
                              label="no skew: illegal mass spread over the illegal squares"))
    ax.set_yscale("log")
    log_ticks(ax)
    ax.set_xlabel("move index $t$")
    ax.set_ylabel("illegal mass ÷ matched null")
    ax.text(0.03, 0.97, "↑ the model over-weights this rule", transform=ax.transAxes,
            va="top", **NOTE)
    ax.text(0.03, 0.03, f"within-rule ÷ between-rule spread = {spread:.2f}\n"
            "(SD of log$_{10}$ skew over the 6 categories)", transform=ax.transAxes,
            va="bottom", **NOTE)
    return handles


def skew_of_plays(ax, skew: pd.DataFrame) -> list:
    hatch = {"mass": "", "sampled": "//", "greedy": "xx"}
    pooled, keys = skew[skew.level == "rule"], list(RULES)
    x, width, handles = np.arange(len(keys)), 0.26, []
    for i, arm in enumerate(hatch):
        rows = pooled[pooled.arm == arm].set_index("name").loc[keys]
        style, events = GENERATION_STYLE[arm], rows.n_events.iloc[0]
        n = (f"n = {int(rows.n_positions.iloc[0])} positions" if arm == "mass"
             else f"n = {int(events)} illegal moves")
        handles.append(ax.bar(x + (i - 1) * width, rows.ratio.fillna(0), width * 0.9,
                              color=style["color"], alpha=0.85, edgecolor=INK, lw=0.6,
                              hatch=hatch[arm], zorder=3, label=f"{style['label']}, {n}"))
        ax.errorbar(x + (i - 1) * width, rows.ratio,
                    yerr=[rows.ratio - rows.lo, rows.hi - rows.ratio],
                    lw=0, elinewidth=1.0, capsize=2.5, ecolor=INK, zorder=6)
    handles.append(ax.axhline(1.0, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=2,
                              label="no skew (null)"))
    ax.set_xticks(x, [RULES[k][0] for k in keys])
    ax.set_ylabel("share of illegal events ÷ matched null")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.text(0.03, 0.97, "bars: 95% bootstrap CI over games\nan empty bar is a rule the arm "
            "never played", transform=ax.transAxes, va="top", **NOTE)
    return handles


def figure_two(tables: dict[str, pd.DataFrame], camera_ready: bool) -> Path:
    spread = float(tables["overall"].set_index("metric")
                   .value["rule_skew_within_over_between"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    panels = [rules_vs_move(axes[0], tables["by_move"]),
              skew_vs_move(axes[1], tables["by_move"], tables["rule_skew"], spread),
              skew_of_plays(axes[2], tables["rule_skew"])]
    for i, ax in enumerate(axes):
        panel_letter(ax, i)
    if not camera_ready:
        fig.suptitle("The residual illegal mass is not spread evenly: R3 (turn) is "
                     "over-weighted and R1 (placement) is suppressed")
    fig.tight_layout(rect=(0, LEGEND_BAND, 1, 1 if camera_ready else 0.94))
    for i, handles in enumerate(panels):
        legend_band(fig, handles, i)
    rule_key(fig, ["R1", "R2", "R3"])
    return save_figure(fig, FIGURES / "generation_legality_by_rule", camera_ready)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    tables = read(args.results)
    for path in (figure_one(tables, args.camera_ready), figure_two(tables, args.camera_ready)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
