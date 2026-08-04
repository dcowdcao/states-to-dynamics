"""The selection disclosure: what the fork figure's game is the argmax of.

Any figure drawn on one hand-picked game owes the reader the population it was
picked from. Panel (a) is that population, (b) is how far the four candidate
criteria disagree about it, and (c) is whether a game chosen in raw layer 5 stays
chosen once the SAE dictionaries re-score it.

Reads `results/` from score_games.py.

Usage: python analysis/violation_divergence/game_selection/make_figures.py
       [--camera-ready]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from spade.figures.paper_style import (
    GRID, INK, MUTED, OKABE_ITO, REFERENCE, apply_style, sae_label, save_figure)

RESULTS = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
INCUMBENT = 0
BINS = 45

# Hue per SAE family, matching sae_spaces/make_figures.py so a dictionary keeps
# one identity across the study; marker is the redundant cue.
SPACE_STYLE = {
    "StandardTrainer-othello": {"color": OKABE_ITO[0], "marker": "o"},
    "TopKTrainer-othello":     {"color": OKABE_ITO[2], "marker": "s"},
    "SpaDE-lam1e-03-gam1e-04": {"color": OKABE_ITO[3], "marker": "^"},
}
PICK_STYLE = {"color": OKABE_ITO[1], "marker": "D"}
CRITERION_LABEL = {
    "s1_illegal_vs_control": "$S_1$ illegal vs. control (pooled within-group SDs)",
    "s2_branch_distinctness": "$S_2$ branch distinctness",
    "s3_category_differentiation": "$S_3$ category differentiation",
    "s4_delta_hat_end": r"$S_4$ $\hat{\delta}$ at $r=24$",
}
S3, S1 = "s3_category_differentiation", "s1_illegal_vs_control"

KEY = (
    r"$S_3$ — mean pairwise $\|X_i-X_j\|$ over the six illegal branches $\div$ "
    r"RMS $\|X\|$, where $X_b$ stacks $D(t)=z_b(t)-z_{\mathrm{ref}}(t)$ over "
    r"$r\in[0,24]$ at $t_0$'s parity, in full $\mathbb{R}^{512}$."
    "\n"
    r"0 = the six illegal branches displace identically; $\sqrt{2}$ = mutually "
    r"orthogonal. $S_1$ — $\|\mu_{\mathrm{illegal}}-\mu_{\mathrm{control}}\|$ "
    r"$\div$ pooled within-group RMS radius, same stacked vectors."
    "\n"
    "Panels (a) and (b) are the whole population of complete games, not a sample, "
    "so the spread is the population IQR and not an error bar.")


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: (v if k == "space" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


MARK_TOP = 0.66  # the two game rules stop below the legend, in axes fraction


def marks(ax, x_pick: float, x_incumbent: float) -> None:
    """A vertical rule per game, capped below the legend and headed by its marker."""
    ax.axvline(x_incumbent, ymax=MARK_TOP, color=REFERENCE["color"], lw=1.6, zorder=4,
               label=f"incumbent: game {INCUMBENT} (list order)")
    ax.axvline(x_pick, ymax=MARK_TOP, color=PICK_STYLE["color"], lw=1.6,
               ls=(0, (4, 1.5)), zorder=5, label="selected: argmax over the population")
    for x, st in [(x_incumbent, REFERENCE), (x_pick, PICK_STYLE)]:
        ax.plot([x], [ax.get_ylim()[1] * MARK_TOP], marker=st["marker"], ms=7,
                color=st["color"], markeredgecolor="white", markeredgewidth=0.7,
                zorder=6)


def distribution_panel(ax, s3: np.ndarray, pick: dict, incumbent: dict) -> None:
    q25, med, q75 = np.percentile(s3, [25, 50, 75])
    ax.axvspan(q25, q75, color=GRID, alpha=0.8, lw=0, zorder=0,
               label=f"IQR of all {len(s3)} games")
    ax.hist(s3, bins=BINS, color=GRID, edgecolor=MUTED, linewidth=0.5, zorder=2)
    ax.axvline(med, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3,
               label=f"median = {med:.3f}")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
    marks(ax, pick[S3], incumbent[S3])
    ax.set_xlabel(CRITERION_LABEL[S3])
    ax.set_ylabel("games")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(pick[S3], ax.get_ylim()[1] * 0.58,
            f"  game {int(pick['game'])}\n  $t_0$={int(pick['t0'])}\n"
            f"  $S_3$={pick[S3]:.3f}", fontsize=7.5, color=PICK_STYLE["color"], va="top")
    ax.text(incumbent[S3], ax.get_ylim()[1] * 0.58,
            f"game {INCUMBENT}  \n$t_0$={int(incumbent['t0'])}  \n"
            f"$S_3$={incumbent[S3]:.3f}  ", fontsize=7.5, color=INK, va="top", ha="right")


def disagreement_panel(ax, rows: list[dict], pick: dict, incumbent: dict) -> None:
    x = np.array([r[S3] for r in rows])
    y = np.array([r[S1] for r in rows])
    ax.scatter(x, y, s=5, color=MUTED, alpha=0.35, linewidths=0, rasterized=True,
               zorder=1, label=f"complete game (n = {len(rows)})")
    ax.axhline(np.median(y), color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2,
               label=f"$S_1$ median = {np.median(y):.3f}")
    for g, st, lab in [(incumbent, REFERENCE, f"game {INCUMBENT} (incumbent)"),
                       (pick, PICK_STYLE, f"game {int(pick['game'])} (selected)")]:
        ax.plot([g[S3]], [g[S1]], marker=st["marker"], ms=9, color=st["color"],
                markeredgecolor="white", markeredgewidth=0.8, zorder=5, ls="none",
                label=lab)
    rho = spearmanr(x, y).statistic
    ax.set_xlabel(CRITERION_LABEL[S3])
    ax.set_ylabel(CRITERION_LABEL[S1])
    ax.legend(loc="upper right", fontsize=7.5,
              title=f"Spearman $\\rho$ = {rho:+.3f}", title_fontsize=7.5)


def transfer_panel(ax, rows: list[dict], raw: dict[int, float], pick_game: int) -> None:
    by_space: dict[str, list[dict]] = {}
    for r in rows:
        by_space.setdefault(r["space"], []).append(r)
    lo = min(min(raw[int(r["game"])] for r in rows), min(r[S3] for r in rows))
    hi = max(max(raw[int(r["game"])] for r in rows), max(r[S3] for r in rows))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1,
            label="identity (no rank shift)")
    for space, rs in by_space.items():
        st = SPACE_STYLE[space]
        x = np.array([raw[int(r["game"])] for r in rs])
        y = np.array([r[S3] for r in rs])
        rho = spearmanr(x, y).statistic
        ax.plot(x, y, ls="none", marker=st["marker"], ms=3.2, color=st["color"],
                alpha=0.55, markeredgewidth=0, zorder=2, rasterized=True,
                label=f"{sae_label(space.replace('Trainer-othello', ''))}  "
                      f"$\\rho$={rho:.3f}")
        pick_row = next(r for r in rs if int(r["game"]) == pick_game)
        ax.plot([raw[pick_game]], [pick_row[S3]], marker=st["marker"], ms=8,
                color=st["color"], markeredgecolor=PICK_STYLE["color"],
                markeredgewidth=1.4, zorder=5, ls="none")
    ax.set_xlabel("$S_3$ in raw layer 5 (the selector)")
    ax.set_ylabel("$S_3$ in the SAE latent space")
    n = len(by_space[list(by_space)[0]])
    ax.legend(loc="upper left", fontsize=7.5,
              title=f"n = {n} games (random sample, seed 0)", title_fontsize=7.5)
    ax.text(0.97, 0.05, f"outlined: game {pick_game},\nrank 1 of {n} in every space",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=PICK_STYLE["color"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    apply_style(args.camera_ready)

    rows = read(RESULTS / "ranking.csv")
    transfer = read(RESULTS / "transfer_check.csv")
    pick = min(rows, key=lambda r: r["rank_s3"])
    incumbent = next(r for r in rows if int(r["game"]) == INCUMBENT)
    raw = {int(r["game"]): r[S3] for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    distribution_panel(axes[0], np.array([r[S3] for r in rows]), pick, incumbent)
    disagreement_panel(axes[1], rows, pick, incumbent)
    transfer_panel(axes[2], transfer, raw, int(pick["game"]))
    for ax, title in zip(axes, [
            "(a) the population the game was chosen from",
            "(b) the criteria disagree",
            "(c) the choice survives the SAE spaces"]):
        ax.set_title(title, loc="left", fontsize=10)
    fig.text(0.005, -0.02, KEY, ha="left", va="top", fontsize=7.5, color=MUTED,
             linespacing=1.5)
    if not args.camera_ready:
        fig.suptitle(f"Selecting the fork figure's game: {int(pick['game'])} is the "
                     f"argmax of $S_3$ over {len(rows)} complete games, scored in "
                     f"raw layer 5 in full $\\mathbb{{R}}^{{512}}$", fontsize=11)
    fig.tight_layout()
    print(f"wrote {save_figure(fig, FIG_DIR / 'selection_disclosure', args.camera_ready)}")


if __name__ == "__main__":
    main()
