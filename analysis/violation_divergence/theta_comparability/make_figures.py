"""What the proposed overlay would look like, and why it must not be drawn.

One figure, two panels: the level the two sampling rules sit at over the game,
and the per-class contrast each of them reports. Both are the axis a figure would
share between the straightness class baselines and the fork study's branches.

The two rules are different quantities, so they take different hues (§3). The
per-ply rule is the incumbent and keeps the reference styling; the stride-2 rule
takes a palette hue and its own marker, and the fork study's own R^21 curve is
drawn in that same hue, faint, because it is the same quantity measured on one
game.

Usage: python analysis/violation_divergence/theta_comparability/make_figures.py
       [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spade.figures.paper_style import (
    CLASS_STYLE, MUTED, OKABE_ITO, REFERENCE, apply_style, save_figure)
from spade.figures.straightness_panels import DRAWN_CLASSES
from spade.straightness.build_games import CLASSES
from spade.violation_divergence import curvature_plane as dsc
from spade.violation_divergence.branches import FIGURE_GAME
from spade.violation_divergence.fork_plane import RAW_SPACE

from measure import MEAN_THETA, RULES, delta_ci, per_game

FIG_DIR = Path(__file__).resolve().parent / "figures"
# Two ply-samplings of one estimator: the incumbent is the reference, the fork's
# takes a palette hue and its own marker (FIGURE_GUIDELINES §1.1, §3).
STYLE = {RULES[0]: {"color": REFERENCE["color"], "marker": "o", "lw": REFERENCE["lw"]},
         RULES[1]: {"color": OKABE_ITO[1], "marker": "s", "lw": 1.8}}
FORK_LABEL = "$R^{21}$ windowed mean θ, the fork study's own curve (game %d, 9 branches, w=8; n=1 game)"
GAP_AT = 41  # mid-game, where both profiles are flat and both rules have a vertex


def level_panel(ax, g: dict, gid: int) -> None:
    """Mean turning angle vs board position, legal games, under each rule, with
    the fork study's published curve where it actually lands."""
    legal = CLASSES.index("legal")
    for rule in RULES:
        st, t = STYLE[rule], g[f"vertex {rule}"]
        mu, sem = g[f"profile {rule}"][legal], g[f"profile sem {rule}"][legal]
        ax.fill_between(t, mu - 1.96 * sem, mu + 1.96 * sem, color=st["color"],
                        alpha=0.25, lw=0, zorder=2)
        ax.plot(t, mu, color=st["color"], lw=st["lw"], marker=st["marker"], ms=4,
                markevery=(0, 6), markeredgecolor="white", markeredgewidth=0.5,
                zorder=4, label=f"{rule}, legal games (n=2000)")

    fork = dsc.figure_game(gid, dsc.W)
    phi = dsc.windowed(fork[RAW_SPACE], dsc.W)[:, :, MEAN_THETA]
    ok = np.isfinite(phi).all(0)
    for row in phi:
        ax.plot(fork["pos"][ok], row[ok], color=STYLE[RULES[1]]["color"], lw=0.8,
                alpha=0.55, zorder=3)
    ax.plot(fork["pos"][ok], phi[:, ok].mean(0), color=STYLE[RULES[1]]["color"],
            lw=1.6, ls=(0, (1, 1.2)), zorder=5, label=FORK_LABEL % gid)

    at = GAP_AT
    hi, lo = (g[f"profile {r}"][legal][g[f"vertex {r}"] == at].item() for r in RULES)
    ax.annotate("", xy=(at, hi), xytext=(at, lo),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text(at - 1.5, (hi + lo) / 2, f"{hi - lo:.1f}° apart at $t$={at};\nno drawn class differs\n"
            "from legal by more than 1.3°", fontsize=7.5, color=MUTED, style="italic",
            ha="right", va="center")
    ax.set_xlabel("board position $t$ (the angle's vertex ply)")
    ax.set_ylabel("mean turning angle (deg)")


def contrast_panel(ax, g: dict) -> None:
    """Δ mean θ against legal, per class, under each rule."""
    classes = DRAWN_CLASSES[1:]
    y = np.arange(len(classes))
    ax.axvline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for j, rule in enumerate(RULES):
        st = STYLE[rule]
        off = (j - 0.5) * 0.26
        vals = np.array([delta_ci(g, rule, c) for c in classes])
        ax.errorbar(vals[:, 0], y + off, xerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]],
                    fmt=st["marker"], ms=6, color=st["color"], ecolor=st["color"],
                    elinewidth=1.2, capsize=2.5, lw=0, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=4)
    ax.set_yticks(y, [CLASS_STYLE[c]["label"] for c in classes])
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.invert_yaxis()
    lim = 1.15 * np.abs(ax.get_xlim()).max()
    ax.set_xlim(-lim, lim)
    ax.text(-lim * 0.95, -0.5, "← straighter than legal", fontsize=7.5, color=MUTED,
            style="italic", va="center")
    ax.text(lim * 0.95, -0.5, "curvier than legal →", fontsize=7.5, color=MUTED,
            style="italic", ha="right", va="center")
    ax.set_xlabel("Δ mean turning angle vs. legal (deg)")
    ax.grid(axis="y", visible=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = per_game()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    level_panel(axes[0], g, args.game)
    axes[0].set_title("(a) the level each rule reports", loc="left")
    contrast_panel(axes[1], g)
    axes[1].set_title("(b) the contrast each rule reports", loc="left")
    if not args.camera_ready:
        fig.suptitle("Two ply-samplings, one name: the straightness and fork studies' "
                     "“mean turning angle” are different quantities", fontsize=11, y=1.02)
    fig.text(0.5, 0.945, "layer 5, raw activations, same estimator "
             "(spade.straightness.geometry.temporal_geometry), different sampled sequence.  "
             "(a) mean ± 95% CI over games;\n(b) 95% percentile bootstrap over games, 1000 "
             "resamples, n = 2000 per class.  Sign flips in (b) on every class: no rescale "
             "or reindex reconciles them.",
             fontsize=7.5, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.07), columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    print(f"wrote {save_figure(fig, FIG_DIR / 'theta_sampling_mismatch', args.camera_ready)}")


if __name__ == "__main__":
    main()
