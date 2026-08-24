"""The detector score across one game's spliced move, for the three rule arms and its control.

An illustrative game, not a typical one. `select_game.py` picked game 2498 because all three
rules have a branch at its t0, all of t0 +- 5 sit inside the scanned band, raw's score peaks
at t0 on every rule arm and NOT on the matched legal control, and among the games that hold
it has the largest margin over its own runner-up. That is a selection over games, so this
panel shows what a clean detection looks like and says nothing about how often one happens:
the AUROC and histogram figures of this study are the population.

Four branches of one prefix, so there is nothing to average and no interval to draw -- the
uncertainty lives in the population figures. The matched legal control is the reference: it
is the same prefix with a *legal* move at the same t0, which is what makes the peak readable
as "the violation" rather than "that position".

The score at candidate c reads the M steps either side of step c - CAND_OFFSET, and the
spliced move first enters the trajectory at step t0 - CAND_OFFSET, so every branch is
bit-identical out to c = t0 - M; that stretch is shaded rather than left to look like a
coincidence. tau is refit on this population by `detector.operating_point`, never quoted.

Usage: PYTHONPATH=src python analysis/violation_divergence/illegal_move_detector/figure_trace.py
       [--camera-ready]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import detector
from detector import CAND_OFFSET, FIGURES, M, RULES
from spade.figures.paper_style import (
    INK, MUTED, NULL, PRE_EVENT_FACE, PROBE_CLASS_STYLE, apply_style, save_figure)

HALF = 5   # plies drawn either side of t0; select_game.py guaranteed they fit in the band

# Four series is past what hue and marker carry in grayscale (FIGURE_GUIDELINES 1), so the
# rules take a dash pattern too and the control keeps NULL's. Same patterns as
# move_legality/localization.py, so a rule looks the same in both event-locked panels.
RULE_DASH = {"R1": (0, (6, 1.5, 1, 1.5)), "R2": (0, (4, 2)), "R3": (0, (1, 1.6))}
ARM_STYLE = {
    **{r: {**PROBE_CLASS_STYLE[r], "ls": RULE_DASH[r], "lw": 1.8, "zorder": 4} for r in RULES},
    "control": {**NULL, "label": "matched legal control (baseline)"},
}


def draw(game: dict, move: np.ndarray, s: np.ndarray, tau: float, camera_ready: bool) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))   # NeurIPS full width, drawn at printed size
    t0 = game["t0"]
    window = np.arange(t0 - HALF, t0 + HALF + 1)
    take = np.searchsorted(move, window)
    assert (move[take] == window).all(), "the trace window left the scanned band"
    left, right = window[0] - 0.5, window[-1] + 0.5

    ax.axvspan(left, t0 - M + 0.5, color=PRE_EVENT_FACE, zorder=0, lw=0)
    ax.axvline(t0, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.axhline(tau, color=MUTED, ls=(0, (1, 2)), lw=1.0, zorder=1)

    rows = {**game["rows"], "control": game["control_row"]}
    for name, row in rows.items():
        style = ARM_STYLE[name]
        ax.plot(window, s[row, take], color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=style["lw"], markersize=4.2, markeredgecolor="white",
                markeredgewidth=0.5, zorder=style["zorder"], label=style["label"])

    ax.set_xlim(left, right)
    # Headroom above the peak so the legend clears the curves rather than sitting on them.
    trace = s[list(rows.values())][:, take]
    ax.set_ylim(trace.min() - 0.55, trace.max() + 1.1)
    ax.set_xticks(window)
    ax.set_xlabel(f"move position in the game   ($t_0 = {t0}$ is the spliced move)")
    ax.set_ylabel("paired MOSUM of standardized step-length drop\n"
                  f"($M = {M}$ steps, SD units) — higher $=$ more illegal")
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick == t0:
            text.set(color=INK, fontweight="bold", fontsize=9.5)

    ax.text(left + 0.15, ax.get_ylim()[0] + 0.15,
            f"$t \\leq t_0 - {M}$: score reads no post-splice step, identical in every branch",
            color=MUTED, fontsize=7.2, ha="left", va="bottom")

    top = max(RULES, key=lambda r: s[rows[r], take[HALF]])
    ax.annotate(f"peak at $t_0$, score {s[rows[top], take[HALF]]:.2f}\n"
                f"({ARM_STYLE[top]['label']})",
                xy=(t0, s[rows[top], take[HALF]]), xytext=(0.99, 1.03),
                textcoords="axes fraction", ha="right", va="bottom", fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9, "shrinkB": 4})
    # The threshold line keeps its label in the legend rather than in-panel prose: at this
    # game's scale every clear stretch of the tau line is crossed by a curve.
    ax.plot([], [], color=MUTED, ls=(0, (1, 2)), lw=1.0,
            label=f"detector threshold $\\tau$ = {tau:.3f}")
    ax.legend(title=f"game {game['game_id']}, one branch per arm (illustrative)",
              loc="upper left", fontsize=7, title_fontsize=7, handlelength=2.0,
              labelspacing=0.4, borderaxespad=0.5)
    fig.tight_layout()
    return save_figure(fig, FIGURES / "detector_trace", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    game = json.loads((Path(__file__).resolve().parent / "figure_game.json").read_text())
    data = detector.load()
    cand = detector.band()
    s = detector.score(data, cand)
    tau = detector.operating_point(s, cand, data, detector.arms(data))
    for name, row in {**game["rows"], "control": game["control_row"]}.items():
        assert data["game_id"][row] == game["game_id"] and data["t0"][row] == game["t0"], \
            f"{name}: figure_game.json is stale against the corpus"
    print(draw(game, cand + CAND_OFFSET, s, tau, args.camera_ready))


if __name__ == "__main__":
    main()
