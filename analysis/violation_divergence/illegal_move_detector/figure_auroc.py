"""Does the detector score rank an illegal move above the legal move it replaced?

One bar per arm: the pooled illegal branch and the three rules, each scored against the SAME
comparison set, the matched legal control. The contrast is paired within game -- same source
game, same t0, same prefix, one legal move swapped for one illegal one -- so neither the game
nor the move index is on the axis, only the move.

Ranked on the score AT ITS OWN t0, one value per branch. That is a different statistic from
the parent study's per-position AUROC, which scores every candidate in the band, and from its
peak-based detection AUROC, which ranks a game by its argmax; all three are legitimate and
none of them is the other's estimate.

The evaluation set is `detector.stratified`: the same number of games at every t0 in the
scored range, so the bars are flat over move index rather than 46% t0 = 20, which is what a
prefix of the tree's game ids is. n is the smallest stratum times the number of positions.

Bars stay in rule order rather than sorting by height (FIGURE_GUIDELINES 4.5): R1-R3 are an
ordered variable, and the finding is that the AUROC ordering is NOT monotone in it, which a
sort would hide.

Numbers, with n per bar, go to results/fig03_detector_example/auroc_by_rule.csv. The games are
asserted against results/shared/eval_games.csv, which `eval_games.py` writes.

Usage: python analysis/violation_divergence/illegal_move_detector/figure_auroc.py
           [--camera-ready]
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import detector
from spade.figures.paper_style import (
    DIRECTION_STYLE, MUTED, NULL, RULE_STYLE, apply_style, save_figure)

BOOT_SEED = 42
BAR_WIDTH = 0.72
TOP = 1.0
# RULE_STYLE has no entry for the pooled arm -- it is the union of the three, not a fourth
# rule -- so it takes DIRECTION_STYLE's `illegal`, the repo's hue for exactly that aggregate.
BAR_STYLE = {"illegal": DIRECTION_STYLE["illegal"],
             "R1": RULE_STYLE["R1 placement"],
             "R2": RULE_STYLE["R2 bracketing"],
             "R3": RULE_STYLE["R3 turn"]}
TICK = {"illegal": "legal / illegal", "R1": "R1", "R2": "R2", "R3": "R3"}


@dataclass
class Bar:
    arm: str
    auroc: float
    lo: float
    hi: float
    n_games: int


def measure(positive: np.ndarray, negative: np.ndarray, game: np.ndarray, arm: str) -> Bar:
    """Point AUROC and a 95% percentile bootstrap CI resampling source games."""
    order, counts = detector.by_game(game)
    statistic = lambda draw: detector.auroc(*(v[detector.take(order, counts, draw)]
                                              for v in (positive, negative)))
    lo, hi = detector.ci(statistic, detector.resamples(len(game), BOOT_SEED))
    return Bar(arm, detector.auroc(positive, negative), lo, hi, len(game))


def table() -> list[Bar]:
    data = detector.load()
    cand, sel = detector.band(), detector.arms(data)
    s = detector.score(data, cand)
    game_of = data["game_id"][sel["reference"]]
    for arm, rows in sel.items():
        assert np.array_equal(data["game_id"][rows], game_of), f"{arm} is not aligned by game"
    keep = detector.stratified(data, sel)
    at_t0 = lambda arm: detector.at_t0(s, cand, data["t0"], sel[arm][keep])
    negative = at_t0("control")
    game = data["game_id"][sel["reference"][keep]]
    detector.assert_eval_set(game, data["t0"][sel["reference"][keep]])
    return [measure(at_t0(arm), negative, game, arm) for arm in detector.ARMS]


def panel(ax, rows: list[Bar]) -> None:
    """The bars themselves, so a figure that puts them beside others draws this and not a copy."""
    ax.axhline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=2,
               label="chance")
    for x, bar in enumerate(rows):
        ax.bar(x, bar.auroc, width=BAR_WIDTH, color=BAR_STYLE[bar.arm]["color"],
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(x, bar.auroc, yerr=[[bar.auroc - bar.lo], [bar.hi - bar.auroc]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4,
                    label="95% CI" if x == 0 else None)
        ax.text(x, bar.hi + TOP * 0.02, f"{bar.auroc:.3f}", ha="center", va="bottom",
                fontsize=6.6, color=MUTED, zorder=4, rotation=90)

    ax.set_xticks(range(len(rows)), [TICK[b.arm] for b in rows], fontsize=8.4)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, TOP)
    ax.set_ylabel("AUROC")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.8,
              frameon=False, handlelength=2.4)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def draw(rows: list[Bar], camera_ready: bool):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    panel(ax, rows)
    fig.tight_layout()
    return save_figure(fig, detector.FIGURES / "auroc_by_rule", camera_ready)


def write_csv(rows: list[Bar]) -> Path:
    detector.EXAMPLE_RESULTS.mkdir(parents=True, exist_ok=True)
    out = detector.EXAMPLE_RESULTS / "auroc_by_rule.csv"
    round5 = lambda v: round(v, 5) if isinstance(v, float) else v
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows([{k: round5(v) for k, v in asdict(r).items()} for r in rows])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    rows = table()
    print(write_csv(rows))
    print(draw(rows, args.camera_ready))


if __name__ == "__main__":
    main()
