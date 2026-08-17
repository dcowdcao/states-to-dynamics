"""At the spliced move, how well does each probe head separate illegal from legal?

One bar chart. Every bar is a paired AUROC at t0: one arm's score against the score the
same game's matched legal control gets at the same position of the same prefix, one pair
per game. The 2-class head is read as P(illegal), the 4-class head as 1 - P(legal) and
split by which rule the branch actually broke, so a rule the probe cannot catch shows up as
its own short bar rather than being averaged away.

AUROC is threshold-free, so a head's hit rate and the false alarms it pays for it are one
number rather than two bars. The legal-vs-legal null -- a game's legal control against the
second legal control it carries -- is still measured, since it is what makes the 0.5 line
empirical rather than assumed, but it is a number under the panel and not a bar of its own.

Usage: python analysis/move_legality/detection_auroc.py [--space raw] [--camera_ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from spade.common.paths import repo_root
from spade.figures.paper_style import (
    CLASS_STYLE, MUTED, NULL, PROBE_CLASS_STYLE, apply_style, rule_key, save_figure)
from spade.move_legality import localize

FIGURES = repo_root() / "analysis" / "move_legality" / "figures"
RESULTS = repo_root() / "analysis" / "move_legality" / "results"
RULES = localize.RULES
SEED = 42
GAP = 0.8   # blank slot between the two heads

ARM_STYLE = {
    "illegal": {**CLASS_STYLE["violation"], "label": "illegal,\nany rule"},
    "control": NULL,   # the legal-vs-legal null: measured, printed in the note, not a bar
    **{r: {**PROBE_CLASS_STYLE[r],
           "label": PROBE_CLASS_STYLE[r]["label"].replace(" ", "\n")} for r in RULES},
}


def bar(head: str, score: torch.Tensor, arm: str, meta: dict, selection: dict) -> dict:
    """One arm's AUROC against the legal branch it is paired with in every game.

    The control arm is scored against `control_legal_2`, the second legal branch every game
    carries: pairing it with itself would print a tautological 0.5 where the reader needs an
    empirical legal-vs-legal null.
    """
    rows = selection[arm]
    null = selection["control2" if arm == "control" else "control"]
    paired = localize.detection(score, meta, rows,
                                localize.matched_control(meta, rows, null), seed=SEED)
    return {"head": head, "arm": arm, "style": ARM_STYLE[arm], **paired}


def bars(space: str, meta: dict, selection: dict) -> list[dict]:
    """Each head's illegal arms, plus the legal-vs-legal null the chance line stands for."""
    heads = [("2-class", localize.read(space, "p_illegal"), ["illegal"]),
             ("4-class", 1.0 - localize.read(space, "proba4")[..., 0], RULES)]
    return [bar(head, score, arm, meta, selection)
            for head, score, illegal_arms in heads
            for arm in [*illegal_arms, "control"]]


def draw(space: str, rows: list[dict], camera_ready: bool) -> Path:
    """The illegal arms as bars; the legal-vs-legal null stays a number under them.

    Drawn, the null was two bars at the chance line saying the same thing the line says. It
    is still measured -- it is what licenses that line being read as empirical -- so it goes
    in the note and in the CSV.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    null = {r["head"]: r for r in rows if r["arm"] == "control"}
    rows = [r for r in rows if r["arm"] != "control"]
    x = np.array([i + (GAP if r["head"] == "4-class" else 0.0) for i, r in enumerate(rows)])

    split = next(i for i, r in enumerate(rows) if r["head"] == "4-class")
    ax.axhline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=1)
    ax.text(x[split - 1: split + 1].mean(), 0.51, "chance", color=NULL["color"], fontsize=7.5,
            ha="center", va="bottom")   # the blank slot between the heads is the only clear space
    for pos, row in zip(x, rows):
        ax.bar(pos, row["auroc"], width=0.72, color=row["style"]["color"],
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(pos, row["auroc"], yerr=[[row["auroc"] - row["lo"]],
                                             [row["hi"] - row["auroc"]]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(pos, row["hi"] + 0.02, f"{row['auroc']:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=MUTED, zorder=4)

    for head in ["2-class", "4-class"]:
        at = x[[i for i, r in enumerate(rows) if r["head"] == head]]
        ax.text(at.mean(), 1.10, f"{head} probe", ha="center", va="bottom", fontsize=9)
        ax.plot([at[0] - 0.45, at[-1] + 0.45], [1.07, 1.07], color=MUTED, lw=0.8,
                clip_on=False)

    ax.set_xticks(x, [f"{r['style']['label']}\n($n$ = {r['n_games']})" for r in rows],
                  fontsize=7.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylabel("paired AUROC at $t_0$")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    fig.text(0.005, -0.005,
             "AUROC at $t_0$ of an illegal branch against the same game's legal control, one "
             "pair per game: above 0.5\nthe head scores the illegal move above the legal one, "
             "and AUROCs are never flipped, so a bar below 0.5\nis the finding. 2-class head "
             "read as $P$(illegal), 4-class as $1-P$(legal); bars are 95% percentile\nbootstrap "
             "CIs over paired games (1,000 resamples). The chance line is empirical: that same "
             f"control\nagainst the second legal control the game carries scores "
             f"{null['2-class']['auroc']:.3f} and {null['4-class']['auroc']:.3f}.",
             ha="left", va="top", fontsize=7, color=MUTED)
    rule_key(fig, RULES, y=-0.205)
    return save_figure(fig, FIGURES / f"{space}_detection_auroc", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="raw")
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    meta = localize.branches()
    selection = localize.arms(meta, SEED)
    rows = bars(args.space, meta, selection)

    print(draw(args.space, rows, args.camera_ready))
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"space": args.space, **{k: v for k, v in r.items() if k != "style"}}
                          for r in rows])
    csv = RESULTS / f"detection_auroc_{args.space}.csv"
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
