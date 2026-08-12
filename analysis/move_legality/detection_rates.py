"""At the spliced move, how often does each probe head get the call right?

One bar chart. True-positive rate is the share of illegal branches the probe calls illegal
at t0; true-negative rate is the share of that same game's matched legal control it calls
legal. The 2-class head answers legal-vs-illegal directly; the 4-class head answers it by
argmax != legal, split by which rule the branch actually broke, so a rule the probe cannot
catch shows up as its own short bar rather than being averaged away.

Both heads are read at their argmax -- the operating point a classifier is actually used at,
not the threshold-free AUROC that `localization_<space>.csv` already carries.

Usage: python analysis/move_legality/detection_rates.py [--space raw] [--camera_ready]
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


def at_t0(score: torch.Tensor, meta: dict) -> torch.Tensor:
    return score.gather(1, meta["t0"].unsqueeze(1)).squeeze(1)


def rate(correct: torch.Tensor, meta: dict, rows: torch.Tensor) -> dict:
    """Share of `rows` the probe calls right, with a 95% CI over source games."""
    mean, lo, hi = localize.cluster_bootstrap(
        correct[rows].float().unsqueeze(1), meta["game_id"][rows], seed=SEED)
    return {"value": float(mean[0]), "lo": float(lo[0]), "hi": float(hi[0]), "n": len(rows)}


def bars(space: str, meta: dict, selection: dict) -> list[dict]:
    """Six bars: each head's true-positive rates and the true-negative rate it pays for them."""
    illegal2 = at_t0(localize.read(space, "p_illegal"), meta) >= 0.5
    illegal4 = at_t0(localize.read(space, "proba4").argmax(-1).float(), meta) > 0

    rows = [{"head": "2-class", "arm": "illegal", "kind": "TPR",
             "label": "illegal,\nany rule", "style": CLASS_STYLE["violation"],
             **rate(illegal2, meta, selection["illegal"])},
            {"head": "2-class", "arm": "control", "kind": "TNR",
             "label": "legal\ncontrol", "style": NULL,
             **rate(~illegal2, meta, selection["control"])}]
    rows += [{"head": "4-class", "arm": r, "kind": "TPR",
              "label": PROBE_CLASS_STYLE[r]["label"].replace(" ", "\n"),
              "style": PROBE_CLASS_STYLE[r], **rate(illegal4, meta, selection[r])}
             for r in RULES]
    rows.append({"head": "4-class", "arm": "control", "kind": "TNR",
                 "label": "legal\ncontrol", "style": NULL,
                 **rate(~illegal4, meta, selection["control"])})
    return rows


def draw(space: str, rows: list[dict], camera_ready: bool) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.array([i + (GAP if r["head"] == "4-class" else 0.0) for i, r in enumerate(rows)])

    split = next(i for i, r in enumerate(rows) if r["head"] == "4-class")
    ax.axhline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=1)
    ax.text(x[split - 1: split + 1].mean(), 0.51, "chance", color=NULL["color"], fontsize=7.5,
            ha="center", va="bottom")   # the blank slot between the heads is the only clear space
    for pos, row in zip(x, rows):
        ax.bar(pos, row["value"], width=0.72, color=row["style"]["color"],
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(pos, row["value"], yerr=[[row["value"] - row["lo"]],
                                             [row["hi"] - row["value"]]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(pos, row["hi"] + 0.02, f"{row['value']:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=MUTED, zorder=4)

    for head in ["2-class", "4-class"]:
        at = x[[i for i, r in enumerate(rows) if r["head"] == head]]
        ax.text(at.mean(), 1.10, f"{head} probe", ha="center", va="bottom", fontsize=9)
        ax.plot([at[0] - 0.45, at[-1] + 0.45], [1.07, 1.07], color=MUTED, lw=0.8,
                clip_on=False)

    ax.set_xticks(x, [f"{r['label']}\n({r['kind']})" for r in rows], fontsize=7.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylabel("called correctly at $t_0$")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    fig.text(0.005, -0.005,
             "TPR: an illegal branch called illegal.  TNR: that same game's legal control "
             "called legal.\nargmax of each head at $t_0$; bars are 95% percentile bootstrap "
             f"CIs over source games (1,000 resamples,\ngames as clusters), $n$ = "
             f"{rows[0]['n']} illegal branches against the same games' controls",
             ha="left", va="top", fontsize=7, color=MUTED)
    rule_key(fig, RULES, y=-0.155)
    return save_figure(fig, FIGURES / f"{space}_detection_rates", camera_ready)


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
    csv = RESULTS / f"detection_rates_{args.space}.csv"
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
