"""Do the SAE dictionaries keep what the residual stream has, and does any of them sharpen it?

Two products. A 3x4 grid per score -- one panel per space, the same event-locked curve
`localization.py` draws for one -- and a scalar summary that puts every space on three axes
at once: how well the probe separates the classes on held-out fit-set positions, how often
its argmax lands on the spliced move, and how far the illegal move outranks the legal
control at the same position.

Unlike the embedding grids elsewhere in this repo, **these panels are comparable
point-for-point**. The y axis is a posterior probability, not a coordinate in a fitted
basis, so 0.5 means the same thing in SpaDE's simplex code as in the raw residual. What is
not comparable is the input dimension behind them, which is why `random_proj` is on the
grid: a ReLU of a dense 512 -> 8192 projection preserves nearly everything, so it bounds
what "a probe reads this space" can show, and `random_topk` at L0 80 is the sparsity-matched
null for the k=80 dictionaries. They are not one gray dashed pair (paper_style.SPACE_STYLE).

The rightmost column of the summary is not a space at all: it is `model_baseline`, the
model's own 1 - P(the move played), which needs no probe. It is drawn last because that is
where the reader looks after the eleven, and it is the number the eleven have to beat.

Usage: python analysis/move_legality/space_comparison.py [--camera_ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.figures import move_legality_panels as panels
from spade.figures.paper_style import (
    CLASS_STYLE, NULL, PROBE_CLASS_STYLE, PROBE_HEAD_STYLE, REFERENCE, SPACE_STYLE,
    SPACES, apply_style, rule_key, save_figure)
from spade.move_legality import localize
from spade.move_legality.model_baseline import NAME as BASELINE

FIGURES = repo_root() / "analysis" / "move_legality" / "figures"
RESULTS = repo_root() / "analysis" / "move_legality" / "results"
RULES = localize.RULES
SEED = 42
GRID = (3, 4)
# The no-probe floor gets its own column on the summary, not a panel on the grid: it is a
# detector, not a representation space, and the grid is about what the dictionaries keep.
BASELINE_TICK = "model logits\n(no probe)"
BASELINE_SCORE = "1-P(played)"

# (slug, arm, human label, how to pull the score out of the read-out tensors)
SCORES = [
    ("2class", "illegal", "$P(\\mathrm{illegal})$, 2-class probe",
     lambda p2, p4: p2),
    ("4class_combined", None, "$1 - P(\\mathrm{legal})$, 4-class probe",
     lambda p2, p4: 1.0 - p4[..., 0]),
    *[(f"4class_{r}", r, f"$P(\\mathrm{{{r}}})$, 4-class probe",
       (lambda i: lambda p2, p4: p4[..., i])(i))
      for i, r in enumerate(RULES, start=1)],
]
ARM_STYLE = {
    "control": {**NULL, "label": "legal control (null)"},
    "illegal": {**CLASS_STYLE["violation"],
                "label": "illegal (pooled), 2-class $P(\\mathrm{illegal})$"},
    **{r: {**PROBE_CLASS_STYLE[r],
           "label": f"{PROBE_CLASS_STYLE[r]['label']}, 4-class $P(\\mathrm{{{r}}})$"}
       for r in RULES},
}


def available() -> list[str]:
    """SPACES in paper_style order -- reference first, untrained controls last."""
    have = {p.stem.removeprefix("p_illegal_")
            for p in (study_dir("move_legality") / "readout").glob("p_illegal_*.pt")}
    missing = [s for s in SPACES if s not in have]
    assert not missing, f"no read-out for {missing}; run spade.move_legality.readout"
    return list(SPACES)


def grid(slug: str, arm: str | None, label: str, pull, meta: dict, selection: dict,
         spaces: list[str], camera_ready: bool) -> Path:
    """One panel per space, same axes, reference space first, controls last."""
    fig, axes = plt.subplots(*GRID, figsize=(13.4, 8.6), sharex=True, sharey=True)
    flat = axes.flatten()
    drawn = ["control", *( [arm] if arm else RULES)]
    for ax, space in zip(flat, spaces):
        score = pull(localize.read(space, "p_illegal"), localize.read(space, "proba4"))
        for name in drawn:
            mean, lo, hi = localize.curve(score, meta, selection[name], SEED)
            panels.curve(ax, localize.WINDOW, mean, lo, hi, ARM_STYLE[name],
                         label=ARM_STYLE[name]["label"], markevery=5)
        panels.event_axes(ax, localize.WINDOW, label)
        ax.set_title(SPACE_STYLE[space]["short"].replace("\n", " "), fontsize=8.5,
                     color=SPACE_STYLE[space]["color"])
        ax.set_ylim(0, 1)
    # One axis label per edge: 11 copies of a full-sentence x label collide, and with
    # sharex/sharey the inner ones say nothing the edge does not.
    for i, ax in enumerate(flat[:len(spaces)]):
        ax.set_xlabel("$r = t - t_0$" if i >= len(spaces) - GRID[1] else "")
        if i % GRID[1]:
            ax.set_ylabel("")
    for ax in flat[len(spaces):]:
        ax.axis("off")

    handles = flat[0].get_legend_handles_labels()[0] + panels.region_key()
    flat[len(spaces)].legend(handles=handles, loc="center left", fontsize=8.5,
                             title="$r$: move index relative to the spliced move.\nMean over "
                                   "source games; band = 95% bootstrap CI\n"
                                   f"({len(selection['control'])} games, resampled as clusters)",
                             title_fontsize=8)
    fig.suptitle(f"{label} at every position of a game with one spliced illegal move, "
                 f"across 11 representation spaces", fontsize=11)
    rule_key(fig, ["R1", "R2", "R3"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return save_figure(fig, FIGURES / f"space_grid_{slug}", camera_ready)


def summary(table: pd.DataFrame, quality: pd.DataFrame, spaces: list[str],
            camera_ready: bool) -> Path:
    """Probe quality, localization and paired detection, one row of panels, one x per space."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    columns = [*spaces, BASELINE]
    x = np.arange(len(columns))
    labels = [*[SPACE_STYLE[s]["short"] for s in spaces], BASELINE_TICK]

    for n_classes, style in PROBE_HEAD_STYLE.items():
        rows = quality[(quality["n_classes"] == n_classes) & (quality["split"] == "test")]
        values = [float(rows[rows["space"] == s]["macro_f1"].iloc[0]) for s in spaces]
        axes[0].plot(x[:len(spaces)], values, marker=style["marker"], color=style["color"],
                     ls=style["ls"], markersize=4.5, label=style["label"])
    axes[0].axhline(0.5, **{k: NULL[k] for k in ["color", "ls", "lw"]}, zorder=2)
    axes[0].text(len(spaces) - 0.5, 0.5, " chance (2-class)", color=NULL["color"],
                 fontsize=7.5, ha="right", va="bottom")
    axes[0].axhline(0.25, color=NULL["color"], ls=(0, (1, 1.6)), lw=1.2, zorder=2)
    axes[0].text(len(spaces) - 0.5, 0.25, " chance (4-class)", color=NULL["color"],
                 fontsize=7.5, ha="right", va="bottom")
    axes[0].set_ylabel("macro-F1, held-out games")
    axes[0].set_title("(a) does the probe fit at all?")
    axes[0].set_ylim(0, 1.02)

    for arm in ["illegal", *RULES]:
        values = [pick(table, s, arm) for s in columns]
        axes[1].errorbar(x, [v["hit_rate"] for v in values],
                         yerr=[[v["hit_rate"] - v["hit_lo"] for v in values],
                               [v["hit_hi"] - v["hit_rate"] for v in values]],
                         color=ARM_STYLE[arm]["color"], marker=ARM_STYLE[arm]["marker"],
                         ls="-" if arm == "illegal" else (0, (4, 2)), markersize=4.5,
                         elinewidth=0.8, capsize=1.5, label=ARM_STYLE[arm]["label"])
        axes[2].errorbar(x, [v["auroc_at_t0"] for v in values],
                         yerr=[[v["auroc_at_t0"] - v["auroc_lo"] for v in values],
                               [v["auroc_hi"] - v["auroc_at_t0"] for v in values]],
                         color=ARM_STYLE[arm]["color"], marker=ARM_STYLE[arm]["marker"],
                         ls="-" if arm == "illegal" else (0, (4, 2)), markersize=4.5,
                         elinewidth=0.8, capsize=1.5, label=ARM_STYLE[arm]["label"])
    axes[1].plot(x, [pick(table, s, "control")["hit_rate"] for s in columns],
                 color=NULL["color"], marker=NULL["marker"], ls=NULL["ls"], lw=NULL["lw"],
                 zorder=2, label="legal control (false alarms)")
    axes[1].set_ylabel("argmax over $t \\leq t_0$ lands on $t_0$")
    axes[1].set_title("(b) does it find the position?")
    axes[1].set_ylim(0, 1.02)
    axes[2].axhline(0.5, **{k: NULL[k] for k in ["color", "ls", "lw"]}, zorder=2)
    axes[2].text(len(spaces) - 0.5, 0.5, " chance", color=NULL["color"], fontsize=7.5,
                 ha="right", va="bottom")
    axes[2].set_ylabel("AUROC at $t_0$ vs. the same game's legal control")
    axes[2].set_title("(c) does it separate the move?")
    axes[2].set_ylim(0, 1.02)

    for ax in axes:
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7.5)
        ax.axvline(0.5, color=REFERENCE["color"], lw=0.6, alpha=0.35, zorder=1)
        ax.axvline(len(spaces) - 2.5, color=REFERENCE["color"], lw=0.6, alpha=0.35, zorder=1)
        ax.axvline(len(spaces) - 0.5, color=REFERENCE["color"], lw=0.6, alpha=0.35, zorder=1)
    axes[0].legend(loc="lower left", fontsize=7.4)
    handles = axes[1].get_legend_handles_labels()[0]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=5,
               fontsize=8.2, title="(b) and (c): every bar is a 95% percentile bootstrap CI "
                                   "over source games (1000 resamples, games as clusters)",
               title_fontsize=8)
    axes[0].text(0.0, 1.02, "reference", transform=axes[0].get_xaxis_transform(),
                 fontsize=7, color=REFERENCE["color"], ha="left")
    fig.suptitle("Move-legality probe across 11 spaces, against the no-probe detector: the thin "
                 "vertical rules separate the raw reference (left), the two untrained "
                 "controls, and the model's own next-move distribution (right)", fontsize=10.5)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.145)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return save_figure(fig, FIGURES / "space_summary", camera_ready)


def pick(table: pd.DataFrame, space: str, arm: str) -> pd.Series:
    """The row a summary point comes from: each arm is drawn with its own natural score --
    the 2-class P(illegal) for the pooled arm, the 4-class P(R) for a rule -- and the
    no-probe baseline has only one score, so the lookup is per (space, arm)."""
    score = BASELINE_SCORE if space == BASELINE else (
        "P(illegal)" if arm in ("illegal", "control") else f"P({arm})")
    row = table[(table["space"] == space) & (table["arm"] == arm) & (table["score"] == score)]
    assert len(row) == 1, f"{space}/{arm}/{score}: {len(row)} rows"
    return row.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    spaces = available()
    meta = localize.branches()
    selection = localize.arms(meta, SEED)

    rows = []
    for space in [*spaces, BASELINE]:
        p2 = localize.read(space, "p_illegal")
        rows += localize.scalar_rows(space, BASELINE_SCORE if space == BASELINE else "P(illegal)",
                                     p2, meta, selection, ["control", "illegal", *RULES], SEED)
        if space == BASELINE:
            print(f"scored {space} (no probe, one score and no rule heads)", flush=True)
            continue
        p4 = localize.read(space, "proba4")
        for i, rule in enumerate(RULES, start=1):
            rows += localize.scalar_rows(space, f"P({rule})", p4[..., i], meta, selection,
                                         [rule], SEED)
        rows += localize.scalar_rows(space, "1-P(legal)", 1.0 - p4[..., 0], meta, selection,
                                     ["control", *RULES], SEED)
        print(f"scored {space}", flush=True)
    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "localization_by_space.csv", index=False)

    quality = pd.read_csv(RESULTS / "probe_quality.csv")
    for slug, arm, label, pull in SCORES:
        print(grid(slug, arm, label, pull, meta, selection, spaces, args.camera_ready))
    print(summary(table, quality, spaces, args.camera_ready))
    print(table[table["arm"] != "control"].to_string(index=False))


if __name__ == "__main__":
    main()
