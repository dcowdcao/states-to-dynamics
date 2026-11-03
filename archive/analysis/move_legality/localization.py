"""Where does a legality probe put its score when one move of a game is illegal?

Two figures per space, off the 34,546-branch violation-divergence tree: the 2-class head's
P(illegal) and the 4-class head's 1 - P(legal), each event-locked to the spliced move at t0
and drawn against the reference game and the same game's matched legal control. `--space raw`
is the headline -- whether OthelloGPT's own residual stream carries this at all -- and only
raw is rendered and committed.

Two CSVs carry the numbers the README tables cite, for whatever space is asked for:
`localization_<space>.csv` (localization rate, rank of t0, paired AUROC at t0, and the
pick-a-position chance the rate is read against) and `rule_posterior_<space>.csv` (the mean
4-class posterior at t0 per arm, where the probe's legal/R1 leak shows).

The pre-event window is shaded because r < 0 is bit-identical across a game's branches
(causal attention), so it is a paired baseline and not an approximation of one.

The window STOPS at r = 0. A branch keeps the source game's tokens[t0+1:] verbatim -- the
suffix assumes the *reference* move, never the spliced one -- so past t0 it is replayed onto
a board that no longer exists and no move there has a legality label to plot against. The
probe cannot be read as judging those positions either: every fit-set row's tokens[:t] is a
legal game, so a prefix containing an illegal move is a state it was never fit on. Drawing
the tail invited both readings; the CSVs already score t <= t0 only.

Usage: python analysis/move_legality/localization.py [--space raw] [--camera_ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from spade.common.paths import repo_root
from spade.figures.paper_style import (
    CLASS_STYLE, INK, MUTED, NULL, PRE_EVENT_FACE, PROBE_CLASS_STYLE, REFERENCE, apply_style,
    rule_key, save_figure)
from spade.move_legality import localize
from spade.move_legality.fit_set import CLASSES

FIGURES = repo_root() / "analysis" / "move_legality" / "figures"
RESULTS = repo_root() / "analysis" / "move_legality" / "results"
RULES = localize.RULES
SEED = 42
NOTE = ("mean over source games, one branch drawn per game per arm (seed 42), never a mean "
        "over its\ncategories; bands are 95% percentile bootstrap CIs, 1,000 resamples with "
        "games as clusters")

# Five series on one axis is more than hue and marker carry in grayscale (FIGURE_GUIDELINES
# 1): R1/R2/R3 sit inside a 32-level gray band, so the rules also take a dash pattern and the
# reference is left as the only solid heavy line.
RULE_DASH = {"R1": (0, (6, 1.5, 1, 1.5)), "R2": (0, (4, 2)), "R3": (0, (1, 1.6))}

# The reference is the only branch that is a legal game end to end, so it takes REFERENCE;
# control_legal is a legal move at the same t0 of the same prefix, so it is a matched control
# and takes NULL's gray. The illegal arm keeps the straightness study's violation identity,
# the rules the rules study's.
ARM_STYLE = {
    "reference": {**REFERENCE, "label": "reference (baseline)"},
    "control": {**NULL, "label": "matched legal control"},
    "illegal": {**CLASS_STYLE["violation"], "label": "illegal, any rule"},
    **{r: {**PROBE_CLASS_STYLE[r], "ls": RULE_DASH[r]} for r in RULES},
}


def draw(space: str, slug: str, ylabel: str, score: torch.Tensor, meta: dict,
         selection: dict, arms: list[str], camera_ready: bool) -> Path:
    """One standalone panel: the event-locked mean of every arm with its bootstrap band."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))   # NeurIPS full width, drawn at printed size
    offsets = np.array(localize.WINDOW)
    left, right = offsets[0] - 0.5, offsets[-1] + 0.5
    zero = localize.WINDOW.index(0)

    ax.axvspan(left, -0.5, color=PRE_EVENT_FACE, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)

    means = {}
    for i, name in enumerate(arms):
        rows, style = selection[name], ARM_STYLE[name]
        means[name], lo, hi = localize.curve(score, meta, rows, SEED)
        ax.fill_between(offsets, lo, hi, color=style["color"], alpha=0.18, lw=0, zorder=2)
        ax.plot(offsets, means[name], color=style["color"], marker=style["marker"],
                ls=style.get("ls", "-"), lw=style.get("lw", 1.8), markersize=4.2,
                markevery=(i, 4), markeredgecolor="white", markeredgewidth=0.5,
                zorder=style.get("zorder", 4),
                label=f"{style['label']}\n($n$ = {len(rows)} games)")
        # The event is the whole figure, so every arm carries its marker there whatever the
        # stagger lands on -- at r = 0 the marker is what separates the arms in grayscale.
        ax.plot(0, means[name][zero], color=style["color"], marker=style["marker"],
                markersize=4.6, markeredgecolor="white", markeredgewidth=0.5,
                zorder=style.get("zorder", 4) + 1)

    ax.set_xlim(left, right)
    ax.set_ylim(0, 1)
    ax.set_xlabel("move index relative to the spliced move  ($r = t - t_0$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(int(offsets[0]) + 1, int(offsets[-1]) + 1, 2))
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick == 0:
            text.set(color=INK, fontweight="bold", fontsize=9.5)

    # r < 0 is bitwise identical across the arms, so the reference's level is every arm's.
    baseline = float(means["reference"][:zero].mean())
    ax.axhline(baseline, color=MUTED, ls=(0, (1, 2)), lw=1.0, zorder=2)
    ax.text(left + 0.3, baseline + 0.012, f"pre-event level {baseline:.3f}", color=MUTED,
            fontsize=7.2, ha="left", va="bottom")
    ax.text((left - 0.5) / 2, 0.015, "$r<0$: shared prefix, identical in every branch",
            color=MUTED, fontsize=7.2, ha="center", va="bottom")

    top = max((n for n in arms if n not in ("reference", "control")),
              key=lambda n: means[n][zero])
    ax.annotate(f"illegal move spliced in at $t_0$\n"
                f"peak {float(means[top][zero]):.3f} ({ARM_STYLE[top]['label']})",
                xy=(0, float(means[top][zero])), xytext=(0.99, 1.03),
                textcoords="axes fraction", ha="right", va="bottom", fontsize=8.5,
                color=INK, arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.9,
                                       "shrinkB": 4})
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6, labelspacing=0.4,
              borderaxespad=0.5)

    fig.tight_layout()
    fig.text(0.005, -0.005, NOTE, ha="left", va="top", fontsize=7, color=MUTED)
    if any(name in RULES for name in arms):
        rule_key(fig, RULES, y=-0.105)
    return save_figure(fig, FIGURES / f"{space}_{slug}", camera_ready)


def posterior_at_t0(space: str, proba4: torch.Tensor, meta: dict,
                    selection: dict) -> pd.DataFrame:
    """Mean 4-class posterior at t0 per arm: where the probe's legal/R1 leak shows.

    Not an argmax confusion matrix. The mean posterior keeps the leak an argmax hides, and
    the leak is the pair of rows the study's headline rests on -- the legal branches and the
    R1 branches, which the probe does not separate.
    """
    index = meta["t0"][:, None, None].expand(-1, 1, proba4.shape[-1])
    at_t0 = proba4.gather(1, index).squeeze(1)
    return pd.DataFrame([
        {"space": space, "arm": name, "n_games": len(selection[name]),
         **{f"P({c})": round(float(v), 5)
            for c, v in zip(CLASSES, at_t0[selection[name]].mean(0))}}
        for name in ["reference", "control", *RULES]])


def chance(meta: dict, rows: torch.Tensor) -> float:
    """Pick-a-position chance for an arm: mean 1/(t0+1) over its games, the null the
    localization rate is read against."""
    return round(float((1.0 / (meta["t0"][rows].double() + 1)).mean()), 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="raw")
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    space = args.space
    meta = localize.branches()
    selection = localize.arms(meta, SEED)
    rows = []

    p_illegal = localize.read(space, "p_illegal")
    print(draw(space, "2class", "$P(\\mathrm{illegal})$    ($1$ = illegal)", p_illegal, meta,
               selection, ["reference", "control", "illegal"], args.camera_ready))
    rows += localize.scalar_rows(space, "P(illegal)", p_illegal, meta, selection,
                                 ["control", "illegal", *RULES], SEED)

    proba4 = localize.read(space, "proba4")
    for i, rule in enumerate(RULES, start=1):
        rows += localize.scalar_rows(space, f"P({rule})", proba4[..., i], meta, selection,
                                     ["control", *RULES], SEED)
    any_illegal = 1.0 - proba4[..., 0]
    print(draw(space, "4class", "$1 - P(\\mathrm{legal})$    ($1$ = illegal)", any_illegal,
               meta, selection, ["reference", "control", *RULES], args.camera_ready))
    rows += localize.scalar_rows(space, "1-P(legal)", any_illegal, meta, selection,
                                 ["control", *RULES], SEED)

    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table["chance_hit_rate"] = [chance(meta, selection[arm]) for arm in table["arm"]]
    table.to_csv(RESULTS / f"localization_{space}.csv", index=False)
    posterior = posterior_at_t0(space, proba4, meta, selection)
    posterior.to_csv(RESULTS / f"rule_posterior_{space}.csv", index=False)
    print(f"wrote {RESULTS}/localization_{space}.csv and rule_posterior_{space}.csv")
    print(table.to_string(index=False))
    print(posterior.to_string(index=False))


if __name__ == "__main__":
    main()
