"""Does the probe put its highest score on the move that is actually illegal?

Six figures per space, off the 34,546-branch violation-divergence tree: one for the 2-class
probe, one per constraint rule from the 4-class probe, one combining the three rules, and
one asking whether the right rule head fires. `--space raw` is the headline -- whether
OthelloGPT's own residual stream carries this at all -- and every space in the roster is
drawn the same way.

Every panel is event-locked to the spliced move at t0. The pre-event window is shaded
because r < 0 is bit-identical across a game's branches (causal attention), so it is a
paired baseline and not an approximation of one; the post-event region is hatched because
the branch replays the source game's suffix onto a board that no longer exists, so nothing
there carries a ground-truth label. Localization is therefore scored over t <= t0 only, and
the legal control branch of the same game at the same t0 is the false-alarm rate of exactly
that statistic.

Usage: python analysis/move_legality/localization.py [--space raw] [--camera_ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from spade.common.paths import repo_root
from spade.figures import move_legality_panels as panels
from spade.figures.paper_style import (
    CLASS_STYLE, NULL, PROBE_CLASS_STYLE, REFERENCE, apply_style, rule_key, save_figure)
from spade.move_legality import localize
from spade.move_legality.fit_set import CLASSES

FIGURES = repo_root() / "analysis" / "move_legality" / "figures"
RESULTS = repo_root() / "analysis" / "move_legality" / "results"
RULES = localize.RULES
HEATMAP_ROWS = 300
SEED = 42

# The illegal class pooled over its six categories keeps the straightness study's violation
# identity; the three rules keep the rules study's. The reference is the only branch that is
# a legal game end to end, so it takes REFERENCE; control_legal is legal at t0 with a spliced
# suffix, so it is the matched null and takes NULL.
ARM_STYLE = {
    "reference": {**REFERENCE, "short": "reference (legal)",
                  "label": "reference: the game as played (baseline)"},
    "control": {**NULL, "short": "legal control",
                "label": "control: another legal move at $t_0$ (null)"},
    "illegal": {**CLASS_STYLE["violation"], "short": "illegal (pooled)",
                "label": "illegal move (pooled over R1-R3)"},
    **{r: {**PROBE_CLASS_STYLE[r], "short": PROBE_CLASS_STYLE[r]["label"],
           "label": f"illegal move, {PROBE_CLASS_STYLE[r]['label']}"} for r in RULES},
}


def heatmap_matrix(score: torch.Tensor, meta: dict, rows: torch.Tensor) -> np.ndarray:
    """`HEATMAP_ROWS` branches spread evenly over the t0-sorted arm."""
    order = rows[meta["t0"][rows].argsort(stable=True)]
    take = torch.linspace(0, len(order) - 1, HEATMAP_ROWS).long()
    return localize.peri_event(score, meta["t0"], localize.WINDOW)[order[take]].numpy()


def draw(space: str, slug: str, label: str, score: torch.Tensor, meta: dict,
         focus: dict[str, torch.Tensor], selection: dict, title: str,
         camera_ready: bool) -> Path:
    """Items, event-locked mean, localization against t0, paired detection."""
    fig, ((ax_items, ax_curve), (ax_rate, ax_auroc)) = plt.subplots(2, 2, figsize=(11.4, 8.6))
    control = selection["control"]
    first = next(iter(focus))

    panels.item_heatmap(ax_items, heatmap_matrix(score, meta, focus[first]),
                        localize.WINDOW, label, fig=fig)
    ax_items.set_title(f"(a) {len(focus[first])} {first} branches, {HEATMAP_ROWS} drawn "
                       f"evenly over $t_0$")

    baseline = None
    for name in ["reference", "control", *focus]:
        rows = selection[name]
        mean, lo, hi = localize.curve(score, meta, rows, SEED)
        panels.curve(ax_curve, localize.WINDOW, mean, lo, hi, ARM_STYLE[name],
                     label=f"{ARM_STYLE[name]['label']}  ($n$ = {len(rows)} games)")
        if name == first:
            baseline = float(mean[:localize.WINDOW.index(0)].mean())
    panels.event_axes(ax_curve, localize.WINDOW, label)
    ax_curve.axhline(baseline, color=ARM_STYLE[first]["color"], ls=(0, (1, 2)), lw=1.0, zorder=2)
    ax_curve.set_title(f"(b) event-locked mean; dotted line is the {first} arm's pre-event "
                       f"level, {baseline:.3f}")

    rates = [{"label": ARM_STYLE[n]["label"], "style": ARM_STYLE[n],
              **localize.hit_curve(score, meta, selection[n], SEED)}
             for n in ["control", *focus]]
    panels.rate_by_t0(ax_rate, rates[0]["t0"], rates, chance=rates[0]["chance"])
    ax_rate.set_title("(c) localization: does the argmax over $t \\leq t_0$ land on $t_0$?")
    ax_rate.legend(handles=[ax_rate.get_legend_handles_labels()[0][0]], loc="center left",
                   fontsize=7.4)

    # Every arm, not only the focus: which rule the probe is blind to is the finding, and a
    # one-row panel is a layout bug (FIGURE_GUIDELINES 5).
    bars = [{"label": f"{ARM_STYLE[n]['short']}  ($n$ = {len(selection[n])})",
             "style": ARM_STYLE[n],
             **localize.detection(score, meta, selection[n],
                                  localize.matched_control(meta, selection[n], control), seed=SEED)}
            for n in ["illegal", *RULES]]
    panels.auroc_bars(ax_auroc, bars)
    ax_auroc.set_title("(d) detection at $t_0$, paired with the same game's legal control")

    handles = ax_curve.get_legend_handles_labels()[0] + panels.region_key()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=3,
               fontsize=8.2,
               title="every band and bar is a 95% percentile bootstrap CI over source games "
                     "(1000 resamples, games resampled as clusters)", title_fontsize=8)
    fig.suptitle(title, fontsize=11)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.115)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return save_figure(fig, FIGURES / f"{space}_{slug}", camera_ready)


def rule_identity(space: str, proba4: torch.Tensor, meta: dict, selection: dict,
                  camera_ready: bool) -> Path:
    """When the probe is right that something is wrong, does it name the right rule?"""
    fig, (ax_conf, ax_auroc) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    index = meta["t0"][:, None, None].expand(-1, 1, proba4.shape[-1])
    at_t0 = proba4.gather(1, index).squeeze(1)

    names = ["reference", "control", *RULES]
    matrix = np.stack([at_t0[selection[n]].mean(0).numpy() for n in names])
    panels.confusion(ax_conf, matrix,
                     [f"{n} ($n$={len(selection[n])})" for n in names], list(CLASSES), fig=fig)
    ax_conf.set_title("(a) mean 4-class posterior at $t_0$")

    bars = [{"label": f"P({r}) on {r} branches", "style": ARM_STYLE[r],
             **localize.detection(proba4[..., i], meta, selection[r],
                                  localize.matched_control(meta, selection[r], selection["control"]),
                                  seed=SEED)}
            for i, r in enumerate(RULES, start=1)]
    panels.auroc_bars(ax_auroc, bars)
    ax_auroc.set_title("(b) does the matching head fire?")
    fig.suptitle(f"4-class probe in {space}: rule identity at the spliced move", fontsize=11)
    rule_key(fig, ["R1", "R2", "R3"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return save_figure(fig, FIGURES / f"{space}_rule_identity", camera_ready)


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
    print(draw(space, "2class", "$P(\\mathrm{illegal})$", p_illegal, meta,
               {"illegal": selection["illegal"]}, selection,
               f"2-class move-legality probe in {space}: legal vs illegal",
               args.camera_ready))
    rows += localize.scalar_rows(space, "P(illegal)", p_illegal, meta, selection,
                                 ["control", "illegal", *RULES], SEED)

    proba4 = localize.read(space, "proba4")
    for i, rule in enumerate(RULES, start=1):
        print(draw(space, f"4class_{rule}", f"$P(\\mathrm{{{rule}}})$", proba4[..., i], meta,
                   {rule: selection[rule]}, selection,
                   f"4-class move-legality probe in {space}: the {rule} posterior",
                   args.camera_ready))
        rows += localize.scalar_rows(space, f"P({rule})", proba4[..., i], meta, selection,
                                     ["control", *RULES], SEED)

    any_illegal = 1.0 - proba4[..., 0]
    print(draw(space, "4class_combined", "$1 - P(\\mathrm{legal})$", any_illegal, meta,
               {r: selection[r] for r in RULES}, selection,
               f"4-class move-legality probe in {space}: all three rules, one legal control",
               args.camera_ready))
    rows += localize.scalar_rows(space, "1-P(legal)", any_illegal, meta, selection,
                                 ["control", *RULES], SEED)
    print(rule_identity(space, proba4, meta, selection, args.camera_ready))

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv = RESULTS / f"localization_{space}.csv"
    table = pd.DataFrame(rows)
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
