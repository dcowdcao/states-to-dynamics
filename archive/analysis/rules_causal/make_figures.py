"""Figures for the rules causal experiments (R5 un-flip edit, R5 pair
patching, R4 forfeit patching). Reads results/*.csv, writes figures here.

Conventions: FIGURE_GUIDELINES.md §3 (the random-direction edit and the
corrupted run are nulls, not conditions), §4.1 (layer sweeps), §4.4 (the
legal-mass histogram is bounded and skewed, so median/IQR annotates it).

Usage: python analysis/rules_causal/make_figures.py [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from spade.figures.paper_style import (
    COMPONENT_STYLE, GRID, INK, MUTED, NULL, POPULATION_STYLE, REFERENCE,
    apply_style, save_figure)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POP_ORDER = ["flipped", "settled", "placed", "random"]
# The un-flip edit scores two disjoint square sets; the pair reads in grayscale
# as filled vs hatched, so population keeps the color channel to itself.
SERIES = [("d_lp_cf", "counterfactual-only legal", None),
          ("d_lp_real", "real-only legal", "////")]
LAYERS = "transformer block of the patched component (0 = first block)"


def ci95(x: pd.Series) -> float:
    x = x.dropna()
    return 1.96 * x.std() / max(len(x), 1) ** 0.5


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def unflip_panel(ax, d: pd.DataFrame, title: str) -> None:
    pops = [p for p in POP_ORDER if p in set(d.population)]
    x = np.arange(len(pops))
    if "random" in pops:  # mark the null's slot, not just its color
        ax.axvspan(x[pops.index("random")] - 0.5, x[-1] + 0.5, color=GRID,
                   alpha=0.6, lw=0, zorder=0)
    for off, (col, _, hatch) in zip((-0.18, 0.18), SERIES):
        for xi, p in zip(x, pops):
            style, is_null = POPULATION_STYLE[p], p == "random"
            vals = d[d.population == p][col]
            ax.bar(xi + off, vals.mean(), 0.34, yerr=ci95(vals), capsize=2.5,
                   facecolor="white" if hatch else style["color"],
                   edgecolor=style["color"], hatch=hatch, lw=1.0,
                   ls=NULL["ls"] if is_null else "-", zorder=3,
                   error_kw={"lw": 0.9, "ecolor": INK, "zorder": 4})
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_xticks(x, [f"{POPULATION_STYLE[p]['short']}\nn={(d.population == p).sum()}"
                      for p in pops], fontsize=8)
    ax.set_ylabel("$\\Delta$ log probability mass (edited $-$ base)")
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor="white" if h else MUTED, edgecolor=MUTED,
                             hatch=h, label=lab) for _, lab, h in SERIES],
              title="bars: mean $\\pm$ 95% CI over items\n"
                    "$\\uparrow$ mass onto the counterfactual board's legal set",
              title_fontsize=7.5, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 0.94))


def unflip_figure(camera_ready: bool) -> None:
    """Populations x edit magnitude; the second row holds the item set fixed,
    since settled controls only exist once squares can be 10 moves old."""
    df = pd.read_csv(RESULTS / "unflip_edit.csv")
    betas = sorted(df.beta.unique())
    # Shared y within a column (same edit magnitude, all items vs the matched
    # subset); never across betas, whose scales differ by design.
    fig, axes = plt.subplots(2, len(betas), figsize=(4.6 * len(betas), 7.4),
                             squeeze=False, sharey="col")
    for c, beta in enumerate(betas):
        d = df[df.beta == beta]
        matched = d[d.item.isin(d[d.population == "settled"].item)]
        for r, (rows, what) in enumerate([(d, "all items"),
                                          (matched, "items with a settled control")]):
            unflip_panel(axes[r][c], rows, f"$\\beta$={beta:g} — {what}")
            panel_letter(axes[r][c], r * len(betas) + c)
    if not camera_ready:
        fig.suptitle("R5 un-flip edit: reverting one square along its board-state "
                     "probe direction", y=0.998)
    fig.tight_layout()
    save_figure(fig, HERE / "unflip_edit", camera_ready)


def component_curves(ax, df: pd.DataFrame, value: str, variants: list[tuple]) -> None:
    """One curve per (variant, component): color = component, dash = variant."""
    for mask, ls, suffix in variants:
        for comp, style in COMPONENT_STYLE.items():
            d = df[mask & df.patch.str.startswith(comp)]
            g = d.assign(layer=d.patch.str[len(comp):].astype(int)).groupby("layer")[value]
            mean, err = g.mean(), g.apply(ci95)
            ax.errorbar(mean.index, mean, yerr=err, ls=ls, marker=style["marker"],
                        ms=4.5, color=style["color"], elinewidth=0.8, capsize=2,
                        ecolor=style["color"], markeredgecolor="white",
                        markeredgewidth=0.6, zorder=3,
                        label=style["label"] + suffix)
    ax.set_xticks(range(int(df.patch.str[-1].astype(int).max()) + 1))
    ax.set_xlabel(LAYERS)


def pair_patch_figure(camera_ready: bool) -> None:
    df = pd.read_csv(RESULTS / "pair_patch.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.axhline(1, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1,
               label="full-residual patch $\\equiv$ 1 (shared prefix)")
    ax.axhline(0, color=MUTED, lw=0.9, ls=NULL["ls"], zorder=1,
               label="corrupted run (no recovery)")
    component_curves(ax, df, "recovery", [(df.variant == "all", "-", " (all diag.)"),
                                          (df.variant == "strict", "--", " (strict)")])
    ax.set_ylabel("score recovery (0 = corrupted, 1 = clean)")
    ax.set_ylim(-0.08, 1.55)  # headroom so the legend clears the ceiling line
    ax.legend(title=f"bars: 95% CI over pairs (n={df.pair.nunique()})",
              title_fontsize=8, fontsize=8, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="none")
    if not camera_ready:
        ax.set_title("R5 pair patching: where flip consequences are written")
    fig.tight_layout()
    save_figure(fig, HERE / "pair_patch", camera_ready)


def forfeit_behavior(ax, per_event: pd.DataFrame) -> None:
    """Legal-set mass at forfeit positions vs t-matched non-forfeit controls."""
    bins = np.linspace(0, 1, 41)
    for col, style, label in [
            ("corr_legal_mass", REFERENCE, "matched non-forfeit (baseline)"),
            ("clean_legal_mass", POPULATION_STYLE["flipped"], "forfeit position")]:
        mass = np.exp(per_event[col])
        q1, med, q3 = np.percentile(mass, [25, 50, 75])
        ax.hist(mass, bins=bins, histtype="stepfilled", lw=1.4, zorder=3,
                color=style["color"], alpha=0.35,
                label=f"{label}\nmedian {med:.3f}, IQR [{q1:.3f}, {q3:.3f}]")
        ax.hist(mass, bins=bins, histtype="step", lw=1.4, zorder=4,
                color=style["color"])
    ax.set_xlabel("probability mass on the true legal set")
    ax.set_ylabel(f"events (n={len(per_event)})")
    ax.legend(fontsize=7.5, loc="upper left")


def forfeit_figure(camera_ready: bool) -> None:
    df = pd.read_csv(RESULTS / "forfeit_patch.csv")
    per_event = df[df.patch == "resid0"]  # one row per event; also the sanity patch
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    forfeit_behavior(axes[0], per_event)
    axes[0].set_title("behavioral: top-1 legal at forfeits "
                      f"{per_event.clean_top1_legal.mean():.1%}", fontsize=9.5)

    ax = axes[1]
    ax.axhline(per_event.s_clean.mean(), lw=1.2, ls="--", zorder=1,
               color=REFERENCE["color"],
               label="clean run (= full-residual patch, sanity $\\checkmark$)")
    ax.axhline(per_event.s_corr.mean(), lw=1.2, ls=NULL["ls"], zorder=1,
               color=NULL["color"], label="corrupted run (no patch)")
    component_curves(ax, df, "s_patch", [(df.patch.notna(), "-", "")])
    ax.set_ylabel("color score: log mass(e-only) $-$ log mass(d-only)")
    ax.set_ylim(top=per_event.s_clean.mean() + 1.8)  # room for the legend
    ax.legend(title="bars: 95% CI over events\n"
                    "$\\uparrow$ model treats e as the mover (correct at a forfeit)",
              title_fontsize=7.5, fontsize=8, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="none")
    ax.set_title("causal: turn-assignment transfer", fontsize=9.5)
    for i, ax in enumerate(axes):
        panel_letter(ax, i)
    if not camera_ready:
        fig.suptitle("R4 forfeit: behaviorally solved, computed in the mid-late MLPs",
                     y=1.0)
    fig.tight_layout()
    save_figure(fig, HERE / "forfeit", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    unflip_figure(args.camera_ready)
    pair_patch_figure(args.camera_ready)
    forfeit_figure(args.camera_ready)


if __name__ == "__main__":
    main()
