"""The four figures of the legality-probe study. Drawing only.

F1  what each space supports    — board and legality probes, against two untrained controls
F2  HEADLINE, what a move costs — paired macro-F1 against raw over the window r
F3  which rule each space names — correct-rule recall on the shared pre-move board, r = -1
F4  which standard metric sees it — L0, variance explained, board probe

Everything is read from `results/{probe_fit,readout,space_quality}.csv`; nothing here
computes a statistic except the Spearman rho annotated on F4 and F3's within/between
spread, both of which are summaries of numbers already on the page.

Read the CSVs with keep_default_na=False: `category` and `rule` are empty strings on the
aggregate rows, and NaN there silently drops every aggregate.

Legality is a deterministic function of the board and the board decodes at ~0.98, so no
figure presents a macro-F1 *level* as a result. What is read is always a difference:
between a space and raw at the same offset, or between r < 0 and r = 0 within a space.

The two untrained controls are not interchangeable and are never drawn as one gray pair.
`random_topk` has L0 80, exactly TopK's k, and is the sparsity-matched null. `random_proj`
is a ReLU of a dense 512 -> 8192 projection at L0 4101; it preserves nearly all of the
input, so it bounds what "a probe reads this space" can demonstrate. F1 makes the
distinction the panel's subject, because on the study's own headline metric
(`drop_agreement_with_raw`) `random_proj` is the best space in the study.

`excess_macro_f1` is in readout.csv and is deliberately not plotted: it is not scale-free.
SpaDE's is negative only because its r < 0 baseline is already 0.42, so there is less left
to lose. F2 shows the r < 0 level and the r = 0 step in the same panel instead, and F4
shows the agreement baseline beside its drop, so no floor effect can hide inside a
difference.

Usage: python analysis/legality_probe/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spade.figures.paper_style import (
    CONTROL_SPACES, DICTIONARIES, GRID, INK, MUTED, PROBE_CLASS_STYLE, REFERENCE_SPACE,
    RULE_STYLE, SPACE_FAMILIES, SPACE_STYLE, SPACES, apply_style, rule_key, save_figure)
from spade.othello.taxonomy import RULE_MEMBERS

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

CLASSES = list(PROBE_CLASS_STYLE)
RULES = list(RULE_MEMBERS)
POOLED = "empty_pooled"
ILLEGAL = "illegal"
# Per-game paired difference from raw at the same offset; it is what removes the
# ply-driven class-balance drift from F2. raw's own rows are identically zero, which is a
# free correctness check on the pairing and is asserted and drawn rather than dropped.
DELTA_METRIC = "macro_f1_minus_raw"
# The shared pre-move board: every branch of a game coincides here bitwise, so anything
# read at r = -1 is a property of the space, not an effect of the injected move. It is
# also the phase of the mover-relative frame in which the illegal move is about to be
# played -- the frame alternates each ply, so even offsets read the opponent's board and
# R3 collapses there by construction (FIGURE_GUIDELINES 4.8).
PREFIX_R = -1
BARS = "bars: 95% bootstrap CI over source games"
# SpaDE's paired difference is ~5x the widest of the other three families, so one shared y
# axis would flatten three panels of four. The three ordinary families share an axis; the
# SpaDE panel gets its own and draws the shared range as a band, so the change of scale is
# stated on the page rather than left in the ticks.
WIDE_SCALE_FAMILY = "SpaDE"


def styled(space: str) -> dict:
    return SPACE_STYLE[space]


def read(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = pd.read_csv(results / "probe_fit.csv", keep_default_na=False)
    readout = pd.read_csv(results / "readout.csv", keep_default_na=False)
    quality = pd.read_csv(results / "space_quality.csv", keep_default_na=False)
    for column in ("l0", "l0_window", "variance_explained"):
        quality[column] = pd.to_numeric(quality[column], errors="coerce")
    assert set(fit.space) == set(readout.space) == set(quality.space) == set(SPACES), \
        f"the CSVs and paper_style.SPACES disagree: {set(readout.space) ^ set(SPACES)}"
    return fit, readout, quality.set_index("space")


def macro(fit: pd.DataFrame, target: str) -> pd.DataFrame:
    return fit[(fit.metric == "macro_f1") & (fit["class"] == "")
               & (fit.target == target)].set_index("space")


def pooled_curves(readout: pd.DataFrame, metric: str) -> pd.DataFrame:
    return readout[(readout.role == ILLEGAL) & (readout.category == POOLED)
                   & (readout.metric == metric)]


def dot_rows(ax, spaces: list[str]) -> np.ndarray:
    """Spaces top-to-bottom on the y axis, raw first, the untrained controls last."""
    y = np.arange(len(spaces))
    ax.set_yticks(y, [styled(s)["short"] for s in spaces], fontsize=7.5)
    ax.set_ylim(len(spaces) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    for row in y:  # one guide per row, so a marker hanging below a tick still reads
        ax.axhline(row, color=GRID, lw=0.5, zorder=0)
    ax.axhline(0.5, color=MUTED, lw=0.6, zorder=0)                        # raw | trained
    ax.axhline(len(spaces) - len(CONTROL_SPACES) - 0.5, color=MUTED, lw=0.6, zorder=0)
    return y


def n_text(d: pd.DataFrame) -> str:
    lo, hi = int(d.n.min()), int(d.n.max())
    return f"n = {lo}" if lo == hi else f"n = {lo}-{hi}"


def point(ax, x, row, space: str, value, lo, hi, size: float = 7.5) -> None:
    """A space's marker on a dot-row panel: the value runs along x, the row indexes y."""
    style = styled(space)
    ax.errorbar(x, row, xerr=[[value - lo], [hi - value]], marker=style["marker"], ms=size,
                lw=0, elinewidth=1.1, capsize=2, color=style["color"], ecolor=style["color"],
                markeredgecolor="white", markeredgewidth=0.6, zorder=style["zorder"])


def scatter_point(ax, x, r: pd.Series, space: str) -> None:
    """A space's marker on an x-vs-y scatter, with the CI on y where the quantity is."""
    style = styled(space)
    ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]],
                marker=style["marker"], ms=8, lw=0, elinewidth=1.1, capsize=2,
                color=style["color"], ecolor=style["color"], markeredgecolor="white",
                markeredgewidth=0.7, zorder=style["zorder"])


# --- F1: what each space supports, against two untrained controls -----------

def f1_probes_per_space(fit: pd.DataFrame, camera_ready: bool) -> None:
    """A dumbbell per space, board endpoint open, legality endpoint filled, with both
    untrained controls drawn as levels across the trained rows.

    The dumbbell puts the eye on the segment rather than on the two levels, which is what
    the study can claim. The two vertical rules are the point of the panel: the
    sparsity-matched null is 0.15 below TopK at the same L0, and the unmatched wide random
    expansion sits above three of the eight trained dictionaries.
    """
    legality, board = macro(fit, "legality"), macro(fit, "board")
    per_class = fit[(fit.metric == "per_class_f1")
                    & (fit.target == "legality")].set_index(["space", "class"])
    hyper = legality.loc[REFERENCE_SPACE]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=True)
    y = dot_rows(axes[0], SPACES)
    dot_rows(axes[1], SPACES)

    matched, ceiling = (legality.loc[s, "value"] for s in CONTROL_SPACES)
    wrong = macro(fit, "legality_wrong_player").loc[REFERENCE_SPACE, "value"]
    for space, x in zip(CONTROL_SPACES, (matched, ceiling)):
        style = styled(space)
        axes[0].axvline(x, color=style["color"], ls=style["ls"], lw=style["lw"], zorder=1)
    axes[0].text(0.362, 9.85, "the two vertical rules, both untrained:\n"
                 f"gray dashed — sparsity-matched null, $L_0$ 80 ({matched:.3f})\n"
                 f"dark dotted — wide random expansion, $L_0$ 4101 ({ceiling:.3f})",
                 fontsize=6.5, color=MUTED, style="italic", ha="left", va="center",
                 linespacing=1.6)

    for row, space in zip(y, SPACES):
        style = styled(space)
        b, rules = board.loc[space], legality.loc[space]
        axes[0].plot([b.value, rules.value], [row, row], color=style["color"], lw=1.8,
                     zorder=3, solid_capstyle="butt")
        axes[0].plot(b.value, row, marker=style["marker"], ms=6, lw=0,
                     markerfacecolor="white", markeredgecolor=style["color"],
                     markeredgewidth=1.3, zorder=4)
        point(axes[0], rules.value, row, space, rules.value, rules.lo, rules.hi)
        for code in CLASSES:
            class_style = PROBE_CLASS_STYLE[code]
            axes[1].plot(per_class.loc[(space, code), "value"], row,
                         marker=class_style["marker"], ms=5, lw=0,
                         markerfacecolor="white", markeredgecolor=class_style["color"],
                         markeredgewidth=1.1, zorder=3,
                         label=class_style["label"] if row == 0 else None)

    topk = legality.loc["TopKTrainer-othello", "value"]
    axes[0].annotate("", xy=(matched, 6.48), xytext=(topk, 6.48),
                     arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9})
    axes[0].text(matched - 0.012, 6.48, f"+{topk - matched:.3f} at matched $L_0$ = 80",
                 fontsize=7, color=INK, ha="right", va="center")

    axes[0].set_xlim(0.35, 1.0)
    axes[1].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("macro-F1, held-out positions"
                       r"   ($\rightarrow$ more of the target is linearly decodable)")
    axes[1].set_xlabel("per-class F1 of the legality probe")
    axes[0].set_title("(a)  the segment is what the rules cost on top of the board",
                      loc="left", fontsize=9.5)
    axes[1].set_title("(b)  and it is one boundary that pays for it", loc="left", fontsize=9.5)
    axes[1].legend(fontsize=7, loc="upper left", labelspacing=0.35, handletextpad=0.4,
                   borderpad=0.3, title="legality probe class", title_fontsize=7)
    axes[1].text(0.02, 10.12, "R1 includes the 4 starting squares, readable off\nthe role "
                 "embedding alone; dropping them moves\nR1 by $\\leq$ 0.02 in every space.",
                 fontsize=6.5, color=MUTED, style="italic", ha="left", va="center",
                 linespacing=1.6)

    fig.text(0.5, 0.945, "Open marker: the "
             f"{int(board.loc[REFERENCE_SPACE, 'n_classes'])}-way board probe "
             "(empty / theirs / mine). Filled marker with bars: the "
             f"{int(hyper.n_classes)}-way legality probe, $d_f$ = {int(hyper.filler_dim)}. "
             "Both fitted in the same space with the same machinery, OthelloGPT layer 5. "
             f"x truncated at 0.35 in (a); no marker is clipped.\n"
             f"Raw's legality scored for the WRONG player reads {wrong:.3f}, within 0.004 of "
             "the untrained ceiling.\n"
             f"Bars are 95% bootstrap CI over source games ({n_text(legality)}). Read the "
             "segments and the spread across rows, never the level: legality is a "
             "deterministic function of a board that already decodes at 0.98, so a space "
             "that loses board and legality equally is merely lossy.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top", linespacing=1.5)
    if not camera_ready:
        fig.suptitle("An untrained wide random expansion supports the legality probe "
                     "better than three of the eight trained dictionaries", fontsize=11,
                     y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    rule_key(fig, ["R1", "R2", "R3"])
    save_figure(fig, FIGURES / "f1_probes_per_space", camera_ready)


# --- F2: the headline ------------------------------------------------------

def curve(ax, d: pd.DataFrame, space: str, label: str | None, offset: int) -> None:
    """The CI rides on the markers, not on a ribbon: eleven series share these axes and
    eleven fills covered the curves they annotated. `offset` staggers both."""
    style, d = styled(space), d.sort_values("r")
    ax.errorbar(d.r, d["mean"], yerr=[d["mean"] - d.lo, d.hi - d["mean"]],
                color=style["color"], marker=style["marker"], ms=5, lw=style["lw"],
                ls=style["ls"], markevery=(offset, 5), errorevery=(offset, 5),
                elinewidth=0.8, capsize=1.8, ecolor=style["color"],
                markeredgecolor="white", markeredgewidth=0.5, zorder=style["zorder"],
                label=label)


def event_axes(ax, window: np.ndarray) -> None:
    ax.axvspan(window[0], 0, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.set_xlim(window[0], window[-1])
    ax.set_xticks(range(int(window[0]), int(window[-1]) + 1, 8))


def f2_readout_window(readout: pd.DataFrame, camera_ready: bool) -> None:
    """Panel (a) is raw and the untrained controls absolute; (b)-(e) are paired differences.

    The 4-way class balance is a function of absolute ply -- R1 is exactly (ply + 5)/64 and
    R2 collapses late -- and t0 ranges over [10, 34], so an absolute macro-F1 curve drifts
    against r for reasons no dictionary controls. The drift is common to every space, so a
    per-game paired difference against raw at the same r removes it exactly.

    Both untrained controls are drawn in every family panel, because the study's sharpest
    contrast is the step at r = 0: the six ordinary dictionaries each lose an extra 0.02-0.06
    macro-F1 at the violation itself, and neither untrained control loses anything there.
    The SpaDE pair moves the other way, and that is a floor artifact rather than a
    counterexample -- their r < 0 baselines are 0.49 and 0.43, so raw has more to lose than
    they do. RED_TEAM.md marks the excess statistic as not scale-free for this reason.
    """
    absolute = pooled_curves(readout, "macro_f1")
    delta = pooled_curves(readout, DELTA_METRIC)
    reference = delta[delta.space == REFERENCE_SPACE]
    assert (reference[["mean", "lo", "hi"]].abs().to_numpy() == 0).all(), \
        "raw's paired difference from raw is not identically zero: the pairing is broken"
    window = np.sort(absolute.r.unique())

    # Everything here is a loss, so a symmetric axis would waste half of every panel; the
    # zero line is raw's own series and is in every panel's legend, so the asymmetry cannot
    # be misread. The headroom above zero is where the panel legends go.
    def bounds(spaces: list[str]) -> tuple[float, float]:
        d = delta[delta.space.isin(spaces)]
        return d.lo.min(), d.hi.max()

    extent = {family: -bounds(members + CONTROL_SPACES)[0] for family, members in
              SPACE_FAMILIES.items()}
    assert extent[WIDE_SCALE_FAMILY] == max(extent.values()), \
        f"{WIDE_SCALE_FAMILY} is no longer the family that needs its own y axis: {extent}"
    ordinary = [s for f, m in SPACE_FAMILIES.items() if f != WIDE_SCALE_FAMILY for s in m]
    floor, ceiling = bounds(ordinary + CONTROL_SPACES)
    pad = 0.06 * (ceiling - floor)
    shared = (floor - pad, ceiling + 7 * pad)
    wide = (bounds(SPACE_FAMILIES[WIDE_SCALE_FAMILY] + CONTROL_SPACES)[0] - pad,
            shared[1] + 0.30 * (shared[1] - shared[0]))

    fig = plt.figure(figsize=(11.6, 5.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30, hspace=0.44,
                          left=0.07, right=0.99, top=0.76, bottom=0.13)
    ref_ax = fig.add_subplot(gs[:, 0])
    family_axes = [fig.add_subplot(gs[i // 2, 1 + i % 2]) for i in range(len(SPACE_FAMILIES))]

    event_axes(ref_ax, window)
    for j, space in enumerate([REFERENCE_SPACE] + CONTROL_SPACES):
        curve(ref_ax, absolute[absolute.space == space], space, styled(space)["label"], j)
    ref_ax.set_ylim(0.0, 1.0)
    ref_ax.set_ylabel("macro-F1 against the engine's legality")
    ref_ax.set_title("(a)  raw and the two untrained controls, absolute", loc="left",
                     fontsize=9.5)
    ref_ax.text(window[0] + 0.4, 0.32, "bitwise-shared prefix:\nevery branch of a game\n"
                "coincides here by construction", fontsize=6.8, color=MUTED, style="italic",
                ha="left", va="top")
    ref_ax.annotate("this slope is class-balance drift with absolute ply,\n"
                    "not decay — read it only as the panels' reference",
                    xy=(16, absolute[(absolute.space == REFERENCE_SPACE)
                                     & (absolute.r == 16)]["mean"].iloc[0]),
                    xytext=(window[0] + 1, 0.50), fontsize=6.8, color=MUTED, style="italic",
                    ha="left", arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
    ref_ax.legend(fontsize=6.6, loc="lower left", labelspacing=0.3, borderpad=0.3,
                  bbox_to_anchor=(0.0, 0.05))
    ref_ax.text(window[-1] - 0.5, 0.14, "both controls are redrawn in (b)-(e),\n"
                "in these same two styles", fontsize=6.4, color=MUTED, style="italic",
                ha="right", va="bottom", linespacing=1.5)

    for i, (ax, (family, members)) in enumerate(zip(family_axes, SPACE_FAMILIES.items())):
        event_axes(ax, window)
        span = wide if family == WIDE_SCALE_FAMILY else shared
        if family == WIDE_SCALE_FAMILY:
            ax.axhspan(*shared, color=GRID, alpha=0.9, lw=0, zorder=0)
            ax.text(window[0] + 0.5, shared[0], "y range of (b)-(d)", fontsize=6.4,
                    color=MUTED, ha="left", va="top")
        curve(ax, reference, REFERENCE_SPACE, "raw $-$ raw = 0 (pairing check)", 0)
        for j, space in enumerate(CONTROL_SPACES):
            curve(ax, delta[delta.space == space], space, None, 3 + 2 * j)
        for j, space in enumerate(members):
            curve(ax, delta[delta.space == space], space, styled(space)["label"], 2 * j)
        ax.set_ylim(*span)
        ax.set_title(f"({chr(98 + i)})  {family}", loc="left", fontsize=9.5)
        ax.legend(fontsize=6.6, labelspacing=0.28, borderpad=0.2, handlelength=2.4,
                  handletextpad=0.4,
                  loc="upper right")

    step = delta[(delta.space == "TopKTrainer-othello") & (delta.r == 0)]["mean"].iloc[0]
    family_axes[2].annotate("an extra 0.02–0.06 lost here and\nrecovered by $r$ = 1, in the six "
                            "ordinary\ndictionaries; in neither untrained control",
                            xy=(0.2, step), xytext=(3.0, 0.55 * shared[0]), fontsize=6.6,
                            color=INK, style="italic", ha="left", va="center",
                            arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8})
    for ax in family_axes[:2]:
        ax.set_xticklabels([])
    for ax in family_axes[::2]:
        ax.set_ylabel(r"$\Delta$ macro-F1 vs raw" "\n" r"(paired, same source game)")

    fig.supxlabel("position relative to the injected move, $r = s - t_0$", fontsize=9, y=0.035)
    fig.text(0.5, 0.925, "Pooled empty-square violations (the four categories whose square "
             "is empty before the move). (a) absolute macro-F1. (b)-(e) every dictionary as "
             "a per-game paired difference from raw at the same $r$, which cancels the "
             "class-balance drift exactly because both spaces are scored on the same "
             "positions against the same labels.\n"
             f"{BARS}, at every fifth offset; {n_text(absolute)} source games, constant "
             "across the window by construction ($t_0$ round-robin over [10, 34]). "
             f"Half-widths are $\\leq$ {((delta.hi - delta.lo) / 2).max():.3f}, mostly "
             "shorter than the marker. Gray span: $r < 0$, bitwise identical across "
             "branches. (e) has its own y scale.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top", linespacing=1.5)
    if not camera_ready:
        fig.suptitle("Every dictionary that tracks raw loses macro-F1 at the injected move; "
                     "neither untrained control does, and the SpaDE pair is at its floor",
                     fontsize=11, y=0.99)
    rule_key(fig, ["R1", "R2", "R3"], y=0.005)
    save_figure(fig, FIGURES / "f2_readout_window", camera_ready)


# --- F3: which rule each space names ---------------------------------------

RULE_OFFSET = dict(zip(RULES, (-0.27, 0.0, 0.27)))


def spread(values: np.ndarray, gap: float) -> np.ndarray:
    """Push overlapping labels apart along one axis, keeping their order."""
    order = np.argsort(-values)
    out = values[order].copy()
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1] - gap)
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] + gap)
    placed = np.empty_like(out)
    placed[order] = out
    return placed


def spread_ratio(means: pd.DataFrame, members: pd.DataFrame) -> float:
    """Within-rule / between-rule spread, averaged over spaces (FIGURE_GUIDELINES 1.7).

    Near 1 the three rules are a label rather than a grouping, and the panel says so.
    """
    ratios = []
    for space in SPACES:
        within = np.sqrt(np.mean([members.loc[space, RULE_MEMBERS[r]].to_numpy().std() ** 2
                                  for r in RULES]))
        ratios.append(within / means.loc[space].to_numpy().std())
    return float(np.mean(ratios))


def f3_per_rule(readout: pd.DataFrame, fit: pd.DataFrame, camera_ready: bool) -> None:
    """(a) all three rules on one row per space; (b) R3 against the space's own overall level.

    Read at r = -1, the shared pre-move board. At r = 0 the illegal move has been played,
    every space reads that square as occupied, and correct-rule recall for R3 is 0.000-0.007
    everywhere -- a property of the target, not of the spaces. Both panels are therefore
    read-out properties of the SPACE on a board every branch shares bitwise.

    Panel (a) puts the three rules on one row so the reader can see whether a space's R3
    loss exceeds what its R1 and R2 predict. Panel (b) is the same question against the
    space's overall level, which is what a uniform-degradation account would predict.
    """
    d = readout[(readout.metric == "p_pred_correct_rule") & (readout.role == ILLEGAL)
                & (readout.r == PREFIX_R)]
    means = d[d.category == ""].pivot(index="space", columns="rule", values="mean")[RULES]
    bounds = {b: d[d.category == ""].pivot(index="space", columns="rule", values=b)[RULES]
              for b in ("lo", "hi")}
    members = d[d.category != ""].pivot(index="space", columns="category", values="mean")
    assert set(means.index) == set(SPACES), "readout.csv has no rule-level rows (category empty)"

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), width_ratios=[1.55, 1.0])
    y = dot_rows(axes[0], SPACES)
    for row, space in zip(y, SPACES):
        for rule in RULES:
            at, style = row + RULE_OFFSET[rule], RULE_STYLE[rule]
            axes[0].plot(members.loc[space, RULE_MEMBERS[rule]], np.full(len(RULE_MEMBERS[rule]),
                         at + 0.10), marker=".", ms=3.5, lw=0, alpha=0.55,
                         color=style["color"], zorder=2)
            value = means.loc[space, rule]
            axes[0].errorbar(value, at, marker=style["marker"], ms=6, lw=0, elinewidth=1.0,
                             capsize=2, color=style["color"], ecolor=style["color"],
                             markeredgecolor="white", markeredgewidth=0.6, zorder=4,
                             xerr=[[value - bounds["lo"].loc[space, rule]],
                                   [bounds["hi"].loc[space, rule] - value]],
                             label=f"{rule}  (mean of {len(RULE_MEMBERS[rule])})"
                             if row == 0 else None)
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_xlabel("P(the probe names the violated rule at the square the branch "
                       "is about to play)")
    axes[0].set_title(f"(a)  all three rules per space, at $r$ = {PREFIX_R}", loc="left",
                      fontsize=9.5)
    axes[0].legend(fontsize=7, loc="upper left", labelspacing=0.3, handletextpad=0.4,
                   borderpad=0.3, title="small dots: the rule's member categories",
                   title_fontsize=6.8)
    axes[0].text(0.02, 6.2, "within-rule $\\div$ between-rule spread = "
                 f"{spread_ratio(means, members):.2f}\n(mean over the 11 spaces; near 1 the "
                 "three rules\nwould be a label rather than a grouping)", fontsize=6.8,
                 color=MUTED, style="italic", ha="left", va="center", linespacing=1.5)

    level = macro(fit, "legality")["value"]
    r3 = RULES[-1]
    label_at, column = spread(means[r3].to_numpy(), 0.031), 1.02
    for space, label_y in zip(means.index, label_at):
        x, value = level[space], means.loc[space, r3]
        point(axes[1], x, value, space, value, bounds["lo"].loc[space, r3],
              bounds["hi"].loc[space, r3], size=8)
        axes[1].plot([x, column - 0.012], [value, label_y], color=styled(space)["color"],
                     lw=0.5, zorder=1)
        axes[1].text(column, label_y, styled(space)["short"].replace("\n", " "), fontsize=6.4,
                     color=styled(space)["color"], ha="left", va="center")
    axes[1].set_xlim(0.36, 1.34)
    axes[1].set_ylim(-0.06, 1.0)
    axes[1].set_xticks(np.arange(0.4, 1.01, 0.1))
    axes[1].set_xlabel("the same space's held-out legality macro-F1 (F1a)")
    axes[1].set_ylabel(f"P(names {r3}) at $r$ = {PREFIX_R}", color=RULE_STYLE[r3]["color"])
    axes[1].set_title(f"(b)  {r3} against the space's overall level", loc="left",
                      fontsize=9.5, color=RULE_STYLE[r3]["color"])
    axes[1].text(0.38, 0.30, "The two SpaDE variants sit far below any monotone\nreading of "
                 "the other nine. Whether that excess survives a\nmultiplicative account of "
                 "the same lossiness — R3 needs\nseveral squares of mover-relative colour at "
                 "once — is not\nsettled by these data; the panel shows the differential\n"
                 "and asserts no mechanism.", fontsize=6.8, color=MUTED, style="italic",
                 ha="left", va="top", linespacing=1.5)

    fig.text(0.5, 0.945, f"Read at $r$ = {PREFIX_R}, the board every branch of a game shares "
             "bitwise, one ply before the illegal move: this is a property of the "
             "representation space, not an effect of the violation. The mover-relative frame "
             "alternates each ply, so R3 (turn) is near zero at even offsets by construction "
             f"and {PREFIX_R} is the phase in which the branch's move is about to be played.\n"
             f"{BARS} ({n_text(d[d.category == ''])}). Rows and markers are F1's and F2's; a "
             "one-member rule is labelled as such so it is not read as an average.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top", linespacing=1.5)
    if not camera_ready:
        fig.suptitle("Both SpaDE variants keep R1 and R2 at the sparsity-matched null's "
                     "level and lose R3 entirely", fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    rule_key(fig, ["R1", "R2", "R3"])
    save_figure(fig, FIGURES / "f3_per_rule", camera_ready)


# --- F4: which standard metric sees any of this? ---------------------------
# L0 spans 50x across these dictionaries, variance explained is trapped in 0.971-0.999, and
# the board probe is the one that shares the legality probe's units. Two rows because the
# two things a space can lose are answered by different x: the LEVEL it supports on the
# shared prefix, and the DROP at the violation.

METRIC_PANELS = [
    ("l0_window", "log", r"$L_0$ over the read-out window (log scale)"),
    ("variance_explained", "linear", "variance explained of the layer-5 residual"),
    ("board", "linear", "macro-F1 of the board probe in the same space"),
]


def rho(x: pd.Series, y: pd.Series) -> str:
    """Spearman over the trained dictionaries only: the untrained controls have no
    reconstruction objective, so they cannot be in a 'does this training metric predict'
    correlation. Descriptive at n = 8, and labelled as such."""
    d = pd.concat([x[DICTIONARIES], y[DICTIONARIES]], axis=1).dropna()
    return f"$\\rho$ = {spearmanr(d.iloc[:, 0], d.iloc[:, 1]).statistic:+.2f}  (n = {len(d)})"


def f4_metric_vs_rules(fit: pd.DataFrame, readout: pd.DataFrame, quality: pd.DataFrame,
                       camera_ready: bool) -> None:
    """Top row: the level each space supports on the shared prefix. Bottom row: how much of
    raw's read-out it stops reproducing at the violation.

    The two rows answer to different x, which is the figure. `drop_agreement_with_raw` on the
    bottom row is not scale-free -- a space that never agreed with raw has little to drop, and
    SpaDE starts at 0.66 -- so it is read against the top row, where that floor is the y value.
    """
    level = pooled_curves(readout, "macro_f1")
    level = level[level.r == PREFIX_R].set_index("space")
    drop = readout[(readout.role == ILLEGAL) & (readout.category == "illegal_pooled")
                   & (readout.metric == "drop_agreement_with_raw")].set_index("space")
    variance = quality.variance_explained.copy()
    variance[REFERENCE_SPACE] = 1.0  # raw is the reconstruction target, not a reconstruction
    x_of = {"l0_window": quality.l0_window, "variance_explained": variance,
            "board": macro(fit, "board")["value"]}

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6), sharey="row", sharex="col")
    for column, (metric, scale, xlabel) in enumerate(METRIC_PANELS):
        x = x_of[metric]
        for row, (ax, d) in enumerate(zip(axes[:, column], (level, drop))):
            for space in SPACES:
                if np.isnan(x.get(space, np.nan)):  # no reconstruction objective, no x
                    style = styled(space)
                    ax.axhspan(d.loc[space, "lo"], d.loc[space, "hi"], color=style["color"],
                               alpha=0.18, lw=0, zorder=0)
                    ax.axhline(d.loc[space, "mean"], color=style["color"], ls=style["ls"],
                               lw=style["lw"], zorder=1)
                else:
                    scatter_point(ax, x[space], d.loc[space], space)
            ax.set_title(f"({chr(97 + 3 * row + column)})", loc="left", fontsize=9)
            ax.text(0.97, 0.93 if row else 0.06, rho(x, d["mean"]), transform=ax.transAxes,
                    fontsize=7.5, color=INK, ha="right", va="top" if row else "bottom")
        axes[0][column].set_xscale(scale)
        axes[1][column].set_xlabel(xlabel)
    axes[0][0].set_ylim(0.3, 1.0)
    axes[1][0].set_ylim(-0.006, 0.095)
    axes[0][0].set_ylabel(f"legality macro-F1 at $r$ = {PREFIX_R}\n"
                          r"($\rightarrow$ the space supports the read-out)")
    axes[1][0].set_ylabel("drop in agreement with raw's read-out\n"
                          r"($\downarrow$ the violation disturbs it less)")
    axes[0][0].annotate("$L_0$ spans 50$\\times$ and sees nothing of\nthe top row; it is most "
                        "of what the\nbottom row tracks", xy=(0.03, 0.58),
                        xycoords="axes fraction", fontsize=7, color=INK, style="italic",
                        ha="left", va="top")
    axes[1][1].annotate("raw is 0 by construction: it agrees with itself\neverywhere. This "
                        "row is not scale-free — SpaDE\nagreed with raw on only 0.66 of "
                        "squares before\nthe move, so read it against the row above.",
                        xy=(0.03, 0.04), xycoords="axes fraction", fontsize=6.8, color=MUTED,
                        style="italic", ha="left", va="bottom")
    fig.legend(handles=[plt.Line2D([], [], color=styled(s)["color"], marker=styled(s)["marker"],
                                   ls=styled(s)["ls"], lw=styled(s)["lw"], ms=6,
                                   label=styled(s)["label"]) for s in SPACES],
               loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.10),
               labelspacing=0.4, columnspacing=1.4, handletextpad=0.5)

    fig.text(0.5, 0.955, "$L_0$ and variance explained are measured on this study's own "
             "read-out window (results/space_quality.csv); raw sits at $L_0$ = 512 and "
             "VE = 1 by construction, and the two untrained controls have no reconstruction "
             "objective, so on (b) and (e) each carries a horizontal level instead of an x "
             "position. The sparsity-matched null sits at exactly TopK's $L_0$ in (a) and (d).\n"
             f"{BARS} ({n_text(level)}), and the same interval shades each control level. "
             "$\\rho$ is Spearman over the eight trained "
             "dictionaries only and is descriptive at that n. The top row is a level, which "
             "this study cannot claim as a result; the panels are about which x tracks it.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top", linespacing=1.5)
    if not camera_ready:
        fig.suptitle("What a space supports is a board property; what the violation costs it "
                     "is a reconstruction-fidelity property", fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, FIGURES / "f4_metric_vs_rules", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results",
                    help="directory holding probe_fit.csv, readout.csv, space_quality.csv")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)

    fit, readout, quality = read(args.results)
    f1_probes_per_space(fit, args.camera_ready)
    f2_readout_window(readout, args.camera_ready)
    f3_per_rule(readout, fit, args.camera_ready)
    f4_metric_vs_rules(fit, readout, quality, args.camera_ready)
    print(f"4 figures in {FIGURES}")


if __name__ == "__main__":
    main()
