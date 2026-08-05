"""The four figures of the legality-probe study. Drawing only.

F1  board vs legality, per space   — the gap between the two probes is the quantity
F2  HEADLINE, what survives a move — macro-F1 against the engine over the window r
F3  which rule is lost             — correct-rule recall at the played square, r = 0
F4  which standard metric predicts rule preservation — L0, variance explained, board

Everything is read from `results/probe_fit.csv` and `results/readout.csv` (schemas
frozen in the study contract) plus the committed SAE acceptance table; nothing here
computes a statistic. Conventions: FIGURE_GUIDELINES.md 3 (reference/null styling),
4.1 (five series per panel) and 4.2 (event-locked panels).

Legality is a deterministic function of the board and the board decodes at ~0.98, so
no figure here is allowed to present a macro-F1 *level* as a result. F1 draws every
space against its own board probe and against raw's wrong-player control; F2 and F4
read differences (r < 0 against r = 0, space against space).

F2 facets by SAE family rather than drawing eleven lines on one axis, and plots
each dictionary as a per-game paired difference from raw: the 4-way class balance
is a function of absolute ply, so an absolute macro-F1 curve drifts against r for
reasons no dictionary controls. Panel (a) carries raw's absolute curve so the
drift is shown rather than hidden.

Two things the CSVs must carry that the contract's first pass did not name:
rule-level aggregates in `readout.csv` as rows with an empty `category`, and the
paired difference as `metric = macro_f1_minus_raw`. Both are asserted here rather
than recomputed.

Usage: python analysis/legality_probe/make_figures.py [--results DIR] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spade.common.config import D_MODEL
from spade.common.paths import repo_root
from spade.figures.paper_style import (
    GRID, INK, MUTED, NULL_SPACES, PROBE_CLASS_STYLE, REFERENCE,
    REFERENCE_SPACE, RULE_STYLE, SPACE_FAMILIES, SPACE_STYLE, SPACES, apply_style,
    rule_key, save_figure)
from spade.othello.taxonomy import RULE_MEMBERS

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
ACCEPTANCE = repo_root() / "results" / "acceptance" / "all_results.csv"

CLASSES = list(PROBE_CLASS_STYLE)
RULES = list(RULE_MEMBERS)
POOLED = "empty_pooled"
ILLEGAL = "illegal"
# Per-game paired difference from raw at the same offset. Not in the contract's first
# metric list; it is what removes the ply-driven class-balance drift from F2.
DELTA_METRIC = "macro_f1_minus_raw"
BAND = "band: 95% bootstrap CI over source games"


def styled(space: str) -> dict:
    return SPACE_STYLE[space]


def dot_rows(ax, spaces: list[str]) -> np.ndarray:
    """Spaces top-to-bottom on the y axis, raw first, nulls last."""
    y = np.arange(len(spaces))
    ax.set_yticks(y, [styled(s)["short"] for s in spaces], fontsize=8)
    ax.set_ylim(len(spaces) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    for row in y:  # one guide per row, so a marker hanging below a tick still reads
        ax.axhline(row, color=GRID, lw=0.5, zorder=0)
    ax.axhline(0.5, color=MUTED, lw=0.6, zorder=0)                # raw | dictionaries
    ax.axhline(len(spaces) - len(NULL_SPACES) - 0.5, color=MUTED, lw=0.6, zorder=0)
    return y


def dots(ax, d: pd.DataFrame, spaces: list[str], value: str, y: np.ndarray) -> None:
    """One point per space with its 95% CI, in each space's own style. No legend:
    the y tick labels already name the rows, so one would only repeat them."""
    at = d.set_index("space")
    for row, space in zip(y, spaces):
        r, style = at.loc[space], styled(space)
        ax.errorbar(r[value], row, xerr=[[r[value] - r.lo], [r.hi - r[value]]],
                    marker=style["marker"], ms=7, lw=0, elinewidth=1.1, capsize=2,
                    color=style["color"], ecolor=style["color"],
                    markeredgecolor="white", markeredgewidth=0.6, zorder=style["zorder"])


def n_text(d: pd.DataFrame) -> str:
    lo, hi = int(d.n.min()), int(d.n.max())
    return f"n = {lo}" if lo == hi else f"n = {lo}–{hi}"


# --- F1: board against legality, per space ----------------------------------

def f1_board_vs_legality(fit: pd.DataFrame, camera_ready: bool) -> None:
    """A dumbbell per space, board endpoint open, legality endpoint filled.

    Two adjacent bars would put the reader's eye on the two levels; the levels are
    exactly what this study cannot claim, because legality is a deterministic
    function of a board that already decodes at ~0.98. A dumbbell makes the
    segment — the loss from board to legality — the thing that is read.
    """
    macro = fit[(fit.metric == "macro_f1") & (fit["class"].isna())]
    at = macro.set_index(["space", "target"])
    per_class = fit[(fit.metric == "per_class_f1") & (fit.target == "legality")]
    per_class = per_class.set_index(["space", "class"])
    legality = macro[macro.target == "legality"]
    hyper = legality.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)
    y = dot_rows(axes[0], SPACES)
    dot_rows(axes[1], SPACES)

    for row, space in zip(y, SPACES):
        style = styled(space)
        board, rules = at.loc[(space, "board")], at.loc[(space, "legality")]
        axes[0].plot([board.value, rules.value], [row, row], color=style["color"],
                     lw=1.8, zorder=3, solid_capstyle="butt")
        axes[0].plot(board.value, row, marker=style["marker"], ms=6, lw=0,
                     markerfacecolor="white", markeredgecolor=style["color"],
                     markeredgewidth=1.3, zorder=4)
        axes[0].errorbar(rules.value, row,
                         xerr=[[rules.value - rules.lo], [rules.hi - rules.value]],
                         marker=style["marker"], ms=7.5, lw=0, elinewidth=1.1,
                         capsize=2, color=style["color"], ecolor=style["color"],
                         markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        for code in CLASSES:
            class_style = PROBE_CLASS_STYLE[code]
            axes[1].plot(per_class.loc[(space, code), "value"], row,
                         marker=class_style["marker"], ms=5, lw=0,
                         markerfacecolor="white", markeredgecolor=class_style["color"],
                         markeredgewidth=1.1, zorder=3,
                         label=class_style["label"] if row == 0 else None)

    wrong = at.loc[(REFERENCE_SPACE, "legality_wrong_player"), "value"]
    axes[0].axvline(wrong, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    axes[0].text(0.85, len(SPACES) / 2, "raw, legality scored for the WRONG player "
                 f"({wrong:.3f})\n— what board-reading alone already buys",
                 fontsize=7, color=MUTED, ha="right", va="center", linespacing=1.4)
    for ax, xlabel in zip(axes, [
            "macro-F1, held-out positions"
            r"   ($\rightarrow$ more of the target is linearly decodable)",
            "per-class F1 of the legality probe"]):
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(xlabel)
    axes[0].set_title("(a)  the segment is what legality costs on top of the board",
                      loc="left", fontsize=9.5)
    axes[1].set_title("(b)  and which class pays for it", loc="left", fontsize=9.5)
    axes[1].legend(fontsize=7, loc="upper left", labelspacing=0.35,
                   handletextpad=0.4, borderpad=0.3, title="legality probe class",
                   title_fontsize=7)

    board_classes = int(at.loc[(REFERENCE_SPACE, "board"), "n_classes"])
    fig.text(0.5, 0.945, "Open marker: the "
             f"{board_classes}-way board probe (empty / theirs / mine). Filled marker "
             f"with bars: the {int(hyper.n_classes)}-way legality probe, "
             f"$d_f$ = {int(hyper.filler_dim)}. Both fitted in the same space with the "
             "same machinery, OthelloGPT layer 5.\n"
             f"Bars are 95% bootstrap CI over source games ({n_text(legality)}). "
             "Read the segments and the spread across rows, never the level: a space "
             "that loses board and legality equally is merely lossy.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    if not camera_ready:
        fig.suptitle("What each representation space loses when the target goes from "
                     "the board to the rules", fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    rule_key(fig, ["R1", "R2", "R3"])
    save_figure(fig, FIGURES / "f1_board_vs_legality", camera_ready)


# --- F2: the headline ------------------------------------------------------

def curve(ax, d: pd.DataFrame, style: dict, label: str, offset: int) -> None:
    d = d.sort_values("r")
    ax.fill_between(d.r, d.lo, d.hi, color=style["color"], alpha=0.18, lw=0,
                    zorder=style["zorder"] - 1)
    ax.plot(d.r, d["mean"], color=style["color"], marker=style["marker"], ms=5,
            lw=style["lw"], ls=style["ls"], markevery=(offset, 5),
            markeredgecolor="white", markeredgewidth=0.5, zorder=style["zorder"],
            label=label)


def event_axes(ax, window: np.ndarray) -> None:
    ax.axvspan(window[0], 0, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, ls=(0, (4, 3)), lw=0.9, zorder=1)
    ax.set_xlim(window[0], window[-1])
    ax.set_xticks(range(int(window[0]), int(window[-1]) + 1, 8))


def f2_readout_window(readout: pd.DataFrame, camera_ready: bool) -> None:
    """Panel (a) is raw's absolute curve; the family panels are paired differences.

    The 4-way class balance is a function of absolute ply — R1 is exactly
    (ply + 5)/64 and R2 collapses late — and t0 ranges over [10, 34], so an
    absolute macro-F1 curve drifts against r for reasons no dictionary controls.
    The drift is common to every space, so a per-game paired difference against raw
    at the same r removes it exactly. Panel (a) shows the drift rather than hiding
    it; the family panels are where the study's claim is read.
    """
    pooled = (readout.role == ILLEGAL) & (readout.category == POOLED)
    absolute = readout[pooled & (readout.metric == "macro_f1")]
    delta = readout[pooled & (readout.metric == DELTA_METRIC)]
    assert set(delta.space) == set(SPACES) - {REFERENCE_SPACE}, \
        f"readout.csv is missing {DELTA_METRIC} rows for {set(SPACES) - set(delta.space)}"
    window = np.sort(absolute.r.unique())
    dictionaries = delta[~delta.space.isin(NULL_SPACES)]
    span = 1.15 * max(dictionaries["hi"].abs().max(), dictionaries["lo"].abs().max())

    fig = plt.figure(figsize=(11.4, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32, hspace=0.42,
                          left=0.07, right=0.99, top=0.78, bottom=0.13)
    ref_ax = fig.add_subplot(gs[:, 0])
    family_axes = [fig.add_subplot(gs[i // 2, 1 + i % 2]) for i in range(len(SPACE_FAMILIES))]

    event_axes(ref_ax, window)
    for j, space in enumerate([REFERENCE_SPACE] + NULL_SPACES):
        curve(ref_ax, absolute[absolute.space == space], styled(space),
              styled(space)["label"], offset=j)
    ref_ax.set_ylim(0.0, 1.0)
    ref_ax.set_ylabel("macro-F1 against the engine's legality")
    ref_ax.set_title("(a)  the reference space, absolute", loc="left", fontsize=9.5)
    ref_ax.text(window[0] + 0.4, 0.70, "bitwise-shared prefix:\nevery branch of a game\n"
                "coincides here by construction", fontsize=6.8, color=MUTED,
                style="italic", ha="left", va="top")
    ref_ax.annotate("this slope is class-balance drift with absolute ply,\n"
                    "not decay — read it only as the panels' reference",
                    xy=(16, absolute[(absolute.space == REFERENCE_SPACE)
                                     & (absolute.r == 16)]["mean"].iloc[0]),
                    xytext=(window[0] + 1, 0.46), fontsize=6.8, color=MUTED,
                    style="italic", ha="left",
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8})
    ref_ax.legend(fontsize=7, loc="lower right", labelspacing=0.3, borderpad=0.3,
                  bbox_to_anchor=(1.0, 0.06))

    for i, (ax, (family, members)) in enumerate(zip(family_axes, SPACE_FAMILIES.items())):
        event_axes(ax, window)
        ax.axhline(0, color=REFERENCE["color"], lw=REFERENCE["lw"], zorder=2)
        for j, space in enumerate(members):
            curve(ax, delta[delta.space == space], styled(space),
                  styled(space)["label"], offset=2 * j)
        ax.set_ylim(-span, span)
        ax.set_title(f"({chr(98 + i)})  {family}", loc="left", fontsize=9.5)
        ax.legend(fontsize=6.6, loc="upper left", labelspacing=0.28, borderpad=0.2,
                  handlelength=2.4, handletextpad=0.4)
    family_axes[0].text(window[-1] - 0.5, 0.004, "raw (reference)", fontsize=6.8,
                        color=INK, ha="right", va="bottom")
    for ax in family_axes[:2]:
        ax.set_xticklabels([])
    for ax in family_axes[::2]:
        ax.set_ylabel(r"$\Delta$ macro-F1 vs raw" "\n" r"(paired, same source game)")

    fig.supxlabel("position relative to the injected move, $r = s - t_0$", fontsize=9,
                  y=0.035)
    fig.text(0.5, 0.925, "(a) raw and the two random controls, absolute. (b)-(e) every "
             f"dictionary as a per-game paired difference from raw at the same $r$, "
             "which cancels the class-balance drift exactly because both spaces are "
             "scored on the same positions against the same labels.\n"
             f"{BAND}; {n_text(absolute)} source games, constant across the whole window "
             "by construction ($t_0$ round-robin over [10, 34]), so no offset is better "
             "sampled than another. Gray span: $r < 0$, bitwise identical across branches.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    if not camera_ready:
        fig.suptitle("What each space loses relative to the raw residual stream when an "
                     "illegal move is injected", fontsize=11, y=0.99)
    rule_key(fig, ["R1", "R2", "R3"], y=0.005)
    save_figure(fig, FIGURES / "f2_readout_window", camera_ready)


# --- F3: which rule is lost ------------------------------------------------

def f3_per_rule(readout: pd.DataFrame, camera_ready: bool) -> None:
    d = readout[(readout.metric == "p_pred_correct_rule") & (readout.role == ILLEGAL)
                & (readout.r == 0)]
    means = d[d.category.isna()]
    assert set(means.rule) == set(RULES), \
        f"readout.csv has no rule-level rows (category empty) for {set(RULES) - set(means.rule)}"
    fig, axes = plt.subplots(1, len(RULES), figsize=(10.2, 4.0), sharey=True)

    for i, (ax, rule) in enumerate(zip(axes, RULES)):
        y = dot_rows(ax, SPACES)
        members = RULE_MEMBERS[rule]
        by_category = d[d.category.isin(members)].set_index(["space", "category"])
        for row, space in zip(y, SPACES):
            for category in members:
                ax.plot(by_category.loc[(space, category), "mean"], row + 0.26,
                        marker=".", ms=4, lw=0, alpha=0.6,
                        color=styled(space)["color"], zorder=2)
        dots(ax, means[means.rule == rule], SPACES, "mean", y)
        ax.set_title(f"({chr(97 + i)})  {rule}  (mean of {len(members)})", loc="left",
                     color=RULE_STYLE[rule]["color"], fontsize=9.5)
        ax.set_xlim(0.0, 1.0)

    fig.supxlabel("P(the probe names the violated rule at the played square)", fontsize=9)
    axes[0].text(0.97, len(SPACES) - 1.6, "small dots: the rule's\nmember categories",
                 fontsize=7, color=MUTED, style="italic", ha="right", va="center")
    fig.text(0.5, 0.94, "Read at the illegally played square, $r$ = 0. Filled marker is "
             "the mean over the rule's member categories, small dots are those members; "
             f"a one-member rule is labelled as such so it is not read as an average.\n"
             f"{BAND} ({n_text(means)}). Rows are the eleven spaces, in the same order "
             "as F1; each space keeps the marker and colour it carries in F1 and F2.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    if not camera_ready:
        fig.suptitle("Which of the three constraint rules each space still names, at the "
                     "square the violation was played on", fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    rule_key(fig, ["R1", "R2", "R3"])
    save_figure(fig, FIGURES / "f3_per_rule", camera_ready)


# --- F4: which standard metric predicts rule preservation? ------------------
# Three x axes on one y. L0 spans 50x across these dictionaries, variance explained
# is trapped in 0.971-0.999 and cannot separate them at all, and the board probe is
# the one that shares the legality probe's units. Reading left to right is the
# argument: the standard reconstruction metrics do not see rule preservation.

SPARSITY_PANELS = [
    ("l0", "log", r"$L_0$ (active latents per position, log scale)"),
    ("variance_explained", "linear", "variance explained of the layer-5 residual"),
    ("board", "linear", "macro-F1 of the board probe in the same space"),
]


def sparsity_table() -> pd.DataFrame:
    """The committed seed-42 acceptance metrics, keyed by space name."""
    d = pd.read_csv(ACCEPTANCE)
    d["space"] = d.wandb_name.str.split("_trainer_").str[0]
    return d.set_index("space")


def f4_metric_vs_rules(fit: pd.DataFrame, readout: pd.DataFrame, camera_ready: bool) -> None:
    survives = (readout.metric == "macro_f1") & (readout.role == ILLEGAL) \
        & (readout.category == POOLED)
    d = readout[survives & (readout.r == 0)].set_index("space")
    prefix = readout[survives & (readout.r == -1)].set_index("space")
    board = fit[(fit.metric == "macro_f1") & (fit["class"].isna())
                & (fit.target == "board")].set_index("space")["value"]
    sparsity = sparsity_table()
    dictionaries = [s for f in SPACE_FAMILIES.values() for s in f]
    raw_x = {"l0": D_MODEL, "variance_explained": 1.0}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3), sharey=True)

    for i, (ax, (column, scale, xlabel)) in enumerate(zip(axes, SPARSITY_PANELS)):
        # The random controls have no L0 and no variance explained, so they can only
        # be a level; keeping them one on all three panels also keeps (c)'s x range
        # on the dictionaries instead of stretching it down to a control at 0.35.
        for space in NULL_SPACES:
            r, style = d.loc[space], styled(space)
            ax.axhspan(r.lo, r.hi, color=style["color"], alpha=0.20, lw=0, zorder=0)
            ax.axhline(r["mean"], color=style["color"], ls=style["ls"], lw=style["lw"],
                       zorder=1, label=f"{style['label']}, no $L_0$/VE (n={int(r.n)})")
        ax.axhline(d.loc[REFERENCE_SPACE, "mean"], color=REFERENCE["color"],
                   ls=(0, (1, 2.2)), lw=0.9, zorder=1)
        x_of = {s: board[s] if column == "board"
                else raw_x[column] if s == REFERENCE_SPACE else sparsity.loc[s, column]
                for s in [REFERENCE_SPACE] + dictionaries}
        # Point labels stagger by rank in x: TopK and BatchTopK differ by 0.2 in L0.
        for j, space in enumerate(sorted(x_of, key=x_of.get)):
            style, x, r = styled(space), x_of[space], d.loc[space]
            ax.plot([x, x], [prefix.loc[space, "mean"], r["mean"]], color=style["color"],
                    lw=0.8, ls=(0, (2, 1.5)), zorder=2)
            ax.plot(x, prefix.loc[space, "mean"], marker=style["marker"], ms=5, lw=0,
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.0, zorder=3)
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]],
                        marker=style["marker"], ms=8, lw=0, elinewidth=1.0, capsize=2,
                        color=style["color"], ecolor=style["color"],
                        markeredgecolor="white", markeredgewidth=0.7, zorder=4)
            # Four stagger levels: TopK and BatchTopK differ by 0.2 in L0, and seven
            # of the nine spaces sit inside 0.04 of board macro-F1.
            ax.annotate(style["short"], (x, r["mean"]), textcoords="offset points",
                        xytext=(0, -11 - 10 * (j % 4)), ha="center", va="top", fontsize=6.6,
                        color=style["color"], zorder=5,
                        arrowprops={"arrowstyle": "-", "color": style["color"],
                                    "lw": 0.5, "shrinkA": 1, "shrinkB": 3})
        ax.set_xscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"({chr(97 + i)})", loc="left", fontsize=9)

    axes[0].set_ylabel("macro-F1 at $r$ = 0, pooled empty-square violations\n"
                       r"($\rightarrow$ more rule state survives the violation)")
    axes[0].annotate("open marker + dashes: the same space at $r$ = $-$1, on the\n"
                     "bitwise-shared prefix. The drop is what the violation costs.\n"
                     "Dotted line: raw at $r$ = 0. Markers and colours are F1's and F2's.",
                     xy=(0.03, 0.03), xycoords="axes fraction", fontsize=6.8,
                     color=MUTED, style="italic", ha="left", va="bottom")
    axes[1].legend(fontsize=6.8, loc="lower left", ncol=1, labelspacing=0.35,
                   handletextpad=0.6, borderpad=0.3)
    fig.text(0.5, 0.945, "$L_0$ and variance explained are the committed seed-42 "
             "acceptance metrics (results/acceptance/all_results.csv); raw is plotted at "
             f"$L_0$ = {D_MODEL} and VE = 1 by construction. The board probe in (c) is "
             "F1's open marker, fitted in the same space.\n"
             f"{BAND} ({n_text(d)}); nine labelled points need no legend. The two random "
             "controls have no reconstruction objective, so on (a) and (b) each carries "
             "a horizontal level rather than an x position.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    if not camera_ready:
        fig.suptitle("Neither sparsity nor variance explained sees rule preservation; "
                     "the board probe does", fontsize=11, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, FIGURES / "f4_metric_vs_rules", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results",
                    help="directory holding probe_fit.csv and readout.csv")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)

    fit = pd.read_csv(args.results / "probe_fit.csv")
    readout = pd.read_csv(args.results / "readout.csv")
    f1_board_vs_legality(fit, args.camera_ready)
    f2_readout_window(readout, args.camera_ready)
    f3_per_rule(readout, args.camera_ready)
    f4_metric_vs_rules(fit, readout, args.camera_ready)
    print(f"4 figures in {FIGURES}")


if __name__ == "__main__":
    main()
