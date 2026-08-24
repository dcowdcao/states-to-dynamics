"""The two headline figures for the bracket-flip study.

Fig 1  where each square's read-out moves when a flanking move is played, in the
       probe's native mover-relative basis.
Fig 2  that the displacement carries a flip-specific direction, and that editing
       along that direction changes what the model plays.

Drawing only: everything is read from the committed CSVs in results/. The SAE
arm is drawn by sae_pca_figure.py. Conventions: FIGURE_GUIDELINES.md 4.4 and 4.7.

Usage: python analysis/bracket_flip/make_figures.py [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from spade.figures.figure_othello_rules import mover_note
from spade.figures.othello_board import draw_board, mark_square, square_xy
from spade.figures.paper_style import (
    INK, MUTED, NULL, PROBE_STYLE, SQUARE_ROLE_STYLE, apply_style, save_figure)
from spade.figures.straightness_panels import kde_ci

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

ROLES = ["flipped", "unflipped_own", "unflipped_opp", "placed", "empty"]
# m2_field.csv still carries both label bases; only the mover-relative one is drawn.
BASIS = "relative"
# The sign gloss each axis used to carry -- >0 is owned by the player to move, >0 is
# occupied -- is in the README's figure notes, next to the probe that produces the logits.
X_AXIS = r"$\ell_{\rm mine}-\ell_{\rm theirs}$ (logits)"
Y_AXIS = r"$\frac{1}{2}(\ell_{\rm mine}+\ell_{\rm theirs})-\ell_{\rm empty}$"


def letters_below(fig, axes) -> None:
    """(a) (b) under the panels, on one baseline measured off what they actually occupy.

    A fixed axes-fraction offset lands in different places here: the board carries a one-line
    x label and the field panel a two-line one. Sharing the lower edge of the two tight boxes
    keeps the letters level.
    """
    fig.canvas.draw()
    boxes = [ax.get_tightbbox(fig.canvas.get_renderer())
               .transformed(fig.transFigure.inverted()) for ax in axes]
    base = min(b.y0 for b in boxes) - 0.03
    for i, b in enumerate(boxes):
        fig.text((b.x0 + b.x1) / 2, base, f"({chr(97 + i)})", ha="center", va="top",
                 color=INK, fontsize=10)


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def field_panel(ax, field: pd.DataFrame, annotate: bool = False) -> None:
    """One arrow per square role, tail = before, head = after the flanking move.

    The bars are on the displacement, not on the head: an arrow asserts a change,
    the change is paired within event, and the two endpoints are anticorrelated
    across events, so an endpoint interval is not one for the arrow. The flipped
    square carries a second, dashed arrow to where the matched non-flipping move
    leaves it, and the gap between those two heads is the panel's one contrast.

    The probe's held-out accuracy and its occupancy-only ceiling live in the
    study README, not the panel title: they are findings, not a description of
    what is drawn.
    """
    d = field[field.basis == BASIS].set_index("role")
    for role in ROLES:
        r, style = d.loc[role], SQUARE_ROLE_STYLE[role]
        ax.annotate("", xy=(r.after_x, r.after_y), xytext=(r.before_x, r.before_y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=style["color"],
                                    shrinkA=0, shrinkB=4, alpha=0.9))
        ax.errorbar(r.after_x, r.after_y,
                    xerr=[[r.d_after_x - r.d_after_x_lo], [r.d_after_x_hi - r.d_after_x]],
                    yerr=[[r.d_after_y - r.d_after_y_lo], [r.d_after_y_hi - r.d_after_y]],
                    marker=style["marker"], ms=8, color=style["color"], lw=0,
                    elinewidth=0.9, ecolor=INK, markeredgecolor="white",
                    markeredgewidth=1.0, zorder=5, label=style["label"])
        ax.plot(r.before_x, r.before_y, marker=style["marker"], ms=6, lw=0,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.2, zorder=4)
        if role != "flipped":
            continue
        ax.annotate("", xy=(r.control_x, r.control_y), xytext=(r.before_x, r.before_y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, ls=(0, (4, 2)),
                                    color=style["color"], shrinkA=0, shrinkB=4))
        ax.errorbar(r.control_x, r.control_y,
                    xerr=[[r.d_control_x - r.d_control_x_lo],
                          [r.d_control_x_hi - r.d_control_x]],
                    yerr=[[r.d_control_y - r.d_control_y_lo],
                          [r.d_control_y_hi - r.d_control_y]],
                    marker="X", ms=9, color=style["color"], lw=0, elinewidth=0.9,
                    ecolor=INK, markeredgecolor="white", markeredgewidth=1.0,
                    zorder=6, label="flipped square, after $m'$")
        if annotate:  # the fill convention is the panel's whole grammar
            for x, y, text, offset, ha, va in [
                    (r.before_x, r.before_y, "before", (-4, -10), "right", "top"),
                    (r.after_x, r.after_y, "after", (5, 8), "left", "bottom"),
                    (r.control_x, r.control_y, "after $m'$\n(no flip)", (-7, -6),
                     "right", "top")]:
                ax.annotate(text, (x, y), textcoords="offset points", xytext=offset,
                            ha=ha, va=va, fontsize=7.5, color=style["color"], zorder=7)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.margins(y=0.12)
    ax.set_xlabel(X_AXIS)
    ax.set_ylabel(Y_AXIS)


# The two roles (b) draws blue arrows for. They are the only ones tinted: see board_panel.
TINTED_ROLES = ("unflipped_own", "unflipped_opp")


def board_panel(ax, event: pd.DataFrame, meta: pd.Series) -> None:
    """The board just before the flanking move, drawn the way every other board here is.

    Same renderer and same marking grammar as `figures/rules_and_generation`: cell outline
    plus a corner marker, never a fill, so the disc underneath stays visible -- on `s` it is
    the evidence, since `s` is the square whose disc the move flips. `m'` is the matched
    non-flipping move the control branch plays instead, outlined dashed to match the dashed
    arrow it produces in the next panel.

    The two unflipped roles keep their tint, because they are the panel's one ambiguous
    reading: which discs the blue arrows in (b) average over is not recoverable from the
    board otherwise, and unlike `flipped` and `placed` there is no single square to outline.
    `empty` and the two singleton roles are not tinted -- the empties would wash the board and
    the singletons are already outlined and named.
    """
    s, m = int(meta.square_s), int(meta.square_m)
    tinted = event[event.role.isin(TINTED_ROLES)]
    for _, r in tinted.iterrows():
        col, row = square_xy(int(r.square))
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, lw=0, zorder=0,
                                   color=SQUARE_ROLE_STYLE[r.role]["color"], alpha=0.16))
    draw_board(ax, event.sort_values("square").disc_before.to_numpy())
    ax.set_ylabel("rank")
    for square, label, role, ls in [(s, "$s$", "flipped", "-"),
                                    (m, "$m$", "placed", "-"),
                                    (int(meta.square_m_prime), "$m'$", "flipped",
                                     (0, (3, 2)))]:
        # A name above the cell falls off the board on rank 1, where m' sits in this event,
        # and matplotlib clips it away without saying so.
        mark_square(ax, square, SQUARE_ROLE_STYLE[role], label=label, ls=ls,
                    below=square_xy(square)[1] == 0)
    mover_note(ax, int(meta.mover))


def figure_one(camera_ready: bool) -> None:
    field = pd.read_csv(RESULTS / "m2_field.csv")
    event = pd.read_csv(RESULTS / "f2_event.csv")
    meta = pd.read_csv(RESULTS / "f2_event_meta.csv").iloc[0]
    # Board is square, so the left column's height sets the figure's: any extra
    # becomes dead space beside it.
    fig = plt.figure(figsize=(9.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.0], wspace=0.32,
                          bottom=0.24, top=0.87)
    board = fig.add_subplot(gs[0, 0])
    native = fig.add_subplot(gs[0, 1])

    board_panel(board, event, meta)
    field_panel(native, field, annotate=True)
    fig.legend(*native.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.50, 0.015), ncol=3, fontsize=7, alignment="left",
               labelspacing=0.3, columnspacing=1.0)
    letters_below(fig, (board, native))
    save_figure(fig, FIGURES / "fig1_flip_and_basis", camera_ready)


def share_inset(ax, shares: pd.DataFrame, n_events: int) -> None:
    """The magnitude the cosine density cannot show: how much of the squared
    displacement lies along the flip direction, against the same null."""
    d = shares[(shares.contrast == "after-control")
               & (shares.direction_source == "tpr")].set_index("component")
    assert (d.n_events == n_events).all(), "shares and cosines cover different events"
    pct = 100 * d[["share", "ci_lo", "ci_hi"]]
    ins = ax.inset_axes([0.64, 0.62, 0.34, 0.26])
    drawn = [("flip_null", NULL), ("flip", PROBE_STYLE["tpr"])]
    for y, (component, style) in enumerate(drawn):
        r = pct.loc[component]
        ins.barh(y, r.share, height=0.6, color=style["color"], edgecolor=INK,
                 lw=0.5, hatch="///" if style is NULL else None,
                 xerr=[[r.share - r.ci_lo], [r.ci_hi - r.share]],
                 error_kw=dict(elinewidth=0.9, ecolor=INK, capsize=2))
        ins.text(r.ci_hi + 0.1, y, f"{r.share:.2f}%", va="center", fontsize=6.5,
                 color=INK)
    ins.set_yticks([1, 0], ["flip at $s$", "null"], fontsize=6.5)
    ins.set_xlim(0, 1.75 * pct.loc[[c for c, _ in drawn], "ci_hi"].max())
    ins.xaxis.set_major_locator(MaxNLocator(3))
    ins.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ins.tick_params(labelsize=6.5, length=2, pad=1.5)
    ins.grid(axis="y", visible=False)
    ins.set_title(r"share of $\|\Delta'\|^2$ along it", fontsize=6.5, pad=3)


def specificity_panel(ax, cos: pd.DataFrame, shares: pd.DataFrame) -> None:
    """Alignment of the flip-attributable displacement with the predicted flip
    direction at s, against one unflipped disc per event. Strict controls only.

    The null is a single disc, not a per-event median over the ~29 unflipped
    discs: a median is the tighter estimator, and against it the separation is
    partly arithmetic rather than evidence.
    """
    d = cos[cos.control_kind == "strict"]
    series = [("cos_flip_after_control_tpr", PROBE_STYLE["tpr"],
               "flipped sq., TPR direction"),
              ("cos_flip_after_control_linear", PROBE_STYLE["linear"],
               "flipped sq., linear-probe direction"),
              ("cos_null1_after_control_tpr", NULL,
               "one unflipped disc per event (null)")]
    vals = {col: d[col].to_numpy() for col, _, _ in series}
    pooled = np.concatenate(list(vals.values()))
    pad = 0.03 * np.ptp(pooled)
    grid = np.linspace(pooled.min() - pad, pooled.max() + pad, 512)
    peak = 0.0
    for i, (col, style, label) in enumerate(series):
        dens, lo, hi = kde_ci(vals[col], grid)
        is_null = style is NULL
        body = np.flatnonzero(dens > 0.02 * dens.max())  # no markers on flat tails
        at = body[16 * i::96]
        peak = max(peak, hi.max())
        # The bootstrap rides on the markers; three fills over one peak hid it.
        ax.errorbar(grid[at], dens[at], yerr=[dens[at] - lo[at], hi[at] - dens[at]],
                    lw=0, elinewidth=0.9, capsize=1.8, ecolor=style["color"],
                    zorder=1 if is_null else 3)
        ax.plot(grid, dens, color=style["color"], lw=1.6, ls=style.get("ls", "-"),
                marker=style["marker"], ms=4, markevery=at.tolist(),
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=2 if is_null else 4,
                label=f"{label}: {np.median(vals[col]):+.3f}")
    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.set_ylim(0, 1.8 * peak)  # headroom for the legend and the share inset
    ax.set_xlabel(r"cos($\Delta'$, predicted flip direction at $s$)"
                  "\n" r"$\Delta'$ = act(P+$m$) $-$ act(P+$m'$), flip-attributable")
    ax.set_ylabel(f"density over events (n = {len(d):,})")
    ax.set_title("alignment with the flip direction")
    ax.legend(loc="upper left", fontsize=6.5, labelspacing=0.3)
    share_inset(ax, shares, len(d))


def causal_panel(ax, summary: pd.DataFrame) -> None:
    """Effect against collateral over both edit sweeps, with the damage control
    the effect has to survive drawn behind it.

    Up and to the left is a clean intervention. The solid and dotted curves are
    the two sides of one switch — mass onto squares only the counterfactual
    board allows, mass off squares only the real board allows — so the dotted
    fall at high magnitude is the effect deepening, not damage. Damage is the
    band: log-mass on the whole legal set, which a broken edit moves and this one
    does not. `collateral_{lin,tpr}` are swapped in so neither family is graded
    by the decoder it was built from, and x is a measured quantity, which is what
    puts the additive TPR knob and the probe's project-to-mean knob on one axis.
    """
    series = [(("tpr", "flipped"), PROBE_STYLE["tpr"], "TPR edit", "lin"),
              (("linear", "flipped"), PROBE_STYLE["linear"], "linear-probe edit", "tpr"),
              (("tpr", "random"), NULL,
               "random direction, magnitude-matched (null)", "lin")]
    edits = summary[summary.population.eq("flipped")
                    & summary.method.isin(["tpr", "linear"])]
    # Both sets are fixed by the event, not by the edit, so one n covers a sweep.
    assert edits.d_lp_cf_n.nunique() == 1 and edits.d_lp_real_n.nunique() == 1
    handles = []
    for (method, pop), style, label, decoder in series:
        d = summary[(summary.method == method)
                    & (summary.population == pop)].sort_values("scale")
        x = d[f"collateral_{decoder}_mean"]
        xerr = [x - d[f"collateral_{decoder}_lo"], d[f"collateral_{decoder}_hi"] - x]
        is_null = style is NULL
        handles.append(ax.errorbar(
            x, d.d_lp_cf_mean,
            yerr=[d.d_lp_cf_mean - d.d_lp_cf_lo, d.d_lp_cf_hi - d.d_lp_cf_mean],
            xerr=xerr, marker=style["marker"], ms=5, lw=1.4,
            ls=style.get("ls", "-"), color=style["color"], elinewidth=0.8,
            ecolor=style["color"], markeredgecolor="white", markeredgewidth=0.6,
            zorder=2 if is_null else 4, label=label))
        if is_null:  # the null has no switch to read the other side of
            continue
        ax.errorbar(x, d.d_lp_real_mean,
                    yerr=[d.d_lp_real_mean - d.d_lp_real_lo,
                          d.d_lp_real_hi - d.d_lp_real_mean], xerr=xerr,
                    marker=style["marker"], ms=4.5, lw=1.2, ls=(0, (1, 1.6)),
                    color=style["color"], elinewidth=0.8, ecolor=style["color"],
                    markerfacecolor="white", markeredgewidth=0.9, zorder=3)
        for rel, xi, yi in zip(d.rel_edit_norm_mean, x, d.d_lp_cf_mean):
            ax.annotate(f"{rel:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(-5, 4), ha="right", va="bottom", fontsize=6,
                        color=MUTED, zorder=5)
    bound = edits[["d_lp_legal_lo", "d_lp_legal_hi"]].abs().to_numpy().max()
    band = ax.axhspan(-bound, bound, color=MUTED, alpha=0.35, lw=0, zorder=0,
                      label="damage control: legal set "
                            f"(n = {int(edits.d_lp_legal_n.iloc[0]):,}), "
                            rf"$|\Delta| \leq {bound:.2f}$")
    handles.insert(-1, band)  # the null keeps the last slot
    ax.set_xscale("log")
    ax.set_xlabel("collateral: other squares (of 63) whose decoded class changed"
                  "\n" r"$\leftarrow$ cleaner intervention")
    ax.set_ylabel(r"$\Delta$ next-move log-mass (nats)"
                  "\n" r"$\uparrow$ onto the un-flipped board's squares,"
                  "\n" r"$\downarrow$ off the real board's")
    ax.set_title("edit effect against collateral")
    ax.set_ylim(top=1.5 * edits.d_lp_cf_hi.max())  # room for the legend above the curves
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, labelspacing=0.3,
              title_fontsize=6.5,
              title="solid = counterfactual-only squares, dotted = real-only")


def figure_two(camera_ready: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    specificity_panel(axes[0], pd.read_csv(RESULTS / "flip_cos_per_event.csv"),
                      pd.read_csv(RESULTS / "variance_shares.csv"))
    causal_panel(axes[1], pd.read_csv(RESULTS / "intervention_summary.csv"))
    for i, ax in enumerate(axes):
        panel_letter(ax, i)
    if not camera_ready:
        fig.suptitle("Where the flip-attributable displacement points, and what "
                     "editing along it does", y=1.0)
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig2_direction_and_cause", camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    figure_one(args.camera_ready)
    figure_two(args.camera_ready)


if __name__ == "__main__":
    main()
