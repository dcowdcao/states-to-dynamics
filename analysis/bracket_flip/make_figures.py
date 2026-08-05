"""The two headline figures for the bracket-flip study.

Fig 1  where each square's read-out moves when a flanking move is played, in the
       probe's native mover-relative basis and relabeled to absolute colour.
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

from spade.figures.othello_board import draw_board, square_xy
from spade.figures.paper_style import (
    INK, MUTED, NULL, PROBE_STYLE, SQUARE_ROLE_STYLE, apply_style, role_key,
    save_figure)
from spade.figures.straightness_panels import kde_ci

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

ROLES = ["flipped", "unflipped_own", "unflipped_opp", "placed", "empty"]
# Two probes of the same architecture and layer, one per label convention, so the
# panels are two measurements rather than one relabeled twice.
BASIS_TITLE = {"relative": "TPR trained on mine / theirs / empty",
               "absolute": "a second TPR, trained on black / white"}
AXIS = {
    "relative": (r"$\ell_{\rm mine}-\ell_{\rm theirs}$ (logits)"
                 "\n" r"$>0$: owned by the player to move",
                 r"$\frac{1}{2}(\ell_{\rm mine}+\ell_{\rm theirs})-\ell_{\rm empty}$"
                 "\n" r"$>0$: occupied"),
    "absolute": (r"$(\ell_{\rm black}-\ell_{\rm white})\times$ mover of $m$ (logits)"
                 "\n" r"$>0$: the colour the flanking player plays",
                 r"$\frac{1}{2}(\ell_{\rm black}+\ell_{\rm white})-\ell_{\rm empty}$"
                 "\n" r"$>0$: occupied"),
}


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def field_panel(ax, field: pd.DataFrame, basis: str, accuracy: float,
                note_xy: tuple[float, float], annotate: bool = False) -> None:
    """One arrow per square role, tail = before, head = after the flanking move.

    The bars are on the displacement, not on the head: an arrow asserts a change,
    the change is paired within event, and the two endpoints are anticorrelated
    across events, so an endpoint interval is not one for the arrow. The flipped
    square carries a second, dashed arrow to where the matched non-flipping move
    leaves it, and the gap between those two heads is the panel's one contrast.
    """
    d = field[field.basis == basis].set_index("role")
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
        lo, hi = sorted(abs(v) for v in (r.gap_x_lo, r.gap_x_hi))
        ax.text(*note_xy, f"$s$ after $m$ vs after $m'$:\n{abs(r.gap_x):.2f} logits "
                          f"apart in $x$\n[95% CI {lo:.2f}, {hi:.2f}]",
                transform=ax.transAxes, ha="right", fontsize=7.5,
                va="top" if note_xy[1] > 0.5 else "bottom",
                color=style["color"], zorder=7)
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
    ax.set_xlabel(AXIS[basis][0])
    ax.set_ylabel(AXIS[basis][1])
    ax.set_title(f"{BASIS_TITLE[basis]}\nheld-out board accuracy {accuracy:.3f}",
                 fontsize=9)


def board_panel(ax, event: pd.DataFrame, meta: pd.Series) -> None:
    """The board just before the flanking move, shaded by the same role colours
    the arrow panels use, so a reader can see which squares each arrow covers.

    m' is the matched non-flipping move the control branch plays instead; it is
    outlined dashed, matching the dashed arrow it produces in the next panel.
    """
    s, m = int(meta.square_s), int(meta.square_m)
    for _, r in event.iterrows():
        col, row = square_xy(int(r.square))
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, lw=0,
                                   color=SQUARE_ROLE_STYLE[r.role]["color"],
                                   alpha=0.18, zorder=0))
    draw_board(ax, event.sort_values("square").disc_before.to_numpy())
    for square, label, role, ls in [(s, "$s$", "flipped", "-"),
                                    (m, "$m$", "placed", "-"),
                                    (int(meta.square_m_prime), "$m'$", "flipped",
                                     (0, (3, 2)))]:
        col, row = square_xy(square)
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                   lw=2.2, zorder=4, ls=ls,
                                   edgecolor=SQUARE_ROLE_STYLE[role]["color"]))
        ax.annotate(label, (col, row - 0.52), ha="center", va="bottom", fontsize=9,
                    color=SQUARE_ROLE_STYLE[role]["color"], zorder=5,
                    fontweight="bold")
    ax.set_title(f"one event ($t$={int(meta.t)}): the board before $m$")
    ax.set_xlabel("discs in absolute colour; shading = role\n"
                  "$m$ flanks $s$, so $s$ turns the mover's colour;\n"
                  "$m'$ (dashed) is the matched non-flipping move", fontsize=7.5)


def figure_one(camera_ready: bool) -> None:
    field = pd.read_csv(RESULTS / "m2_field.csv")
    event = pd.read_csv(RESULTS / "f2_event.csv")
    meta = pd.read_csv(RESULTS / "f2_event_meta.csv").iloc[0]
    layer = int(field.layer.iloc[0])
    accuracy = pd.read_csv(RESULTS / "m1_basis.csv").set_index("layer").loc[layer]
    # Board is square, so the left column's height sets the figure's: any extra
    # becomes dead space beside it.
    fig = plt.figure(figsize=(10.4, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.80, 1.0], wspace=0.26, hspace=0.62)
    board = fig.add_subplot(gs[:, 0])
    # The two bases share both axes, or the absolute panel's 0.2-logit spread
    # blows up to the width of the panel and reads as separation.
    native = fig.add_subplot(gs[0, 1])
    absolute = fig.add_subplot(gs[1, 1], sharex=native, sharey=native)

    board_panel(board, event, meta)
    # The note goes in whichever corner the basis leaves empty.
    field_panel(native, field, "relative", accuracy.acc_tpr_relative, (0.98, 0.04),
                annotate=True)
    field_panel(absolute, field, "absolute", accuracy.acc_tpr_absolute, (0.98, 0.96))
    for i, ax in enumerate((board, native, absolute)):
        panel_letter(ax, i)
    fig.legend(*native.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.44, 0.015), ncol=2, fontsize=7, alignment="left",
               title=f"layer-{layer} read-out; arrow = the same squares before "
                     r"$\rightarrow$ after $m$, open tail = before, filled head = after"
                     "\nbars = 95% CI of the paired displacement (cluster bootstrap over "
                     f"{int(field.n_events.iloc[0]):,} strict-control events)",
               title_fontsize=7, labelspacing=0.3, columnspacing=1.0)
    if not camera_ready:
        fig.suptitle(f"Layer {layer} codes the board mover-relatively: a TPR trained "
                     "on absolute colour separates occupied from empty and no more",
                     y=0.99)
    role_key(fig, ROLES)
    save_figure(fig, FIGURES / "fig1_flip_and_basis", camera_ready)


def specificity_panel(ax, cos: pd.DataFrame) -> None:
    """Alignment of the flip-attributable displacement with the predicted flip
    direction at s, against the unflipped-disc null. Strict controls only."""
    d = cos[cos.control_kind == "strict"]
    series = [("cos_flip_after_control_tpr", PROBE_STYLE["tpr"],
               "flipped square, TPR direction"),
              ("cos_flip_after_control_linear", PROBE_STYLE["linear"],
               "flipped square, linear-probe direction"),
              ("cos_null_after_control_tpr", NULL, "unflipped discs (null)")]
    grid = np.linspace(-0.25, 0.45, 512)
    for col, style, label in series:
        vals = d[col].to_numpy()
        dens, lo, hi = kde_ci(vals, grid)
        is_null = style is NULL
        ax.fill_between(grid, lo, hi, color=style["color"], alpha=0.22, lw=0,
                        zorder=1 if is_null else 3)
        ax.plot(grid, dens, color=style["color"], lw=1.6, ls=style.get("ls", "-"),
                marker=style["marker"], ms=4, markevery=64,
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=2 if is_null else 4,
                label=f"{label}: median {np.median(vals):+.3f}")
    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.set_xlabel(r"cos($\Delta'$, predicted flip direction at $s$)"
                  "\n" r"$\Delta'$ = act(P+$m$) $-$ act(P+$m'$), flip-attributable")
    ax.set_ylabel("density over events (n = 1,596)")
    ax.set_title("the displacement has a flip-specific direction")
    ax.legend(loc="upper left", fontsize=7,
              title="band: 95% bootstrap over events", title_fontsize=7,
              labelspacing=0.3)


def causal_panel(ax, summary: pd.DataFrame) -> None:
    """Effect against collateral over the edit-scale sweep. Up and to the left is
    a clean intervention; to the right the effect is bought with board damage."""
    series = [(("tpr", "flipped"), PROBE_STYLE["tpr"], "TPR edit"),
              (("linear", "flipped"), PROBE_STYLE["linear"], "linear-probe edit"),
              (("tpr", "random"), NULL, "random direction (null)")]
    for (method, pop), style, label in series:
        d = summary[(summary.method == method)
                    & (summary.population == pop)].sort_values("scale")
        ax.errorbar(d.collateral_tpr_mean, d.d_lp_cf_mean,
                    yerr=[d.d_lp_cf_mean - d.d_lp_cf_lo,
                          d.d_lp_cf_hi - d.d_lp_cf_mean],
                    xerr=[d.collateral_tpr_mean - d.collateral_tpr_lo,
                          d.collateral_tpr_hi - d.collateral_tpr_mean],
                    marker=style["marker"], ms=5, lw=1.4, ls=style.get("ls", "-"),
                    color=style["color"], elinewidth=0.8, ecolor=style["color"],
                    markeredgecolor="white", markeredgewidth=0.6,
                    zorder=2 if style is NULL else 4, label=label)
    ax.set_xlabel("collateral: other squares whose decoded class changed (of 63)"
                  "\n" r"$\leftarrow$ cleaner intervention")
    ax.set_ylabel(r"$\Delta$ log-mass onto the counterfactual legal set"
                  "\n" r"$\uparrow$ the model plays the un-flipped board")
    ax.set_title("and editing along it changes what the model plays")
    ax.legend(loc="upper left", fontsize=7,
              title="one point per edit scale (1-16); bars = 95% CI over events",
              title_fontsize=7, labelspacing=0.3)


def figure_two(camera_ready: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    specificity_panel(axes[0], pd.read_csv(RESULTS / "flip_cos_per_event.csv"))
    causal_panel(axes[1], pd.read_csv(RESULTS / "intervention_summary.csv"))
    for i, ax in enumerate(axes):
        panel_letter(ax, i)
    if not camera_ready:
        fig.suptitle("The flip is a specific direction in the residual stream, and "
                     "the model uses it", y=1.0)
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
