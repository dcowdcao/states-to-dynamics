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
BASIS_TITLE = {"relative": "native basis: mine / theirs / empty",
               "absolute": "relabeled to absolute colour"}
AXIS = {
    "relative": (r"$\ell_{\rm mine}-\ell_{\rm theirs}$ (logits)",
                 r"$\frac{1}{2}(\ell_{\rm mine}+\ell_{\rm theirs})-\ell_{\rm empty}$"),
    "absolute": (r"$\ell_{\rm mover}-\ell_{\rm opponent}$ (logits)",
                 r"$\frac{1}{2}(\ell_{\rm mover}+\ell_{\rm opp})-\ell_{\rm empty}$"),
}


def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def field_panel(ax, field: pd.DataFrame, basis: str, annotate: bool = False) -> None:
    """One arrow per square role, tail = before, head = after the flanking move.

    The flipped square carries a second, dashed arrow to where the matched
    non-flipping move leaves it; the gap between those two heads is the claim.
    """
    d = field[field.basis == basis].set_index("role")
    for role in ROLES:
        r, style = d.loc[role], SQUARE_ROLE_STYLE[role]
        ax.annotate("", xy=(r.after_x, r.after_y), xytext=(r.before_x, r.before_y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=style["color"],
                                    shrinkA=0, shrinkB=4, alpha=0.9))
        ax.errorbar(r.after_x, r.after_y,
                    xerr=[[r.after_x - r.after_x_lo], [r.after_x_hi - r.after_x]],
                    yerr=[[r.after_y - r.after_y_lo], [r.after_y_hi - r.after_y]],
                    marker=style["marker"], ms=8, color=style["color"], lw=0,
                    elinewidth=0.9, ecolor=INK, markeredgecolor="white",
                    markeredgewidth=1.0, zorder=5, label=style["label"])
        ax.plot(r.before_x, r.before_y, marker=style["marker"], ms=6, lw=0,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.2, zorder=4)
        if role == "flipped":
            ax.annotate("", xy=(r.control_x, r.control_y),
                        xytext=(r.before_x, r.before_y),
                        arrowprops=dict(arrowstyle="-|>", lw=1.1, ls=(0, (4, 2)),
                                        color=style["color"], shrinkA=0, shrinkB=4))
            ax.plot(r.control_x, r.control_y, marker="X", ms=9, lw=0,
                    color=style["color"], markeredgecolor="white",
                    markeredgewidth=1.0, zorder=6,
                    label="flipped square, matched non-flipping move")
            if annotate:  # the fill convention is the panel's whole grammar
                for x, y, text, dy, va in [
                        (r.before_x, r.before_y, "before", -9, "top"),
                        (r.after_x, r.after_y, "after", 9, "bottom"),
                        (r.control_x, r.control_y, "after $m'$\n(no flip)", -9, "top")]:
                    ax.annotate(text, (x, y), textcoords="offset points",
                                xytext=(0, dy), ha="center", va=va, fontsize=7.5,
                                color=style["color"], zorder=7)
    ax.axhline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.set_xlabel(AXIS[basis][0])
    ax.set_ylabel(AXIS[basis][1])
    ax.set_title(BASIS_TITLE[basis])


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
    # Board is square, so the left column's height sets the figure's: any extra
    # becomes dead space beside it.
    fig = plt.figure(figsize=(10.4, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.80, 1.0], wspace=0.26, hspace=0.55)
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 1])]

    board_panel(axes[0], event, meta)
    for i, basis in enumerate(["relative", "absolute"]):
        field_panel(axes[i + 1], field, basis, annotate=(basis == "relative"))
    for i, ax in enumerate(axes):
        panel_letter(ax, i)
    axes[1].legend(loc="lower left", fontsize=6.5, ncol=2,
                   title="open tail = before, filled head = after; "
                         "bars = 95% CI of the mean over 2,000 events",
                   title_fontsize=6.5, labelspacing=0.25, columnspacing=1.0)
    if not camera_ready:
        fig.suptitle("Where each square's read-out moves when a flanking move is "
                     "played, and how relabeling to absolute colour inverts it", y=0.99)
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


def base_mass_gap(per_event: pd.DataFrame) -> tuple[float, float]:
    """How far apart the two diagnostic sets start, and what that costs in
    resolution: the median real-minus-counterfactual base log-mass, and the
    Delta log-mass the real-only set would show at the smallest TPR edit if it
    had donated the whole counterfactual gain."""
    d = per_event[per_event.method.eq("tpr") & per_event.scale.eq(1.0)
                  & per_event.population.eq("flipped")].dropna(
                      subset=["base_lp_cf", "base_lp_real"])
    gain = np.exp(d.base_lp_cf + d.d_lp_cf) - np.exp(d.base_lp_cf)
    return (float((d.base_lp_real - d.base_lp_cf).median()),
            float(np.log1p(-gain / np.exp(d.base_lp_real)).mean()))


def causal_panel(ax, summary: pd.DataFrame, per_event: pd.DataFrame) -> None:
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
    gap, floor = base_mass_gap(per_event)
    ax.annotate(f"the two sets start {gap:.1f} nats apart, so equal "
                r"$\Delta$ log-mass is not" "\n"
                "equal probability: at the smallest edit even a complete\n"
                f"transfer would move the dotted curve by {floor:+.4f} nats,\n"
                "well inside its own CI",
                (0.02, 0.02), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=6.5, color=MUTED, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("collateral: other squares (of 63) whose decoded class changed,\n"
                  "scored by the decoder the edit was not built from (n = 1,370)"
                  "\n" r"$\leftarrow$ cleaner intervention")
    ax.set_ylabel(r"$\Delta$ next-move log-mass (nats)"
                  "\n" r"$\uparrow$ onto the un-flipped board's squares,"
                  "\n" r"$\downarrow$ off the real board's")
    ax.set_title("and editing along it moves mass toward the un-flipped board,\n"
                 "not off the legal set")
    ax.set_ylim(top=1.45 * edits.d_lp_cf_hi.max())  # a clear band for the legend
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, labelspacing=0.3,
              title_fontsize=6.5,
              title="solid, filled = counterfactual-only legal squares "
                    f"(n = {int(edits.d_lp_cf_n.iloc[0]):,})\n"
                    "dotted, open = real-only legal squares "
                    f"(n = {int(edits.d_lp_real_n.iloc[0]):,}); one switch, two sides\n"
                    r"bars = 95% bootstrap CI; labels = $\|\delta\|/\|h\|$ "
                    "(the knobs differ)")


def figure_two(camera_ready: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    specificity_panel(axes[0], pd.read_csv(RESULTS / "flip_cos_per_event.csv"))
    causal_panel(axes[1], pd.read_csv(RESULTS / "intervention_summary.csv"),
                 pd.read_csv(RESULTS / "intervention_per_event.csv"))
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
