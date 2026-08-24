"""The magnitude a fork panel cannot show: how far each branch runs from the
reference at every offset r, in all nine representation spaces.

The companion to the fork grid in `../fork_and_board` and `../game_trajectory`.
A 3-D projection shows the shape of a fork and never its size, so a reader
looking at those panels cannot tell how far a branch actually went
(FIGURE_GUIDELINES 4.8). This is that number, with the same nine cells in the
same order, the same rule grouping, and the same `branch_style` colours and
markers, so nothing has to be re-learnt moving between the two.

Two curves per class, and they are not equals. The **2-D plane** is the answer:
it is the distance between a branch and the reference in the same plane the fork
panel draws them in, so it is the number that panel poses and cannot give. Solid,
filled markers, and what the y-axis is about. The **full space** rides underneath
as the subordinate reference -- dotted, open markers, thinner -- because a plane
holds only part of a displacement and holds a different part of each branch, so a
reader who takes the in-plane curve for the whole divergence overstates it. The
gap between the pair is the size of that overstatement.

Every curve is divided by the same cell's legal control -- that control's own
mean over r >= 0 -- which is the only form in which nine refit planes may be set
side by side: raw distances differ ~585x across the grid for the same branch.
`../fork_and_board`'s panel (c) is exactly these curves collapsed to one number
per rule; the r axis is what this figure adds, and the population and the
interval over games.

`--plane r21` draws the same nine cells over `measure.py`'s R^21 sweep, where the
reference is subtracted after the curvature descriptor rather than in the
representation space. Everything else -- panels, colours, markers, the
ratio-to-the-legal-control, the CI -- is the same, and the x window is read off
the npz because phi's trailing W-window costs the first two offsets.

`--raw` drops the divisor and draws the measured distances, so the magnitudes the
ratio hides are on the page. Nothing else about the figure changes: the same panel
grid, styling, legend and furniture, minus only the pieces the divisor defined --
the line at 1, the y cap, and the two callouts written in multiples of the control.
The displacement pair then gets a per-panel y axis, because what it draws spans
1360x: SpaDE's curves sit at 0.0097-0.023 where raw layer 5's run 7-13, and on one
linear axis the two SpaDE panels are a flat line on zero. `--shared-y` is that pair
on one autoscaled linear axis anyway, i.e. with no deviation from the ratio
figure's format at all, so the two can be judged against each other. R^21 needs
neither: `curvature_plane.fit_plane` scales its 21 columns to unit variance per
game, so every panel's peak lands within 1.3x of every other's -- and its units are
per-game descriptor SDs, not activations.

Reads `measure.py`'s npz. Writes the figure and one tidy CSV of every drawn
point. Prose belongs in README.md (FIGURE_GUIDELINES 2, 5); what stays inside
the figure is what a reader cannot decode while looking at it.

`--l0-matched` draws `measure.py --l0-matched`'s sweep: the same nine cells with
the eight SAE slots seated at `l0_operating_point`'s shared L0 rather than at each
type's own default sparsity, so a difference between two cells is the architecture
and not how sparse it happens to be. Every file it writes carries an `_l0_matched`
suffix, so the canonical figures and CSVs are untouched. Each cell's title carries
its own L0, which is what the panels are matched on.

Usage: python analysis/violation_divergence/divergence_by_space/make_figures.py
       [--plane displacement|r21] [--variant spliced|rollout] [--raw [--shared-y]]
       [--l0-matched] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.figures.branch_style import branch_style
from spade.figures.paper_style import (
    GRID, INK, MUTED, NULL, apply_style, save_figure)
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_MEMBERS

HERE = Path(__file__).resolve().parent
VARIANTS = {"spliced": "", "rollout": "rollout"}
NCOL = 3
CELL_W, CELL_H = 3.05, 2.45
CONTROL = "control_legal"
# Which space the distance was measured in rides on line style and marker fill,
# so hue stays free for the rule and a panel carries four classes, not eight
# series. The plane is the figure's quantity and takes the solid line.
PLANE = {"ls": "-", "lw": 1.7, "ms": 4.2, "fill": True}
MEASURES = {"plane": PLANE}
# Drawn: the plane only. Tabulated: both, because the gap between them is how
# much of a branch's run the fork panel actually shows, and the README quotes it.
TABLE_MEASURES = ("plane", "full")
SATURATED_R = 8  # rollout: from here the illegal/control ratio is within 0.5% of 1
# One shared y-axis is the whole point of dividing by the control, and the r = 0
# spike in the two SpaDE codes is ~3x the next panel's. Capping at 2 keeps the
# plateau -- where the claim is -- readable in all nine cells; a curve that runs
# past the cap is labelled with the value it reaches, so nothing is hidden.
YMAX = 2.0
RATIO_YLABEL = "distance from the reference\n(legal control = 1)"
# --raw has no divisor, so the label has to name the units. R^21's 21 columns are
# scaled to unit variance per game inside `curvature_plane.fit_plane`, which
# `measure.curvature_distances` measures both its distances in, so that pair is in
# per-game descriptor SDs and is not physically raw.
RAW_YLABEL = {
    "displacement": "distance from the reference\n(each space's own units)",
    "r21": "distance from the reference\n($R^{21}$ descriptor SDs,\nper game)",
}


def series_rows(z) -> dict[str, list[int]]:
    """Row indices per drawn series: one per rule, then the control."""
    role = list(z["role"])
    return {rule: [role.index(c) for c in members]
            for rule, members in RULE_MEMBERS.items()} | {CONTROL: [role.index(CONTROL)]}


def normalised(z, key: str) -> np.ndarray:
    """[S, G, role, T]: every branch's distance from the reference, as a multiple
    of the same space and game's legal control averaged over r >= 0.

    The divisor is a scalar per cell, not a curve, so the control keeps its shape
    and the number a reader reads off a curve is the number
    `../fork_and_board`'s panel (c) puts in a bar.
    """
    control = z[key][:, :, list(z["role"]).index(CONTROL)][..., z["window"] >= 0].mean(-1)
    return z[key] / control[:, :, None, None]


def rule_means(z, key: str, raw: bool) -> np.ndarray:
    """[S, G, series, T]: the measured distances, or `normalised`, with each rule
    collapsed onto the mean of its member categories, so the resampling unit stays
    the source game."""
    x = z[key] if raw else normalised(z, key)
    return np.stack([x[:, :, members].mean(2) for members in series_rows(z).values()],
                    axis=2)


def mean_ci(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over games and the half-width of its 95% CI. Axis 0 is the game, and
    one game is one item in every series, so no clustering correction is needed."""
    return x.mean(0), 1.96 * x.std(0, ddof=1) / np.sqrt(len(x))


def table(z, variant: str, raw: bool) -> pd.DataFrame:
    """Every drawn point, so the README can quote a number without reading a curve."""
    col = "distance" if raw else "ratio"
    rows = []
    for key in TABLE_MEASURES:
        r = rule_means(z, key, raw)
        for si, space in enumerate(z["space"]):
            for ci, name in enumerate(series_rows(z)):
                mu, half = mean_ci(r[si, :, ci])
                rows.append(pd.DataFrame({
                    "variant": variant, "space": space, "series": name,
                    "measured_in": key, "r": z["window"], "n_games": r.shape[1],
                    col: mu, f"{col}_lo": mu - half, f"{col}_hi": mu + half}))
    return pd.concat(rows, ignore_index=True)


def style_of(name: str) -> dict:
    """The fork grid's own styling, through `branch_style`, so a colour and a
    marker mean the same thing in both figures."""
    if name == CONTROL:
        return branch_style(LEGAL, 1)
    st = branch_style(CATEGORIES.index(RULE_MEMBERS[name][0]), 0, by_rule=True)
    return st | {"label": f"{st['label']} (mean of {len(RULE_MEMBERS[name])})"}


def draw_series(ax, w, mu, half, name: str, how: dict, offset: int) -> None:
    st = style_of(name)
    ax.fill_between(w, mu - half, mu + half, color=st["color"], alpha=0.18, lw=0,
                    zorder=st["zorder"] - 1)
    ax.plot(w, mu, color=st["color"], marker=st["marker"], ls=how["ls"], lw=how["lw"],
            ms=how["ms"], markevery=(offset % 3, 3), zorder=st["zorder"],
            markerfacecolor=st["color"] if how["fill"] else "white",
            markeredgecolor="white" if how["fill"] else st["color"],
            markeredgewidth=0.6 if how["fill"] else 1.0)


def event_axes(ax, w: np.ndarray, rollout: bool) -> None:
    """The pre-branch window shaded, the branch position marked, and on the
    rollout tree the offsets where the comparison has saturated (README)."""
    ax.axvspan(w[0] - 1, -1, color=GRID, alpha=0.75, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if rollout:
        ax.axvspan(SATURATED_R, w[-1] + 1, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xlim(w[0] - 1, w[-1] + 1)
    # Anchored on the branch, not on w[0]: R^21's window starts at r = -4, and a
    # tick run from there would skip the event the axis is locked to.
    ax.set_xticks([r for r in range(int(w[0]), int(w[-1]) + 1) if r % 8 == 0])


def spread_note(z) -> str:
    """1.7's within- ÷ between-rule spread, measured on the population drawn here.
    Raw layer 5, the reference cell, as `rule_groups` quotes it. Printed rather
    than drawn: the figure carries no footer text, and the README quotes it."""
    r = z["spread"][0]
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    return ("within- ÷ between-rule spread of the displacements a rule mean averages: "
            f"median {med:.3f} [IQR {q1:.3f}–{q3:.3f}], raw layer 5 over the {len(r)} "
            "games drawn (rule_groups)")


def legend_marks(n_games: int) -> tuple[list, list]:
    """Two keys in one legend: hue and marker = class, exactly as the fork grid
    styles it; line style and marker fill = which space the distance is in."""
    handles, labels = [], []
    for name in list(RULE_MEMBERS) + [CONTROL]:
        st = style_of(name)
        handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"], ms=5,
                                  lw=1.7, markeredgecolor="white", markeredgewidth=0.6))
        labels.append(st["label"])
    return handles, labels


TITLE = {
    ("displacement", "spliced"): "How far the fork panels' branches actually run — and "
                                 "how much of that separation is the plane",
    ("displacement", "rollout"): "With a sampled suffix the branch move stops setting "
                                 "the distance, in every space",
    ("r21", "spliced"): "How far the branches run with the reference subtracted in "
                        "R$^{21}$ curvature space, not in the representation",
    ("r21", "rollout"): "With a sampled suffix, R$^{21}$'s trailing window puts the "
                        "separation where the comparison has already saturated",
}
# Spans are of what is drawn -- every rule mean and the control, over r >= 0 -- not
# of the control alone, which is the ~585x `measure.py` reports and a smaller number.
RAW_TITLE = {
    ("displacement", "spliced"): "The same branches un-normalised: nine spaces, nine "
                                 "scales, 1360$\\times$ from SpaDE to raw layer 5",
    ("displacement", "rollout"): "Un-normalised on the rollout tree: the same nine "
                                 "scales, 1671$\\times$ from SpaDE to raw layer 5",
    ("r21", "spliced"): "Un-normalised in R$^{21}$: column-scaling per game puts every "
                        "panel's peak within 1.3$\\times$, so one axis holds all nine",
    ("r21", "rollout"): "Un-normalised in R$^{21}$ on the rollout tree, where every "
                        "panel's peak is within 1.5$\\times$ and one axis still holds",
}


# A variant figure's title says what changed, not what the canonical one claims: the
# roster moved, and whether the claim survives is the thing the reader is here to check.
L0_TITLE = ("Each SAE type seated at the shared operating point $L_0$ = 219.2 — "
            "architecture and sparsity no longer confounded")


def figure(z, plane: str, variant: str, suffix: str, raw: bool, shared_y: bool,
           l0_matched: bool, camera_ready: bool) -> Path:
    w = z["window"]
    rollout = variant == "rollout"
    n_games = z["plane"].shape[1]
    curves = {key: rule_means(z, key, raw) for key in MEASURES}
    nrow = int(np.ceil(len(z["space"]) / NCOL))
    # What raw displacement draws spans 1360x, which no one linear axis holds;
    # --shared-y is that same pair on one anyway, to be judged against.
    free_y = raw and plane == "displacement" and not shared_y

    fig, axes = plt.subplots(nrow, NCOL, figsize=(CELL_W * NCOL, CELL_H * nrow),
                             sharex=True, sharey=not free_y)
    for si, space in enumerate(z["space"]):
        ax = axes.flat[si]
        event_axes(ax, w, rollout)
        if not raw:
            ax.axhline(1, color=NULL["color"], lw=0.9, ls=(0, (5, 2)), zorder=1)
        for ci, (name, rows) in enumerate(series_rows(z).items()):
            for key, how in MEASURES.items():
                draw_series(ax, w, *mean_ci(curves[key][si, :, ci]), name, how, ci)
        ax.set_title(space, loc="left", fontsize=8)
        if not raw:
            ax.set_ylim(-0.09, YMAX)
            off = curves["plane"][si].mean(0).max(0)
            if off.max() > YMAX:
                ri = int(off.argmax())
                ax.annotate(f"{off[ri]:.1f}$\\times$", xy=(w[ri], YMAX * 0.985),
                            xytext=(w[ri] + 2.5, YMAX * 0.86), fontsize=7.5, color=INK,
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        if si % NCOL == 0:
            ax.set_ylabel(RAW_YLABEL[plane] if raw else RATIO_YLABEL, fontsize=8.5)
    axes.flat[-NCOL + 1].set_xlabel("position relative to the branch move "
                                    "($r = s - t_0$)", fontsize=9)
    # Autoscaled, since YMAX is a multiple of a divisor --raw does not have, but
    # keeping the ratio panels' -0.09/2.0 of headroom under zero, so the shared
    # prefix is visibly zero rather than sitting on the spine. After the loop: on a
    # shared axis the first set_ylim would freeze the autoscale mid-grid.
    if raw:
        for ax in axes.flat[:len(z["space"])]:
            ax.set_ylim(bottom=-0.045 * ax.get_ylim()[1])

    lead = axes.flat[0]
    top = lead.get_ylim()[1]
    if not raw:
        lead.text(1, 0.06, "above 1: the rule runs further from the reference\nthan a "
                  "legal alternative does at the same prefix", fontsize=7, color=MUTED,
                  style="italic", ha="left", va="bottom")
    lead.text(w[0] + 1, 0.06 * top, "$\\equiv 0$ (shared prefix)", fontsize=7,
              rotation=90, color=MUTED, style="italic", ha="center", va="bottom")
    if not raw:
        peak = curves["plane"][0].mean(0)[:-1]  # series x r; the control is the last row
        ci, ri = np.unravel_index(np.argmax(np.where(w >= 0, peak, -np.inf)), peak.shape)
        lead.annotate(f"{peak[ci, ri]:.2f}$\\times$ at $r$={w[ri]}",
                      xy=(w[ri], peak[ci, ri]), xytext=(4, 0.90 * top), fontsize=7.5,
                      color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    if rollout:
        sat = axes.flat[NCOL - 1]
        sat.text(SATURATED_R + 1, 0.06 * sat.get_ylim()[1], f"$r \\geq {SATURATED_R}$: "
                 "saturated,\nnot evidence", fontsize=7, color=MUTED,
                 style="italic", ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.12, 1, 0.97 if camera_ready else 0.94))
    legend = fig.legend(*legend_marks(n_games), ncol=3, fontsize=8, frameon=False,
                        loc="lower center", bbox_to_anchor=(0.5, 0.025),
                        title=f"mean over n = {n_games} source games, band = 95% CI")
    legend.get_title().set_fontsize(8)
    if not camera_ready:
        fig.suptitle(L0_TITLE if l0_matched
                     else (RAW_TITLE if raw else TITLE)[(plane, variant)],
                     fontsize=11, y=0.995)
    name = ("divergence_by_space" + ("" if plane == "displacement" else f"_{plane}")
            + (f"_{variant}" if rollout else "") + suffix
            + ("_raw" if raw else "") + ("_sharedy" if shared_y else "")
            + ("_l0_matched" if l0_matched else ""))
    return save_figure(fig, HERE / "figures" / name, camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plane", default="displacement", choices=["displacement", "r21"],
                    help="the space measure.py subtracted the reference in")
    ap.add_argument("--variant", default="spliced", choices=list(VARIANTS))
    ap.add_argument("--n_games", type=int, default=None,
                    help="draw measure.py's n-game population instead of the whole tree")
    ap.add_argument("--raw", action="store_true",
                    help="draw the measured distances, not the ratio to the control")
    ap.add_argument("--shared-y", dest="shared_y", action="store_true",
                    help="--raw --plane displacement on one autoscaled linear y instead "
                         "of a per-panel one")
    ap.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                    help="draw measure.py --l0-matched's sweep, where the eight SAE cells "
                         "sit at one shared L0")
    ap.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = ap.parse_args()
    if args.shared_y and not args.raw:
        ap.error("--shared-y only means anything with --raw")
    apply_style(args.camera_ready)

    suffix = f"_n{args.n_games}" if args.n_games else ""
    l0 = "_l0_matched" if args.l0_matched else ""
    tag = "" if args.plane == "displacement" else f"_{args.plane}"
    base = variant_dir("divergence", variant=VARIANTS[args.variant])
    z = dict(np.load(base / "divergence_by_space" / f"curves{tag}{suffix}{l0}.npz"))
    z["space"], z["role"] = list(z["space"]), list(z["role"])

    out = results_dir("violation_divergence")
    out.mkdir(parents=True, exist_ok=True)
    # --shared-y is the same numbers on a different axis, so it writes no second CSV.
    csv = out / (f"divergence_by_space{tag}"
                 f"{'_' + args.variant if args.variant != 'spliced' else ''}"
                 f"{suffix}{'_raw' if args.raw else ''}{l0}.csv")
    df = table(z, args.variant, args.raw)
    df.to_csv(csv, index=False, float_format="%.6g")

    col = "distance" if args.raw else "ratio"
    post = df[(df.r >= 0) & (df.series != CONTROL)]
    print(f"mean over r >= 0, plane / full space, {z['plane'].shape[1]} games:")
    for space in z["space"]:
        cell = post[post.space == space].groupby(["series", "measured_in"])[col].mean()
        print(f"  {space:<34} " + "  ".join(
            f"{rule.split()[0]} {cell[rule, 'plane']:5.3g}|{cell[rule, 'full']:5.3g}"
            for rule in RULE_MEMBERS))
    if "spread" in z:
        print(spread_note(z))
    png = figure(z, args.plane, args.variant, suffix, args.raw, args.shared_y,
                 args.l0_matched, args.camera_ready)
    print(f"wrote {png} and {csv}")


if __name__ == "__main__":
    main()
