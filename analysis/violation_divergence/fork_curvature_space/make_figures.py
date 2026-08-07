"""The branch fork drawn in R^21, the ReStraV curvature descriptor space.

The incumbent grid (../game_trajectory, `space_fork_grid_pca`) projects the
displacement D(t) = z_b(t) - z_ref(t) straight onto a PCA of D. Here the game is
first mapped into R^21 -- the 7 raw step lengths, 6 raw turning angles and their
8 moments of src/spade/straightness/geometry.py -- and the plane is taken there.

Two variants, both shipped, in two planes each:

  windowed  R^21 over the W=8 sampled positions ending at t, so the fork keeps
            its curve over move index. The figure draws
            Δφ(t) = φ(z_b(t)) - φ(z_ref(t)), not φ(D): D is exactly zero over the
            shared prefix, so its steps are zero, its angles undefined, and the
            reference branch has no descriptor at all (measure.py: 10 of 21
            components NaN). Differencing in R^21 instead keeps both of the
            incumbent figure's invariants -- reference at the origin, prefix
            collapsed -- and they are verified exactly.
  whole     one R^21 point per branch over the post-branch trajectory. No move
            axis; this is analysis/straightness/feature_pca's object, cut down to
            one game's branches, with a jackknife spread over which sampled
            position is dropped.

Eight branches are drawn -- the played move, one legal control (control 2 is the
null for control 1 and belongs where control 1 is measured), and one move per
illegal category, the six of them collapsed onto the three rules they violate.
phi is nonlinear, so it is applied to each member's own trajectory and the rule
means are taken afterwards, in the plane; the README carries the within- over
between-rule spread that says how little the grouping is worth.

Each windowed grid ends in a read-out strip: how far each rule's mean path runs
from the reference, averaged over the drawn offsets r >= 0, DIVIDED BY the legal
control's run in the same field. A 3-D projection shows shape and never
magnitude, so the strip is the quantitative companion FIGURE_GUIDELINES 4.8 asks
for. The ratio is what makes it readable across the grid at all: the plane is
refit per space and carries its own units, and only a within-panel contrast
survives the move from one cell to the next. Beside each plane number sits the
same statistic in full R^21 -- the gap between them is what the plane threw away
-- with a bar spanning the 11 step-length and the 10 turning-angle components,
which disagree. The whole grids get no strip: one point per branch over r >= 0
IS the average, so the number would restate the panel (README).

The grids carry one line of prose: that the game was selected. Everything else --
the spread ratio, where the mean over K rollouts is taken, the saturation horizon,
the cells-are-not-comparable disclaimer -- is the figures' caption and lives in
README.md (FIGURE_GUIDELINES 2, 5).

  pca       top-2 uncentred PCs of the column-scaled descriptor, refit per space,
            so cells are comparable in arrangement only.
  fixed     mean d against mean θ, the two components the straightness study
            states its headline in. The same named pair in every cell; θ is in
            degrees and comparable across cells, d is on each space's own
            activation scale and is not.

`--variant rollout` draws the same four grids on the branch set whose suffixes
are sampled from the model (rollout_branches.py) instead of spliced from the
source game, into `*_rollout` files. It stores K=4 rows per branch, so a branch
is drawn as the mean over its K descriptors -- taken AFTER φ, since φ of an
averaged trajectory is the curvature of a path no rollout took -- and the
windowed grids stop at ROLLOUT_MAX_R, past which two rollouts of one branch
differ by more than the branch differs from the reference.

Usage: python analysis/violation_divergence/fork_curvature_space/make_figures.py
       [--game 1235] [--w 8] [--variant spliced|rollout] [--camera-ready]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import torch

from spade.figures.branch_panels import PLANE_TICKS, axis_labels, plane_displacement
from spade.figures.branch_style import (
    CONTOUR_LABEL, MEMBER, MEMBER_LABEL, below, branch_style, legend_proxy,
    ordered_legend, rule_member_key)
from spade.figures.embedding_panels import class_ellipse
from spade.figures.paper_style import GRID, INK, MUTED, NULL, apply_style, save_figure
from spade.othello.taxonomy import LEGAL
from spade.straightness.geometry import RESTRAV_FEATURE_NAMES
from spade.violation_divergence import curvature_plane as dsc
from spade.violation_divergence.fork_plane import RAW_SPACE
from spade.violation_divergence.branches import FIGURE_GAME
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

FIG_DIR = Path(__file__).resolve().parent / "figures"
ELEV, AZIM = 18, -58  # matched to ../game_trajectory so the two grids read alike
NCOL = 3
# FIGURE_GUIDELINES 1.4: units, and the direction of "good" where it is not obvious.
D_AXIS = "mean step length $d$ (activation norm)"
THETA_AXIS = "mean turning angle θ (deg, ↑ curvier)"
# 3-D panels have no room for the long forms; the units and direction survive.
D_AXIS_3D = "Δ mean $d$ (activation norm)"
THETA_AXIS_3D = "Δ mean θ (deg, ↑ curvier)"
Z_AXIS = "$t-t_0$ (plies)"
PREFIX_LABEL = "shared prefix, window runs off the board (no descriptor)"

VARIANT_DIRS = {"spliced": "", "rollout": "rollout"}
FULL_R = 24              # the study window's last offset; the spliced grids run to it
# Past this offset a branch's K rollouts differ from each other by more than the
# branch differs from the reference -- median over the nine spaces of
# ||Δφ_branch|| / ||φ_k − φ_k'|| is 0.93 at r=8 and falls monotonically after
# (README, "Where the rollout curve stops being about the branch").
ROLLOUT_MAX_R = 8
ROLLOUT_CONTOUR = "50% Gaussian contour of {n} rollout descriptors (sampling spread, not a CI)"
# The one thing a reader cannot look up while looking at the panels: this game was
# chosen, so its clean arrangement is not what a draw from the population gives.
SELECTED = (f"game {FIGURE_GAME} is selected, not typical: the argmax over 1550 "
            "branch-complete games of its illegal branches' mutual separation "
            "(../game_selection). One game, n = 1.")

# The read-out strip under the windowed grids: the number the fork panel makes the
# reader guess at. R^21's two halves are measured apart because they disagree --
# no component name carries both letters, so the split is exact.
STRIP_H = 3.3            # inches of figure height the strip and its legend take
STRIP_LEGEND = 0.85      # of which, the row above the axes its legend sits in
DODGE = 0.15             # x offset that keeps the plane mark off the R^21 mark
RUN_TICKS = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]   # a ratio axis, so doubling steps
D_COLS = [i for i, n in enumerate(RESTRAV_FEATURE_NAMES) if "d" in n]
THETA_COLS = [i for i, n in enumerate(RESTRAV_FEATURE_NAMES) if "θ" in n]
RUN_AXIS = ("run from the reference\n÷ the control's (log, ↑ further)")
RUN_LEGEND = {
    "plane": "in the plane drawn above (what the eye is judging)",
    "full": "in full $R^{21}$ — the plane keeps only part of it",
    "block": f"the same split into its {len(D_COLS)} step-length and "
             f"{len(THETA_COLS)} turning-angle components",
}


@dataclass(frozen=True)
class Draw:
    """The branches drawn, each as the rows a caller must reduce over."""
    rows: list[dict]
    groups: list[list[int]]   # the rows averaged into each drawn branch
    ref: int
    contour: str              # what the whole grid's ellipses are, for the legend

    @property
    def rollout(self) -> bool:
        """A drawn branch is K sampled continuations rather than one spliced row."""
        return len(self.groups[0]) > 1


@dataclass(frozen=True)
class View:
    """How far the curve runs, and the filename suffix that keeps the two
    variants' outputs apart."""
    r_max: int
    suffix: str


def one_control(g: dict) -> Draw:
    """The branch types drawn, each as the rows that make it up.

    The spliced set stores one row per branch type; the rollout set stores K, so
    a caller must reduce over them explicitly. Control 2 is dropped: it is the
    null *for* control 1 and belongs where control 1 is measured, not in an
    illustration of the construction. Dropping it also drops it from the plane
    fit, which is fit on what is drawn.
    """
    key = list(zip(g["category"].tolist(), g["is_control"].tolist()))
    kinds = [k for k in sorted(set(key)) if k[1] != 2]
    rows = [{"category": c, "is_control": k} for c, k in kinds]
    groups = [[i for i, k in enumerate(key) if k == kind] for kind in kinds]
    ref = next(i for i, r in enumerate(rows)
               if r["category"] == LEGAL and not r["is_control"])
    n_pos = len(g["pos"]) - int(np.flatnonzero(g["pos"] == g["t0"])[0])
    contour = (ROLLOUT_CONTOUR.format(n=len(groups[0])) if len(groups[0]) > 1
               else CONTOUR_LABEL.format(n=n_pos))
    return Draw(rows, groups, ref, contour)


def phi_mean(F: np.ndarray, groups: list[list[int]], axis: int = 0) -> np.ndarray:
    """Average each branch's rows along `axis`, after the descriptor has been taken.

    φ is nonlinear, so this never touches a trajectory: callers apply `windowed`,
    `descriptor` or `jackknife` to every row and reduce here. One row per branch --
    the spliced set -- makes it the identity.
    """
    return np.stack([F.take(ix, axis).mean(axis) for ix in groups], axis)


def scaled(phi: np.ndarray) -> np.ndarray:
    """Column-scaled as fit_plane scales: R^21 mixes degrees with activation-scale
    lengths, so a Euclidean statistic on the raw columns is one about θ alone."""
    sd = np.nanstd(phi.reshape(-1, phi.shape[-1]), axis=0)
    return phi / np.where(sd > 0, sd, 1)


def runs(F: np.ndarray, r: np.ndarray, ref: int, ctl: int,
         groups: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Per rule, `[mean path, member, member, ...]` runs from the reference in the
    field F, each divided by the control's run in the same field.

    The rule's own entry is the run of the MEAN PATH the panel draws, not the mean
    of its members' runs: the two differ whenever the members leave in different
    directions, and the drawn line is the mean path. Dividing by the control makes
    the field's units cancel, which is the only thing that lets nine refit planes
    be read side by side (README).
    """
    order = ([F[ref], F[ctl]] + [F[g].mean(0) for g in groups.values()]
             + [F[i] for g in groups.values() for i in g])
    run = plane_displacement(np.stack(order), r, 0, 0)
    out, at = {}, 2 + len(groups)
    for k, (rule, g) in enumerate(groups.items()):
        out[rule] = np.concatenate([run[[2 + k]], run[at:at + len(g)]]) / run[1]
        at += len(g)
    return out


@dataclass(frozen=True)
class Run:
    """One space's answer to "how far does each branch run from the reference",
    every entry a ratio to that space's own control (`runs`)."""
    plane: dict[str, np.ndarray]    # in the 2-D plane the panel draws
    full: dict[str, np.ndarray]     # in all 21 components
    d: dict[str, np.ndarray]        # in the 11 step-length components
    theta: dict[str, np.ndarray]    # in the 10 turning-angle components


def read_out(metric: np.ndarray, z: np.ndarray, r: np.ndarray, d: Draw) -> Run:
    """`metric` is the plane in the same column scaling as `z`, so plane ÷ full is
    the share of the run the plane keeps rather than a change of units."""
    ctl = next(i for i, row in enumerate(d.rows) if row["is_control"])
    groups = rule_rows(d.rows)
    return Run(*(runs(F, r, d.ref, ctl, groups)
                 for F in (metric, z, z[:, :, D_COLS], z[:, :, THETA_COLS])))


def run_strip(ax, names: list[str], read: list[Run], d: Draw, r: np.ndarray) -> None:
    """The quantitative companion the 3-D panels owe the reader (§4.8).

    Ratios, not distances: each panel's plane is refit and carries its own units,
    so only "this branch against this panel's legal control" survives the move
    from one cell to the next.
    """
    x = np.arange(len(names))
    ax.axhline(1.0, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=1,
               label="the legal control itself (1 by construction)")
    for rule, members in rule_rows(d.rows).items():
        st = branch_style(d.rows[members[0]]["category"], 0, by_rule=True)
        # No label: the rule colours and markers are the grid's, and the figure
        # legend below already carries them with their n (FIGURE_GUIDELINES 5).
        ax.plot(x - DODGE, [q.plane[rule][0] for q in read], color=st["color"],
                marker=st["marker"], ms=5, lw=1.4, markeredgecolor="white",
                markeredgewidth=0.5, zorder=4)
        for k in range(len(members)):
            ax.plot(x - DODGE, [q.plane[rule][1 + k] for q in read], ls="none",
                    marker=st["marker"], ms=3, color=st["color"], **MEMBER)
        lo = np.minimum([q.d[rule][0] for q in read], [q.theta[rule][0] for q in read])
        hi = np.maximum([q.d[rule][0] for q in read], [q.theta[rule][0] for q in read])
        full = [q.full[rule][0] for q in read]
        ax.vlines(x + DODGE, lo, hi, color=st["color"], lw=0.9, zorder=3)
        ax.plot(x + DODGE, full, ls="none", marker=st["marker"], ms=5,
                markerfacecolor="white", markeredgecolor=st["color"],
                markeredgewidth=1.1, zorder=4)
    for key, kw in (("plane", {"marker": "o", "ls": "-"}),
                    ("full", {"marker": "o", "ls": "none",
                              "markerfacecolor": "white", "markeredgewidth": 1.1}),
                    ("block", {"marker": "|", "ls": "none", "ms": 9})):
        ax.plot([], [], color=MUTED, lw=1.4, markeredgecolor=MUTED,
                label=RUN_LEGEND[key], **kw)
    vals = [v for q in read for rule in rule_rows(d.rows)
            for v in [*q.plane[rule], q.full[rule][0], q.d[rule][0], q.theta[rule][0]]]
    ax.set_yscale("log")
    ax.set_ylim(min(vals) / 1.35, max(vals) * 1.35)
    ax.set_yticks(RUN_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_ylabel(RUN_AXIS, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6.5)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=7.5,
              frameon=False, title=f"mean over the {len(r)} drawn offsets "
              f"$t-t_0$ = {r[0]}…{r[-1]}; the reference is 0 in every field by "
              f"construction and is not a point here",
              title_fontsize=7.5)


def _path(ax, seg: np.ndarray, z: np.ndarray, st: dict, label: str | None) -> None:
    """One branch or one rule mean: the line, its sparse markers, its end marker."""
    ax.plot(seg[:, 0], seg[:, 1], z, ls=st["ls"], color=st["color"], lw=st["lw"],
            marker=st["marker"], ms=3.0, markevery=3, markerfacecolor=st["mfc"],
            markeredgecolor=st["mec"], markeredgewidth=0.3, zorder=st["zorder"],
            label=label)
    ax.plot([seg[-1, 0]], [seg[-1, 1]], [z[-1]], marker=st["marker"], ms=6,
            color=st["color"], markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
            markeredgewidth=0.7, zorder=st["zorder"] + 1)


def fork_panel(ax, coords: np.ndarray, rows: list[dict], r: np.ndarray,
               legend: bool, named: bool, zlabel: bool) -> None:
    """coords [n_branches, n_pos, 2]; offset from the branch is the vertical axis.

    One path per rule -- the mean of its member categories' Δφ -- over the members
    drawn faint. φ is nonlinear, so it is applied to each member's own trajectory
    and the means are taken afterwards; the projection is linear, so the mean of
    the projected members is the projection of their mean Δφ.

    `named` means the two plane axes carry units of their own (the fixed pair), so
    they keep their own ranges; a PCA plane is isotropic and unitless and gets one
    shared range. Either way both carry three ticks symmetric about zero, a
    within-panel scale bar for a plane that is refit per cell (§4.6, §4.8).
    """
    k = int(np.flatnonzero(np.isfinite(coords).all(-1).all(0))[0])
    ax.plot([0, 0], [0, 0], [r[0], r[k]], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label=PREFIX_LABEL if legend else None)
    for i, row in enumerate(rows):
        if row["category"] != LEGAL:
            continue
        st = branch_style(row["category"], row["is_control"])
        _path(ax, coords[i, k:], r[k:], st, st["label"] if legend else None)
    for members in rule_rows(rows).values():
        st = branch_style(rows[members[0]]["category"], 0, by_rule=True)
        for i in members:
            ax.plot(coords[i, k:, 0], coords[i, k:, 1], r[k:], color=st["color"],
                    **MEMBER)
        _path(ax, coords[members, k:].mean(axis=0), r[k:], st,
              f"{st['label']} (mean of {len(members)})" if legend else None)
    ax.plot([0], [0], [0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    if legend:
        legend_proxy(ax)
    if not named:
        lo, hi = np.nanmin(coords), np.nanmax(coords)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(PLANE_TICKS, symmetric=True))
        axis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.tick_params(axis="x", labelsize=6.5, pad=-3)
    ax.tick_params(axis="y", labelsize=6.5, pad=-3)
    ax.set_zlim(r[0], r[-1])
    if zlabel:
        ax.set_zlabel(Z_AXIS, labelpad=-6, fontsize=7)
    ax.tick_params(axis="z", labelsize=6.5, pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def point_panel(ax, P: np.ndarray, jack: np.ndarray, rows: list[dict],
                contour: str) -> None:
    """One R^21 point per branch, with the spread `contour` names around it.

    `jack` is [n_replicates, n_branches, 2]: leave-one-position-out replicates on
    the spliced set, the K rollout descriptors on the rollout one. A rule is one
    point, the mean of its members', with the replicates of that mean; the members
    are drawn faint. φ is applied per member and never to a mean trajectory, and
    the projection is affine, so mean-then-project agrees.
    """
    for i, row in enumerate(rows):
        if row["category"] != LEGAL:
            continue
        st = branch_style(row["category"], row["is_control"])
        is_ref = not row["is_control"]
        class_ellipse(ax, jack[:, i], st["color"], 2.0 if is_ref else 1.2)
        ax.plot(P[i, 0], P[i, 1], ls="none", marker=st["marker"],
                ms=9 if is_ref else 7, color=st["color"], markerfacecolor=st["mfc"],
                markeredgecolor=st["mec"], markeredgewidth=1.0,
                zorder=st["zorder"] + 3, label=st["label"])
    for members in rule_rows(rows).values():
        st = branch_style(rows[members[0]]["category"], 0, by_rule=True)
        ax.plot(P[members, 0], P[members, 1], ls="none", marker=st["marker"], ms=5,
                color=st["color"], **MEMBER)
        class_ellipse(ax, jack[:, members].mean(1), st["color"], 1.2)
        ax.plot(P[members, 0].mean(), P[members, 1].mean(), ls="none",
                marker=st["marker"], ms=7, color=st["color"],
                markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
                markeredgewidth=1.0, zorder=st["zorder"] + 3,
                label=f"{st['label']} (mean of {len(members)})")
    ax.plot([], [], ls="none", marker="o", color=MUTED, alpha=MEMBER["alpha"],
            label=MEMBER_LABEL)
    # The ellipse's semantics ride in the legend, not in a prose block: it is the
    # only uncertainty one game supports and a reader must not read it as a CI.
    ax.plot([], [], color=MUTED, lw=1.3, label=contour)


def spreads(g: dict, w: int, d: Draw, r_max: int) -> dict[str, float]:
    """Within-rule ÷ between-rule mean pairwise distance, one per variant, in the
    reference space and in full R^21 -- never on the 2-D plane the figure draws.
    `displacement` is the same statistic on the raw activations, the number
    ../fork_and_board reports, so the two figures can be compared."""
    c = int(g["n_context"])
    r = g["pos"][c:] - g["t0"]
    post = (r >= 0) & (r <= r_max)
    Z = g[RAW_SPACE]
    phi = phi_mean(dsc.windowed(Z, w)[:, c:], d.groups)
    k0 = c + int(np.flatnonzero(r >= 0)[0])
    whole = phi_mean(dsc.descriptor(torch.from_numpy(Z[:, k0:])), d.groups)
    D = phi_mean(Z, d.groups)[:, c:]
    return {"windowed": spread_ratio(scaled(phi - phi[d.ref])[:, post], d.rows),
            "whole": spread_ratio(scaled(whole), d.rows),
            "displacement": spread_ratio((D - D[d.ref])[:, post], d.rows)}


def windowed_grid(g: dict, w: int, plane: str, d: Draw, v: View,
                  camera_ready: bool) -> Path:
    c = int(g["n_context"])
    r = g["pos"][c:] - g["t0"]
    drawn = r <= v.r_max
    r = r[drawn]
    h = 3.0 * NCOL + STRIP_H
    fig = plt.figure(figsize=(3.9 * NCOL, h))
    # Two boxes rather than one gridspec: the strip is a 2-D axes and needs a left
    # margin for its y label, which the 3-D cells give away to their panes.
    cells = fig.add_gridspec(NCOL, NCOL, left=0.02, right=0.97, top=1 - 0.72 / h,
                             bottom=STRIP_H / h, wspace=0.04, hspace=0.18)
    read = []
    for j, (name, _) in enumerate(dsc.spaces()):
        phi = phi_mean(dsc.windowed(g[name], w)[:, c:], d.groups)[:, drawn]
        dphi = phi - phi[d.ref]
        flat = dphi.reshape(-1, dphi.shape[2])
        sd = np.nanstd(flat, axis=0)
        z = dphi / np.where(sd > 0, sd, 1)
        if plane == "pca":
            p = dsc.fit_plane(flat, centred=False)
            coords = p.project(flat).reshape(dphi.shape[:2] + (2,))
            metric = coords          # project() divides by the same column SDs as z
            labels = axis_labels("pca", p.var)
        else:
            ax_i = [dsc.names_of(dphi.shape[2]).index(a) for a in dsc.FIXED_AXES]
            coords = dphi[:, :, ax_i]
            # Drawn in named units, measured in each axis's own SD: degrees and an
            # activation norm do not add, and the panel's independent x/y autoscale
            # is already showing the scaled arrangement.
            metric = z[:, :, ax_i]
            labels = [D_AXIS_3D, THETA_AXIS_3D]
        ax = fig.add_subplot(cells[j // NCOL, j % NCOL], projection="3d")
        ax.set_box_aspect((1, 1, 0.85))
        fork_panel(ax, coords, d.rows, r, legend=(j == 0), named=(plane == "fixed"),
                   zlabel=(j % NCOL == 0))
        ax.set_xlabel(labels[0], labelpad=-2, fontsize=6.5)
        ax.set_ylabel(labels[1], labelpad=-2, fontsize=6.5)
        ax.set_title(name, loc="left", fontsize=8)
        read.append(read_out(metric, z, r, d))
    strip = fig.add_subplot(fig.add_gridspec(1, 1, left=0.07, right=0.97,
                                             bottom=0.95 / h,
                                             top=(STRIP_H - STRIP_LEGEND) / h)[0])
    run_strip(strip, [name for name, _ in dsc.spaces()], read, d, r[r >= 0])
    leg = ordered_legend(fig, fig.axes[0], 4, loc="lower center",
                         bbox_to_anchor=(0.5, -0.035))
    if not camera_ready:
        fig.suptitle(f"The fork inside the 21-D ReStraV descriptor space, "
                     f"game {g['game']} ({plane} plane)", fontsize=11)
    selection_and_key(fig, leg, int(g['game']))
    print(run_table(plane, [name for name, _ in dsc.spaces()], read, d))
    return save_figure(fig, FIG_DIR / f"fork_r21_windowed_{plane}{v.suffix}",
                       camera_ready)


def run_table(plane: str, names: list[str], read: list[Run], d: Draw) -> str:
    """The strip's numbers as text, so the README can quote them without a reader
    measuring them off a log axis."""
    rules = list(rule_rows(d.rows))
    head = (f"\nrun from the reference / the control's run, {plane} plane\n"
            f"{'space':34}" + "".join(f"{rule + ' plane/21/d/θ':>30}" for rule in rules))
    return "\n".join([head] + [
        f"{name:34}" + "".join(
            f"{q.plane[rule][0]:8.2f}{q.full[rule][0]:8.2f}"
            f"{q.d[rule][0]:7.2f}{q.theta[rule][0]:7.2f}" for rule in rules)
        for name, q in zip(names, read)])


def whole_grid(g: dict, plane: str, d: Draw, v: View, camera_ready: bool) -> Path:
    c = int(g["n_context"])
    k = c + int(np.flatnonzero(g["pos"][c:] - g["t0"] == 0)[0])
    fig, axes = plt.subplots(NCOL, NCOL, figsize=(3.4 * NCOL, 3.2 * NCOL))
    for ax, (name, _) in zip(axes.flat, dsc.spaces()):
        Z = g[name][:, k:]
        rowphi = dsc.descriptor(torch.from_numpy(Z))
        phi = phi_mean(rowphi, d.groups)
        # The spliced set has one row per branch and no sampling spread, so its
        # replicates are the leave-one-position-out jackknife; the rollout set's
        # are its K descriptors, which is the spread a reader has to see.
        jack = (np.stack([rowphi[ix] for ix in d.groups], 1) if d.rollout
                else phi_mean(dsc.jackknife(Z), d.groups, axis=1))
        if plane == "pca":
            p = dsc.fit_plane(phi, centred=True)
            P, J = p.project(phi), p.project(jack)
            xlab, ylab = axis_labels("pca", p.var)
            ax.set_title(f"{name}  ·  arb. units, refit per space", loc="left",
                         fontsize=8)
        else:
            i = [dsc.names_of(phi.shape[1]).index(a) for a in dsc.FIXED_AXES]
            P, J = phi[:, i], jack[:, :, i]
            xlab, ylab = D_AXIS, THETA_AXIS
            ax.set_title(name, loc="left", fontsize=8)
        ax.set_xlabel(xlab, fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        point_panel(ax, P, J, d.rows, d.contour)
        ax.tick_params(labelsize=6.5)
    leg = ordered_legend(fig, axes.flat[0], 3, loc="lower center",
                         bbox_to_anchor=(0.5, -0.045))
    if not camera_ready:
        fig.suptitle(f"One R$^{{21}}$ point per branch, game {g['game']} "
                     f"({plane} plane)", fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    selection_and_key(fig, leg, int(g['game']))
    return save_figure(fig, FIG_DIR / f"fork_r21_whole_{plane}{v.suffix}", camera_ready)


def selection_and_key(fig, leg, game: int) -> None:
    """The selection line, then the rule key under it.

    Everything else a reader needs -- where the mean over K is taken, the spread
    ratio, the saturation horizon -- is the caption's job and lives in README.md
    (§5). The selection is not: a reader who does not know the game was chosen
    over-reads the panels in front of them. `--game` can draw an unselected one,
    and then the line must not claim the argmax."""
    note = SELECTED if game == FIGURE_GAME else f"game {game}, n = 1: shape only."
    y = below(fig, fig.text(0.005, below(fig, leg), note, ha="left", va="top",
                            fontsize=7.5, color=MUTED))
    rule_member_key(fig, y=y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, default=dsc.W)
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--variant", default="spliced", choices=list(VARIANT_DIRS))
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = dsc.figure_game(args.game, args.w, variant=VARIANT_DIRS[args.variant])
    rollout = args.variant == "rollout"
    d = one_control(g)
    suffix = "_rollout" if rollout else ""
    windowed = View(ROLLOUT_MAX_R if rollout else FULL_R, suffix)
    whole = View(FULL_R, suffix)
    # The ratios are the figures' caption, not the figures': README.md quotes them.
    for r_max in ([ROLLOUT_MAX_R, FULL_R] if rollout else [FULL_R]):
        print(f"within/between rule spread ratio, game {args.game}, {args.variant}, "
              f"t-t0 in [0, {r_max}]: "
              + ", ".join(f"{k} {x:.3f}"
                          for k, x in spreads(g, args.w, d, r_max).items()))
    for plane in ("pca", "fixed"):
        print(f"wrote {windowed_grid(g, args.w, plane, d, windowed, args.camera_ready)}")
        print(f"wrote {whole_grid(g, plane, d, whole, args.camera_ready)}")


if __name__ == "__main__":
    main()
