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

The grids carry no prose block. The spread ratio, the selection of the game, the
n=1 caveat and the cells-are-not-comparable disclaimer are the figures' captions
and live in README.md (FIGURE_GUIDELINES 2, 5).

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
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from spade.figures.branch_panels import PLANE_TICKS, axis_labels
from spade.figures.branch_style import (
    CONTOUR_LABEL, MEMBER, MEMBER_LABEL, below, branch_style, legend_proxy,
    ordered_legend, rule_member_key)
from spade.figures.embedding_panels import class_ellipse
from spade.figures.paper_style import INK, MUTED, apply_style, save_figure
from spade.othello.taxonomy import LEGAL
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
ROLLOUT_NOTE = (
    "rollout branch set: each branch is the mean over its K = 4 sampled continuations, "
    "taken AFTER φ — φ is nonlinear, and the mean of the trajectories has a curvature no "
    f"rollout has.\nDrawn to $t-t_0$ = {ROLLOUT_MAX_R}: past it two rollouts of one branch "
    "differ by more than the branch differs from the reference, so the curve is sampling "
    "noise (README).")
ROLLOUT_WHOLE_NOTE = (
    "rollout branch set: each point is the mean over its K = 4 sampled continuations, taken "
    "AFTER φ — φ is nonlinear, and the mean of the trajectories has a curvature no rollout "
    f"has.\nThe descriptor needs 8 sampled positions, so it cannot be confined to the "
    f"$t-t_0 \\leq {ROLLOUT_MAX_R}$ region where the branch still outweighs the rollout "
    "sampling spread; the ellipses are that spread (README).")
ROLLOUT_CONTOUR = "50% Gaussian contour of {n} rollout descriptors (sampling spread, not a CI)"


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
    """How far the curve runs, the in-figure note, and the filename suffix that
    keeps the two variants' outputs apart."""
    r_max: int
    note: str
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
    fig = plt.figure(figsize=(3.9 * NCOL, 3.0 * NCOL))
    for j, (name, _) in enumerate(dsc.spaces()):
        phi = phi_mean(dsc.windowed(g[name], w)[:, c:], d.groups)[:, drawn]
        dphi = phi - phi[d.ref]
        flat = dphi.reshape(-1, dphi.shape[2])
        if plane == "pca":
            p = dsc.fit_plane(flat, centred=False)
            coords = p.project(flat).reshape(dphi.shape[:2] + (2,))
            labels = axis_labels("pca", p.var)
        else:
            ax_i = [dsc.names_of(dphi.shape[2]).index(a) for a in dsc.FIXED_AXES]
            coords = dphi[:, :, ax_i]
            labels = [D_AXIS_3D, THETA_AXIS_3D]
        ax = fig.add_subplot(NCOL, NCOL, j + 1, projection="3d")
        ax.set_box_aspect((1, 1, 0.85))
        fork_panel(ax, coords, d.rows, r, legend=(j == 0), named=(plane == "fixed"),
                   zlabel=(j % NCOL == 0))
        ax.set_xlabel(labels[0], labelpad=-2, fontsize=6.5)
        ax.set_ylabel(labels[1], labelpad=-2, fontsize=6.5)
        ax.set_title(name, loc="left", fontsize=8)
    leg = ordered_legend(fig, fig.axes[0], 4, loc="lower center",
                         bbox_to_anchor=(0.5, -0.035))
    if not camera_ready:
        fig.suptitle(f"The fork inside the 21-D ReStraV descriptor space, "
                     f"game {g['game']} ({plane} plane)", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.07,
                        wspace=0.04, hspace=0.18)
    rule_member_key(fig, y=note_and_key(fig, leg, v.note))
    return save_figure(fig, FIG_DIR / f"fork_r21_windowed_{plane}{v.suffix}",
                       camera_ready)


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
    rule_member_key(fig, y=note_and_key(fig, leg, v.note))
    return save_figure(fig, FIG_DIR / f"fork_r21_whole_{plane}{v.suffix}", camera_ready)


def note_and_key(fig, leg, note: str) -> float:
    """y for the rule key, with `note` written above it when the variant has one.

    Where a mean is taken is apparatus a reader needs while looking at the panels,
    so it sits with the rule key rather than in the README caption (§5)."""
    y = below(fig, leg)
    if not note:
        return y
    return below(fig, fig.text(0.005, y, note, ha="left", va="top", fontsize=7.5,
                               color=MUTED, linespacing=1.5))


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
    windowed = View(ROLLOUT_MAX_R if rollout else FULL_R,
                    ROLLOUT_NOTE if rollout else "", suffix)
    whole = View(FULL_R, ROLLOUT_WHOLE_NOTE if rollout else "", suffix)
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
