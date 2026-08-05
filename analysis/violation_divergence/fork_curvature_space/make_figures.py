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

Usage: python analysis/violation_divergence/fork_curvature_space/make_figures.py
       [--game 1235] [--w 8] [--camera-ready]
"""

import argparse
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


def branches(g: dict) -> tuple[int, list[dict]]:
    rows = [{"category": int(c), "is_control": int(k)}
            for c, k in zip(g["category"], g["is_control"])]
    ref = next(i for i, r in enumerate(rows)
               if r["category"] == LEGAL and not r["is_control"])
    return ref, rows


def one_control(g: dict) -> dict:
    """Drop control 2. It is the null *for* control 1 and belongs where control 1
    is measured, not in an illustration of the construction; dropping the row also
    drops it from the plane fit, which is fit on what is drawn."""
    keep = np.flatnonzero(g["is_control"] != 2)
    per_branch = ["category", "is_control", *(name for name, _ in dsc.spaces())]
    return {k: v[keep] if k in per_branch else v for k, v in g.items()}


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
                n_pos: int) -> None:
    """One R^21 point per branch, with its leave-one-position-out jackknife spread.

    A rule is one point, the mean of its members', with the jackknife of that mean;
    the members are drawn faint. φ is applied per member and never to a mean
    trajectory, and the projection is affine, so mean-then-project agrees.
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
    ax.plot([], [], color=MUTED, lw=1.3, label=CONTOUR_LABEL.format(n=n_pos))


def spreads(g: dict, w: int) -> dict[str, float]:
    """Within-rule ÷ between-rule mean pairwise distance, one per variant, in the
    reference space and in full R^21 -- never on the 2-D plane the figure draws.
    `displacement` is the same statistic on the raw activations, the number
    ../fork_and_board reports, so the two figures can be compared."""
    ref, rows = branches(g)
    c = int(g["n_context"])
    post = g["pos"][c:] - g["t0"] >= 0
    Z = g[RAW_SPACE]
    phi = dsc.windowed(Z, w)[:, c:]
    whole = dsc.descriptor(torch.from_numpy(Z[:, c + int(np.flatnonzero(post)[0]):]))
    return {"windowed": spread_ratio(scaled(phi - phi[ref])[:, post], rows),
            "whole": spread_ratio(scaled(whole), rows),
            "displacement": spread_ratio((Z - Z[ref])[:, c:][:, post], rows)}


def windowed_grid(g: dict, w: int, plane: str, camera_ready: bool) -> Path:
    ref, rows = branches(g)
    c = int(g["n_context"])
    r = g["pos"][c:] - g["t0"]
    fig = plt.figure(figsize=(3.9 * NCOL, 3.0 * NCOL))
    for j, (name, _) in enumerate(dsc.spaces()):
        phi = dsc.windowed(g[name], w)[:, c:]
        dphi = phi - phi[ref]
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
        fork_panel(ax, coords, rows, r, legend=(j == 0), named=(plane == "fixed"),
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
    rule_member_key(fig, y=below(fig, leg))
    return save_figure(fig, FIG_DIR / f"fork_r21_windowed_{plane}", camera_ready)


def whole_grid(g: dict, plane: str, camera_ready: bool) -> Path:
    ref, rows = branches(g)
    c = int(g["n_context"])
    k = c + int(np.flatnonzero(g["pos"][c:] - g["t0"] == 0)[0])
    n_pos = len(g["pos"]) - k
    fig, axes = plt.subplots(NCOL, NCOL, figsize=(3.4 * NCOL, 3.2 * NCOL))
    for ax, (name, _) in zip(axes.flat, dsc.spaces()):
        Z = g[name][:, k:]
        phi = dsc.descriptor(torch.from_numpy(Z))
        jack = dsc.jackknife(Z)
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
        point_panel(ax, P, J, rows, n_pos)
        ax.tick_params(labelsize=6.5)
    leg = ordered_legend(fig, axes.flat[0], 3, loc="lower center",
                         bbox_to_anchor=(0.5, -0.045))
    if not camera_ready:
        fig.suptitle(f"One R$^{{21}}$ point per branch, game {g['game']} "
                     f"({plane} plane)", fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    rule_member_key(fig, y=below(fig, leg))
    return save_figure(fig, FIG_DIR / f"fork_r21_whole_{plane}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, default=dsc.W)
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = one_control(dsc.figure_game(args.game, args.w))
    # The ratios are the figures' caption, not the figures': README.md quotes them.
    print(f"within/between rule spread ratio, game {args.game}: "
          + ", ".join(f"{k} {v:.3f}" for k, v in spreads(g, args.w).items()))
    for plane in ("pca", "fixed"):
        print(f"wrote {windowed_grid(g, args.w, plane, args.camera_ready)}")
        print(f"wrote {whole_grid(g, plane, args.camera_ready)}")


if __name__ == "__main__":
    main()
