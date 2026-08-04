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
            one game's nine branches, with a jackknife spread over which sampled
            position is dropped.

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

from spade.figures.embedding_panels import class_ellipse
from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, REFERENCE, apply_style, rule_key,
    save_figure)
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence import curvature_plane as dsc

FIG_DIR = Path(__file__).resolve().parent / "figures"
ELEV, AZIM = 18, -58  # matched to ../game_trajectory so the two grids read alike
NCOL = 3
CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}
# FIGURE_GUIDELINES 1.4: units, and the direction of "good" where it is not obvious.
D_AXIS = "mean step length $d$ (activation norm)"
THETA_AXIS = "mean turning angle θ (deg, ↑ curvier)"
# 3-D panels have no room for the long forms; the units and direction survive.
D_AXIS_3D = "Δ mean $d$ (activation norm)"
THETA_AXIS_3D = "Δ mean θ (deg, ↑ curvier)"
Z_AXIS = "$t-t_0$ (plies)"


def branch_style(category: int, is_control: int) -> dict:
    if is_control:
        # Both controls are nulls and share NULL's gray; the second is drawn open
        # so two series are never distinguishable by their legend text alone.
        return {"color": NULL["color"], "marker": NULL["marker"], "ls": NULL["ls"],
                "lw": NULL["lw"], "zorder": NULL["zorder"],
                "mfc": "none" if is_control == 2 else NULL["color"],
                "label": CONTROL_LABEL[is_control]}
    if category == 0:
        return {"color": REFERENCE["color"], "marker": REFERENCE["marker"], "ls": "-",
                "lw": REFERENCE["lw"], "zorder": REFERENCE["zorder"],
                "mfc": REFERENCE["color"], "label": "reference (the game as played)"}
    s = CATEGORY_STYLE[CATEGORIES[category]]
    return {"color": s["color"], "marker": s["marker"], "ls": "-", "lw": 1.5,
            "zorder": 3, "mfc": s["color"], "label": s["label"]}


def legend_order(labels: list[str]) -> list[int]:
    """Reference first, nulls last (FIGURE_GUIDELINES 3); taxonomy order between."""
    names = [CATEGORY_STYLE[c]["label"] for c in CATEGORIES[1:]]
    rank = {}
    for i, lab in enumerate(labels):
        if lab.startswith("window runs off"):
            rank[i] = 0
        elif lab.startswith("reference"):
            rank[i] = 1
        elif lab.startswith("branch point"):
            rank[i] = 8
        elif lab.startswith("control"):
            rank[i] = 9 + ("2" in lab)
        else:
            rank[i] = 2 + (names.index(lab) if lab in names else len(names))
    return sorted(range(len(labels)), key=lambda i: rank[i])


def ordered_legend(fig, ax, y: float, title: str):
    handles, labels = ax.get_legend_handles_labels()
    order = legend_order(labels)
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, y), title=title, title_fontsize=7.5)


def branches(g: dict) -> tuple[int, list[dict]]:
    rows = [{"category": int(c), "is_control": int(k)}
            for c, k in zip(g["category"], g["is_control"])]
    ref = next(i for i, r in enumerate(rows)
               if r["category"] == 0 and not r["is_control"])
    return ref, rows


def fork_panel(ax, coords: np.ndarray, rows: list[dict], r: np.ndarray,
               legend: bool, named: bool, zlabel: bool) -> None:
    """coords [n_branches, n_pos, 2]; offset from the branch is the vertical axis.

    `named` means the two plane axes carry units of their own (the fixed pair), so
    they get ticks and their own ranges; a PCA plane is isotropic and unitless, so
    it gets one shared range and no ticks.
    """
    k = int(np.flatnonzero(np.isfinite(coords).all(-1).all(0))[0])
    ax.plot([0, 0], [0, 0], [r[0], r[k]], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label="window runs off the board (no descriptor)" if legend else None)
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        seg, z = coords[i, k:], r[k:]
        ax.plot(seg[:, 0], seg[:, 1], z, ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.0, markevery=3, mfc=st["mfc"],
                markeredgecolor="white", markeredgewidth=0.3, zorder=st["zorder"],
                label=st["label"] if legend else None)
        ax.plot([seg[-1, 0]], [seg[-1, 1]], [z[-1]], marker=st["marker"], ms=6,
                color=st["color"], mfc=st["mfc"], markeredgecolor="white",
                markeredgewidth=0.7, zorder=st["zorder"] + 1)
    ax.plot([0], [0], [0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    if named:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=6, pad=-3)
    else:
        lo, hi = np.nanmin(coords), np.nanmax(coords)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_zlim(r[0], r[-1])
    ax.set_zlabel(Z_AXIS, labelpad=-6, fontsize=7)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def point_panel(ax, P: np.ndarray, jack: np.ndarray, rows: list[dict]) -> None:
    """One R^21 point per branch, with its leave-one-position-out jackknife spread."""
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        is_ref = row["category"] == 0 and not row["is_control"]
        class_ellipse(ax, jack[:, i], st["color"], 2.0 if is_ref else 1.2)
        ax.plot(P[i, 0], P[i, 1], ls="none", marker=st["marker"],
                ms=9 if is_ref else 7, color=st["color"], mfc=st["mfc"],
                markeredgecolor="white" if st["mfc"] != "none" else st["color"],
                markeredgewidth=1.0, zorder=st["zorder"] + 3, label=st["label"])


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
            labels = [f"PC{i + 1} ({p.var[i]:.0%} var, {p.lead[i]})" for i in (0, 1)]
        else:
            ax_i = [dsc.names_of(dphi.shape[2]).index(a) for a in dsc.FIXED_AXES]
            coords = dphi[:, :, ax_i]
            labels = [D_AXIS_3D, THETA_AXIS_3D]
        ax = fig.add_subplot(NCOL, NCOL, j + 1, projection="3d")
        ax.set_box_aspect((1, 1, 0.85))
        fork_panel(ax, coords, rows, r, legend=(j == 0), named=(plane == "fixed"),
                   zlabel=(j % NCOL == 0))
        ax.set_xlabel(labels[0], labelpad=-10, fontsize=5.8)
        ax.set_ylabel(labels[1], labelpad=-10, fontsize=5.8)
        ax.set_title(name, loc="left", fontsize=8)
    axis_note = ("plane refit per space (arb. units, not comparable across cells)"
                 if plane == "pca" else
                 "the same named pair of R$^{21}$ axes in every cell; θ in degrees "
                 "is comparable across cells, d is not")
    ordered_legend(fig, fig.axes[0], -0.045,
                   f"game {g['game']}, $t_0$={g['t0']}; Δφ(t) = φ(z_b(t)) − φ(z_ref(t)) "
                   f"in R$^{{21}}$ over the {w} sampled positions (={dsc.STRIDE * (w - 1)} "
                   f"plies) ending at t; {axis_note};\nview elev={ELEV}, azim={AZIM}; "
                   f"every {dsc.STRIDE}nd ply. One game (n=1) and no error bars: an "
                   f"illustration of the construction, not a claim — the separation "
                   f"numbers, with CIs over 40 games, are in the README.")
    if not camera_ready:
        fig.suptitle(f"The fork inside the 21-D ReStraV descriptor space, "
                     f"game {g['game']} ({plane} plane)", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.07,
                        wspace=0.04, hspace=0.18)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.085)
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
            ax.set_xlabel(f"PC1 ({p.var[0]:.0%} var, leads on {p.lead[0]})",
                          fontsize=7.5)
            ax.set_ylabel(f"PC2 ({p.var[1]:.0%} var, leads on {p.lead[1]})",
                          fontsize=7.5)
            ax.set_title(f"{name}  ·  arb. units, refit per space", loc="left",
                         fontsize=8)
        else:
            i = [dsc.names_of(phi.shape[1]).index(a) for a in dsc.FIXED_AXES]
            P, J = phi[:, i], jack[:, :, i]
            ax.set_xlabel(D_AXIS, fontsize=7.5)
            ax.set_ylabel(THETA_AXIS, fontsize=7.5)
        point_panel(ax, P, J, rows)
        if plane == "fixed":
            ax.set_title(name, loc="left", fontsize=8)
        ax.tick_params(labelsize=6.5)
    ordered_legend(fig, axes.flat[0], -0.055,
                   f"game {g['game']}, $t_0$={g['t0']}; one R$^{{21}}$ point per branch "
                   f"over the {n_pos} sampled positions from the branch to $t_0$+24. "
                   f"Ellipses are the 50% contour of a Gaussian fit to the {n_pos} "
                   f"leave-one-position-out jackknife replicates\n— a dispersion over "
                   f"which move enters the descriptor, not a confidence interval, and "
                   f"the only uncertainty one game supports. "
                   + ("PCA refit per space, so cells are comparable in arrangement "
                      "only." if plane == "pca" else
                      "Axes are the same named pair in every cell; θ is in degrees, "
                      "d is on each space's own activation scale."))
    if not camera_ready:
        fig.suptitle(f"One R$^{{21}}$ point per branch, game {g['game']} "
                     f"({plane} plane)", fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.955))
    return save_figure(fig, FIG_DIR / f"fork_r21_whole_{plane}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, default=dsc.W)
    parser.add_argument("--game", type=int, default=dsc.FIGURE_GAME)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = dsc.figure_game(args.game, args.w)
    for plane in ("pca", "fixed"):
        print(f"wrote {windowed_grid(g, args.w, plane, args.camera_ready)}")
        print(f"wrote {whole_grid(g, plane, args.camera_ready)}")


if __name__ == "__main__":
    main()
