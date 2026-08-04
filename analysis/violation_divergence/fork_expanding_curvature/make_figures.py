"""The fork drawn in the curvature representation space R^21, in nine spaces.

Same game and same branches as analysis/violation_divergence/game_trajectory,
but the plane is a PCA of the displacement between ReStraV descriptors,
D21(t) = r21(branch, t0..t) - r21(reference, t0..t), instead of a PCA of the
raw activation displacement. The window is expanding and anchored at the branch:
the descriptor at t is "the trajectory since t0", which is causal and never
mixes a pre-branch ply into a post-branch point.

The price is on the figure. R^21 needs 8 sampled positions to fill d1..d7 and
theta1..theta6, so the curve cannot start until 14 plies after t0 -- the shared
prefix and the spray at t0, which are the whole content of the original figure,
are outside the descriptor's domain and are drawn as an explicit undefined
segment. Past that point d1..d7 and theta1..theta6 are the *first* steps since
the branch and never change again, so only the 8 moments move and the curve is
nearly a point. Panel (b) of the companion figure is that number.

Coordinates are rescaled to unit RMS before the plane is fit: d is in
activation-norm units and theta in degrees, so an unscaled PCA of R^21 reports
whichever block is larger. Rescaling is diagonal, so zero displacement is still
the origin and the reference is still the vertical line there.

Usage: python analysis/violation_divergence/fork_expanding_curvature/make_figures.py
       [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, REFERENCE, apply_style, rule_key,
    save_figure)
from spade.othello.taxonomy import CATEGORIES

from cache_features import load
from window import MIN_WINDOW, R21_DIM, STRIDE, expanding, rescale, travel

FIG_DIR = Path(__file__).resolve().parent / "figures"
ELEV, AZIM = 18, -58  # the fork figure's camera, so the two are comparable
CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}
UNDEFINED = "$R^{21}$ undefined\n(< 8 sampled positions since $t_0$)"


def branch_style(category: int, is_control: int) -> dict:
    if is_control:
        return {"color": NULL["color"], "marker": NULL["marker"], "ls": NULL["ls"],
                "lw": NULL["lw"], "zorder": NULL["zorder"], "label": CONTROL_LABEL[is_control]}
    if category == 0:
        return {"color": REFERENCE["color"], "marker": REFERENCE["marker"], "ls": "-",
                "lw": REFERENCE["lw"], "zorder": REFERENCE["zorder"],
                "label": "reference (the game as played)"}
    s = CATEGORY_STYLE[CATEGORIES[category]]
    return {"color": s["color"], "marker": s["marker"], "ls": "-", "lw": 1.5,
            "zorder": 3, "label": s["label"]}


def legend_order(labels: list[str]) -> list[int]:
    """Reference first, nulls last (FIGURE_GUIDELINES 3); taxonomy order between."""
    names = [CATEGORY_STYLE[c]["label"] for c in CATEGORIES[1:]]
    rank = {}
    for i, lab in enumerate(labels):
        if lab.startswith("$R^{21}$ undefined"):
            rank[i] = 0
        elif lab.startswith("observed median"):
            rank[i] = 0.5
        elif lab.startswith("time-shuffled null"):
            rank[i] = 11 + ("range" in lab)
        elif lab.startswith("reference"):
            rank[i] = 1
        elif lab.startswith("branch point"):
            rank[i] = 8
        elif lab.startswith("control"):
            rank[i] = 9 + ("2" in lab)
        else:
            rank[i] = 2 + (names.index(lab) if lab in names else len(names))
    return sorted(range(len(labels)), key=lambda i: rank[i])


def ordered_legend(fig, axes, ncol: int, y: float, title: str):
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        h, l = ax.get_legend_handles_labels()
        handles, labels = handles + h, labels + l
    order = legend_order(labels)
    return fig.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="lower center", ncol=ncol, fontsize=8, frameon=False,
                      bbox_to_anchor=(0.5, y), title=title, title_fontsize=7.5)


def pca2(X: np.ndarray):
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T, (S ** 2 / (S ** 2).sum())[:2]


def plane(F: np.ndarray, k0: int, ref: int):
    """([n, n_points, 2], variance ratio, the sampled-position indices drawn)."""
    D, ks = expanding(F, k0)
    D = rescale(D - D[ref])
    coords, var = pca2(D.reshape(-1, R21_DIM))
    return coords.reshape(D.shape[0], D.shape[1], 2), var, ks


def fork_panel(ax, coords: np.ndarray, rows: list[dict], pos: np.ndarray,
               ks: np.ndarray, t0: int, legend: bool, zlabel: bool = True) -> None:
    lo, hi = coords.min(), coords.max()
    ax.plot([0, 0], [0, 0], [t0, pos[ks[0]]], color=MUTED, lw=3.0, alpha=0.6,
            ls=(0, (1, 1.6)), solid_capstyle="round", zorder=1,
            label=UNDEFINED if legend else None)
    for i, r in enumerate(rows):
        st = branch_style(r["category"], r["is_control"])
        ax.plot(coords[i, :, 0], coords[i, :, 1], pos[ks], ls=st["ls"], color=st["color"],
                lw=st["lw"], marker=st["marker"], ms=3.0, markeredgecolor="white",
                markeredgewidth=0.3, zorder=st["zorder"],
                label=st["label"] if legend else None)
        ax.plot([coords[i, -1, 0]], [coords[i, -1, 1]], [pos[ks[-1]]], marker=st["marker"],
                ms=6, color=st["color"], markeredgecolor="white", markeredgewidth=0.7,
                zorder=st["zorder"] + 1)
    ax.plot([0], [0], [t0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(t0, pos[ks[-1]])
    ax.set_xticks([])
    ax.set_yticks([])
    if zlabel:
        ax.set_zlabel("move index $t$ (window's right edge)", labelpad=-4)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def space_grid(g: dict, rows: list[dict], ref: int, k0: int, camera_ready: bool) -> Path:
    ncol = 3
    nrow = int(np.ceil(len(g["names"]) / ncol))
    fig = plt.figure(figsize=(3.7 * ncol, 3.5 * nrow))
    for j, (name, F) in enumerate(zip(g["names"], g["F"])):
        coords, var, ks = plane(F, k0, ref)
        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        fork_panel(ax, coords, rows, g["pos"], ks, g["t0"],
                   legend=(j == 0), zlabel=(j % ncol == 0))
        ax.set_title(f"{name}  ({var[0]:.0%}/{var[1]:.0%} var)", loc="left", fontsize=8.5)
        ax.set_xlabel("PC1 of $\\Delta_{21}$", labelpad=-8)
        ax.set_ylabel("PC2 of $\\Delta_{21}$", labelpad=-8)
    ordered_legend(fig, fig.axes[0], 4, -0.06,
        f"game {g['game']}, $t_0$={g['t0']}; $\\Delta_{{21}}(t)$ = ReStraV descriptor of "
        f"positions $t_0..t$ (every {STRIDE}nd ply) minus the reference's, coordinates "
        f"rescaled to unit RMS,\nuncentred PCA refit per space (arb. units, not comparable "
        f"across cells); view elev={ELEV}, azim={AZIM}. The descriptor needs "
        f"{MIN_WINDOW} sampled positions, so the curve\nstarts {(MIN_WINDOW - 1) * STRIDE} "
        f"plies after $t_0$ and only the 8 moment coordinates move along it. One game "
        f"(n=1): shape only, not a claim.")
    if not camera_ready:
        fig.suptitle("The fork in the curvature space $R^{21}$, expanding window anchored "
                     f"at $t_0$ (game {g['game']})", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.06, wspace=0.06, hspace=0.16)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.085)
    return save_figure(fig, FIG_DIR / "space_fork_grid_r21_expanding", camera_ready)


def magnitude_panel(ax, F: np.ndarray, rows: list[dict], pos: np.ndarray,
                    k0: int, t0: int, ref: int) -> None:
    D, ks = expanding(F, k0)
    ref21 = np.linalg.norm(D[ref], axis=-1)
    dh = np.linalg.norm(D - D[ref], axis=-1) / ref21
    r = pos[ks] - t0
    ax.axvspan(0, r[0], color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        ax.plot(r, dh[i], ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.5, markeredgecolor="white",
                markeredgewidth=0.3, zorder=st["zorder"], label=st["label"])
    ax.text(0.4, ax.get_ylim()[1] * 0.5, UNDEFINED, fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("position relative to the branch, $t-t_0$")
    ax.set_ylabel(r"$\|\Delta_{21}\|\,/\,\|r_{21}(\mathrm{ref})\|$")


def motion_panel(ax, g: dict, rows: list[dict], k0: int, ref: int) -> None:
    """How far the curve travels, against a null with the temporal order destroyed."""
    keep = [i for i in range(len(rows)) if i != ref]
    jitter = np.linspace(-0.2, 0.2, len(keep))
    for j, F in enumerate(g["F"]):
        obs, null = travel(F, k0, ref)
        ax.fill_between([j - 0.36, j + 0.36], null.min(), null.max(), color=GRID,
                        alpha=0.9, lw=0, zorder=0,
                        label="time-shuffled null, range over branches" if j == 0 else None)
        ax.plot([j - 0.36, j + 0.36], [np.median(null)] * 2, color=NULL["color"],
                ls=NULL["ls"], lw=NULL["lw"], zorder=NULL["zorder"],
                label="time-shuffled null, median (20 shuffles)" if j == 0 else None)
        ax.plot([j - 0.36, j + 0.36], [np.median(obs)] * 2, color=REFERENCE["color"], lw=1.8,
                zorder=6, label="observed median over branches" if j == 0 else None)
        for m, i in enumerate(keep):
            st = branch_style(rows[i]["category"], rows[i]["is_control"])
            ax.plot(j + jitter[m], obs[m], marker=st["marker"], ms=5, color=st["color"],
                    ls="none", markeredgecolor="white", markeredgewidth=0.4,
                    zorder=st["zorder"])
    ax.set_xticks(range(len(g["names"])))
    ax.set_xticklabels(g["names"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("path length after the first point\n$/\\;\\|\\Delta_{21}\\|$ there"
                  "   (higher = more curve)")
    ax.set_ylim(bottom=0)


def companion(g: dict, rows: list[dict], ref: int, k0: int, camera_ready: bool) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    magnitude_panel(ax1, g["F"][0], rows, g["pos"], k0, g["t0"], ref)
    ax1.set_title(f"(a) magnitude in {g['names'][0]}", loc="left", fontsize=10)
    motion_panel(ax2, g, rows, k0, ref)
    ax2.set_title("(b) how far the curve travels, against a time-shuffled null",
                  loc="left", fontsize=10)
    fig.tight_layout()
    ordered_legend(fig, [ax1, ax2], 4, -0.26,
        "One game (n=1), 8 non-reference branches; each marker in (b) is one branch, the "
        "black bar its median. The null recomputes the same\nexpanding window over a random "
        "permutation of the post-$t_0$ positions, so growing the window is the only thing "
        "left that can move the\ndescriptor. Both panels use the expanding window anchored "
        "at $t_0$; coordinates rescaled to unit RMS.")
    if not camera_ready:
        fig.suptitle("What the $R^{21}$ curve can and cannot show", fontsize=11, y=1.04)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.50)
    return save_figure(fig, FIG_DIR / "r21_expanding_magnitude", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = load()
    rows = [{"category": int(c), "is_control": int(s)}
        for c, s in zip(g["category"], g["is_control"])]
    ref = int(np.flatnonzero((g["category"] == 0) & (g["is_control"] == 0))[0])
    k0 = int(np.flatnonzero(g["pos"] == g["t0"])[0])
    print(f"wrote {space_grid(g, rows, ref, k0, args.camera_ready)}")
    print(f"wrote {companion(g, rows, ref, k0, args.camera_ready)}")


if __name__ == "__main__":
    main()
