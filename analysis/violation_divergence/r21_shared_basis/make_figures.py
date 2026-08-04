"""Can the nine fork panels share one plane? Route the fork through R^21.

`violation_divergence/game_trajectory` draws one game's fork in nine
representation spaces and must disclaim that the planes are refit per space, so
positions mean nothing across cells (FIGURE_GUIDELINES 4.6). The reason is
structural: R^512 and R^8192 have no coordinates in common.

The ReStraV descriptor removes that obstacle. Each space's displacement
trajectory is summarised by the same 21 named components (7 step lengths, 6
turning angles, 4 moments of each; spade/straightness/geometry.py), so "mean θ"
is the same quantity in raw layer 5 and in CosineSpaDE and a single plane can be
applied to every panel. Three planes are drawn and scored against each other:

  refit   per-space top-2 of that space's own descriptors (the status quo)
  shared  one top-2 plane fit on the nine spaces' pooled descriptors
  named   no fit at all: the axes are (mean d, mean θ)

All three act on the same z-scored displacement. Z-scoring is mandatory in R^21:
the components mix degrees with activation-scale lengths whose variances differ
~450x across dictionaries, so an unscaled plane reports units (see
analysis/straightness/feature_pca). It is per space, so a shared plane makes
SHAPES comparable across cells and not absolute magnitudes -- the panels are in
each space's own descriptor SD.

A point is the descriptor of the W=8 sampled positions ending at t: 8 is the
smallest window with a full 21-D vector, since restrav_features slices d[:, :7]
and silently returns a shorter vector below that. At the study's window that
costs the figure its zero-by-construction prefix -- the first window already
straddles t0.

Score: illegal-vs-control centroid distance in pooled within-group SDs, over
anchors t >= t0, against the exact permutation null over the 28 ways to split
the 8 non-reference branches 6/2.

Usage: python analysis/violation_divergence/r21_shared_basis/make_figures.py [--camera-ready]
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, OKABE_ITO, REFERENCE, apply_style,
    rule_key, sae_label, save_figure)
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.straightness.geometry import (
    RESTRAV_FEATURE_NAMES, restrav_features, temporal_geometry)
from spade.violation_divergence.compute_divergence import sae_encoder

FIG_DIR = Path(__file__).resolve().parent / "figures"
CACHE = study_dir("divergence") / "fork_r21" / "game_features.npz"

W = 8                    # smallest window with a full 21-D descriptor (d[:, :7])
STRIDE = 2               # one mover per curve; see the parent sub-study
ELEV, AZIM = 18, -58     # same camera as the parent, so the two read together
WINDOW = (-8, 24)        # the study's analysis window, r = t - t0
FIRST_POS, LAST_POS = 1, 58   # position 0 is the attention sink
ISOMAP_NEIGHBORS = 12
RAW_SPACE = "raw layer 5 (reference space)"
MEAN_D = RESTRAV_FEATURE_NAMES.index("mean d")
MEAN_THETA = RESTRAV_FEATURE_NAMES.index("mean θ")
CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}

# Plane choices are this figure's series. Colors come from paper_style's palette;
# the status-quo activation-space plane is what the R^21 planes are read against,
# so it takes REFERENCE styling.
PLANE_STYLE = {
    "activation refit": {"color": REFERENCE["color"], "marker": REFERENCE["marker"],
                         "ls": (0, (1, 1.4)),
                         "label": "activation space, refit per space (status quo)"},
    "refit":  {"color": OKABE_ITO[0], "marker": "s", "ls": "-",
               "label": "$R^{21}$, plane refit per space"},
    "shared": {"color": OKABE_ITO[1], "marker": "D", "ls": "-",
               "label": "$R^{21}$, one shared fitted plane"},
    "named":  {"color": OKABE_ITO[2], "marker": "^", "ls": "-",
               "label": "$R^{21}$, fixed named axes (mean $d$, mean $\\theta$)"},
}


# ---------------------------------------------------------------- features

def build_cache(path: Path, device: str) -> None:
    """Game 0's branches in every space, at the study's sampled positions."""
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)
    n_types = len(CATEGORIES) + 2
    game = next(g for g in meta["game_id"].unique().tolist()
                if int((meta["game_id"] == g).sum()) == n_types)
    idx = (meta["game_id"] == game).nonzero().flatten()
    t0 = int(meta["t0"][idx[0]])
    p = np.arange(max(FIRST_POS, t0 + WINDOW[0]), min(LAST_POS, t0 + WINDOW[1]) + 1)
    pos = p[(p - t0) % STRIDE == 0]
    Z = acts[idx][:, pos, :].float()

    blobs = {"game": game, "t0": t0, "pos": pos, RAW_SPACE: Z.numpy(),
             "category": np.array([int(meta["category"][i]) for i in idx]),
             "is_control": np.array([int(meta["is_control"][i]) for i in idx])}
    for e in seed42_entries():
        if e.config in EXCLUDED_SAE:
            continue
        with torch.no_grad():
            f = sae_encoder(e, device)(Z.flatten(0, 1).to(device)).unflatten(0, Z.shape[:2])
        blobs[sae_label(e.config.replace("Trainer-othello", ""))] = f.cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **blobs)


def game_features(device: str) -> dict:
    if not CACHE.exists():
        build_cache(CACHE, device)
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


def space_names() -> list[str]:
    return [RAW_SPACE] + [sae_label(e.config.replace("Trainer-othello", ""))
                          for e in seed42_entries() if e.config not in EXCLUDED_SAE]


# ---------------------------------------------------------------- R^21

def descriptors(F: np.ndarray) -> np.ndarray:
    """[n, T, width] -> [n, T-W+1, 21]; a point is the W samples ending at t."""
    x = torch.from_numpy(F).unfold(1, W, 1).permute(0, 1, 3, 2).flatten(0, 1)
    f = restrav_features(*temporal_geometry(x))
    assert f.shape[1] == len(RESTRAV_FEATURE_NAMES), f"width {f.shape[1]}, need 21"
    assert torch.isfinite(f).all(), "a zero-length step left an undefined angle"
    return f.reshape(F.shape[0], -1, f.shape[1]).numpy()


def displacement21(F: np.ndarray, ref: int) -> np.ndarray:
    """[n, nw, 21] descriptor displacement from the reference, in per-space SDs.

    The mean cancels in the difference, so z-scoring leaves the reference exactly
    at the origin and only rescales each named component.
    """
    phi = descriptors(F)
    sd = phi.reshape(-1, phi.shape[-1]).std(0)
    assert (sd > 0).all(), f"constant component: {np.array(RESTRAV_FEATURE_NAMES)[sd == 0]}"
    return (phi - phi[ref]) / sd


def top2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uncentred top-2 plane (basis [2, k], variance ratios [2]).

    Uncentred because zero displacement is a real point: centring would move the
    reference off the origin it is drawn at.
    """
    _, s, vt = np.linalg.svd(X.reshape(-1, X.shape[-1]), full_matrices=False)
    v = vt[:2]
    sign = np.sign(v[np.arange(2), np.abs(v).argmax(1)])
    return v * sign[:, None], (s ** 2 / (s ** 2).sum())[:2]


def captured(X: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of one space's descriptor variance an orthonormal plane keeps."""
    flat = X.reshape(-1, X.shape[-1])
    return float(((flat @ basis.T) ** 2).sum() / (flat ** 2).sum())


def named_basis() -> np.ndarray:
    return np.eye(len(RESTRAV_FEATURE_NAMES))[[MEAN_D, MEAN_THETA]]


def activation_plane(F: np.ndarray, ref: int) -> np.ndarray:
    """The status-quo comparison: top-2 of the raw displacement, [n, T, 2]."""
    D = F - F[ref]
    basis, _ = top2(D)
    return D @ basis.T


def isomap(X: np.ndarray, fitted=None):
    """Shared nonlinear embedding: fit on the non-degenerate pooled rows.

    The reference is exactly zero at every anchor, and duplicate rows form a
    disconnected clique in the kNN graph, so they are transformed, not fit.
    """
    from sklearn.manifold import Isomap
    flat = X.reshape(-1, X.shape[-1])
    if fitted is None:
        fitted = Isomap(n_neighbors=ISOMAP_NEIGHBORS, n_components=2, n_jobs=-1)
        fitted.fit(flat[np.linalg.norm(flat, axis=1) > 0])
    return fitted.transform(flat).reshape(*X.shape[:-1], 2), fitted


# ---------------------------------------------------------------- score

def separation(P: np.ndarray, ill: np.ndarray, ctl: np.ndarray) -> float:
    """Illegal-vs-control centroid distance in pooled within-group SDs.

    The pooled SD is per coordinate -- sum of squared deviations over
    (n - 2) * dim -- so a 2-D plane and the full 21-D space are on one scale.
    """
    a, b = P[ill].reshape(-1, P.shape[-1]), P[ctl].reshape(-1, P.shape[-1])
    ss = ((a - a.mean(0)) ** 2).sum() + ((b - b.mean(0)) ** 2).sum()
    s = np.sqrt(ss / ((len(a) + len(b) - 2) * P.shape[-1]))
    return float(np.linalg.norm(a.mean(0) - b.mean(0)) / s)


def permutation_null(P: np.ndarray, branches: np.ndarray) -> np.ndarray:
    """Separation under every 6/2 relabelling of the non-reference branches (n=28)."""
    return np.array([separation(P, np.array([b for b in branches if b not in pair]),
                                np.array(pair))
                     for pair in combinations(branches, 2)])


# ---------------------------------------------------------------- panels

def branch_style(category: int, is_control: int) -> dict:
    if is_control:
        return {"color": NULL["color"], "marker": NULL["marker"], "ls": NULL["ls"],
                "lw": NULL["lw"], "zorder": NULL["zorder"],
                "label": CONTROL_LABEL[is_control]}
    if category == 0:
        return {"color": REFERENCE["color"], "marker": REFERENCE["marker"], "ls": "-",
                "lw": REFERENCE["lw"], "zorder": REFERENCE["zorder"],
                "label": "reference (the game as played)"}
    s = CATEGORY_STYLE[CATEGORIES[category]]
    return {"color": s["color"], "marker": s["marker"], "ls": "-", "lw": 1.5,
            "zorder": 3, "label": s["label"]}


def legend_order(labels: list[str]) -> list[int]:
    """Reference first, nulls last (FIGURE_GUIDELINES 3); taxonomy order between."""
    rank = {}
    for i, lab in enumerate(labels):
        if lab.startswith("shared prefix"):
            rank[i] = 0
        elif lab.startswith("reference"):
            rank[i] = 1
        elif lab.startswith("branch point"):
            rank[i] = 8
        elif lab.startswith("control"):
            rank[i] = 9 + ("2" in lab)
        else:
            names = [CATEGORY_STYLE[c]["label"] for c in CATEGORIES[1:]]
            rank[i] = 2 + (names.index(lab) if lab in names else len(names))
    return sorted(range(len(labels)), key=lambda i: rank[i])


def ordered_legend(fig, ax, ncol: int, y: float, title: str):
    handles, labels = ax.get_legend_handles_labels()
    order = legend_order(labels)
    return fig.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="lower center", ncol=ncol, fontsize=8, frameon=False,
                      bbox_to_anchor=(0.5, y), title=title, title_fontsize=7.5)


def fork_panel(ax, coords: np.ndarray, meta: dict, lim, ticks: bool,
               legend: bool, zlabel: bool) -> None:
    """coords [n_branches, n_anchors, 2]; move index is the vertical axis."""
    pos, anchors, t0 = meta["pos"], meta["anchors"], meta["t0"]
    # zero displacement, which is the origin in every plane but Isomap's
    ox, oy = coords[meta["ref"], 0]
    ax.plot([ox, ox], [oy, oy], [pos[0], anchors[0]], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label=f"shared prefix, then the {W}-sample warm-up" if legend else None)
    for i in range(coords.shape[0]):
        st = branch_style(meta["category"][i], meta["is_control"][i])
        ax.plot(coords[i, :, 0], coords[i, :, 1], anchors, ls=st["ls"], color=st["color"],
                lw=st["lw"], marker=st["marker"], ms=3.0, markevery=3,
                markeredgecolor="white", markeredgewidth=0.3, zorder=st["zorder"],
                label=st["label"] if legend else None)
        ax.plot([coords[i, -1, 0]], [coords[i, -1, 1]], [anchors[-1]], marker=st["marker"],
                ms=6, color=st["color"], markeredgecolor="white", markeredgewidth=0.7,
                zorder=st["zorder"] + 1)
    ax.plot([ox], [oy], [t0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9, label="branch point $t_0$" if legend else None)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(pos[0], anchors[-1])
    if ticks:
        # inside the limits, so the x and y labels do not collide at the front corner
        for axis in (ax.set_xticks, ax.set_yticks):
            axis(np.linspace(0.65 * lim[0], 0.65 * lim[1], 3).round())
        ax.tick_params(axis="both", labelsize=6, pad=-3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if zlabel:
        ax.set_zlabel("move index $t$", labelpad=-4)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def space_grid(plane: str, coords: dict[str, np.ndarray], meta: dict, note: str,
               axis_labels: tuple[str, str], titles: dict[str, str], shared_lim: bool,
               camera_ready: bool) -> Path:
    ncol = 3
    nrow = -(-len(coords) // ncol)
    lim = None
    if shared_lim:
        span = max(np.abs(c).max() for c in coords.values()) * 1.05
        lim = (-span, span)
    fig = plt.figure(figsize=(3.7 * ncol, 3.5 * nrow))
    for j, (name, c) in enumerate(coords.items()):
        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        panel_lim = lim or (c.min() * 1.05, c.max() * 1.05)
        fork_panel(ax, c, meta, panel_lim, ticks=shared_lim,
                   legend=(j == 0), zlabel=(j % ncol == 0))
        ax.set_xlabel(axis_labels[0], labelpad=-6 + 4 * shared_lim, fontsize=7.5)
        ax.set_ylabel(axis_labels[1], labelpad=-6 + 4 * shared_lim, fontsize=7.5)
        ax.set_title(titles[name], loc="left", fontsize=8.5)
    ordered_legend(fig, fig.axes[0], 4, -0.06, note)
    if not camera_ready:
        fig.suptitle(f"The fork in the 21-D curvature descriptor, game {meta['game']} "
                     f"({plane} plane)", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.06,
                        wspace=0.06, hspace=0.16)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.105)
    return save_figure(fig, FIG_DIR / f"space_fork_grid_r21_{plane}", camera_ready)


def plane_series(ax, x, values: dict[str, list[float]], label: bool = True) -> None:
    for key, style in PLANE_STYLE.items():
        ax.plot(x, values[key], color=style["color"], marker=style["marker"],
                ls=style["ls"], ms=5, lw=1.6, markeredgecolor="white",
                markeredgewidth=0.5, label=style["label"] if label else None,
                zorder=5 if key == "activation refit" else 4)


def separation_figure(scores: dict[str, list[float]], nulls: dict[str, np.ndarray],
                      names: list[str], camera_ready: bool) -> Path:
    """The number the grids cannot show, against the null that explains most of it.

    R^21 raises the raw gap and the null by nearly the same factor, so (b) divides
    each plane's score by its own permutation-null median.
    """
    x = np.arange(len(names))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    plane_series(ax, x, scores)
    for key, face in [("activation refit", "none"), ("shared", NULL["color"])]:
        lo, mid, hi = np.percentile(nulls[key], [5, 50, 95], axis=1)
        ax.fill_between(x, lo, hi, facecolor=face, alpha=0.18 if face != "none" else 1.0,
                        lw=0.5, zorder=1, hatch=None if face != "none" else "///",
                        edgecolor=NULL["color"],
                        label="null, 5–95% (" + ("activation plane)" if face == "none"
                                                 else "$R^{21}$ planes)"))
        ax.plot(x, mid, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=1)
    ax.set_ylabel("illegal vs. control centroid gap\n(pooled within-group SDs), ↑ better")
    ax.set_title("(a) raw separation, against the null of the same plane",
                 loc="left", fontsize=9.5)

    plane_series(bx, x, {k: np.array(scores[k]) / np.median(nulls[k], axis=1)
                         for k in PLANE_STYLE}, label=False)
    bx.axhline(1.0, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=1,
               label="a random 6/2 relabelling of the same 8 branches (null median)")
    bx.set_ylabel("gap ÷ the plane's own null median")
    bx.set_title("(b) the same, with each plane's null divided out",
                 loc="left", fontsize=9.5)
    for a in (ax, bx):
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        a.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    both = [h + b for h, b in zip(ax.get_legend_handles_labels(),
                                  bx.get_legend_handles_labels())]
    fig.legend(*both, loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.34))
    if not camera_ready:
        fig.suptitle("$R^{21}$ moves the branches apart faster than it moves the null",
                     fontsize=10.5)
    fig.tight_layout()
    fig.text(0.5, -0.44, "one game (n=1), anchors $t\\geq t_0$: 6 illegal x 10 windows "
             "vs 2 control x 10 windows per space. The null is exact — all 28 ways of "
             "relabelling the 8 non-reference branches 6/2 — so the smallest attainable "
             "p is 1/28 = 0.04.", ha="center", va="top", fontsize=7.5, color=MUTED)
    return save_figure(fig, FIG_DIR / "separation_by_plane", camera_ready)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = game_features(select_device())
    names = space_names()
    meta = {k: g[k] for k in ("game", "t0", "pos", "category", "is_control")}
    meta["anchors"] = meta["pos"][W - 1:]
    ref = meta["ref"] = int(
        np.flatnonzero((meta["category"] == 0) & (meta["is_control"] == 0))[0])
    branches = np.array([i for i in range(len(meta["category"])) if i != ref])
    ill = np.array([i for i in branches if not meta["is_control"][i]])
    ctl = np.array([i for i in branches if meta["is_control"][i]])
    keep = meta["anchors"] >= meta["t0"]

    X = {name: displacement21(g[name], ref) for name in names}
    n, nw, k = X[RAW_SPACE].shape
    print(f"game {meta['game']}, t0 {meta['t0']}, positions {meta['pos'][0]}-"
          f"{meta['pos'][-1]}, W={W} -> {nw} anchors {meta['anchors'][0]}-"
          f"{meta['anchors'][-1]}; {n * nw} rows per space in R^{k}, "
          f"{int(keep.sum())} anchors >= t0")

    shared, shared_var = top2(np.stack(list(X.values())))
    refit = {name: top2(X[name]) for name in names}
    coords = {"refit": {name: X[name] @ refit[name][0].T for name in names},
              "shared": {name: X[name] @ shared.T for name in names},
              "named": {name: X[name] @ named_basis().T for name in names}}
    coords["isomap"] = dict(zip(names, isomap(np.stack([X[nm] for nm in names]))[0]))
    act = {name: activation_plane(g[name], ref) for name in names}
    coords["activation refit"] = {name: act[name][:, W - 1:] for name in names}

    scores = {p: [separation(c[name][:, keep], ill, ctl) for name in names]
              for p, c in coords.items()}
    scores["full 21-D"] = [separation(X[name][:, keep], ill, ctl) for name in names]
    scores["activation, all pos >= t0"] = [
        separation(act[name][:, meta["pos"] >= meta["t0"]], ill, ctl) for name in names]
    nulls = {p: np.array([permutation_null(c[name][:, keep], branches) for name in names])
             for p, c in coords.items()}

    cols = ["activation, all pos >= t0", "activation refit", "refit", "shared",
            "named", "isomap", "full 21-D"]
    print(f"\n{'space':34}" + "".join(f"{c:>26}" for c in cols))
    for i, name in enumerate(names):
        print(f"{name:34}" + "".join(f"{scores[c][i]:26.2f}" for c in cols))
    print(f"\n{'space':34}{'refit var':>10}{'shared var':>11}{'named var':>10}"
          + "".join(f"{'null50(' + p + ')':>20}{'p(' + p + ')':>14}" for p in nulls))
    for i, name in enumerate(names):
        cells = [f"{np.median(nulls[p][i]):20.2f}"
                 f"{(nulls[p][i] >= scores[p][i]).mean():14.2f}" for p in nulls]
        print(f"{name:34}{refit[name][1].sum():10.2f}"
              f"{captured(X[name], shared):11.2f}{captured(X[name], named_basis()):10.2f}"
              + "".join(cells))
    lead = [RESTRAV_FEATURE_NAMES[j] for j in np.abs(shared).argmax(1)]
    print(f"\nshared plane leads on {lead[0]} (PC1) and {lead[1]} (PC2)")
    print(f"\n{'space':34}{'SD mean d':>11}{'SD mean θ (deg)':>17}{'1-D sep mean d':>16}"
          f"{'1-D sep mean θ':>16}{'Δmean θ illegal':>17}{'Δmean θ control':>17}")
    for name in names:
        phi = descriptors(g[name])
        sd = phi.reshape(-1, 21).std(0)
        dtheta = (phi - phi[ref])[:, keep, MEAN_THETA]
        one_d = [separation(X[name][:, keep][:, :, [c]], ill, ctl)
                 for c in (MEAN_D, MEAN_THETA)]
        print(f"{name:34}{sd[MEAN_D]:11.3f}{sd[MEAN_THETA]:17.2f}{one_d[0]:16.2f}"
              f"{one_d[1]:16.2f}{dtheta[ill].mean():17.2f}{dtheta[ctl].mean():17.2f}")

    n_state = (f"one game (n=1): {len(branches)} non-reference branches x {nw} windows "
               f"= {n * nw} rows per space in $R^{{{k}}}$ — a shape illustration, not a "
               f"claim; the separation each plane keeps is in analysis/violation_divergence/r21_shared_basis.")
    common = (f"game {meta['game']}, $t_0$={meta['t0']}; a point is the ReStraV "
              f"descriptor of the {W} sampled positions (every {STRIDE}nd ply) ending "
              f"at $t$, minus the reference's, z-scored per space; view elev={ELEV}, "
              f"azim={AZIM}.\n")
    grids = {
        "refit": ("Top-2 plane REFIT PER SPACE: arb. units, not comparable across "
                  "cells — only the shape of the fork is.", ("PC1", "PC2"), False,
                  {name: f"{name}  ({refit[name][1][0]:.0%}/{refit[name][1][1]:.0%} var)"
                   for name in names}),
        "shared": ("ONE plane, fit on the nine spaces' pooled descriptors and applied "
                   "unchanged, on shared axes: every cell carries the same 21 named "
                   "coordinates, so arrangement and direction ARE comparable cell to "
                   "cell. Units are each space's own descriptor SD, so a longer "
                   "excursion is not a larger absolute change.",
                   (f"shared PC1 ({shared_var[0]:.0%} of pooled var)",
                    f"shared PC2 ({shared_var[1]:.0%} of pooled var)"), True,
                   {name: name for name in names}),
        "named": ("NO FIT: the axes are two named components of the descriptor, on "
                  "shared axes. Comparable cell to cell in the same sense as the shared "
                  "plane — same coordinates, each space's own SD as the unit.",
                  ("$\\Delta$ mean $d$ (SD)", "$\\Delta$ mean $\\theta$ (SD)"), True,
                  {name: name for name in names}),
        "isomap": (f"ONE Isomap ({ISOMAP_NEIGHBORS} neighbours) fit on the pooled "
                   "non-degenerate rows and applied to every cell, on shared axes. "
                   "Isomap has no origin and no variance ratio: axes are arb. units.",
                   ("Isomap-1 (arb. units)", "Isomap-2 (arb. units)"), True,
                   {name: name for name in names}),
    }
    for plane, (note, labels, shared_lim, titles) in grids.items():
        out = space_grid(plane, coords[plane], meta, f"{common}{note}\n{n_state}",
                         labels, titles, shared_lim, args.camera_ready)
        print(f"wrote {out}")
    print(f"wrote {separation_figure(scores, nulls, names, args.camera_ready)}")


if __name__ == "__main__":
    main()
