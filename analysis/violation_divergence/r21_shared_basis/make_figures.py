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

`run_from_reference` answers the other question the grids raise and none of them
can: how far a branch runs from the reference, and how much of that run each
plane keeps. It is a ratio to the legal control's run, because every field is
z-scored per space and `refit` is fit per space, so nothing absolute carries
across cells. (a) puts the four planes against full R^21; (b) splits full R^21
into its 11 step-length and 10 turning-angle components, which disagree. Isomap
is in neither: it is not a linear image of R^21 and its reference is not at the
origin, so a distance in it is not a share of anything.

--variant rollout reads the resampled-continuation copy of the study, where every
branch INCLUDING the reference samples its own K=4 suffixes instead of replaying
the source game's. A branch is then K rows, and the mean over them is taken AFTER
phi -- phi is nonlinear, so averaging the four trajectories first would describe a
path no rollout took. The plane projection is linear, so it does not care which
side of it the mean is on; the descriptor does. Every anchor this figure draws is
at r >= +6, which is deep in the regime the rollout study found uninformative, so
the rollout grids are a shape check, not a result. The anchors are labelled by
move index, so nothing in the drawn window shows that -- ROLLOUT_CAVEAT is the one
line that does, and the rest of the caption is README.md's.

Usage: python analysis/violation_divergence/r21_shared_basis/make_figures.py
           [--variant {spliced,rollout}] [--camera-ready]
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.figures.paper_style import (
    CATEGORY_STYLE, FULL_STYLE, GRID, INK, MUTED, NULL, PLANE_STYLE, QUANTITY_STYLE,
    REFERENCE, apply_style, rule_key, sae_label, save_figure)
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.straightness.geometry import RESTRAV_FEATURE_NAMES
from spade.violation_divergence.branches import complete_games
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.fork_plane import RAW_SPACE, positions, spaces
from spade.violation_divergence.fork_r21 import (
    BLOCKS, MEAN_D, MEAN_THETA, W, activation_plane, descriptors, displacement21,
    fold_rollouts, named_basis, run_ratio, top2)

FIG_DIR = Path(__file__).resolve().parent / "figures"
VARIANTS = {"spliced": "", "rollout": "rollout"}   # CLI name -> variant_dir() suffix
# Nothing in the drawn window says this: the anchors are labelled by move index t,
# so a reader cannot see that every one of them is past the r >= 8 saturation the
# rollout read-out found. One line; the numbers are in README.
ROLLOUT_CAVEAT = ("\nrollout: every anchor drawn is at $r\\geq+6$, where illegal and "
                  "control have already saturated — shape, not a separation.")

ELEV, AZIM = 18, -58     # same camera as the parent, so the two read together
ISOMAP_NEIGHBORS = 12
CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}

RUN_AXIS = ("run from the reference ÷ the control's\n(log, ↑ further than another "
            "legal move)")
RUN_TICKS = [0.25, 0.5, 1, 2, 4, 8]    # a ratio axis, so doubling steps


# ---------------------------------------------------------------- features

def cache_path(variant: str) -> Path:
    """Variant-scoped: an unscoped path would let a rollout run overwrite the
    spliced features and silently redraw every spliced figure."""
    return variant_dir("divergence", variant=variant) / "fork_r21" / "game_features.npz"


def build_cache(path: Path, device: str, variant: str) -> None:
    """Game 0's branches in every space, at the study's sampled positions.

    Under --variant rollout a branch is K contiguous rows, so this caches all of
    them and the `rollout` column that says so; the mean over them is taken after
    the descriptor, not here.
    """
    base = variant_dir("divergence", variant=variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # mmap: the rollout activation file is 4 GB and one game is 36 of its rows.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    game = complete_games(meta)[0]
    idx = (meta["game_id"] == game).nonzero().flatten()
    t0 = int(meta["t0"][idx[0]])
    pos = positions(t0)
    Z = acts[idx][:, pos, :].float()

    blobs = {"game": game, "t0": t0, "pos": pos, RAW_SPACE: Z.numpy(),
             "category": np.array([int(meta["category"][i]) for i in idx]),
             "is_control": np.array([int(meta["is_control"][i]) for i in idx])}
    if "rollout" in meta:
        blobs["rollout"] = meta["rollout"][idx].numpy()
    for e in seed42_entries():
        if e.config in EXCLUDED_SAE:
            continue
        with torch.no_grad():
            f = sae_encoder(e, device)(Z.flatten(0, 1).to(device)).unflatten(0, Z.shape[:2])
        blobs[sae_label(e.config.replace("Trainer-othello", ""))] = f.cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **blobs)


def space_names() -> list[str]:
    return [name for name, _ in spaces()]


def game_features(device: str, variant: str = "") -> dict:
    path = cache_path(variant)
    if not path.exists():
        build_cache(path, device, variant)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def captured(X: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of one space's descriptor variance an orthonormal plane keeps."""
    flat = X.reshape(-1, X.shape[-1])
    return float(((flat @ basis.T) ** 2).sum() / (flat ** 2).sum())


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

def run_args(meta: dict) -> dict:
    """The three fields `fork_r21.run_ratio` reads, out of this figure's meta."""
    return {k: meta[k] for k in ("ref", "anchors", "t0")}


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
               camera_ready: bool, suffix: str = "") -> Path:
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
    # The legend box grows upward from its anchor, so a longer note would climb into
    # the bottom row's axis labels; drop the anchor by the lines the note gained.
    drop = 0.011 * note.count("\n")
    ordered_legend(fig, fig.axes[0], 4, -0.05 - drop, note)
    if not camera_ready:
        fig.suptitle(f"The fork in the 21-D curvature descriptor, game {meta['game']} "
                     f"({plane} plane)", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.06,
                        wspace=0.06, hspace=0.16)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.065 - drop)
    return save_figure(fig, FIG_DIR / f"space_fork_grid_r21_{plane}{suffix}", camera_ready)


def plane_series(ax, x, values: dict[str, list[float]], label: bool = True) -> None:
    for key, style in PLANE_STYLE.items():
        ax.plot(x, values[key], color=style["color"], marker=style["marker"],
                ls=style["ls"], ms=5, lw=1.6, markeredgecolor="white",
                markeredgewidth=0.5, label=style["label"] if label else None,
                zorder=5 if key == "activation refit" else 4)


def separation_figure(scores: dict[str, list[float]], nulls: dict[str, np.ndarray],
                      names: list[str], camera_ready: bool, suffix: str = "",
                      foot: str = "", title: str = "$R^{21}$ moves the branches apart "
                                                   "faster than it moves the null") -> Path:
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
               bbox_to_anchor=(0.5, -0.26))
    if not camera_ready:
        fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    fig.text(0.5, -0.32, "one game (n=1): 6 illegal vs 2 control branches, 10 windows "
             "each, per space." + foot, ha="center", va="top", fontsize=7.5, color=MUTED)
    return save_figure(fig, FIG_DIR / f"separation_by_plane{suffix}", camera_ready)


def run_figure(coords: dict, X: dict, names: list[str], ill: np.ndarray,
               c1: int, c2: int, meta: dict, camera_ready: bool, suffix: str,
               foot: str) -> Path:
    """The number the fork grids make the reader guess at: how far a branch runs.

    (a) asks how much of that run each candidate plane keeps; (b) asks what the
    single R^21 number is averaging over, and answers that its two halves
    disagree. Isomap is absent from both on purpose: its embedding is not a
    linear image of R^21 and puts the reference nowhere in particular, so a
    distance in it is not a share of anything (README).
    """
    x = np.arange(len(names))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    series = [(PLANE_STYLE[k], [run_ratio(coords[k][n], ill, c1, **run_args(meta)) for n in names])
              for k in PLANE_STYLE]
    series.append((FULL_STYLE, [run_ratio(X[n], ill, c1, **run_args(meta)) for n in names]))
    for st, y in series:
        ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"], ms=5,
                lw=st.get("lw", 1.6), markeredgecolor="white", markeredgewidth=0.5,
                label=st["label"], zorder=5 if st is FULL_STYLE else 4)
    ax.set_title("(a) how much of the run each candidate plane keeps",
                 loc="left", fontsize=9.5)

    members = np.array([[run_ratio(X[n], np.array([i]), c1, **run_args(meta)) for n in names]
                        for i in ill])
    bx.plot(x, members.T, color=MUTED, lw=0.7, alpha=0.38, zorder=2)
    bx.plot([], [], color=MUTED, lw=0.7, alpha=0.38,
            label="one illegal category (the 6 the mean is over)")
    for key, st in (("full", FULL_STYLE), ("d", QUANTITY_STYLE["d"]),
                    ("theta", QUANTITY_STYLE["theta"])):
        cols = slice(None) if key == "full" else BLOCKS[key]
        y = [run_ratio(X[n][:, :, cols], ill, c1, **run_args(meta)) for n in names]
        bx.plot(x, y, color=st["color"], marker=st["marker"], ls=st.get("ls", "-"),
                ms=5, lw=st.get("lw", 1.6), markeredgecolor="white",
                markeredgewidth=0.5, zorder=4, label=st["label"] if key == "full"
                else f"{st['label']} only ({len(BLOCKS[key])} of the 21)")
    bx.set_title("(b) the two halves of $R^{21}$ disagree about the same branches",
                 loc="left", fontsize=9.5)

    for a in (ax, bx):
        a.axhline(1.0, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"], zorder=1,
                  label="the legal control itself (1 by construction)")
        a.plot(x, [run_ratio(X[n], np.array([c2]), c1, **run_args(meta)) for n in names],
               color=NULL["color"], marker=NULL["marker"], ls="none", ms=5, zorder=3,
               label="a third legal move ÷ the control, full $R^{21}$ (empirical null)")
        a.set_yscale("log")
        a.set_yticks(RUN_TICKS)
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.minorticks_off()
        a.set_ylabel(RUN_AXIS, fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
        a.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    both = [h + b for h, b in zip(ax.get_legend_handles_labels(),
                                  bx.get_legend_handles_labels())]
    seen = {lab: h for h, lab in zip(*both)}
    # Nulls last (FIGURE_GUIDELINES 3); the sort is stable, so everything else keeps
    # the order it was drawn in, which is the order the panels read in.
    order = sorted(seen, key=lambda lab: "null" in lab or "control itself" in lab)
    fig.legend([seen[lab] for lab in order], order, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.26))
    if not camera_ready:
        fig.suptitle("How far an illegal branch runs from the reference, and how "
                     "much of it survives the plane", fontsize=10.5)
    fig.tight_layout()
    r = meta["anchors"] - meta["t0"]
    # A ratio axis with no absolute unit, and a reference that is not a mark: both
    # decide how the panel reads, so both stay in front of the reader (§4.2, §4.6).
    fig.text(0.5, -0.30, f"one game (n=1), {int((r >= 0).sum())} anchors "
             f"$t-t_0$ = {r[r >= 0][0]}…{r[-1]}: ratios, so only the within-space "
             f"contrast carries across cells — a mark twice as high is not twice the "
             f"absolute change. The reference is 0 in every field by construction."
             + foot, ha="center", va="top", fontsize=7.5, color=MUTED)
    return save_figure(fig, FIG_DIR / f"run_from_reference{suffix}", camera_ready)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="spliced")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    g = game_features(select_device(), VARIANTS[args.variant])
    suffix = "" if args.variant == "spliced" else f"_{args.variant}"
    n_roll = int(g["rollout"].max()) + 1 if "rollout" in g else 1
    names = space_names()
    meta = {k: g[k] for k in ("game", "t0", "pos")}
    meta["category"], meta["is_control"] = g["category"][::n_roll], g["is_control"][::n_roll]
    meta["anchors"] = meta["pos"][W - 1:]
    ref = meta["ref"] = int(
        np.flatnonzero((meta["category"] == 0) & (meta["is_control"] == 0))[0])
    branches = np.array([i for i in range(len(meta["category"])) if i != ref])
    ill = np.array([i for i in branches if not meta["is_control"][i]])
    ctl = np.array([i for i in branches if meta["is_control"][i]])
    keep = meta["anchors"] >= meta["t0"]

    X = {name: displacement21(g[name], ref, n_roll) for name in names}
    n, nw, k = X[RAW_SPACE].shape
    print(f"game {meta['game']}, t0 {meta['t0']}, positions {meta['pos'][0]}-"
          f"{meta['pos'][-1]}, W={W} -> {nw} anchors {meta['anchors'][0]}-"
          f"{meta['anchors'][-1]}; {n * nw} rows per space in R^{k}, "
          f"{int(keep.sum())} anchors >= t0"
          + (f"; {args.variant}, mean over K={n_roll} rollouts taken after phi"
             if n_roll > 1 else ""))

    shared, shared_var = top2(np.stack(list(X.values())))
    refit = {name: top2(X[name]) for name in names}
    coords = {"refit": {name: X[name] @ refit[name][0].T for name in names},
              "shared": {name: X[name] @ shared.T for name in names},
              "named": {name: X[name] @ named_basis().T for name in names}}
    coords["isomap"] = dict(zip(names, isomap(np.stack([X[nm] for nm in names]))[0]))
    act = {name: activation_plane(g[name], ref, n_roll) for name in names}
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
        phi = fold_rollouts(descriptors(g[name]), n_roll)
        sd = phi.reshape(-1, 21).std(0)
        dtheta = (phi - phi[ref])[:, keep, MEAN_THETA]
        one_d = [separation(X[name][:, keep][:, :, [c]], ill, ctl)
                 for c in (MEAN_D, MEAN_THETA)]
        print(f"{name:34}{sd[MEAN_D]:11.3f}{sd[MEAN_THETA]:17.2f}{one_d[0]:16.2f}"
              f"{one_d[1]:16.2f}{dtheta[ill].mean():17.2f}{dtheta[ctl].mean():17.2f}")

    # What each grid's note has to carry is §4.6's comparability verdict, which
    # differs per plane and cannot be looked up while looking at the cells. The
    # descriptor, the window, the camera and where the K-mean is taken are the
    # caption's, and live in README.md.
    n_state = f"\none game (n=1), {len(branches)} non-reference branches: shape only."
    rollout_note = "" if n_roll == 1 else ROLLOUT_CAVEAT
    grids = {
        "refit": ("top-2 plane refit per space: arb. units, not comparable across "
                  "cells — only the shape of the fork is.", ("PC1", "PC2"), False,
                  {name: f"{name}  ({refit[name][1][0]:.0%}/{refit[name][1][1]:.0%} var)"
                   for name in names}),
        "shared": ("one plane fit on the pooled descriptors and applied unchanged: "
                   "arrangement and direction are comparable cell to cell, magnitude "
                   "is not (the unit is each space's own descriptor SD).",
                   (f"shared PC1 ({shared_var[0]:.0%} of pooled var)",
                    f"shared PC2 ({shared_var[1]:.0%} of pooled var)"), True,
                   {name: name for name in names}),
        "named": ("no fit: two named descriptor components, comparable cell to cell "
                  "in arrangement; magnitude is each space's own SD and is not.",
                  ("$\\Delta$ mean $d$ (SD)", "$\\Delta$ mean $\\theta$ (SD)"), True,
                  {name: name for name in names}),
        "isomap": (f"one Isomap ({ISOMAP_NEIGHBORS} neighbours) fit on the pooled "
                   "non-degenerate rows: arb. units, no origin and no variance ratio.",
                   ("Isomap-1 (arb. units)", "Isomap-2 (arb. units)"), True,
                   {name: name for name in names}),
    }
    for plane, (note, labels, shared_lim, titles) in grids.items():
        out = space_grid(plane, coords[plane], meta, f"{note}{n_state}{rollout_note}",
                         labels, titles, shared_lim, args.camera_ready, suffix)
        print(f"wrote {out}")
    foot = "" if n_roll == 1 else ROLLOUT_CAVEAT
    # The spliced claim inverts here: with every branch resampling its future the
    # displacement is in the activations, and routing it through R^21 costs.
    title = ("$R^{21}$ moves the branches apart faster than it moves the null"
             if n_roll == 1 else
             "When every branch resamples its own future, the activation plane beats "
             "$R^{21}$ against each plane's own null")
    print(f"wrote {separation_figure(scores, nulls, names, args.camera_ready, suffix, foot, title)}")

    c1, c2 = (int(np.flatnonzero(meta["is_control"] == k)[0]) for k in (1, 2))
    print(f"\nrun from the reference / control 1's run, illegal (mean of 6) except the "
          f"last column, anchors >= t0\n{'space':34}"
          + "".join(f"{p:>22}" for p in list(PLANE_STYLE)
                    + ["full 21-D", "11 d", "10 θ", "control 2, full 21-D"]))
    for name in names:
        cells = [run_ratio(coords[p][name], ill, c1, **run_args(meta)) for p in PLANE_STYLE]
        cells += [run_ratio(X[name][:, :, c], ill, c1, **run_args(meta))
                  for c in (slice(None), BLOCKS["d"], BLOCKS["theta"])]
        cells.append(run_ratio(X[name], np.array([c2]), c1, **run_args(meta)))
        print(f"{name:34}" + "".join(f"{v:22.2f}" for v in cells))
    print(f"wrote {run_figure(coords, X, names, ill, c1, c2, meta, args.camera_ready, suffix, foot)}")


if __name__ == "__main__":
    main()
