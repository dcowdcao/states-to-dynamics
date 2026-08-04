"""One game's layer-5 trajectory, forking at t0 into every branch.

The aggregate figures answer "how far apart, on average". This one shows the
object those averages are taken over: a single game as a path, shared up to t0
because the branches are token-identical there, then one path per branch.

Move index is the third axis, so the plane must NOT be a PCA of raw activations
-- move index dominates activation geometry in every geometry study in this
repo, so PC1 would re-encode the vertical axis and the figure would collapse to
a slanted ribbon. The plane is therefore a PCA of the displacement
D(t) = z_branch(t) - z_ref(t), pooled over the game's branches and positions.
The reference is then exactly the vertical line at the origin, the shared prefix
collapses onto it, and the spray at t0 is the whole content of the figure.
Giving move index its own axis is also what discharges FIGURE_GUIDELINES 4.6's
nuisance-variable requirement here: it is displayed, not projected away.

Positions are sampled every STRIDE=2 plies at t0's own parity. At every ply the
path is a sawtooth that hides everything else: the board code is mover-relative,
so the frame inverts each ply and consecutive activations alternate. That ripple
is real -- it is the odd/even split in cos(D(r), D(0)) -- but it is a property of
the representation's frame, not of the trajectory's shape. One mover per curve is
the readable view.

Both a linear and a nonlinear plane are drawn, because the branches need not
separate linearly. PCA is uncentred (zero displacement is a real point, not an
arbitrary origin). Isomap follows the rules study's convention -- PCA-50
pre-reduction, then a kNN geodesic embedding -- but is FIT only on the
non-degenerate rows: D is exactly zero for the reference at every position and
for every branch before t0, and those duplicates form a disconnected clique in
the kNN graph. The fitted embedding then transforms all rows, so the zeros still
land somewhere consistent. Isomap has no origin and no variance ratio, so its
axes are arb. units and the reference is wherever the embedding puts it.

Usage: python analysis/violation_divergence/game_trajectory/make_figures.py
       [--n_games 1] [--embedding pca|isomap] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.branch_style import branch_style, ordered_legend
from spade.figures.paper_style import (
    GRID, INK, MUTED, apply_style, rule_key, sae_label, save_figure)
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.compute_divergence import sae_encoder

FIG_DIR = Path(__file__).resolve().parent / "figures"
ISOMAP_NEIGHBORS, ISOMAP_PRE_DIMS = 12, 50  # spade/rules/embed.py uses 15 on ~4k points
STRIDE = 2               # one mover per curve; see the module docstring
ELEV, AZIM = 18, -58     # fixed so the figure is reproducible
WINDOW = (-8, 24)        # the study's analysis window, r = t - t0
FIRST_POS, LAST_POS = 1, 58  # position 0 is the attention sink


def positions(t0: int) -> np.ndarray:
    """The study's window, at t0's parity, clipped to the usable board."""
    p = np.arange(max(FIRST_POS, t0 + WINDOW[0]), min(LAST_POS, t0 + WINDOW[1]) + 1)
    return p[(p - t0) % STRIDE == 0]


def pca2(X: np.ndarray, basis: np.ndarray | None = None):
    """(scores [n, 2], variance ratio [2], basis) — pass `basis` to reuse a fit.

    Uncentred, because zero displacement is a real point and not an arbitrary
    origin: mean-centring would move the reference off (0, 0) and separate it
    from the prefix line and the branch marker, which are drawn there.
    """
    if basis is None:
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        basis = Vt[:2]
        var = (S ** 2 / (S ** 2).sum())[:2]
    else:
        var = np.array([np.nan, np.nan])
    return X @ basis.T, var, basis


def isomap2(X: np.ndarray, fitted=None):
    """(scores [n, 2], nan var, fitted) — fit on nonzero rows, transform all."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    if fitted is None:
        keep = np.linalg.norm(X, axis=1) > 0
        pre = PCA(n_components=min(ISOMAP_PRE_DIMS, X.shape[1] - 1), random_state=0)
        iso = Isomap(n_neighbors=ISOMAP_NEIGHBORS, n_components=2, n_jobs=-1)
        iso.fit(pre.fit_transform(X[keep]))
        fitted = (pre, iso)
    pre, iso = fitted
    return iso.transform(pre.transform(X)), np.array([np.nan, np.nan]), fitted


EMBEDDINGS = {"pca": pca2, "isomap": isomap2}


def axis_labels(method: str, var: np.ndarray) -> tuple[str, str]:
    if method == "isomap":
        return "Isomap-1 (arb. units)", "Isomap-2 (arb. units)"
    return f"PC1 ({var[0]:.0%} var)", f"PC2 ({var[1]:.0%} var)"


def fork_panel(ax, coords: np.ndarray, rows: list[dict], pos: np.ndarray, t0: int,
               legend: bool, zlabel: bool = True) -> None:
    """coords [n_branches, n_pos, 2]; move index is the vertical axis."""
    k = int(np.flatnonzero(pos == t0)[0])
    lo, hi = coords.min(), coords.max()
    ax.plot([0, 0], [0, 0], [pos[0], t0], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label="shared prefix (branches identical)" if legend else None)
    for i, r in enumerate(rows):
        st = branch_style(r["category"], r["is_control"])
        seg, z = coords[i, k:], pos[k:]
        ax.plot(seg[:, 0], seg[:, 1], z, ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.0, markevery=3, markeredgecolor="white",
                markeredgewidth=0.3, zorder=st["zorder"],
                label=st["label"] if legend else None)
        ax.plot([seg[-1, 0]], [seg[-1, 1]], [z[-1]], marker=st["marker"], ms=6,
                color=st["color"], markeredgecolor="white", markeredgewidth=0.7,
                zorder=st["zorder"] + 1)
    ax.plot([0], [0], [t0], marker="*", ms=15, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    for a, lim in [(ax.set_xlim, (lo, hi)), (ax.set_ylim, (lo, hi))]:
        a(lim)
    ax.set_zlim(pos[0], pos[-1])
    ax.set_xticks([])
    ax.set_yticks([])
    if zlabel:
        ax.set_zlabel("move index $t$", labelpad=-4)
    ax.tick_params(axis="z", pad=0)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def delta_hat(Z: np.ndarray, ref: int) -> np.ndarray:
    d = np.linalg.norm(Z - Z[ref], axis=-1)
    return d / np.linalg.norm(Z[ref], axis=-1)


def marginal_panel(ax, Z: np.ndarray, rows: list[dict], pos: np.ndarray,
                   t0: int, ref: int) -> None:
    """The magnitude a projection cannot show: relative displacement vs offset."""
    dh, r = delta_hat(Z, ref), pos - t0
    ax.axvspan(r[0] - 0.5, -0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, row in enumerate(rows):
        st = branch_style(row["category"], row["is_control"])
        ax.plot(r, dh[i], ls=st["ls"], color=st["color"], lw=st["lw"],
                marker=st["marker"], ms=3.5, markevery=3, markeredgecolor="white",
                markeredgewidth=0.3, zorder=st["zorder"])
    ax.text(r[0] + 0.4, ax.get_ylim()[1] * 0.93, "0 by construction\n(shared prefix)",
            fontsize=7, color=MUTED, va="top")
    ax.set_xlabel("position relative to the branch, $t-t_0$")
    ax.set_ylabel(r"$\hat{\delta}=\|\Delta\|\,/\,\|z_{\mathrm{ref}}\|$")


def one_game(game: int, rows: list[dict], Z: np.ndarray, pos: np.ndarray, t0: int,
             method: str, camera_ready: bool) -> Path:
    ref = next(i for i, r in enumerate(rows) if r["category"] == 0 and not r["is_control"])
    n, T, D = Z.shape
    delta = (Z - Z[ref]).reshape(-1, D)
    coords, var, _ = EMBEDDINGS[method](delta)
    xlab, ylab = axis_labels(method, var)

    fig = plt.figure(figsize=(12.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.24)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax3.set_box_aspect((1, 1, 0.85))
    fork_panel(ax3, coords.reshape(n, T, 2), rows, pos, t0, legend=True)
    ax3.set_xlabel(xlab, labelpad=-8)
    ax3.set_ylabel(ylab, labelpad=-8)
    ax3.set_title("(a) displacement from the reference, by move", loc="left", fontsize=10)

    ax2 = fig.add_subplot(gs[1])
    marginal_panel(ax2, Z, rows, pos, t0, ref)
    ax2.set_title("(b) the magnitude the projection cannot show", loc="left", fontsize=10)

    ordered_legend(fig, ax3, 4, -0.19,
                   f"{method.upper()} plane, a learned basis refit for this game; view "
                   f"elev={ELEV}, azim={AZIM}; every {STRIDE}nd ply. One game (n=1): an "
                   f"illustration of the design, not a claim — the study's numbers, with "
                   f"their CIs over 13,531 branches, are in the parent README.")
    if not camera_ready:
        fig.suptitle(f"Game {game}: one prefix, {n} futures, forking at $t_0$={t0} "
                     f"(layer 5, positions {pos[0]}–{pos[-1]})", fontsize=11)
    fig.tight_layout()
    rule_key(fig, ["R1", "R2", "R3"], y=-0.235)
    return save_figure(fig, FIG_DIR / f"game_{game}_fork_{method}", camera_ready)


def space_features(Z: np.ndarray, entry, device: str) -> np.ndarray:
    """[n, T, width] in one representation space; `entry` None is raw layer 5."""
    if entry is None:
        return Z
    import torch as _t
    n, T, D = Z.shape
    enc = sae_encoder(entry, device)
    with _t.no_grad():
        f = enc(_t.tensor(Z.reshape(-1, D), device=device))
    return f.reshape(n, T, -1).cpu().numpy()


def space_grid(g: dict, method: str, device: str, camera_ready: bool) -> Path:
    """One cell per representation space, one game. Planes are refit per space,
    so positions are not comparable across cells -- only the shape of the fork is."""
    spaces = [("raw layer 5 (reference space)", None)] + \
        [(sae_label(e.config.replace("Trainer-othello", "")), e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    ref = next(i for i, r in enumerate(g["rows"]) if r["category"] == 0 and not r["is_control"])
    ncol = 3
    nrow = int(np.ceil(len(spaces) / ncol))
    fig = plt.figure(figsize=(3.7 * ncol, 3.5 * nrow))
    for j, (name, entry) in enumerate(spaces):
        F = space_features(g["Z"], entry, device)
        n, T, D = F.shape
        coords, var, _ = EMBEDDINGS[method]((F - F[ref]).reshape(-1, D))
        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        fork_panel(ax, coords.reshape(n, T, 2), g["rows"], g["pos"], g["t0"],
                   legend=(j == 0), zlabel=(j % ncol == 0))
        extra = "" if method == "isomap" else f"  ({var[0]:.0%}/{var[1]:.0%} var)"
        ax.set_title(f"{name}{extra}", loc="left", fontsize=8.5)
    ordered_legend(fig, fig.axes[0], 4, -0.06,
                   f"game {g['game']}, $t_0$={g['t0']}; {method.upper()} plane refit per space "
                   f"(arb. units, not comparable across cells); view elev={ELEV}, azim={AZIM}. "
                   f"One game (n=1): shape only — the parent README carries the numbers.")
    if not camera_ready:
        fig.suptitle(f"The fork in every representation space, game {g['game']} "
                     f"({method.upper()})", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.92, bottom=0.06, wspace=0.06, hspace=0.16)
    rule_key(fig, ["R1", "R2", "R3"], y=-0.085)
    return save_figure(fig, FIG_DIR / f"space_fork_grid_{method}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=1)
    parser.add_argument("--embedding", default="pca", choices=["pca", "isomap"])
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)

    n_types = len(CATEGORIES) + 2  # 6 illegal + reference + 2 legal controls
    complete = [g for g in meta["game_id"].unique().tolist()
        if int((meta["game_id"] == g).sum()) == n_types]
    assert complete, "no game carries every branch type"

    collected = []
    for game in complete[:args.n_games]:
        idx = (meta["game_id"] == game).nonzero().flatten()
        t0 = int(meta["t0"][idx[0]])
        pos = positions(t0)
        rows = [{"category": int(meta["category"][i]), "is_control": int(meta["is_control"][i])}
            for i in idx]
        Z = acts[idx][:, pos, :].float().numpy()
        ref = next(i for i, r in enumerate(rows) if r["category"] == 0 and not r["is_control"])
        collected.append({"game": game, "rows": rows, "Z": Z, "pos": pos, "t0": t0,
                          "delta": (Z - Z[ref]).reshape(-1, Z.shape[2])})

    methods = [args.embedding]
    for method in methods:
        for g in collected[:args.n_games]:
            print(f"wrote {one_game(g['game'], g['rows'], g['Z'], g['pos'], g['t0'], method, args.camera_ready)}")
        print(f"wrote {space_grid(collected[0], method, device, args.camera_ready)}")


if __name__ == "__main__":
    main()
