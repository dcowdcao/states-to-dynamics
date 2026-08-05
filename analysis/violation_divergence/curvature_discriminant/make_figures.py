"""The fork, drawn in a supervised curvature plane instead of a PCA of activations.

Same object as analysis/violation_divergence/game_trajectory's `space_fork_grid`:
one game, one shared prefix, nine futures, nine representation spaces. What
changes is the plane. There, axes 1-2 are the top-2 uncentred PCs of the
activation-space displacement D(t) = z_branch(t) - z_ref(t) -- unsupervised, and
so not necessarily the directions that separate the branches. Here:

  axis 1 = the LR-21 probe's discriminant direction in the 21-D ReStraV
           curvature descriptor, applied to the descriptor difference
           phi_branch(t) - phi_ref(t) over a trailing 8-ply window. The
           projection is linear, so the coordinate is exactly the branch's probe
           log-odds minus the reference's: "how much more violation-like than the
           game as played". Zero is the reference, by construction, and so is
           every branch before t0.
  axis 2 = the leading uncentred PC of the same differences with axis 1
           projected out. Unsupervised, refit per space, arb. units.
  axis 3 = move index, as in the parent figure.

A supervised axis separates the classes it was fit on by construction, so the
figure's evidence is in the bottom row, and it is against three nulls, not one:
the two LEGAL controls the probe never saw, a within-game label permutation, and
100 random directions through the same R^21. The last is the one that bites -- a
random line already reaches AUROC 0.56-0.64 because the descriptor difference is
simply larger for a branch that perturbed the trajectory more.

Two further limits are drawn rather than buried. The axis is calibrated at
exactly one move index: it is fit on the window ending at t0, and by r = 2 the
illegal-vs-control AUROC has fallen to chance or inverted, so the 3-D panels give
a full marker only to the branch position and draw the rest of each path faint.
And on a single game the plane clears nothing: 0.09-1.05x its own exact label
permutation null, p 0.36-1.00 across the nine spaces.

--variant rollout redraws all of it on the copy of the study where every branch,
the reference included, samples its own K=4 continuations rather than replaying
the source game's. The axis does not move: it is fit on straightness pairs, which
have no variant. The K rollouts are averaged AFTER phi, because phi is nonlinear
and a mean over the four paths is the curvature of a path none of them took; the
axis projection is linear and agrees on either side of the mean. What that run
answers is whether the illegal-vs-control separation was reading the branch move
or the out-of-distribution suffix a spliced branch inherits -- and the answer is
structural: the descriptor window is trailing, so the one ending at t0 sees only
tokens the rollout copies verbatim, and panel (k) at r <= 0 is the same number in
both variants, to three decimals in all nine spaces (evaluate.variant_agreement
returns the residual, 1e-3 to 1e-1, which is two extraction passes' fp16 noise).
Only r > 0 can differ, and that is the regime the rollout study found
uninformative (ROLLOUT_CAVEAT).

Usage: python analysis/violation_divergence/curvature_discriminant/make_figures.py
           [--variant {spliced,rollout}] [--camera-ready]
"""

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, REFERENCE, RULE_STYLE, apply_style,
    rule_key, sae_label, save_figure)
from spade.othello.taxonomy import CATEGORIES

from curvature_axis import (
    WINDOW_PLIES, fit_axis, residual_plane, spaces, transported_auroc)
from branch_geometry import VARIANTS, PLOT_GAME
from evaluate import (
    CONTROL_BINS, FIT_OFFSET, N_DIRS, OFFSETS, PLOT_STRIDE, RULE_NAMES, boot_ci,
    branch_meta, control_null, control_test, descriptors, deltas, flip_confound,
    landing_groups, magnitude_only, meta, named_axes_plane, permutation_null,
    random_direction_null, rollouts, variant_agreement)

FIG_DIR = Path(__file__).resolve().parent / "figures"
ELEV, AZIM = 18, -58              # the parent figure's camera, so the two compare
UNCAL_ALPHA = 0.22                # r > 0 is drawn as shape, not as a coordinate
ROLLOUT_CAVEAT = (
    "ROLLOUT CAVEAT: with every branch resampling its own future, $\\hat{\\delta}$ climbs "
    "monotonically and illegal and control both saturate near 1.23 against a ~1.41 "
    "unrelated-activations ceiling; the illegal/control ratio is pinned to 1.00 by "
    "$r\\geq8$, and only $r=0$ separates (1.032, sign opposite the spliced 0.979). "
    "Long-horizon rollout geometry is not informative, so the $r>0$ part of these panels "
    "is shape, not a result.")
R0 = int(np.flatnonzero(OFFSETS == 0)[0])
CONTROL_LABEL = {1: "control: another legal move (null)",
                 2: "control 2: a third legal move (null)"}
# Both control bins are nulls, so both take NULL's gray; shape carries the flip
# count, which is the variable the panel exists to expose.
GROUP_STYLE = {CONTROL_BINS[0]: {**NULL, "marker": "v"},
               CONTROL_BINS[1]: {**NULL, "marker": "^"},
               **{RULE_NAMES[r]: RULE_STYLE[RULE_NAMES[r]] for r in (1, 2, 3)}}


# --- branch styling, identical to analysis/violation_divergence/game_trajectory ---

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
        if lab.startswith("shared prefix"):
            rank[i] = 0
        elif lab.startswith("reference"):
            rank[i] = 1
        elif lab.startswith("branch point"):
            rank[i] = 8
        elif lab.startswith("$r>0$"):
            rank[i] = 11
        elif lab.startswith("control"):
            rank[i] = 9 + ("2" in lab)
        else:
            rank[i] = 2 + (names.index(lab) if lab in names else len(names))
    return sorted(range(len(labels)), key=lambda i: rank[i])


def ordered_legend(fig, ax, ncol: int, y: float, title: str):
    handles, labels = ax.get_legend_handles_labels()
    order = legend_order(labels)
    return fig.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="lower center", ncol=ncol, fontsize=8, frameon=False,
                      bbox_to_anchor=(0.5, y), title=title, title_fontsize=7.5)


# --- panels ---

def fork_panel(ax, coords: np.ndarray, rows: dict, t0: int, legend: bool) -> None:
    """coords [n_branches, n_offsets, 2]; move index is the vertical axis.

    Only the branch position carries a full marker, because that is the only move
    index at which the axis was measured to mean anything: panel (k) puts the
    illegal-vs-control AUROC at 0.55-0.86 at r = 0 and at 0.32-0.65 at r = 2. The
    rest of each path is drawn faint, as shape without a calibrated coordinate.
    Sampled every 2nd ply at t0's parity, as in the parent figure -- differencing
    against the reference was meant to cancel the mover-relative frame flip and
    only halves it (`parity_ratio`), so at every ply the sawtooth hides the fork.
    """
    keep = OFFSETS % PLOT_STRIDE == 0
    coords, offsets = coords[:, keep], OFFSETS[keep]
    pos = t0 + offsets
    r_zero = int(np.flatnonzero(offsets == 0)[0])
    lo, hi = coords.min(), coords.max()
    ax.plot([0, 0], [0, 0], [pos[0], t0], color=MUTED, lw=3.0, alpha=0.6,
            solid_capstyle="round", zorder=1,
            label="shared prefix (branches identical)" if legend else None)
    for i in range(len(coords)):
        st = branch_style(rows["category"][i], rows["is_control"][i])
        ax.plot(coords[i, r_zero:, 0], coords[i, r_zero:, 1], pos[r_zero:],
                ls=st["ls"], color=st["color"], lw=st["lw"], alpha=UNCAL_ALPHA,
                zorder=st["zorder"])
        ax.plot([coords[i, r_zero, 0]], [coords[i, r_zero, 1]], [t0],
                marker=st["marker"], ms=7, color=st["color"], ls="none",
                markeredgecolor="white", markeredgewidth=0.7, zorder=st["zorder"] + 3,
                label=st["label"] if legend else None)
    ax.plot([0], [0], [t0], marker="*", ms=13, color=INK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=9,
            label="branch point $t_0$" if legend else None)
    if legend:
        ax.plot([], [], color=MUTED, lw=1.5, alpha=UNCAL_ALPHA,
                label="$r>0$: axis not calibrated there (panel k)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(pos[0], pos[-1])
    ax.set_yticks([])
    ax.tick_params(axis="both", pad=-1, labelsize=6.5)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def band_panel(ax, curves: dict[str, np.ndarray], ylabel: str, title: str,
               ylim: tuple[float, float], legend: bool,
               null: tuple[float, float] | None = None,
               random95: np.ndarray | None = None, null_legend: bool = False) -> None:
    """Raw layer 5 as the reference line; the eight dictionaries as median + IQR."""
    q = np.percentile(np.stack([v for k, v in curves.items() if k != "resid_post_5"]),
                      [25, 50, 75], axis=0)
    ax.set_ylim(*ylim)
    ax.axvspan(0.5, OFFSETS[-1], color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(1.2, ylim[0] + 0.02, "axis not calibrated here", fontsize=7,
            color=MUTED, va="bottom")
    if null is None:
        ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(OFFSETS[-1], 0.505, "chance", fontsize=7, color=MUTED,
                ha="right", va="bottom")
    else:
        ax.axhspan(*null, color=NULL["color"], alpha=0.45, lw=0, zorder=1,
                   label="within-game label-permutation null (95%)")
        ax.text(OFFSETS[-1], null[1] + 0.005, "permutation null", fontsize=7,
                color=MUTED, ha="right", va="bottom")
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.fill_between(OFFSETS, q[0], q[2], color=RULE_STYLE["R1 placement"]["color"],
                    alpha=0.18, lw=0, zorder=2)
    ax.plot(OFFSETS, q[1], color=RULE_STYLE["R1 placement"]["color"], lw=1.6,
            marker="s", ms=3.5, markevery=3, zorder=3,
            label="8 SAE dictionaries (median, IQR)")
    ax.plot(OFFSETS, curves["resid_post_5"], color=REFERENCE["color"],
            lw=REFERENCE["lw"], marker=REFERENCE["marker"], ms=3.5, markevery=3,
            zorder=4, label="raw layer 5 (reference space)")
    if random95 is not None:
        ax.plot(OFFSETS, random95, color=NULL["color"], ls=NULL["ls"], lw=1.4,
                zorder=5, label=f"random direction in $R^{{21}}$, 95th pct "
                                f"({N_DIRS} draws, median over spaces)")
    ax.set_xlabel("position relative to the branch, $t-t_0$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5)
    handles, labels = ax.get_legend_handles_labels()
    pick = [i for i, l in enumerate(labels)
        if ("null" in l or "random" in l) == null_legend]
    if legend and pick:
        ax.legend([handles[i] for i in pick], [labels[i] for i in pick],
                  fontsize=6.6, loc="upper right", framealpha=0.92,
                  facecolor="white", edgecolor="none", handlelength=1.6)


def landing_panel(ax, landing: dict, space_labels: list[str], rho: float) -> None:
    """Where each group lands on axis 1 at r = 0, per space, with 95% CI.

    The two legal-control bins are the point: they are the same class, differing
    only in how many discs the move flips, and they straddle the illegal groups.
    """
    y = np.arange(len(space_labels))
    names = list(GROUP_STYLE)
    ax.axvline(0, color=REFERENCE["color"], lw=1.4, zorder=1)
    for j, name in enumerate(names):
        st, off = GROUP_STYLE[name], (2.0 - j) * 0.17
        m, lo, hi = landing[name]
        ax.errorbar(m, y + off, xerr=[m - lo, hi - m], fmt=st["marker"], ms=4.5,
                    color=st["color"], lw=0, elinewidth=1.1, capsize=2,
                    zorder=6 - j, label=name)
    ax.set_yticks(y, space_labels, fontsize=7)
    ax.set_ylim(len(y) - 0.5, -1.05)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.32 * (hi - lo))  # room for the legend, clear of the data
    ax.text(0, -0.9, " reference = 0 by construction", fontsize=7, color=MUTED,
            va="center")
    ax.set_xlabel("probe log-odds, branch $-$ reference, at $r=0$"
                  "   ($\\rightarrow$ more violation-like)")
    ax.set_title("(l) where the branches the probe never saw land\n"
                 f"     raw L5: Spearman $\\rho$ = {rho:+.2f} with flip count, "
                 "within the legal controls", loc="left", fontsize=9.5)
    ax.legend(fontsize=6.6, loc="upper right", ncol=1, framealpha=0.9,
              facecolor="white", edgecolor="none", handletextpad=0.4)


# --- assembly ---

def collect(m: dict, game: int, variant: str = "") -> dict:
    mb, k = branch_meta(m), rollouts(m)
    idx = np.flatnonzero(mb["game_id"] == game)
    groups = landing_groups(mb, True, variant)
    illegal, control = mb["category"] != 0, mb["is_control"] > 0
    keep = illegal | control
    y = illegal[keep].astype(float)

    out = {}
    for space in spaces():
        axis = fit_axis(space, FIT_OFFSET, m["held_out_legal"])
        delta = deltas(descriptors(space, m, variant), axis, m)
        score = (delta / axis.sd) @ axis.w
        assert np.abs(score[:, OFFSETS < 0]).max() < 1e-6, "branches differ before t0"
        plane = residual_plane(delta.reshape(-1, 21), axis).reshape(*delta.shape[:2], 2)
        out[space] = {
            "coords": plane[idx],
            "rows": {key: mb[key][idx] for key in ("category", "is_control")},
            "auroc": axis.auroc,
            "transport": transported_auroc(axis, list(OFFSETS[OFFSETS >= 0])),
            "control": control_test(score, mb),
            "control_norm": control_test(magnitude_only(delta, axis), mb),
            "random95": random_direction_null(delta, axis, mb),
            "control_ci": boot_ci(y, score[keep, R0], mb["game_id"][keep]),
            "perm": permutation_null(plane, mb, game),
            "perm_named": permutation_null(named_axes_plane(delta, axis), mb, game),
            "control_null": control_null(score[:, R0], mb),
            "landing": {name: (score[sel, R0].mean(),
                               *boot_ci_mean(score[sel, R0], mb["game_id"][sel]))
                        for name, sel in groups.items()},
            "flip_rho": flip_confound(score[:, R0], mb, variant),
        }
        print(f"{space:40} illegal-vs-control {out[space]['control'][R0]:.3f} "
              f"(random dir 95th {out[space]['random95'][R0]:.3f}, norm alone "
              f"{out[space]['control_norm'][R0]:.3f})  one game "
              f"{out[space]['perm']['ratio']:.2f}x null "
              f"(p={out[space]['perm']['p_exact']:.2f})"
              + ("" if k == 1 else
                 f"  r<=0 vs spliced {variant_agreement(space, m, variant):.1e}"))
    return out


def boot_ci_mean(x: np.ndarray, game: np.ndarray, n_boot: int = 1000):
    rng = np.random.default_rng(0)
    _, inv = np.unique(game, return_inverse=True)
    by_game = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    means = [x[np.concatenate([by_game[k] for k in
             rng.integers(0, len(by_game), len(by_game))])].mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def space_label(space: str) -> str:
    if space == "resid_post_5":
        return "raw layer 5 (reference space)"
    return sae_label(space.removeprefix("sae_").removesuffix("Trainer-othello"))


def figure(res: dict, m: dict, game: int, camera_ready: bool, n_roll: int = 1,
           suffix: str = "") -> Path:
    """`m` is the per-branch meta; under the rollout variant a branch's row is the
    mean over its K descriptors, taken after phi."""
    idx = np.flatnonzero(m["game_id"] == game)
    t0 = int(m["t0"][idx[0]])
    n_games = len(np.unique(m["game_id"]))
    names = list(res)
    fig = plt.figure(figsize=(11.5, 13.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.35, 1.0])
    gs = top.add_gridspec(3, 3, hspace=0.26, wspace=0.02,
                          left=0.01, right=0.99, top=0.94, bottom=0.01)

    for j, space in enumerate(names):
        ax = top.add_subplot(gs[j // 3, j % 3], projection="3d")
        ax.set_box_aspect((1, 1, 0.85), zoom=1.22)
        fork_panel(ax, res[space]["coords"], res[space]["rows"], t0, legend=(j == 0))
        ax.set_xlabel("$\\Delta$ probe log-odds", labelpad=-6, fontsize=7.5)
        ax.set_ylabel("residual PC1 (arb.)", labelpad=-8, fontsize=7.5)
        if j % 3 == 0:
            ax.set_zlabel("move index $t$", labelpad=-4, fontsize=7.5)
        ax.set_title(f"({chr(97 + j)}) {space_label(space)}\n"
                     f"ctrl {res[space]['control'][R0]:.2f} · rand "
                     f"{res[space]['random95'][R0]:.2f} · 1 game "
                     f"{res[space]['perm']['ratio']:.2f}$\\times$null",
                     loc="left", fontsize=7.5, y=1.00)

    gsb = bottom.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.58,
                              left=0.055, right=0.995, top=0.84, bottom=0.20)
    band_panel(bottom.add_subplot(gsb[0]),
               {s: np.array([res[s]["transport"].get(r, np.nan) for r in OFFSETS])
                for s in names},
               "AUROC, legal vs violation",
               "(j) does the axis survive the move index?", (0.30, 1.02), legend=True)

    ax = bottom.add_subplot(gsb[1])
    nulls = np.array([res[s]["control_null"][1:] for s in names])
    band_panel(ax, {s: res[s]["control"] for s in names},
               "AUROC, illegal vs legal control",
               "(k) legality, or just a different move?", (0.30, 1.02), legend=True,
               null=(nulls[:, 0].min(), nulls[:, 1].max()),
               random95=np.median([res[s]["random95"] for s in names], axis=0),
               null_legend=True)
    lo, hi = res["resid_post_5"]["control_ci"]
    mid = res["resid_post_5"]["control"][R0]
    ax.errorbar([0], [mid], yerr=[[mid - lo], [hi - mid]], fmt="none",
                ecolor=REFERENCE["color"], elinewidth=1.4, capsize=3, zorder=6)

    landing_panel(bottom.add_subplot(gsb[2]),
                  {name: tuple(np.array([res[s]["landing"][name][i] for s in names])
                               for i in range(3))
                   for name in res[names[0]]["landing"]},
                  [space_label(s) for s in names], res["resid_post_5"]["flip_rho"])

    ordered_legend(fig, fig.axes[0], 5, -0.030, "")
    caption = (
        f"(a)-(i) Game {game}, $t_0$={t0}, every {PLOT_STRIDE}nd ply — one game, so "
        f"shape only. Axis 1 is the LR-21 probe direction, fit on "
        f"the straightness study's violation/source-legal pairs at the window ending at "
        f"$t_0$, on games disjoint from every game scored here; axis 2 is the leading PC "
        f"of the residual, fit on the {n_games}-game population. Neither axis is fit on "
        f"this game. Cells are not comparable point-for-point. View elev={ELEV}, "
        f"azim={AZIM}; descriptor window = trailing {WINDOW_PLIES} plies. Per panel: "
        f"illegal-vs-control AUROC, the 95th percentile of {N_DIRS} random directions in "
        f"$R^{{21}}$, and the one-game centroid gap as a multiple of its own exact "
        f"label-permutation null.\n"
        f"The 17 plotted positions of a branch are windows overlapping in 7 of 8 samples, "
        f"so one game is 8 branches in 21 dimensions, not 136 points — which is why "
        f"neither axis is fit here, and why the permutation null (28 assignments, floor "
        f"p=0.036) can be exact with nothing refit. (j)-(l) aggregate {n_games} held-out "
        f"divergence games ({len(m['t0'])} branches) and held-out straightness pairs; "
        f"band = IQR over the 8 dictionaries, error bars = percentile bootstrap 95% CI "
        f"over games, 1000 resamples.")
    extra = [] if n_roll == 1 else [
        f"ROLLOUT VARIANT: every branch, the reference included, samples its own future "
        f"K={n_roll} times instead of replaying the source game's, and a branch's "
        f"descriptor is the MEAN OVER THOSE {n_roll} ROLLOUTS — taken after phi and "
        f"before the difference from the reference, because phi is nonlinear and a mean "
        f"over the {n_roll} trajectories is the curvature of a path none of them took. "
        f"The axis itself is unchanged: it is fit on straightness pairs, which have no "
        f"variant. The descriptor window is TRAILING, so the one ending at t0 sees only "
        f"tokens the rollout copies verbatim — panel (k) at r ≤ 0, the fitted offset "
        f"among them, is the SAME measurement as the spliced figure's and not a "
        f"replication of it. Its AUROCs agree to three decimals in all nine spaces; the "
        f"descriptors agree to 1e-3 (raw L5) to 1e-1 (GatedSAE), which is the fp16 noise "
        f"of two separate extraction passes and not the suffix. Only r > 0 differs.",
        ROLLOUT_CAVEAT]
    wrapped = [fill(line, 178) for line in caption.split("\n") + extra]
    fig.text(0.005, -0.075, "\n".join(wrapped),
             ha="left", va="top", fontsize=7.5, color=MUTED, linespacing=1.45)
    if not camera_ready:
        fig.suptitle("Most of the separation in $R^{21}$ is magnitude, not direction — "
                     "and on one game the plane does not beat its own null"
                     if n_roll == 1 else
                     "Resampling every branch's future changes nothing at $r=0$: the "
                     "trailing window there never saw the suffix",
                     fontsize=12, y=0.996)
    # The rule key sits under the caption, so it has to fall by whatever the
    # rollout paragraphs added; zero extra lines leaves the spliced figure alone.
    n_extra = sum(w.count("\n") + 1 for w in wrapped[len(caption.split("\n")):])
    rule_key(fig, ["R1", "R2", "R3"], y=-0.178 - 0.0115 * n_extra)
    return save_figure(fig, FIG_DIR / f"space_fork_grid_discriminant{suffix}", camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=PLOT_GAME)
    parser.add_argument("--variant", choices=list(VARIANTS), default="spliced")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    apply_style(args.camera_ready)

    variant = VARIANTS[args.variant]
    suffix = "" if args.variant == "spliced" else f"_{args.variant}"
    m = meta(variant)
    res = collect(m, args.game, variant)
    print(f"wrote {figure(res, branch_meta(m), args.game, args.camera_ready, rollouts(m), suffix)}")


if __name__ == "__main__":
    main()
