"""Figures for the straightness study (CPU; reads geometry npz + results CSVs).

F1 theta_dist_layer5     per-game mean turning angle by class, raw vs centered
F2 auroc_vs_depth        mean-theta AUROC across token_embed + layers 0-7
F3 curvature_vs_t        curvature profile over positions; violation spike at t0
F3b step_vs_t            step-length profile over positions; violation response at t0
F4 mean_vs_var_theta     ReStraV signature scatter, one facet per OOD class
F5 sae_families_auroc    separation per SAE family vs raw layer 5
F6 norm_diagnostics      per-position norms by class + mean-d AUROC vs depth

Usage: python analysis/straightness/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spade.common.paths import repo_root, results_dir
from spade.common.study import study_dir

from spade.figures.straightness_panels import (
    DRAWN_CLASSES, REL_WINDOW, ci_bars, class_blocks, density_panel,
    mean_sem as _mean_sem, profile_panel, value_profile_panel, violation_delta,
    violation_panel)
from spade.figures.paper_style import (
    CLASS_STYLE, GRID, INK, MUTED, NULL, REFERENCE, apply_style, sae_label)

DEPTH_SPACES = ["token_embed"] + [f"resid_post_{i}" for i in range(8)]
DEPTH_LABELS = ["W_E"] + [f"L{i}" for i in range(8)]
OOD_PAIRS = [f"legal_vs_{c}" for c in DRAWN_CLASSES[1:]]

apply_style()

FIG_DIR = Path(__file__).resolve().parent / "figures"
GEOM = study_dir("straightness") / "geometry"


def load(space: str, variant: str) -> dict[str, np.ndarray]:
    return dict(np.load(GEOM / f"{space}_{variant}.npz"))


def mean_theta_per_game(theta: np.ndarray) -> np.ndarray:
    return np.nanmean(theta, axis=1)


def f1_theta_dist(blocks):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, variant, panel in zip(axes, ["raw", "centered"], "ab"):
        mt = mean_theta_per_game(load("resid_post_5", variant)["theta"])
        density_panel(ax, mt, blocks, CLASS_STYLE)
        ax.set_title(f"({panel}) {'raw' if variant == 'raw' else 'per-t centered'}",
                     loc="left")
    axes[1].set_ylabel("")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Only games carrying an illegal move shift the curvature "
                 "distribution (layer 5), and by 0.3°", fontsize=11, y=1.00)
    fig.text(0.5, 0.93, "Gaussian KDE, per-class Scott bandwidth (h = 0.21-0.25°); "
             "triangles mark class medians; n = 2000 games per class\n"
             "the separation these curves show is scored with a bootstrap CI in "
             "auroc_vs_depth.png and sae_families_auroc.png",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(FIG_DIR / "theta_dist_layer5.png", dpi=300, bbox_inches="tight")


DEPTH_SERIES = [f"legal_vs_{c}" for c in ["championship", "violation", "rollout"]]


def f2_auroc_vs_depth(auroc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    for ax, variant, panel in zip(axes, ["raw", "centered"], "ab"):
        sub = auroc[auroc.variant == variant].set_index(["space", "pair"])
        ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axvline(0.5, color=GRID, lw=1.2, zorder=1)  # input space | residual stream

        lo = [sub.loc[(s, "null_legal_halves"), "auc_mean_theta_lo"] for s in DEPTH_SPACES]
        hi = [sub.loc[(s, "null_legal_halves"), "auc_mean_theta_hi"] for s in DEPTH_SPACES]
        ax.fill_between(DEPTH_LABELS, lo, hi, color=NULL["color"], alpha=0.25, lw=0,
                        zorder=1, label="legal vs. legal null (95% CI)")
        y = [sub.loc[(s, "null_legal_halves"), "auc_mean_theta"] for s in DEPTH_SPACES]
        ax.plot(DEPTH_LABELS, y, color=NULL["color"], ls=NULL["ls"], lw=NULL["lw"],
                zorder=NULL["zorder"])

        for pair in DEPTH_SERIES:
            cls = pair.removeprefix("legal_vs_")
            style = CLASS_STYLE[cls]
            y = np.array([sub.loc[(s, pair), "auc_mean_theta"] for s in DEPTH_SPACES])
            lo = np.array([sub.loc[(s, pair), "auc_mean_theta_lo"] for s in DEPTH_SPACES])
            hi = np.array([sub.loc[(s, pair), "auc_mean_theta_hi"] for s in DEPTH_SPACES])
            ci_bars(ax, DEPTH_LABELS, y, [y - lo, hi - y], style["color"], zorder=2)
            ax.plot(DEPTH_LABELS, y, "-", marker=style["marker"], lw=2.0, ms=5,
                    color=style["color"], label=style["label"], zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(f"({panel}) {'raw' if variant == 'raw' else 'per-t centered'}",
                     loc="left")
        ax.set_xlabel("representation space")
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", visible=False)

    fig.suptitle("Trajectory curvature by depth — every class scored against the "
                 "legal-game baseline", fontsize=11, x=0.5, y=0.99)
    axes[0].set_ylabel("AUROC vs. legal\n(per-game mean turning angle)")
    axes[0].text(6.3, 0.90, "↑ curvier than legal", fontsize=8, color=MUTED, style="italic")
    axes[0].text(6.3, 0.13, "↓ straighter than legal", fontsize=8, color=MUTED, style="italic")
    axes[1].text(8.25, 0.5, "chance", fontsize=8, color=MUTED, ha="left", va="center",
                 clip_on=False)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]  # null last: it is a reference, not a condition
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03),
               handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(FIG_DIR / "auroc_vs_depth.png", dpi=300, bbox_inches="tight")


def f3_curvature_vs_t(blocks, data):
    theta = load("resid_post_5", "raw")["theta"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))

    profile_panel(axes[0], theta, blocks)
    axes[0].set_title("(a) curvature profile over the game", loc="left")
    axes[0].legend(fontsize=7.5, loc="lower center", ncol=2, columnspacing=1.0)

    viol = np.arange(blocks["violation"].start, blocks["violation"].stop)
    rels, mu, sem = violation_delta(theta, data["t0"].numpy()[viol],
                                    data["src_game"].numpy()[viol], viol)
    ax = axes[1]
    violation_panel(ax, rels, mu, sem)
    ax.set_ylim(mu.min() - 1.0, 0.7)
    ax.text(0.35, 0.45, "$t_0$: injected\nillegal move", fontsize=8, color=INK,
            ha="left", va="top")
    ax.text((REL_WINDOW[0] - 0.5 + -1.5) / 2, mu.min() * 0.5,
            "prefix identical to\nsource game: Δ ≡ 0\nby construction\n($t-t_0 \\leq -2$)",
            fontsize=7.5, color=MUTED, style="italic", ha="center", va="center")
    ax.set_ylabel("Δ turning angle vs. matched\nsource legal game (deg)")
    ax.set_title("(b) violation-locked response", loc="left")

    fig.suptitle("Curvature over the game and around an injected rule violation "
                 "(layer 5, raw)", fontsize=11, y=1.00)
    fig.text(0.5, 0.935, "error bars are mean ± 95% CI at the drawn markers, "
             "n = 2000 games per class",
             fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "curvature_vs_t.png", dpi=300, bbox_inches="tight")


def f3b_step_vs_t(blocks, data):
    """Raw-activation counterpart of f3: step length instead of turning angle."""
    d = load("resid_post_5", "raw")["d"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))

    value_profile_panel(axes[0], d, blocks, "mean step length ‖Δz‖", offset=2)
    axes[0].set_xlabel("board position t (step ending at t)")
    axes[0].set_title("(a) step-length profile over the game", loc="left")
    axes[0].legend(fontsize=7.5, loc="best", ncol=2, columnspacing=1.0)

    viol = np.arange(blocks["violation"].start, blocks["violation"].stop)
    rels, mu, sem = violation_delta(d, data["t0"].numpy()[viol],
                                    data["src_game"].numpy()[viol], viol)
    ax = axes[1]
    violation_panel(ax, rels, mu, sem, prefix_end=-0.5, fmt="{:.2f}")
    span = (mu.max() - mu.min()) or 1.0
    ax.set_ylim(mu.min() - 0.35 * span, mu.max() + 0.2 * span)
    ax.set_ylabel("Δ step length vs. matched\nsource legal game")
    ax.set_title("(b) violation-locked response", loc="left")

    fig.suptitle("Step length over the game and around an injected rule violation "
                 "(layer 5, raw)", fontsize=11, y=1.00)
    fig.text(0.5, 0.935, "error bars are mean ± 95% CI at the drawn markers, "
             "n = 2000 games per class; "
             "gray span = prefix identical to the source game, Δ ≡ 0 by construction",
             fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "step_vs_t.png", dpi=300, bbox_inches="tight")


def f4_mean_vs_var(blocks):
    theta = load("resid_post_5", "raw")["theta"]
    mt, vt = np.nanmean(theta, 1), np.nanvar(theta, 1)
    ood = DRAWN_CLASSES[1:]
    fig, axes = plt.subplots(1, len(ood), figsize=(2.6 * len(ood), 3.3),
                             sharex=True, sharey=True)
    for ax, c, panel in zip(axes, ood, "abc"):
        for name in ["legal", c]:
            b, style = blocks[name], CLASS_STYLE[name]
            ax.scatter(mt[b], vt[b], s=5, alpha=0.22, lw=0, color=style["color"],
                       label=style["label"], zorder=2 if name == "legal" else 3)
            ax.plot(np.median(mt[b]), np.median(vt[b]), marker=style["marker"],
                    ms=8, color=style["color"], markeredgecolor="white",
                    markeredgewidth=1.1, zorder=6)
        ax.set_title(f"({panel}) vs {c.replace('_', ' ')}", loc="left")
        ax.set_xlabel("mean turning angle (deg)")
        leg = ax.legend(fontsize=6.5, markerscale=2.5, loc="upper left")
        for h in leg.legend_handles:
            h.set_alpha(1)
    axes[0].set_ylabel("turning-angle variance")
    fig.suptitle("Violations shift the mean turning angle; human games shift its "
                 "variance", fontsize=11, y=1.00)
    fig.text(0.5, 0.925, "one point per game (layer 5, raw); large markers are "
             "class medians; n = 2000 per class\nmedians: legal 142.0°/27.7, "
             "violation 141.7°/27.3, championship 142.1°/30.6, rollout 142.0°/28.2",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG_DIR / "mean_vs_var_theta.png", dpi=300, bbox_inches="tight")





def f5_sae_families(auroc: pd.DataFrame):
    spaces = ["resid_post_5"] + sorted(
        s for s in auroc.space.unique() if s.startswith("sae_"))
    families = [s.removeprefix("sae_").removesuffix("Trainer-othello")
                for s in spaces[1:]]
    labels = ["raw L5 (reference)"] + [sae_label(f) for f in families]
    ours = {sae_label(f) for f in families if "SpaDE" in f}
    sub = auroc[auroc.variant == "raw"].set_index(["space", "pair"])
    fig, axes = plt.subplots(1, len(OOD_PAIRS), figsize=(2.9 * len(OOD_PAIRS), 3.6),
                             sharey=True)
    ypos = np.arange(len(spaces))
    for ax, pair, panel in zip(axes, OOD_PAIRS, "abc"):
        c = pair.removeprefix("legal_vs_")
        th = np.array([sub.loc[(s, pair), "auc_mean_theta"] for s in spaces])
        lo = np.array([sub.loc[(s, pair), "auc_mean_theta_lo"] for s in spaces])
        hi = np.array([sub.loc[(s, pair), "auc_mean_theta_hi"] for s in spaces])
        lr = [sub.loc[(s, pair), "auc_lr21"] for s in spaces]
        colors = [REFERENCE["color"] if i == 0 else CLASS_STYLE[c]["color"]
                  for i in range(len(spaces))]
        edges = ["black" if lab in ours else "none" for lab in labels]
        widths = [1.4 if lab in ours else 0 for lab in labels]
        ax.barh(ypos, th, height=0.62, color=colors, edgecolor=edges,
                linewidth=widths, label="mean-θ AUROC", zorder=2)
        ax.errorbar(th, ypos, xerr=[th - lo, hi - th], fmt="none", ecolor=MUTED,
                    elinewidth=1.0, capsize=2.0, zorder=4)
        ax.plot(lr, ypos, "o", ms=5, color=INK, label="LR-21 AUROC", zorder=5)
        ax.axvline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f"({panel}) legal vs {c.replace('_', ' ')}", loc="left")
        ax.set_xlabel("AUROC")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, labels)
    axes[0].invert_yaxis()
    for tick, lab in zip(axes[0].get_yticklabels(), labels):
        if lab in ours:
            tick.set_fontweight("bold")
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Every SAE family keeps the violation separation, and most also "
                 "split human games (layer-5 latents)", fontsize=11, y=1.00)
    fig.text(0.5, 0.945,
             "each panel separates legal games from one other class; rows are the "
             "representation space it is measured in\nbars = AUROC from mean turning "
             "angle alone (bootstrap 95% CI), dots = AUROC from the 21-D feature "
             "vector; <0.5 = that class is straighter than legal",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.85))
    fig.savefig(FIG_DIR / "sae_families_auroc.png", dpi=300, bbox_inches="tight")


def f6_norm_diagnostics(blocks, auroc: pd.DataFrame):
    norm = load("resid_post_5", "raw")["norm"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    positions = np.arange(norm.shape[1]) + 1
    for i, c in enumerate(blocks):
        style, is_ref = CLASS_STYLE[c], c == "legal"
        mu, sem = _mean_sem(norm[blocks[c]])
        at = np.arange(2 * i, len(positions), 9)  # the markevery positions below
        ci_bars(axes[0], positions[at], mu[at], 1.96 * sem[at], style["color"])
        axes[0].plot(positions, mu, lw=2.4 if is_ref else 1.6, color=style["color"],
                     marker=style["marker"], ms=4, markevery=(2 * i, 9),
                     markeredgecolor="white", markeredgewidth=0.5,
                     zorder=5 if is_ref else 3, label=style["label"])
    axes[0].set_xlabel("board position t")
    axes[0].set_ylabel("mean activation norm")
    axes[0].set_title("(a) layer-5 activation norms", loc="left")
    axes[0].legend(fontsize=7.5, loc="upper right", ncol=2, columnspacing=1.0)

    ax = axes[1]
    sub = auroc[auroc.variant == "raw"].set_index(["space", "pair"])
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0.5, color=GRID, lw=1.2, zorder=1)
    lo = [sub.loc[(s, "null_legal_halves"), "auc_mean_d_lo"] for s in DEPTH_SPACES]
    hi = [sub.loc[(s, "null_legal_halves"), "auc_mean_d_hi"] for s in DEPTH_SPACES]
    ax.fill_between(DEPTH_LABELS, lo, hi, color=NULL["color"], alpha=0.25, lw=0,
                    zorder=1, label="legal vs. legal null (95% CI)")
    for pair in DEPTH_SERIES:
        c = pair.removeprefix("legal_vs_")
        style = CLASS_STYLE[c]
        y = np.array([sub.loc[(s, pair), "auc_mean_d"] for s in DEPTH_SPACES])
        lo = np.array([sub.loc[(s, pair), "auc_mean_d_lo"] for s in DEPTH_SPACES])
        hi = np.array([sub.loc[(s, pair), "auc_mean_d_hi"] for s in DEPTH_SPACES])
        ci_bars(ax, DEPTH_LABELS, y, [y - lo, hi - y], style["color"], zorder=2)
        ax.plot(DEPTH_LABELS, y, "-", marker=style["marker"], lw=2.0, ms=5,
                color=style["color"], label=style["label"], zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("representation space")
    ax.set_ylabel("AUROC vs. legal (mean step length)")
    ax.set_title("(b) step-length AUROC by depth", loc="left")
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[i] for i in [1, 2, 3, 0]], [l[i] for i in [1, 2, 3, 0]],
              fontsize=6.5, loc="upper right", ncol=1)

    fig.suptitle("Norm-confound check: the classes share one norm profile, and "
                 "step lengths carry the angle result's signs", fontsize=11, y=1.00)
    fig.text(0.5, 0.925, "error bars are 95% CI (panel a: mean ± CI over games, at "
             "the drawn markers; panel b: stratified bootstrap); n = 2000 per class",
             fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG_DIR / "norm_diagnostics.png", dpi=300, bbox_inches="tight")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    blocks = class_blocks(len(data["cls"]))
    auroc = pd.read_csv(results_dir("straightness") / "auroc.csv")

    f1_theta_dist(blocks)
    f2_auroc_vs_depth(auroc)
    f3_curvature_vs_t(blocks, data)
    f3b_step_vs_t(blocks, data)
    f4_mean_vs_var(blocks)
    if any(auroc.space.str.startswith("sae_")):
        f5_sae_families(auroc)
    f6_norm_diagnostics(blocks, auroc)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
