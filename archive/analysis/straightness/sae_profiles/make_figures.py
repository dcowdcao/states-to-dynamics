"""Trajectory profiles per SAE latent space, per family.

Two figures per seed-42 SAE family, each mirroring a raw-space figure in
analysis/straightness/figures/:

  curvature_vs_t_<Family>.png  (a) per-class curvature profile over the game
                               (b) violation-locked Δ turning angle
  norm_vs_t_<Family>.png       (a) per-class latent-norm profile over the game
                               (b) violation-locked Δ latent norm

Plus three small-multiples comparisons across all families, raw layer 5 included
as the reference: the violation-locked curvature panel (raw ghosted behind each),
the whole-game curvature distribution per class, and all three quantities at once
normalized to each space's own level.

Angles are scale-free, so curvature is comparable across spaces; latent norms
are not (SpaDE codes live on a simplex), so norm figures are per-family only.

Reads the geometry npz written by src/spade/straightness/compute_geometry.py,
which stores each space's own per-position norm — the latent norm for SAE spaces.

Usage: python analysis/straightness/sae_profiles/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.study import study_dir
from spade.figures.straightness_panels import (
    REL_WINDOW, ci_bars, class_blocks, density_panel, event_axes, profile_panel,
    value_delta, value_profile_panel, violation_delta, violation_panel)
from spade.figures.paper_style import (
    CLASS_STYLE, MUTED, QUANTITY_STYLE, VIOLATION_RULE_STYLE, apply_style, sae_label)
from spade.straightness.geometry import RESTRAV_FEATURE_NAMES
from spade.violation_divergence import rule_curvature as rc
from spade.violation_divergence.fork_plane import spaces as fork_spaces

apply_style()

FIG_DIR = Path(__file__).resolve().parent / "figures"
GEOM = study_dir("straightness") / "geometry"
RAW_SPACE = "resid_post_5"
TREE = "branches"  # the spliced suffix; the rules are drawn beside whole games
STYLE = CLASS_STYLE | VIOLATION_RULE_STYLE
# Legend order, reference first. The violation class is what the three rules
# replace, so STYLE still holds it and nothing draws it.
DRAWN = [c for c in CLASS_STYLE if c != "violation"] + list(VIOLATION_RULE_STYLE)
MEAN_THETA = RESTRAV_FEATURE_NAMES.index("mean θ")
# One key for both sources: rule_curvature.space_key names the divergence cache
# and this study's geometry npz with the same stem.
ENTRY = {rc.space_key(e): e for _, e in fork_spaces()}
CI_NOTE = ("error bars are mean ± 95% CI at the drawn markers, n = 2000 games per "
           "class; gray span = prefix identical to the source game, Δ ≡ 0 by "
           "construction")


def families() -> list[tuple[str, str]]:
    """(space, short label) for every SAE family, sorted by label."""
    spaces = sorted(f.name.removesuffix("_raw.npz") for f in GEOM.glob("sae_*_raw.npz"))
    return [(s, s.removeprefix("sae_").removesuffix("Trainer-othello")) for s in spaces]


def field(space: str, name: str) -> np.ndarray:
    return np.load(GEOM / f"{space}_raw.npz")[name]


def theta(space: str) -> np.ndarray:
    return field(space, "theta")


def per_family_step(space: str, label: str, blocks: dict, t0: np.ndarray,
                    src: np.ndarray, viol: np.ndarray) -> None:
    """Step length ‖Δz‖ — the trajectory-geometry counterpart of curvature, and
    the quantity ReStraV actually uses alongside the turning angle."""
    d = field(space, "d")
    rels, mu, sem = violation_delta(d, t0, src, viol)  # d index k is the step ending at k+2

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    value_profile_panel(axes[0], d, blocks, "mean step length ‖Δz‖", offset=2)
    axes[0].set_xlabel("board position t (step ending at t)")
    axes[0].set_title("(a) step-length profile over the game", loc="left")
    axes[0].legend(fontsize=7, loc="best", ncol=2, columnspacing=1.0)

    ax = axes[1]
    violation_panel(ax, rels, mu, sem, prefix_end=-0.5, fmt="{:.2e}")
    span = (mu.max() - mu.min()) or 1.0
    ax.set_ylim(mu.min() - 0.35 * span, mu.max() + 0.2 * span)
    ax.set_ylabel("Δ step length vs. matched\nsource legal game")
    ax.set_title("(b) violation-locked response", loc="left")

    fig.suptitle(f"Step length in {label} latents (layer-5 SAE, raw)",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.935, CI_NOTE, fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / f"step_vs_t_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def per_family_norm(space: str, label: str, blocks: dict, t0: np.ndarray,
                    src: np.ndarray, viol: np.ndarray) -> None:
    """Latent-norm counterpart of per_family."""
    norm = field(space, "norm")
    rels, mu, sem = value_delta(norm, t0, src, viol)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    value_profile_panel(axes[0], norm, blocks, "mean latent norm")
    axes[0].set_title("(a) latent-norm profile over the game", loc="left")
    axes[0].legend(fontsize=7, loc="best", ncol=2, columnspacing=1.0)

    ax = axes[1]
    violation_panel(ax, rels, mu, sem, prefix_end=-0.5, fmt="{:.2e}")
    span = (mu.max() - mu.min()) or 1.0
    ax.set_ylim(mu.min() - 0.35 * span, mu.max() + 0.2 * span)
    ax.set_ylabel("Δ latent norm vs. matched\nsource legal game")
    ax.set_title("(b) violation-locked response", loc="left")

    fig.suptitle(f"Activation norm in {label} latents (layer-5 SAE, raw)",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.935, CI_NOTE, fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / f"norm_vs_t_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def per_family(space: str, label: str, blocks: dict, t0: np.ndarray,
               src: np.ndarray, viol: np.ndarray) -> np.ndarray:
    th = theta(space)
    rels, mu, sem = violation_delta(th, t0, src, viol)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    profile_panel(axes[0], th, blocks)
    axes[0].set_title("(a) curvature profile over the game", loc="left")
    axes[0].legend(fontsize=7, loc="best", ncol=2, columnspacing=1.0)

    ax = axes[1]
    violation_panel(ax, rels, mu, sem)
    ax.set_ylim(mu.min() - 1.0, max(0.7, mu.max() + 0.4))
    ax.set_ylabel("Δ turning angle vs. matched\nsource legal game (deg)")
    ax.set_title("(b) violation-locked response", loc="left")

    fig.suptitle(f"Curvature in {label} latents (layer-5 SAE, raw)",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.935, CI_NOTE, fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / f"curvature_vs_t_{label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return mu


def combined_comparison(blocks: dict, t0: np.ndarray, src: np.ndarray,
                        viol: np.ndarray) -> dict[tuple[str, str], tuple[float, int]]:
    """Small multiples: all three trajectory quantities at once, every space.

    Each Δ is divided by that space's own mean level for that quantity, which is
    what lets the nine dictionaries share an axis — latent norms and step lengths
    differ by two orders of magnitude between them, turning angles do not.
    Returns {(label, quantity): (peak %, offset)}.
    """
    panels = [(RAW_SPACE, "raw L5 (reference)")]
    panels += [(s, sae_label(label)) for s, label in families()]
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.9 * nrow),
                             sharex=True, sharey=True)
    peaks = {}
    for ax, (space, label) in zip(axes.flat, panels):
        event_axes(ax)  # -1.5: only theta is sensitive at t0-1, d and norm are not
        for i, (key, delta) in enumerate([("theta", violation_delta),
                                          ("d", violation_delta),
                                          ("norm", value_delta)]):
            values = field(space, key)
            rels, mu, sem = delta(values, t0, src, viol)
            level = np.nanmean(values[blocks["legal"]])
            mu, sem = 100 * mu / level, 100 * sem / level
            style = QUANTITY_STYLE[key]
            at = np.arange(i, len(rels), 4)  # the markevery positions below
            ci_bars(ax, rels[at], mu[at], 1.96 * sem[at], style["color"], zorder=2)
            ax.plot(rels, mu, "-", marker=style["marker"], lw=1.8, ms=4,
                    markevery=(i, 4), color=style["color"], label=style["label"],
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3 + i)
            at = int(np.abs(mu).argmax())
            peaks[(label, key)] = (float(mu[at]), int(rels[at]))
        ax.set_title(label, loc="left", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("position relative to injected move ($t - t_0$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ vs. matched source game\n(% of that space's mean level)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("One illegal move shortens the step and straightens the turn in "
                 "every dictionary — the norm response splits in sign",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.962, "violation-locked Δ against each violation game's matched "
             "source legal game, as a percentage of that space's own mean level — the "
             "normalization\nthat lets dictionaries share an axis; error bars are mean "
             "± 95% CI at the drawn markers; gray span is Δ ≡ 0 by construction (θ is "
             "sensitive at $t_0-1$, "
             "‖Δz‖ and ‖z‖ at $t_0$)\nn = 2000 violation games. θ alone, in degrees "
             "and with troughs annotated: violation_spike_comparison.png",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    fig.savefig(FIG_DIR / "quantity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return peaks


def whole_game_theta(space: str) -> tuple[np.ndarray, dict[str, slice]]:
    """Per-game mean turning angle for the six drawn classes, and each one's rows.

    The illegal move is drawn as the three constraint rules, which this study
    cannot supply -- its violation class injects on empty squares only and carries
    no R1. They come off the violation-divergence branch population, which holds
    the same quantity as the ReStraV descriptor's `mean θ` component. What that
    costs is in ../feature_pca/README.md, and the caption carries it.
    """
    games = np.nanmean(theta(space), 1)
    rules, _ = rc.descriptors(TREE, ENTRY[space])
    parts = [(c, games[b]) for c, b in class_blocks(len(games)).items()
             if c != "violation"]
    parts += [(r, rules[:, j, MEAN_THETA]) for j, r in enumerate(rc.RULES)]

    blocks, at = {}, 0
    for c, v in parts:
        blocks[c] = slice(at, at + len(v))
        at = blocks[c].stop
    return np.concatenate([v for _, v in parts]), blocks


def theta_dist_comparison() -> None:
    """Small multiples: the whole-game curvature distribution in every space.

    The per-panel gap between the legal median triangle and another class's is
    the separation sae_families_auroc.png scores; here it is readable in degrees.
    Those gaps are printed rather than drawn -- five per panel would not fit a
    title -- and README.md quotes what is printed.
    """
    spaces = [(RAW_SPACE, "raw L5 (reference)")]
    spaces += [(s, sae_label(label)) for s, label in families()]
    ncol = 3
    nrow = -(-len(spaces) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.9 * nrow),
                             sharex=True, sharey=True)
    ood = DRAWN[1:]
    print(f"{'space':34}" + "".join(f"{c.split()[0][:5] + ' Δmed':>13}" for c in ood))
    for ax, (space, label) in zip(axes.flat, spaces):
        per_game, blocks = whole_game_theta(space)
        density_panel(ax, per_game, blocks, STYLE)
        ax.set_title(label, loc="left", fontsize=9)
        legal = np.median(per_game[blocks["legal"]])
        print(f"{label:34}" + "".join(
            f"{np.median(per_game[blocks[c]]) - legal:+12.2f}°" for c in ood))
    for ax in axes.flat[len(spaces):]:
        ax.set_visible(False)
    axes.flat[0].set_ylim(bottom=0)  # after every panel: shared axes freeze on set_ylim

    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Two columns of three: whole-game classes on the left, the three rules on the
    # right, which is the split the reader has to hold to read the panels.
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               columnspacing=2.5, bbox_to_anchor=(0.5, -0.045))
    # Descriptive, not a claim, and no prose block: the published caption is in
    # README.md, as in ../../violation_divergence/rule_curvature.
    fig.suptitle("Whole-game curvature distribution per class, "
                 "per representation space", fontsize=11, y=1.00)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIG_DIR / "theta_dist_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def comparison(profiles: dict[str, np.ndarray], rels: np.ndarray,
               sems: dict[str, np.ndarray], raw_mu: np.ndarray) -> None:
    """Small multiples: the violation-locked panel for every space, shared y."""
    labels = ["raw L5"] + sorted(profiles)
    ncol = 3
    nrow = -(-len(labels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.7 * nrow),
                             sharex=True, sharey=True)
    for ax, label in zip(axes.flat, labels):
        is_raw = label == "raw L5"
        mu = raw_mu if is_raw else profiles[label]
        sem = sems.get(label, np.zeros_like(mu)) if not is_raw else sems["raw L5"]
        violation_panel(ax, rels, mu, sem, ghost=None if is_raw else raw_mu,
                        annotate=False)
        post = mu[rels >= 0]
        trough = int(mu.argmin())
        ax.set_title(f"{label}   trough {mu[trough]:.1f}° at $t_0${rels[trough]:+d}"
                     f"   ·   net {post.mean():+.2f}°", loc="left", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("$t - t_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ turning angle (deg)")
    axes.flat[1].legend(fontsize=7, loc="lower right")

    fig.suptitle("Every dictionary localizes the violation at $t_0\\pm1$; the "
                 "whole-game AUROC tracks the net, not the magnitude",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.955, "violation-locked Δ curvature vs. the matched source legal "
             "game; gray = raw layer 5; shared y-axis. 'net' is the mean Δ over "
             "$t \\geq t_0$, which is what a whole-game average sees.",
             fontsize=8, color=MUTED, style="italic", ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(FIG_DIR / "violation_spike_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    blocks = class_blocks(len(data["cls"]))
    viol = np.arange(blocks["violation"].start, blocks["violation"].stop)
    t0, src = data["t0"].numpy()[viol], data["src_game"].numpy()[viol]

    raw_rels, raw_mu, raw_sem = violation_delta(theta(RAW_SPACE), t0, src, viol)
    profiles, sems = {}, {"raw L5": raw_sem}
    for space, label in families():
        th = theta(space)
        _, mu, sem = violation_delta(th, t0, src, viol)
        assert np.allclose(mu[raw_rels <= -2], 0), \
            f"{label}: prefix delta nonzero — alignment or matching is wrong"
        profiles[label], sems[label] = mu, sem
        per_family(space, label, blocks, t0, src, viol)
        per_family_step(space, label, blocks, t0, src, viol)
        per_family_norm(space, label, blocks, t0, src, viol)
        print(f"{label:24} theta trough {mu.min():+.2f}°")

    comparison(profiles, raw_rels, sems, raw_mu)
    theta_dist_comparison()
    peaks = combined_comparison(blocks, t0, src, viol)
    print(f"\n{'space':34} {'θ':>16} {'‖Δz‖':>16} {'‖z‖':>16}   (peak %, offset)")
    for label in dict.fromkeys(k[0] for k in peaks):
        cells = "".join(f"{peaks[(label, q)][0]:+10.2f}% @{peaks[(label, q)][1]:+2d}"
                        for q in ["theta", "d", "norm"])
        print(f"{label:34}{cells}")
    print(f"{len(list(FIG_DIR.glob('*.png')))} figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
