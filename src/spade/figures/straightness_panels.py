"""Trajectory panels for the straightness study, shared by the raw-space and
per-SAE figures, plus the set of classes those figures draw: profiles over the
game, event-locked responses, and per-class distributions of the whole-game
statistic.

Two panels make up a curvature_vs_t figure: the per-class profile over the game,
and the violation-locked response (Δ turning angle against each violation game's
matched source legal game, aligned at the injected move t0). density_panel is
the whole-game view the AUROCs summarize. Angles are scale-free, so all three
are comparable across representation spaces even when step lengths are not. See
FIGURE_GUIDELINES.md §4.2 and §4.4 for the conventions these encode.
"""

import numpy as np

from spade.figures.paper_style import CLASS_STYLE, GRID, INK, MUTED
from spade.straightness.build_games import CLASSES

REL_WINDOW = (-8, 12)  # positions relative to the injected move
TURN_OFFSET, VALUE_OFFSET = 2, 1  # column -> board position; see align_to_event
KDE_GRID, KDE_BOOT, KDE_SEED = 512, 1000, 0

DRAWN_CLASSES = [c for c in CLASSES if c in CLASS_STYLE]


def class_blocks(n_total: int) -> dict[str, slice]:
    """games.pt block layout, restricted to the classes the figures draw.

    CLASSES fixes the block order in the data and every geometry npz indexes into
    it by position, so a class leaves the study's figures by losing its
    CLASS_STYLE entry, never by leaving CLASSES.
    """
    n = n_total // len(CLASSES)
    blocks = {c: slice(i * n, (i + 1) * n) for i, c in enumerate(CLASSES)}
    return {c: blocks[c] for c in DRAWN_CLASSES}


def mean_sem(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = (~np.isnan(x)).sum(0)
    return np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(np.maximum(n, 1))


def ci_bars(ax, x, y, yerr, color, zorder: int = 6) -> None:
    """95% CI bars at the drawn markers, the uncertainty display these panels use.

    Thin and marker-anchored, so four classes on one axis stay separable.
    """
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.9,
                capsize=1.8, zorder=zorder)


def kde_ci(vals: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian KDE (Scott's rule) with a percentile bootstrap band over games.

    Resamples reweight one fixed kernel matrix, so the bandwidth stays the
    full-sample one instead of shifting with each resample's spread.
    """
    n = len(vals)
    h = vals.std(ddof=1) * n ** -0.2
    kernel = np.exp(-0.5 * ((grid[:, None] - vals) / h) ** 2) / (h * np.sqrt(2 * np.pi))
    w = np.random.default_rng(KDE_SEED).multinomial(n, np.full(n, 1 / n), KDE_BOOT)
    lo, hi = np.percentile(kernel @ w.T / n, [2.5, 97.5], axis=1)
    return kernel.mean(1), lo, hi


def density_panel(ax, per_game: np.ndarray, blocks: dict, styles: dict) -> None:
    """Per-game mean turning angle, one KDE curve per class.

    Medians are marked on the baseline: the gap between two triangles is the
    class separation the AUROC figures score. The grid spans the drawn classes
    only, so an undrawn block cannot stretch the axis. Set the y-floor once after
    every panel is drawn — set_ylim freezes autoscale, which on shared axes would
    clip whichever panel peaks highest.

    `styles` is passed rather than assumed, as in `embedding_panels.class_cloud`:
    a panel may draw the game classes or the constraint rules that replace the
    violation class, and those live in two style tables.

    The curves carry no interval: densities whose medians sit ~0.3° apart overlap
    too much for a per-curve interval to stay readable. The separation they show
    is scored with a bootstrap CI in the AUROC figures instead.
    """
    drawn = np.concatenate([per_game[b] for b in blocks.values()])
    pad = 0.03 * np.ptp(drawn)
    grid = np.linspace(drawn.min() - pad, drawn.max() + pad, KDE_GRID)
    for i, c in enumerate(blocks):
        style, is_ref = styles[c], c == "legal"
        vals = per_game[blocks[c]]
        dens, _, _ = kde_ci(vals, grid)
        body = np.flatnonzero(dens > 0.02 * dens.max())  # no markers on flat tails
        ax.plot(grid, dens, lw=2.4 if is_ref else 1.6, color=style["color"],
                marker=style["marker"], ms=4, markevery=body[12 * i::72].tolist(),
                markeredgecolor="white", markeredgewidth=0.5,
                label=style["label"], zorder=5 if is_ref else 3)
        ax.plot(np.median(vals), 0, marker="v", ms=5, color=style["color"],
                markeredgecolor="white", markeredgewidth=0.5, clip_on=False,
                zorder=6)
    ax.set_xlabel("per-game mean turning angle (deg)")
    ax.set_ylabel("density")


def align_to_event(delta: np.ndarray, t0: np.ndarray,
                   offset: int) -> tuple[np.ndarray, np.ndarray]:
    """[n, T] -> (rels, [n, len(rels)]) with column t0+r at relative offset r.

    `offset` maps a board position to its column: TURN_OFFSET for a turning
    angle, whose column k is the turn at position k+2, and VALUE_OFFSET for a
    quantity attached to a position (norm, L0, step length), whose column j is
    position j+1 because position 0 is dropped.
    """
    rels = np.arange(REL_WINDOW[0], REL_WINDOW[1] + 1)
    aligned = np.full((len(delta), len(rels)), np.nan)
    for i, r in enumerate(rels):
        k = t0 + r - offset
        ok = (k >= 0) & (k < delta.shape[1])
        aligned[ok, i] = delta[np.arange(len(delta))[ok], k[ok]]
    return rels, aligned


def violation_delta(theta: np.ndarray, t0: np.ndarray, src: np.ndarray,
                    viol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rels, mean, sem) of Δ turning angle vs. the matched source game.

    Δ is identically zero for r <= -2: the games share every activation before
    t0, and the turn at t0-2 is the last one whose window excludes t0.
    """
    rels, aligned = align_to_event(theta[viol] - theta[src], t0, TURN_OFFSET)
    return (rels, *mean_sem(aligned))


def value_delta(values: np.ndarray, t0: np.ndarray, src: np.ndarray,
                viol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Event-locked Δ for a per-position quantity indexed by position directly
    (norm, L0), rather than by turn like theta. Δ ≡ 0 for r <= -1 here: the
    activation at t0-1 already excludes the injected token."""
    rels, aligned = align_to_event(values[viol] - values[src], t0, VALUE_OFFSET)
    return (rels, *mean_sem(aligned))


def value_profile_panel(ax, values: np.ndarray, blocks: dict, ylabel: str,
                        offset: int = 1) -> None:
    """A per-position quantity vs. board position, one line per class.

    `offset` maps array index to board position: 1 for values attached to a
    position (norm, L0), 2 for step lengths, whose index k spans positions
    k+1 -> k+2 and is plotted at the position the step ends on.
    """
    positions = np.arange(values.shape[1]) + offset
    for i, c in enumerate(blocks):
        style, is_ref = CLASS_STYLE[c], c == "legal"
        mu, sem = mean_sem(values[blocks[c]])
        at = np.arange(2 * i, len(positions), 9)  # the markevery positions below
        ci_bars(ax, positions[at], mu[at], 1.96 * sem[at], style["color"])
        ax.plot(positions, mu, lw=2.4 if is_ref else 1.6, color=style["color"],
                marker=style["marker"], ms=4, markevery=(2 * i, 9),
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=5 if is_ref else 3, label=style["label"])
    ax.set_xlabel("board position t")
    ax.set_ylabel(ylabel)


def profile_panel(ax, theta: np.ndarray, blocks: dict) -> None:
    """Mean turning angle vs. board position, one line per class."""
    positions = np.arange(theta.shape[1]) + 2
    for i, c in enumerate(blocks):
        style, is_ref = CLASS_STYLE[c], c == "legal"
        mu, sem = mean_sem(theta[blocks[c]])
        at = np.arange(2 * i, len(positions), 9)  # the markevery positions below
        ci_bars(ax, positions[at], mu[at], 1.96 * sem[at], style["color"])
        ax.plot(positions, mu, lw=2.4 if is_ref else 1.6, color=style["color"],
                marker=style["marker"], ms=4, markevery=(2 * i, 9),
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=5 if is_ref else 3, label=style["label"])
    ax.set_xlabel("board position t")
    ax.set_ylabel("mean turning angle (deg)")


def event_axes(ax, prefix_end: float = -1.5) -> None:
    """Event-locked furniture: shaded Δ ≡ 0 prefix, zero line, t0 line, x axis.

    `prefix_end` bounds the shaded region: -1.5 for turning angles (the turn at
    t0-1 already sees t0), -0.5 for step lengths and per-position values like
    norm and L0, which stay zero one position longer.
    """
    ax.axvspan(REL_WINDOW[0] - 0.5, prefix_end, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("position relative to injected move ($t - t_0$)")
    ax.set_xticks(range(REL_WINDOW[0], REL_WINDOW[1] + 1, 4))


def violation_panel(ax, rels: np.ndarray, mu: np.ndarray, sem: np.ndarray,
                    ghost: np.ndarray | None = None, annotate: bool = True,
                    prefix_end: float = -1.5, fmt: str = "{:.1f}°") -> None:
    """Event-locked Δ for one quantity. `ghost` overlays a reference space in gray."""
    color = CLASS_STYLE["violation"]["color"]
    event_axes(ax, prefix_end)
    if ghost is not None:
        ax.plot(rels, ghost, "-", lw=1.4, color="#b0b0b0", zorder=2,
                label="raw L5 (reference)")
    ci_bars(ax, rels, mu, 1.96 * sem, color, zorder=3)
    ax.plot(rels, mu, "-", marker=CLASS_STYLE["violation"]["marker"], lw=2.0, ms=5,
            color=color, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    if annotate:
        peak = int(np.abs(mu).argmax())  # largest excursion, either sign
        dy = 0.12 * (mu.max() - mu.min() or 1.0)  # scale-relative: norms are ~1e-3
        ax.annotate(f"{fmt.format(mu[peak])} at $t_0${rels[peak]:+d}",
                    xy=(rels[peak], mu[peak]),
                    xytext=(rels[peak] + 2.2, mu[peak] - np.sign(mu[peak] or 1) * dy),
                    fontsize=8, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
