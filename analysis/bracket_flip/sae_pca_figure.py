"""Fig 3 — the bracketing flip in PCA, one panel per representation space.

For raw layer-5 and each baseline SAE family, the before/after/control
activations are encoded into that space, per-move-number centered (t owns the
leading component otherwise), and projected to their own top-2 PCs. The question
the grid answers: does the flip's condition structure survive the SAE
bottleneck, and does it differ by family?

Each panel has its own PCA basis, so positions are NOT comparable across panels
(FIGURE_GUIDELINES 4.6 facet rule) — only the within-panel arrangement of the
three conditions is. Runs on CPU.

Usage: python analysis/bracket_flip/sae_pca_figure.py [--camera-ready]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.study import study_dir
from spade.figures.embedding_panels import class_cloud
from spade.figures.paper_style import EVENT_STYLE, INK, apply_style, sae_label, save_figure
from spade.spaces import load_space

warnings.filterwarnings("ignore")
LAYER = 5
CONDITIONS = ["before", "after", "control"]
# raw first (the reference space), then one canonical checkpoint per SAE family.
FAMILIES = ["BatchTopKTrainer-othello", "CosineSpaDETrainer-othello",
            "GatedAnnealTrainer-othello", "GatedSAETrainer-othello",
            "PAnnealTrainer-othello", "SpaDETrainer-othello",
            "StandardTrainer-othello", "TopKTrainer-othello"]


def load_conditions() -> tuple[dict, np.ndarray]:
    ev = study_dir("bracket") / "events"
    cond = {c: torch.load(ev / f"{c}.pt", weights_only=True)[:, LAYER].float()
            for c in CONDITIONS}
    import pandas as pd
    t = pd.read_csv(ev / "events.csv")["t"].to_numpy()
    return cond, t


def entry_for(name: str):
    from spade.spaces import representative_spaces
    return dict(representative_spaces())[name]


def per_t_centered_pca(pooled: np.ndarray, t3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-2 PCs after subtracting each move number's pooled mean. Same
    centering as F3's pooled_centered, so the raw panel matches F3(d)."""
    x = pooled.copy()
    for value in np.unique(t3):
        x[t3 == value] -= x[t3 == value].mean(0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T, s[:2] ** 2 / (s ** 2).sum()


def space_scores(name: str, cond: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fn = (lambda z: z) if name == "raw_pre" else load_space(name, None if name == "raw"
                                                            else entry_for(name), "cpu")
    with torch.no_grad():
        latents = [fn(cond[c]).cpu().numpy().astype(np.float64) for c in CONDITIONS]
    pooled = np.concatenate(latents)
    t3 = np.tile(t, len(CONDITIONS))
    scores, ratios = per_t_centered_pca(pooled, t3)
    return scores, ratios


def draw_panel(ax, fig, scores: np.ndarray, ratios: np.ndarray, title: str) -> None:
    n = len(scores) // len(CONDITIONS)
    labels = np.repeat(np.arange(len(CONDITIONS)), n)
    styles = {i: EVENT_STYLE[c] for i, c in enumerate(CONDITIONS)}
    class_cloud(ax, scores, labels, styles, list(styles), reference=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({ratios[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.1%} var)")
    ax.set_title(title, fontsize=9)


def cached_scores(spaces: list[str], cond: dict, t: np.ndarray) -> dict:
    """Per-space (scores, ratios), cached: encoding loads 8 checkpoints, but the
    2-D coordinates are tiny, so restyling the figure never reloads them."""
    cache = study_dir("bracket") / "sae_pca" / "scores.npz"
    if cache.exists():
        d = np.load(cache)
        return {n: (d[f"{n}_s"], d[f"{n}_r"]) for n in spaces}
    out = {n: space_scores(n, cond, t) for n in spaces}
    for n in spaces:
        print(f"encoded {n}: PC1 {out[n][1][0]:.1%}, PC2 {out[n][1][1]:.1%}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{n}_s": out[n][0] for n in spaces},
             **{f"{n}_r": out[n][1] for n in spaces})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-ready", action="store_true")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)

    cond, t = load_conditions()
    spaces = ["raw"] + FAMILIES
    scored = cached_scores(spaces, cond, t)
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 9.6), squeeze=False)
    flat = axes.ravel()
    for ax, name in zip(flat, spaces):
        scores, ratios = scored[name]
        title = "raw L5 (reference)" if name == "raw" else sae_label(
            name.removesuffix("Trainer-othello"))
        draw_panel(ax, fig, scores, ratios, title)
        if name != "raw":  # one legend, on the reference panel, is enough
            ax.legend_.remove() if ax.get_legend() else None
        print(f"drew {name}: PC1 {ratios[0]:.1%}, PC2 {ratios[1]:.1%}")
    for ax in flat[len(spaces):]:
        ax.set_visible(False)

    flat[0].legend(loc="best", fontsize=7, title="per-t centered; each panel\n"
                   "has its own PCA basis (not comparable across panels)",
                   title_fontsize=6.5, labelspacing=0.3, frameon=True,
                   framealpha=0.85, edgecolor="none")
    if not args.camera_ready:
        fig.suptitle("F8 — the bracketing flip in PCA across representation spaces "
                     "(layer 5; before/after/control, per-t centered)", y=0.998)
    fig.tight_layout()
    save_figure(fig, Path(__file__).resolve().parent / "figures" / "fig3_sae_pca_grid",
                args.camera_ready)


if __name__ == "__main__":
    main()
