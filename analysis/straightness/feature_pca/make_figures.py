"""PCA of the 21-D ReStraV feature vector, one panel per representation space.

Each game is one point: the 21 components are its first 7 step lengths, first 6
turning angles, and the 4 moments of each (src/spade/straightness/geometry.py).
This is the space the LR-21 probe classifies in, viewed unsupervised.

The illegal move is drawn as the three constraint rules it can break rather than
as one violation class, and that costs a source. The straightness study's own
violation class is built from empty-square categories only, so it carries no R1 at
all; the rules come off the violation-divergence branch population instead, where
every one of 2000 games carries a branch for every rule
(spade.violation_divergence.rule_curvature). Both sides drop the attention sink
and measure per ply, which is what makes their descriptors the same number -- the
check is in analysis/violation_divergence/rule_curvature/README.md. What differs
is the construction: a rule branch keeps its source game's real suffix, while the
game classes are whole games. The caption says so.

Every space gets its own PCA, fit on that space's z-scored features over the six
classes drawn. Raw covariance PCA would answer a different question here: the
components mix degrees with activation-scale lengths, whose absolute variances
differ by ~450x across dictionaries (SpaDE step lengths ~0.05, raw layer 5 ~22.8),
so PC1 lands on turning-angle variance in every space and reports units, not
structure.

Usage: python analysis/straightness/feature_pca/make_figures.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.study import study_dir
from spade.figures.embedding_panels import class_cloud
from spade.figures.paper_style import (
    CLASS_STYLE, MUTED, VIOLATION_RULE_STYLE, apply_style)
from spade.figures.straightness_panels import class_blocks
from spade.straightness.geometry import RESTRAV_FEATURE_NAMES, restrav_features
from spade.violation_divergence import rule_curvature as rc
from spade.violation_divergence.fork_plane import spaces as fork_spaces

apply_style()

FIG_DIR = Path(__file__).resolve().parent / "figures"
GEOM = study_dir("straightness") / "geometry"
TREE = "branches"  # the spliced suffix; the rules are drawn beside whole games
STYLE = CLASS_STYLE | VIOLATION_RULE_STYLE
# Legend order, reference first. The study's violation class is what the three
# rules replace, so it is drawn by none of these.
DRAWN = [c for c in CLASS_STYLE if c != "violation"] + list(VIOLATION_RULE_STYLE)


@dataclass
class Fit:
    vecs: np.ndarray          # top-2 loadings, sign-pinned against raw L5
    evr: np.ndarray           # variance ratio per component
    shift: dict[str, np.ndarray]  # class mean minus legal's, per PC, in pooled SD


def spaces() -> list[tuple[str, object]]:
    """(label, SAE entry) for raw layer 5 then every SAE family, sorted by family.

    One list for both sources: `rule_curvature.space_key` names the divergence
    cache and the straightness geometry npz with the same stem.
    """
    sae = sorted(fork_spaces()[1:], key=lambda s: rc.space_key(s[1]))
    return [("raw L5 (reference)", None)] + sae


def features(entry) -> tuple[np.ndarray, np.ndarray]:
    """One space's drawn descriptors, and the class each row belongs to.

    A rule's descriptor is the mean over its member categories of each member
    branch's own 21-D vector, one per game -- the members are not drawn separately.
    """
    g = np.load(GEOM / f"{rc.space_key(entry)}_raw.npz")
    games = restrav_features(torch.from_numpy(g["d"]),
                             torch.from_numpy(g["theta"])).numpy()
    rules, _ = rc.descriptors(TREE, entry)
    assert np.isfinite(rules).all(), (
        f"{rc.space_key(entry)}: a rule descriptor is undefined, but every "
        f"rule-complete game carries every rule")

    blocks = [(c, games[b]) for c, b in class_blocks(len(games)).items()
              if c != "violation"]
    blocks += [(rule, rules[:, j]) for j, rule in enumerate(rc.RULES)]
    return (np.concatenate([f for _, f in blocks]),
            np.concatenate([np.full(len(f), c) for c, f in blocks]))


def pca(f21: np.ndarray, align: np.ndarray | None = None):
    """Top-2 PCs of the drawn games' z-scored features: (scores, ratios, loadings).

    Component sign is arbitrary, so it is pinned: the reference space puts its
    largest loading positive, every other space matches the reference's sign, so
    panels are not mirrored against each other for no reason.
    """
    z = (f21 - f21.mean(0)) / f21.std(0)
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    v = vt[:2]
    sign = np.sign(v @ align.T).diagonal() if align is not None else \
        np.sign(v[np.arange(2), np.abs(v).argmax(1)])
    v = v * sign[:, None]
    return z @ v.T, (sv ** 2 / (sv ** 2).sum())[:2], v


def class_shift(scores: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Each non-legal class mean minus legal's, per PC, in pooled SD.

    The number behind the suptitle: how far a class moves in the plane the panel
    draws, on a scale the panels share even though their bases do not.
    """
    legal = scores[labels == "legal"]
    return {c: (scores[labels == c].mean(0) - legal.mean(0))
            / np.sqrt(0.5 * (scores[labels == c].var(0) + legal.var(0)))
            for c in DRAWN[1:]}


def restrav_pca() -> dict[str, Fit]:
    """Small multiples of the top-2 PC plane. Returns each space's fit."""
    panels = spaces()
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 3.0 * nrow),
                             sharex=True, sharey=True)
    fits, raw_loadings = {}, None
    for ax, (label, entry) in zip(axes.flat, panels):
        f21, labels = features(entry)
        scores, evr, vecs = pca(f21, align=raw_loadings)
        if raw_loadings is None:
            raw_loadings = vecs
        fits[label] = Fit(vecs, evr, class_shift(scores, labels))
        class_cloud(ax, scores, labels, STYLE, DRAWN, reference="legal")
        lead = [RESTRAV_FEATURE_NAMES[i] for i in np.abs(vecs).argmax(1)]
        ax.set_title(f"{label}\nPC1 {evr[0]:.0%} ({lead[0]})  ·  PC2 {evr[1]:.0%} "
                     f"({lead[1]})", loc="left", fontsize=8.5)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Two columns of three: whole-game classes on the left, the three rules on the
    # right, which is the split the reader has to hold to read the panels.
    legend = fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
                        columnspacing=2.5, bbox_to_anchor=(0.5, -0.055))
    for h in legend.legend_handles:
        h.set_alpha(1)
    fig.suptitle("Championship games are the only class that moves in the 21-D "
                 "descriptor's principal plane — the three rules do not",
                 fontsize=11, y=1.00)
    fig.text(0.5, 0.966, "one point per game in the top-2 principal plane of that "
             "space's z-scored ReStraV features, drawn in one shuffled pass; "
             "ellipses enclose 50% of a Gaussian fit\nper class and large markers "
             "are class means. R1/R2/R3 are branched moves inside games, not whole "
             "games: one point is one game's mean over that\nrule's member "
             "categories, on a branch keeping its source game's suffix. Each space "
             "fits its own basis on the six classes drawn, so panels compare\nonly "
             "in within-panel arrangement, never point-for-point (alignment with "
             "raw L5: README). PCA is unsupervised — the supervised LR-21 probe\n"
             "separates classes this plane does not. Raw (uncentered) variant.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.925))
    fig.savefig(FIG_DIR / "restrav_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fits


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fits = restrav_pca()
    raw = fits["raw L5 (reference)"]
    ood = list(raw.shift)
    print(f"{'space':34} {'PC1%':>5} {'PC2%':>5} {'|cos| raw PC1':>14} {'PC2':>5}"
          + "".join(f"{c.split()[0][:5] + ' d(PC1,PC2)':>22}" for c in ood))
    for label, fit in fits.items():
        cos = np.abs((fit.vecs @ raw.vecs.T).diagonal())
        shift = "".join(f"{fit.shift[c][0]:+11.2f},{fit.shift[c][1]:+10.2f}" for c in ood)
        print(f"{label:34} {fit.evr[0]:5.0%} {fit.evr[1]:5.0%} {cos[0]:14.2f} "
              f"{cos[1]:5.2f}" + shift)
    print(f"figure written to {FIG_DIR}")


if __name__ == "__main__":
    main()
