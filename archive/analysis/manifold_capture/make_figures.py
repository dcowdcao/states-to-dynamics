"""Render all manifold-capture figures from results/manifold_capture/ artifacts.

Paper-figure map (arXiv:2604.28119):
  manifold_pca_<concept>.png       Fig 2   ambient PCA colored by concept value
  piecewise_<concept>[_c].png      Fig 5   SAE top-k recons vs ambient PCA
  latent_<concept>.png             (ours)  ambient PCA vs each SAE's latent PCA
  capture_<concept>[_c].png        Fig 6   restricted-R2 curves vs PCA/random
  tuning_<concept>.png             Fig 7   top-atom tuning curves along value
  fourviews_<concept>_<space>.png  Fig 9L  PCA(X)/proj/partial/codes-as-coords
  similarity_<concept>_<space>.png Fig 9R  5 feature-similarity heatmaps
  receptive_<concept>_<space>.png  Fig 8   argmax-atom receptive fields
  landmarks_<concept>.png          App C   SpaDE landmark placement + entropy
  regimes.png                      Def 6   capture/shattering/dilution table

Usage: python analysis/manifold_capture/make_figures.py [--only capture,tuning]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spade.figures.karvonen_plot import SAE_CONFIG, TRAINER_CLASS_MAP
from spade.manifold_capture.concepts import concepts
from spade.manifold_capture.spaces import LOW_VE, POINT_BASED

plt.rcParams.update({"font.family": "serif", "font.size": 10})
DPI = 100
RESULTS = Path(__file__).resolve().parents[2] / "results" / "manifold_capture"
FIG_DIR = Path(__file__).resolve().parent / "figures"
RECON_KS = [1, 2, 4, 8, 16]
SQUARE_COLORS = ["#b0b0b0", "#d62728", "#1f77b4"]  # empty / theirs / mine
SHUFFLE_SEED = 0

CONCEPTS = {c.name: c for c in concepts()}


def display(space: str) -> tuple[str, str, str]:
    """(label, color, linestyle) per space; dashed separates shared-hue families."""
    fixed = {"raw": ("raw", "black", "-"), "random_proj": ("rand-proj", "0.6", ":"),
             "random_topk": ("rand-TopK", "0.4", ":"), "baseline": ("baseline", "black", "--")}
    if space in fixed:
        return fixed[space]
    key = space.removesuffix("-othello")
    if key in TRAINER_CLASS_MAP:  # the 8 baselines
        cfg = SAE_CONFIG[TRAINER_CLASS_MAP[key]]
        dashed = key in {"PAnnealTrainer", "GatedAnnealTrainer", "BatchTopKTrainer"}
        label = cfg["name"]
    else:  # sweep-style config names, e.g. SpaDE-lam1e-03-gam1e-04
        cfg = SAE_CONFIG["cosine_spade" if "CosineSpaDE" in key else "sparsemax_dist"]
        dashed, label = True, key
    if space in LOW_VE:
        label += " (low VE)"
    return label, cfg["color"], "--" if dashed else "-"


def value_colors(concept_name: str, values: np.ndarray):
    """(colors, cmap_or_None) — sequential ramp for continuous, fixed hues for classes."""
    c = CONCEPTS[concept_name]
    if c.kind == "continuous":
        lo, hi = values.min(), values.max()
        return plt.cm.viridis((values - lo) / max(hi - lo, 1e-9)), plt.cm.viridis
    palette = SQUARE_COLORS if len(c.class_names) == 3 else \
        [plt.cm.tab10(i) for i in range(len(c.class_names))]
    return np.array([matplotlib.colors.to_rgba(palette[int(v)]) for v in values]), None


def scatter3(ax, coords, colors, title=""):
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(coords))
    ax.scatter(*coords[order].T.astype(np.float32), c=colors[order], s=2, alpha=0.6,
               linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def sae_spaces_in(npz, prefix: str, variant: str) -> list[str]:
    seen = [k[len(prefix):].rsplit(f"_{variant}", 1)[0] for k in npz.files
            if k.startswith(prefix) and k.endswith(f"_{variant}_k1" if "recon" in prefix
                                                   else f"_{variant}")]
    return sorted(dict.fromkeys(seen))


def fig_manifold_pca():
    for name, c in CONCEPTS.items():
        f = RESULTS / f"views_{name}.npz"
        if not f.exists():
            continue
        npz = np.load(f)
        colors, _ = value_colors(name, npz["values"])
        variants = [v for v in c.variants if f"pca3_{v}" in npz]
        fig = plt.figure(figsize=(4 * len(variants), 4))
        for i, v in enumerate(variants):
            ax = fig.add_subplot(1, len(variants), i + 1, projection="3d")
            evr = npz[f"evr_{v}"].sum()
            scatter3(ax, npz[f"pca3_{v}"], colors, f"{name} ({v}, evr {evr:.2f})")
        _save(fig, f"manifold_pca_{name}.png")


def fig_piecewise():
    for name in CONCEPTS:
        f = RESULTS / f"views_{name}.npz"
        if not f.exists():
            continue
        npz = np.load(f)
        colors, _ = value_colors(name, npz["values"])
        for variant, suffix in [("raw", ""), ("centered", "_c")]:
            spaces_here = sae_spaces_in(npz, "recon_", variant)
            if not spaces_here:
                continue
            fig = plt.figure(figsize=(2.2 * (len(RECON_KS) + 1), 2.2 * len(spaces_here)))
            for r, sp in enumerate(spaces_here):
                label, _, _ = display(sp)
                ax = fig.add_subplot(len(spaces_here), len(RECON_KS) + 1,
                                     r * (len(RECON_KS) + 1) + 1, projection="3d")
                scatter3(ax, npz[f"pca3_{variant}"], colors,
                         f"{label}\nambient PCA" if r == 0 else label)
                for j, k in enumerate(RECON_KS):
                    ax = fig.add_subplot(len(spaces_here), len(RECON_KS) + 1,
                                         r * (len(RECON_KS) + 1) + 2 + j, projection="3d")
                    scatter3(ax, npz[f"recon_{sp}_{variant}_k{k}"], colors,
                             f"k={k}" if r == 0 else "")
            _save(fig, f"piecewise_{name}{suffix}.png")


def fig_latent():
    for name in CONCEPTS:
        f = RESULTS / f"views_{name}.npz"
        if not f.exists():
            continue
        npz = np.load(f)
        colors, _ = value_colors(name, npz["values"])
        spaces_here = sae_spaces_in(npz, "latent_pca3_", "raw")
        if not spaces_here:
            continue
        cols = len(spaces_here) + 1
        fig = plt.figure(figsize=(2.6 * cols, 2.8))
        ax = fig.add_subplot(1, cols, 1, projection="3d")
        scatter3(ax, npz["pca3_raw"], colors, "activation space\n(ambient PCA)")
        for i, sp in enumerate(spaces_here):
            ax = fig.add_subplot(1, cols, i + 2, projection="3d")
            evr = npz[f"latent_evr_{sp}_raw"].sum()
            scatter3(ax, npz[f"latent_pca3_{sp}_raw"], colors,
                     f"{display(sp)[0]}\nlatent PCA (evr {evr:.2f})")
        _save(fig, f"latent_{name}.png")


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_capture():
    path = RESULTS / "capture_curves.csv"
    if not path.exists():
        return
    rows = _read_csv(path)
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["concept"], r["variant"], r["space"], r["method"])].append(
            (int(r["n"]), float(r["r2"])))
    for name in CONCEPTS:
        for variant, suffix in [("raw", ""), ("centered", "_c"), ("curve", "_curve")]:
            methods = [("greedy", "span-projection R$^2$"),
                       ("greedy_codes", "code-masked R$^2$")]
            keys = [k for k in by_key if k[0] == name and k[1] == variant]
            if not keys:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, (method, title) in zip(axes, methods):
                for space in sorted({k[2] for k in keys}):
                    for m in ([method] if space != "baseline" else ["pca", "random"]):
                        pts = sorted(by_key.get((name, variant, space, m), []))
                        if not pts:
                            continue
                        label, color, ls = display(space)
                        if space == "baseline":
                            label, ls = m, {"pca": "--", "random": ":"}[m]
                        ax.plot(*zip(*pts), color=color, ls=ls, lw=1.4, label=label)
                ax.set_xscale("log")
                ax.set_xlabel("restricted support size")
                ax.set_title(title, fontsize=9)
                ax.grid(alpha=0.25, lw=0.5)
            axes[0].set_ylabel("variance explained")
            axes[0].legend(fontsize=6.5, ncol=2, loc="lower right")
            fig.suptitle(f"subspace capture: {name} ({variant})", fontsize=11)
            _save(fig, f"capture_{name}{suffix}.png")


def fig_tuning():
    for name, c in CONCEPTS.items():
        f = RESULTS / f"tuning_{name}.npz"
        if not f.exists():
            continue
        npz = np.load(f)
        spaces_here = sorted({k.rsplit("_curves", 1)[0] for k in npz.files
                              if k.endswith("_curves")})
        if not spaces_here:
            continue
        fig, axes = plt.subplots(1, len(spaces_here),
                                 figsize=(2.4 * len(spaces_here), 2.6), sharey=True)
        for ax, sp in zip(np.atleast_1d(axes), spaces_here):
            curves = npz[f"{sp}_curves"]
            for i, curve in enumerate(curves):
                ax.plot(npz["bin_centers"], curve, lw=1,
                        color=plt.cm.plasma(i / max(len(curves) - 1, 1)), alpha=0.85)
            ax.set_title(display(sp)[0], fontsize=8)
            ax.set_xlabel(name, fontsize=8)
            ax.grid(alpha=0.2, lw=0.4)
        np.atleast_1d(axes)[0].set_ylabel("normalized activation", fontsize=8)
        fig.suptitle(f"top-atom tuning curves along {name}", fontsize=10)
        _save(fig, f"tuning_{name}.png")


def fig_fourviews_similarity():
    for f in sorted(RESULTS.glob("geometry_*.npz")):
        concept_name, space = f.stem[len("geometry_"):].rsplit("_", 1)
        npz = np.load(f)
        colors, _ = value_colors(concept_name, npz["values"])

        fig = plt.figure(figsize=(11, 3))
        titles = [("pca_x", "PCA(X)"), ("pca_proj", "PCA(proj on top atoms)"),
                  ("pca_partial", "PCA(partial recon)"), ("codes_top", "codes as coords")]
        for i, (key, title) in enumerate(titles):
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            scatter3(ax, npz[key], colors, title)
        fig.suptitle(f"{concept_name} / {display(space)[0]}", fontsize=10)
        _save(fig, f"fourviews_{concept_name}_{space}.png")

        order = np.argsort(npz["communities"])
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for ax, key in zip(axes, ["cosine", "coact", "corr", "pmi", "ising"]):
            m = npz[key][np.ix_(order, order)]
            if key == "ising":
                lim = max(np.abs(m).max(), 1e-9)
                im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(m, cmap="magma")
            ax.set_title(key, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"feature similarity (community-ordered): "
                     f"{concept_name} / {display(space)[0]}", fontsize=10)
        _save(fig, f"similarity_{concept_name}_{space}.png")


def fig_receptive():
    for f in sorted(RESULTS.glob("receptive_*.npz")):
        concept_name, space = f.stem[len("receptive_"):].rsplit("_", 1)
        npz = np.load(f)
        atom_colors = plt.cm.tab20(npz["argmax_atom"].astype(np.int64) % 20)
        err = npz["recon_err"].astype(np.float32)
        size = 6 * np.exp(-(err / max(np.median(err), 1e-9)) ** 2)  # low error = big
        variants = [v for v in ["raw", "centered"] if f"coords_{v}" in npz]
        fig = plt.figure(figsize=(4 * len(variants), 4))
        for i, v in enumerate(variants):
            ax = fig.add_subplot(1, len(variants), i + 1, projection="3d")
            order = np.random.default_rng(SHUFFLE_SEED).permutation(len(err))
            ax.scatter(*npz[f"coords_{v}"][order].T.astype(np.float32),
                       c=atom_colors[order], s=size[order], alpha=0.6, linewidths=0)
            ax.set_title(f"{v} ({int(npz['n_distinct_argmax'])} argmax atoms)", fontsize=8)
            ax.set_axis_off()
        fig.suptitle(f"receptive fields: {concept_name} / {display(space)[0]}", fontsize=10)
        _save(fig, f"receptive_{concept_name}_{space}.png")


def fig_landmarks():
    lemma_path = RESULTS / "landmarks_lemma1.csv"
    if not lemma_path.exists():
        return
    lemma = [r for r in _read_csv(lemma_path) if r["metric_space"] == "euclidean"]
    scale = {(r["space"], r["concept"]): float(r["manifold_scale"])
             for r in _read_csv(RESULTS / "landmarks.csv")
             if r["metric_space"] == "euclidean"}
    for space in POINT_BASED:
        rows = [r for r in lemma if r["space"] == space]
        ent_f = RESULTS / f"landmark_entropy_{space}.npz"
        if not rows or not ent_f.exists():
            continue
        ent = np.load(ent_f)
        concept_names = [c for c in CONCEPTS
                         if any(r["concept"] == c for r in rows)]
        fig, axes = plt.subplots(2, len(concept_names),
                                 figsize=(2.6 * len(concept_names), 5.2), squeeze=False)
        for j, cname in enumerate(concept_names):
            crows = [r for r in rows if r["concept"] == cname]
            s = scale[(space, cname)]
            ax = axes[0][j]  # Lemma 1: landmark NN dist to each class vs joint
            xs = np.arange(len(crows))
            ax.bar(xs, [float(r["landmark_nn_to_class_mean"]) / s for r in crows],
                   color="#377eb8", label="to class")
            ax.axhline(float(crows[0]["landmark_nn_to_joint_mean"]) / s,
                       color="black", ls="--", lw=1, label="to joint")
            ax.set_xticks(xs, [r["class"] for r in crows], rotation=45,
                          ha="right", fontsize=6)
            ax.set_title(cname, fontsize=8)
            if j == 0:
                ax.set_ylabel("landmark NN dist / manifold scale", fontsize=7)
                ax.legend(fontsize=6)
            ax = axes[1][j]  # firing-class entropy: 0 = 1-to-1, ln(k) = joint tiling
            e = ent[f"{cname}_entropy"].astype(np.float32)
            n_classes = len({r["class"] for r in crows})
            ax.hist(e, bins=30, color="#ff7f00")
            ax.axvline(np.log(max(n_classes, 2)), color="black", ls=":", lw=1)
            ax.set_xlabel("firing-class entropy", fontsize=7)
            if j == 0:
                ax.set_ylabel("landmarks", fontsize=7)
        fig.suptitle(f"{display(space)[0]}: landmark placement (top: Lemma 1 test; "
                     f"dotted = max entropy)", fontsize=10)
        _save(fig, f"landmarks_{space}.png")


def fig_regimes():
    path = RESULTS / "regimes.csv"
    if not path.exists():
        return
    all_rows = _read_csv(path)
    # headline regimes use the per-concept centroid-curve variant (its k_M is
    # concept-specific); fall back to raw for results predating the curve rows
    rows = [r for r in all_rows if r["variant"] == "curve"] or \
        [r for r in all_rows if r["variant"] == "raw" or
         CONCEPTS[r["concept"]].variants == ("raw",)]
    spaces_here = sorted({r["space"] for r in rows}, key=lambda s: display(s)[0])
    concept_names = [c for c in CONCEPTS if any(r["concept"] == c for r in rows)]
    regime_color = {"capture": "#2ca02c", "shattering": "#d62728",
                    "dilution": "#ff7f00", "mixed": "#b0b0b0"}
    grid = np.full((len(spaces_here), len(concept_names)), np.nan)
    fig, ax = plt.subplots(figsize=(1.1 * len(concept_names) + 3,
                                    0.45 * len(spaces_here) + 1.5))
    for r in rows:
        i, j = spaces_here.index(r["space"]), concept_names.index(r["concept"])
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=regime_color[r["regime"]],
                                   alpha=0.75))
        ax.text(j + 0.5, i + 0.5, f'{r["support_size"]}|{r["k_M"]}\n'
                f'ρ={float(r["rho"]):.2f}' if r["rho"] else "",
                ha="center", va="center", fontsize=6)
    ax.set_xlim(0, len(concept_names)), ax.set_ylim(len(spaces_here), 0)
    ax.set_xticks(np.arange(len(concept_names)) + 0.5, concept_names,
                  rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(spaces_here)) + 0.5,
                  [display(s)[0] for s in spaces_here], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75)
               for c in regime_color.values()]
    ax.legend(handles, regime_color, fontsize=7, bbox_to_anchor=(1.02, 1),
              loc="upper left")
    ax.set_title("capture regimes (cell: support|k_M, ρ = signed cohesion)", fontsize=10)
    _save(fig, "regimes.png")


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}")


FAMILIES = {
    "manifold_pca": fig_manifold_pca,
    "piecewise": fig_piecewise,
    "latent": fig_latent,
    "capture": fig_capture,
    "tuning": fig_tuning,
    "fourviews": fig_fourviews_similarity,
    "receptive": fig_receptive,
    "landmarks": fig_landmarks,
    "regimes": fig_regimes,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma filter over figure families")
    args = parser.parse_args()
    only = args.only.split(",") if args.only else list(FAMILIES)
    for fam in only:
        FAMILIES[fam]()


if __name__ == "__main__":
    main()
