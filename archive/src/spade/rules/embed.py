"""Step 3 of the rules study: 2D PCA + Isomap embeddings of activations per rule.

Spaces: raw (standardized activations), two untrained controls (random
projection, random-init TopK SAE), and one representative checkpoint per SAE
config (seed 42 where weights exist, else the lowest seed). Views: the
combined constraint-rule set and one view per rule (R1 placement, R2
bracketing, R3 turn, R5 legal-only for flip coloring). Each view is embedded
in two variants: raw features and per-t-centered features (t dominates
activation geometry, so rule structure is only interpretable next to the
centered variant). PCA runs on every view item; Isomap on a category-balanced
subsample (shared across spaces) after PCA-50 pre-reduction.

The generic half of what this file used to be — load_space, representative_spaces,
batched_apply, center_by_t — lives in spade.spaces, because manifold_capture and
bracket_flip import it too. What stays here is the rules-specific run.

Writes small npz/json artifacts to analysis/rules_manifolds/embeddings/.
Resumable: spaces with an existing npz are skipped.

Usage: python -m spade.rules.embed [--spaces raw,random_proj] [--isomap_n 4000]
"""

import argparse
import json

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES, LEGAL
from spade.spaces import batched_apply, center_by_t, load_space, representative_spaces

SUBSAMPLE_SEED = 11
ISOMAP_NEIGHBORS = 15
ISOMAP_PRE_DIMS = 50

VIEWS = {  # view -> fine categories included (legal always included)
    "combined": [1, 2, 3, 4, 5, 6],
    "r1_placement": [1, 2],
    "r2_bracketing": [3, 4, 5],
    "r3_turn": [6],
    "r5_flips": [],
}


def view_masks(category: torch.Tensor) -> dict[str, torch.Tensor]:
    legal = category == LEGAL
    return {view: legal | torch.isin(category, torch.tensor(cats))
        for view, cats in VIEWS.items()}


def isomap_subsample(category: torch.Tensor, mask: torch.Tensor, n: int,
                     g: torch.Generator) -> torch.Tensor:
    """Category-balanced item indices within `mask` (legal counts as one group)."""
    groups = [((category == c) & mask).nonzero().flatten()
        for c in category[mask].unique().tolist()]
    cap = n // len(groups)
    picks = [idx[torch.randperm(len(idx), generator=g)[:cap]] for idx in groups]
    return torch.cat(picks).sort().values


def embed_view(feats: np.ndarray, iso_local: np.ndarray) -> tuple:
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    pca = PCA(n_components=2, random_state=0)
    coords_pca = pca.fit_transform(feats)
    pre = PCA(n_components=min(ISOMAP_PRE_DIMS, feats.shape[1] - 1),
              random_state=0).fit_transform(feats[iso_local])
    coords_iso = Isomap(n_neighbors=ISOMAP_NEIGHBORS, n_components=2,
                        n_jobs=-1).fit_transform(pre)
    return coords_pca, coords_iso, pca.explained_variance_ratio_.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default="", help="comma-separated filter (smoke tests)")
    parser.add_argument("--isomap_n", type=int, default=4000)
    args = parser.parse_args()

    device = select_device()
    base = study_dir("rules")
    items = torch.load(base / "pairs" / "items.pt", weights_only=True)
    acts = torch.load(base / "acts" / "acts_final.pt", weights_only=True)
    flips = torch.load(base / "pairs" / "flip_labels.pt", weights_only=True)
    category, t = items["category"], items["t"]
    legal = category == LEGAL

    out_dir = repo_root() / "analysis" / "rules_manifolds" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = view_masks(category)
    g = torch.Generator().manual_seed(SUBSAMPLE_SEED)
    iso_idx = {view: isomap_subsample(category, mask, args.isomap_n, g)
        for view, mask in masks.items()}

    np.savez(out_dir / "labels.npz",
             category=category.numpy(), t=t.numpy(),
             is_control_legal=items["is_control_legal"].numpy(),
             n_flipped=flips["n_flipped"].numpy(),
             n_directions=flips["n_directions"].numpy(),
             **{f"items_{v}": m.nonzero().flatten().numpy() for v, m in masks.items()},
             **{f"iso_{v}": idx.numpy() for v, idx in iso_idx.items()})

    spaces = representative_spaces()
    if args.spaces:
        keep = args.spaces.split(",")
        spaces = [s for s in spaces if s[0] in keep]

    evr_path = out_dir / "evr.json"
    evr = json.loads(evr_path.read_text()) if evr_path.exists() else {}
    for name, entry in spaces:
        f = out_dir / f"{name}.npz"
        if f.exists():
            print(f"skip {name} (exists)")
            continue
        feats = batched_apply(load_space(name if entry is None else "sae", entry, device),
                              acts, device)
        arrays = {}
        for variant, variant_feats in [("raw", feats),
                                       ("centered", center_by_t(feats, t, legal))]:
            va = variant_feats.numpy()
            for view, mask in masks.items():
                view_items = mask.nonzero().flatten().numpy()
                local = np.searchsorted(view_items, iso_idx[view].numpy())
                coords_pca, coords_iso, view_evr = embed_view(va[view_items], local)
                arrays[f"{view}_{variant}_pca"] = coords_pca.astype(np.float16)
                arrays[f"{view}_{variant}_isomap"] = coords_iso.astype(np.float16)
                evr[f"{name}/{view}/{variant}"] = view_evr
                print(f"{name}: {view}/{variant} done")
        np.savez(f, **arrays)
        evr_path.write_text(json.dumps(evr, indent=2))
        print(f"wrote {f.name}")


if __name__ == "__main__":
    main()
