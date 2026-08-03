"""Fig. 9 per concept x SAE space: four 3D manifold views + five atom-similarity
matrices over the supervised top atoms.

Supervised atoms are ranked by mutual information between binarized firing and
the concept value (continuous values in 8 quantile bins) — the paper's ground
truth for comparing similarity metrics. Views use the top-3 atoms as coordinate
axes and are [n, 3] fp16 (raw variant only); similarity matrices are [m, m]
fp32 over the top --top_atoms atoms, plus Louvain communities and per-atom
signed cohesion of the Ising couplings.

Usage: python -m spade.manifold_capture.feature_geometry [--concepts t,n_legal] [--spaces TopKTrainer-othello]
"""

import argparse

import numpy as np

from spade.common.device import select_device
from spade.common.paths import results_dir
from spade.manifold_capture import concepts, ising, spaces
from spade.spaces import batched_apply

TOP_VIEW = 3
VALUE_BINS = 8
MIN_FIRE_RATE = 0.001  # reference active_only floor
MIN_COFIRE = 2


def bin_values(vals: np.ndarray, kind: str) -> np.ndarray:
    if kind == "categorical":
        return vals.astype(np.int64)
    edges = np.quantile(vals, np.linspace(0, 1, VALUE_BINS + 1)[1:-1])
    return np.digitize(vals, edges)


def rank_atoms_mi(fired: np.ndarray, y: np.ndarray, top: int) -> np.ndarray:
    from sklearn.metrics import mutual_info_score
    mi = np.array([mutual_info_score(y, fired[:, a]) for a in range(fired.shape[1])])
    return np.argsort(mi)[::-1][:top]


def cosine_sim(d_cols: np.ndarray) -> np.ndarray:
    """Cosine between decoder columns; d_cols is [512, m]."""
    dn = d_cols / np.linalg.norm(d_cols, axis=0, keepdims=True).clip(1e-8)
    return (dn.T @ dn).astype(np.float32)


def coactivation(fired: np.ndarray) -> np.ndarray:
    """coact[i, j] = P(j fires | i fires)."""
    b = fired.astype(np.float32)
    return ((b.T @ b) / b.sum(0)[:, None].clip(1)).astype(np.float32)


def pmi(fired: np.ndarray) -> np.ndarray:
    """log P(i,j) / P(i)P(j) of binarized firing; joint floored at 1/n."""
    n = len(fired)
    b = fired.astype(np.float64)
    p_joint = (b.T @ b) / n
    p = b.mean(0)
    return np.log(p_joint.clip(1 / n) / np.outer(p, p).clip(1e-12)).astype(np.float32)


def cofire_corr(codes: np.ndarray, fired: np.ndarray) -> np.ndarray:
    """Pearson of code values on rows where both atoms fire; atoms under the
    reference active_only floor are zeroed."""
    m = codes.shape[1]
    active = fired.mean(0) > MIN_FIRE_RATE
    corr = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        if not active[i]:
            continue
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            both = active[j] & fired[:, i] & fired[:, j]
            if both.sum() < MIN_COFIRE:
                continue
            x, y = codes[both, i], codes[both, j]
            if x.std() < 1e-8 or y.std() < 1e-8:
                continue
            corr[i, j] = corr[j, i] = np.corrcoef(x, y)[0, 1]
    return corr


def views(ambient: np.ndarray, codes: np.ndarray, d_cols: np.ndarray,
          bias: np.ndarray, top3: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA

    def pca3(x):
        return PCA(n_components=3, random_state=0).fit_transform(x).astype(np.float16)

    _, sv, vt = np.linalg.svd(d_cols[:, top3].T, full_matrices=False)
    basis = vt[sv > 1e-8]  # orthonormal rows spanning the top-3 decoder columns
    partial = bias + codes[:, top3] @ d_cols[:, top3].T
    return {"pca_x": pca3(ambient),
            "pca_proj": pca3((ambient @ basis.T) @ basis),
            "pca_partial": pca3(partial),
            "codes_top": codes[:, top3].astype(np.float16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", default="", help="comma-separated filter")
    ap.add_argument("--concepts", default="", help="comma-separated filter")
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--top_atoms", type=int, default=40)
    args = ap.parse_args()

    device = select_device()
    out_dir = results_dir("manifold_capture")
    out_dir.mkdir(parents=True, exist_ok=True)

    space_list = [(n, e) for n, e in spaces.spaces() if e is not None]
    if args.spaces:
        space_list = [s for s in space_list if s[0] in args.spaces.split(",")]
    concept_list = concepts.concepts()
    if args.concepts:
        concept_list = [c for c in concept_list if c.name in args.concepts.split(",")]

    corpora = {}
    for concept in concept_list:
        todo = [(n, e) for n, e in space_list
            if not (out_dir / f"geometry_{concept.name}_{n}.npz").exists()]
        for name, _ in set(space_list) - set(todo):
            print(f"skip {concept.name}/{name} (exists)")
        if not todo:
            continue
        if concept.corpus not in corpora:
            corpora[concept.corpus] = concepts.load_corpus(concept.corpus)
        data = corpora[concept.corpus]
        idx = concepts.subsample(len(data["acts"]), args.cap)
        acts = data["acts"][idx]
        vals = concepts.values(concept, data)[idx].numpy()
        y = bin_values(vals, concept.kind)
        ambient = batched_apply(spaces.encode_fn("raw", None, device),
                                acts, device).numpy()

        for name, entry in todo:
            codes_t = batched_apply(spaces.encode_fn(name, entry, device), acts, device)
            spaces.check_codes(name, codes_t)
            codes = codes_t.numpy()
            d_cols, bias = (t.cpu().numpy() for t in spaces.decoder(entry))

            thr = ising.thresholds(codes)
            s_all = ising.binarize(codes, thr)
            top = rank_atoms_mi(s_all > 0, y, args.top_atoms)

            arrays = views(ambient, codes, d_cols, bias, top[:TOP_VIEW])
            arrays["values"] = vals

            fired = s_all[:, top] > 0
            J = ising.fit_ising(s_all[:, top])
            labels = ising.communities(J)
            rho = ising.signed_cohesion(J, labels)
            arrays.update(
                cosine=cosine_sim(d_cols[:, top]),
                coact=coactivation(fired),
                corr=cofire_corr(codes[:, top], fired),
                pmi=pmi(fired),
                ising=J.astype(np.float32),
                communities=labels,
                community_rho=np.array([rho[l] for l in labels], dtype=np.float32),
                atoms=top)
            out = out_dir / f"geometry_{concept.name}_{name}.npz"
            np.savez(out, **arrays)
            print(f"wrote {out.name} ({len(labels)} atoms, "
                  f"{len(set(labels.tolist()))} communities)")


if __name__ == "__main__":
    main()
