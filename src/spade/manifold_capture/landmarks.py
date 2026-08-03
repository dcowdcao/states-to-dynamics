"""App C / Lemma 1 (arXiv:2604.28119): landmark capture for point-based SAEs.

Landmarks are decoder columns (prototypes) in raw uncentered standardized
space — where the prototypes live, so all distances are computed there. Per
(point-based space, concept): the active landmark set on the concept corpus,
directed/symmetric Hausdorff and NN capture distances, landmark kNN-graph
connectivity, manifold scale, and the Lemma 1 tiling test — each active
landmark's NN distance to every class submanifold vs the joint sample, plus
per-landmark class entropy (low = 1-to-1 with a class, high = joint tiling).
CosineSpaDE gets cosine-distance variants of every metric (its score is
cosine, and its decoder norms are not tied to the data's — landmark norms are
recorded per space).

Usage: python -m spade.manifold_capture.landmarks [--concepts x,y] [--cap 30000] [--selftest]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spade.common.device import select_device
from spade.common.paths import results_dir
from spade.manifold_capture.concepts import Concept, concepts, load_corpus, subsample
from spade.manifold_capture.concepts import values as concept_values
from spade.manifold_capture.ising import thresholds as fire_thresholds
from spade.manifold_capture.spaces import POINT_BASED, check_codes, decoder, encode_fn, spaces
from spade.spaces import batched_apply

MIN_SUPPORT_ROWS = 30
POINT_SUBSAMPLE = 5000   # M rows for point-to-set distances
SCALE_SUBSAMPLE = 2000   # M rows for mean pairwise distance
KNN_K = 5
QUARTILE_NAMES = ("q1", "q2", "q3", "q4")


def pdist(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclidean":
        return torch.cdist(a, b)
    an = torch.nn.functional.normalize(a, dim=-1)
    bn = torch.nn.functional.normalize(b, dim=-1)
    return (1 - an @ bn.T).clamp_min(0)


def active_support(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(active atom indices [L], per-row support on those atoms [n, L] bool).
    Study-wide rule: fire iff code >= per-atom 10th percentile of nonzeros,
    active iff firing on >= MIN_SUPPORT_ROWS rows."""
    thr = torch.from_numpy(fire_thresholds(codes.numpy())).float()
    support = codes >= thr
    active = (support.sum(0) >= MIN_SUPPORT_ROWS).nonzero().flatten()
    assert len(active) > 0, "empty active landmark set"
    return active, support[:, active]


def knn_components(landmarks: torch.Tensor, metric: str) -> tuple[int, int]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(landmarks)
    d = pdist(landmarks, landmarks, metric)
    d.fill_diagonal_(torch.inf)
    k = min(KNN_K, n - 1)
    nbrs = d.topk(k, largest=False).indices
    rows = torch.arange(n).repeat_interleave(k).numpy()
    adj = coo_matrix((np.ones(n * k), (rows, nbrs.flatten().numpy())), shape=(n, n))
    n_comp, comp_labels = connected_components(adj, directed=False)
    return n_comp, int(np.bincount(comp_labels).max())


def capture_row(landmarks: torch.Tensor, M_point: torch.Tensor, M_scale: torch.Tensor,
                metric: str) -> tuple[dict, torch.Tensor]:
    """Capture metrics + the landmark->M NN distances (the Lemma 1 joint term)."""
    d = pdist(landmarks, M_point, metric)
    nn_l, nn_m = d.min(1).values, d.min(0).values
    h_l, h_m = nn_l.max().item(), nn_m.max().item()
    ds = pdist(M_scale, M_scale, metric)
    n_comp, largest = knn_components(landmarks, metric)
    row = {"n_active": len(landmarks),
           "hausdorff_sym": max(h_l, h_m),
           "h_landmark_to_M": h_l, "h_M_to_landmark": h_m,
           "nn_mean": nn_l.mean().item(), "nn_p95": nn_l.quantile(0.95).item(),
           "nn_M_to_landmark_mean": nn_m.mean().item(),
           "nn_M_to_landmark_p95": nn_m.quantile(0.95).item(),
           "manifold_scale": (ds.sum() / (len(M_scale) * (len(M_scale) - 1))).item(),
           "n_components": n_comp, "largest_component": largest}
    assert row["hausdorff_sym"] >= max(row["nn_mean"], row["nn_M_to_landmark_mean"])
    return row, nn_l


def class_labels(concept: Concept, vals: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Class index per row: concept classes, or quartile bands for continuous."""
    if concept.kind == "categorical":
        return vals.long(), list(concept.class_names)
    qs = torch.quantile(vals.float(), torch.tensor([0.25, 0.5, 0.75]))
    return torch.bucketize(vals.float(), qs), list(QUARTILE_NAMES)


def landmark_entropy(support: torch.Tensor, labels: torch.Tensor,
                     n_classes: int) -> torch.Tensor:
    """Entropy (nats) of the class distribution of rows each active atom fires on."""
    counts = support.T.float() @ torch.nn.functional.one_hot(labels, n_classes).float()
    p = counts / counts.sum(1, keepdim=True)
    return -(p * p.clamp_min(1e-12).log()).sum(1)


def merge_csv(path: Path, rows: list[dict], space: str) -> None:
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[old["space"] != space], df], ignore_index=True)
    df.to_csv(path, index=False)


def selftest():
    torch.manual_seed(0)
    n_per, d = 300, 5
    M = torch.cat([torch.randn(n_per, d) * 0.1,
                   torch.randn(n_per, d) * 0.1 + torch.tensor([10.0, 0, 0, 0, 0])])
    labels = torch.arange(2).repeat_interleave(n_per)

    on = M[::5].clone()  # landmarks exactly on the manifold, 60 per cluster
    row_on, nn_on = capture_row(on, M, M[::3], "euclidean")
    assert row_on["h_landmark_to_M"] < 0.01, row_on
    assert row_on["hausdorff_sym"] < 1.5, row_on
    assert row_on["n_components"] == 2 and row_on["largest_component"] == 60, row_on
    assert nn_on.max().item() < 0.01

    row_off, _ = capture_row(on + 100.0, M, M[::3], "euclidean")
    assert row_off["h_landmark_to_M"] > 50, row_off
    assert pdist(on[:1], on[:1], "cosine")[0, 0].item() < 1e-6

    # class-specific codes -> entropy 0; joint-uniform codes -> entropy ln 2
    n = 2 * n_per
    codes = torch.zeros(n, 4)
    codes[labels == 0, 0], codes[labels == 0, 1] = 0.6, 0.4
    codes[labels == 1, 2], codes[labels == 1, 3] = 0.6, 0.4
    active, support = active_support(codes)
    assert active.tolist() == [0, 1, 2, 3]
    assert landmark_entropy(support, labels, 2).max().item() < 1e-6
    _, support_j = active_support(torch.full((n, 4), 0.25))
    assert (landmark_entropy(support_j, labels, 2) > 0.6).all()

    # rare atom (support < MIN_SUPPORT_ROWS rows) is excluded
    codes5 = torch.cat([codes, torch.zeros(n, 1)], dim=1)
    codes5[:10] = 0
    codes5[:10, 4] = 1.0
    active5, _ = active_support(codes5)
    assert active5.tolist() == [0, 1, 2, 3]

    # Lemma 1 direction: landmarks planted on class 0 sit near it, far from class 1
    lm0 = M[labels == 0][::5]
    near = pdist(lm0, M[labels == 0], "euclidean").min(1).values.mean().item()
    far = pdist(lm0, M[labels == 1], "euclidean").min(1).values.mean().item()
    assert near < 1.0 < far, (near, far)
    print("landmarks selftest passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts", default="", help="comma-separated filter")
    parser.add_argument("--cap", type=int, default=30000)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    device = select_device()
    torch.set_grad_enabled(False)
    out_dir = results_dir("manifold_capture")
    out_dir.mkdir(parents=True, exist_ok=True)

    point_spaces = [(n, e) for n, e in spaces() if n in POINT_BASED]
    todo = concepts()
    if args.concepts:
        todo = [c for c in todo if c.name in args.concepts.split(",")]

    raw_fn = encode_fn("raw", None, device)
    corpora = {}
    for corpus in sorted({c.corpus for c in todo}):
        data = load_corpus(corpus)
        rows = subsample(len(data["acts"]), args.cap)
        M = batched_apply(raw_fn, data["acts"][rows], device)
        corpora[corpus] = (data, rows, M,
                           subsample(len(rows), POINT_SUBSAMPLE),
                           subsample(len(rows), SCALE_SUBSAMPLE))

    for name, entry in point_spaces:
        npz_path = out_dir / f"landmark_entropy_{name}.npz"
        if npz_path.exists():
            print(f"skip {name} (exists)")
            continue
        D, _ = decoder(entry)
        landmarks_all = D.T.contiguous()  # [F, 512] prototypes in standardized space
        enc = encode_fn(name, entry, device)
        supports = {}
        metric_rows, lemma_rows = [], []
        ent_arrays = {"landmark_norms": landmarks_all.norm(dim=-1).numpy().astype(np.float32)}
        for concept in todo:
            data, rows, M, point_idx, scale_idx = corpora[concept.corpus]
            if concept.corpus not in supports:
                codes = batched_apply(enc, data["acts"][rows], device)
                check_codes(name, codes)
                supports[concept.corpus] = active_support(codes)
                del codes
            active, support = supports[concept.corpus]
            landmarks = landmarks_all[active]
            vals = concept_values(concept, data)[rows]
            labels, cnames = class_labels(concept, vals)
            ent_arrays[f"{concept.name}_atoms"] = active.numpy().astype(np.int32)
            ent_arrays[f"{concept.name}_entropy"] = \
                landmark_entropy(support, labels, len(cnames)).numpy().astype(np.float16)
            ent_arrays[f"{concept.name}_fire_frac"] = \
                support.float().mean(0).numpy().astype(np.float16)
            for metric in ["euclidean"] + (["cosine"] if "Cosine" in name else []):
                row, nn_joint = capture_row(landmarks, M[point_idx], M[scale_idx], metric)
                metric_rows.append({"space": name, "concept": concept.name,
                                    "metric_space": metric, **row})
                for ci, cname in enumerate(cnames):
                    class_rows = (labels == ci).nonzero().flatten()
                    if len(class_rows) < MIN_SUPPORT_ROWS:
                        continue
                    sub = class_rows[subsample(len(class_rows), POINT_SUBSAMPLE)]
                    nn_class = pdist(landmarks, M[sub], metric).min(1).values
                    lemma_rows.append({
                        "space": name, "concept": concept.name, "class": cname,
                        "metric_space": metric,
                        "landmark_nn_to_class_mean": nn_class.mean().item(),
                        "landmark_nn_to_joint_mean": nn_joint.mean().item(),
                        "n_landmarks": len(landmarks)})
            print(f"{name}: {concept.name} done ({len(active)} active landmarks)")
        merge_csv(out_dir / "landmarks.csv", metric_rows, name)
        merge_csv(out_dir / "landmarks_lemma1.csv", lemma_rows, name)
        np.savez(npz_path, **ent_arrays)
        print(f"wrote {npz_path.name} + CSV rows for {name}")


if __name__ == "__main__":
    main()
