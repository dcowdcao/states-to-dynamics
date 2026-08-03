"""Atom tuning curves over continuous concepts (paper Fig 7).

Per (SAE space, continuous concept): the top --n_atoms atoms by support (the
paper's rule — fraction of rows firing above the atom's 10th-percentile-of-
nonzero threshold), each atom's mean activation per concept-value bin,
normalized per-atom to max 1 (raw code magnitudes are not comparable across
families). Tuning uses raw codes; the ambient variants are irrelevant here,
so one result per (space, concept). One npz per concept; resumable.

Usage: python -m spade.manifold_capture.tuning_curves [--spaces a,b] [--concepts x,y]
"""

import argparse

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.paths import results_dir
from spade.manifold_capture import concepts, spaces
from spade.spaces import batched_apply

MAX_VALUE_BINS = 58   # one bin per integer value below this, else equal-width
N_CONT_BINS = 40
SUPPORT_QUANTILE = 0.1
ATOM_CHUNK = 256      # keeps each nanquantile input under torch's 2^24-element limit


def bin_ids_centers(vals: np.ndarray, n_bins: int | None) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(vals)
    if n_bins is None and len(uniq) <= MAX_VALUE_BINS:
        return np.searchsorted(uniq, vals), uniq.astype(np.float32)
    n_bins = n_bins or N_CONT_BINS
    lo, hi = vals.min(), vals.max()
    ids = np.clip(((vals - lo) / (hi - lo) * n_bins).astype(np.int64), 0, n_bins - 1)
    centers = lo + (hi - lo) / n_bins * (np.arange(n_bins) + 0.5)
    return ids, centers.astype(np.float32)


def support(codes: torch.Tensor) -> torch.Tensor:
    """Per-atom fraction of rows above its 10th-percentile-of-nonzero threshold."""
    out = torch.zeros(codes.shape[1])
    for s in range(0, codes.shape[1], ATOM_CHUNK):
        chunk = codes[:, s:s + ATOM_CHUNK]
        thr = torch.nanquantile(torch.where(chunk > 0, chunk, float("nan")),
                                SUPPORT_QUANTILE, dim=0)
        out[s:s + ATOM_CHUNK] = (chunk > thr).float().mean(0)  # dead atom: nan thr -> 0
    return out


def curves(codes: torch.Tensor, atoms: torch.Tensor, bin_ids: np.ndarray,
           n_bins: int) -> np.ndarray:
    """[n_atoms, n_bins] mean activation per bin, normalized per-atom to max 1."""
    sel = codes[:, atoms].numpy()
    out = np.zeros((len(atoms), n_bins), dtype=np.float32)
    for b in range(n_bins):
        rows = bin_ids == b
        if rows.any():
            out[:, b] = sel[rows].mean(0)
    peak = out.max(1)
    assert (peak > 0).all(), "selected atom never fires in any bin"
    return out / peak[:, None]


def selftest():
    rng = np.random.default_rng(0)
    n, F = 3000, 20
    vals = rng.uniform(0, 1, n).astype(np.float32)
    codes = rng.random((n, F), dtype=np.float32) * (rng.random((n, F)) < 0.05)
    codes[:, 7] = np.exp(-((vals - 0.5) / 0.08) ** 2)  # planted atom localized at 0.5

    ids, centers = bin_ids_centers(vals, None)
    assert len(centers) == N_CONT_BINS  # >58 distinct values -> equal-width default
    counts = np.bincount(ids, minlength=len(centers))
    assert counts.sum() == n

    sup = support(torch.from_numpy(codes))
    atoms = sup.topk(5).indices
    assert atoms[0].item() == 7, f"planted atom not top-support: {atoms.tolist()}"
    cur = curves(torch.from_numpy(codes), atoms, ids, len(centers))
    assert cur.shape == (5, len(centers))
    planted = cur[0]
    assert planted.max() == 1.0
    assert abs(centers[planted.argmax()] - 0.5) < 0.1, "bump not at planted value"
    far = np.abs(centers - 0.5) > 0.3
    assert planted[far].max() < 0.05, "planted atom curve not localized"
    print(f"selftest ok: planted atom support={sup[7]:.3f}, "
          f"peak at {centers[planted.argmax()]:.3f}, far-bin max={planted[far].max():.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default="", help="comma-separated SAE-space filter")
    parser.add_argument("--concepts", default="", help="continuous concepts only")
    parser.add_argument("--cap", type=int, default=60000)
    parser.add_argument("--n_atoms", type=int, default=10)
    parser.add_argument("--n_bins", type=int, default=None,
                        help="equal-width bin count; default: per-integer-value bins")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    device = select_device()
    out_dir = results_dir("manifold_capture")
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_list = [c for c in concepts.concepts() if c.kind == "continuous"]
    if args.concepts:
        by_name = {c.name: c for c in concepts.concepts()}
        concept_list = [by_name[n] for n in args.concepts.split(",")]
        assert all(c.kind == "continuous" for c in concept_list), \
            "tuning curves are defined for continuous concepts only"
    sae_spaces = [(n, e) for n, e in spaces.spaces() if e is not None]
    if args.spaces:
        sae_spaces = [(n, e) for n, e in sae_spaces if n in args.spaces.split(",")]

    loaded = (None, None)
    for concept in concept_list:
        out = out_dir / f"tuning_{concept.name}.npz"
        if out.exists():
            print(f"skip {concept.name} (exists)")
            continue
        if loaded[0] != concept.corpus:
            loaded = (concept.corpus, concepts.load_corpus(concept.corpus))
        data = loaded[1]
        vals = concepts.values(concept, data)
        idx = concepts.subsample(len(vals), args.cap)
        acts, vals_v = data["acts"][idx], vals[idx].numpy()
        ids, centers = bin_ids_centers(vals_v, args.n_bins)
        arrays = {"bin_centers": centers,
                  "bin_counts": np.bincount(ids, minlength=len(centers)).astype(np.int64)}
        for name, entry in sae_spaces:
            codes = batched_apply(spaces.encode_fn(name, entry, device), acts, device)
            spaces.check_codes(name, codes)
            atoms = support(codes).topk(min(args.n_atoms, codes.shape[1])).indices
            arrays[f"{name}_curves"] = curves(codes, atoms, ids, len(centers))
            arrays[f"{name}_atoms"] = atoms.numpy().astype(np.int64)
            arrays[f"{name}_fire_frac"] = (codes[:, atoms] > 0).float().mean(0).numpy()
            print(f"{concept.name}: {name} done")
        np.savez(out, **arrays)
        print(f"wrote {out.name} ({len(idx)} rows, {len(centers)} bins)")


if __name__ == "__main__":
    main()
