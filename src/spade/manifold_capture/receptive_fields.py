"""Fig 8 replication (arXiv:2604.28119): atom receptive fields on concept manifolds.

Per concept x trained SAE space: PCA(3) coords of the manifold rows (raw and
per-t-centered ambient), each row's argmax atom, and its reconstruction error
against raw ambient (the SAE reconstructs uncentered standardized space).
Small npz artifacts; figure rendering lives elsewhere.

Usage: python -m spade.manifold_capture.receptive_fields [--spaces A,B]
       [--concepts x,y] [--cap 20000] [--selftest]
"""

import argparse

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.paths import results_dir
from spade.manifold_capture.concepts import center_by_t, concepts, load_corpus, subsample
from spade.manifold_capture.concepts import values as concept_values
from spade.manifold_capture.spaces import check_codes, decoder, encode_fn, spaces
from spade.spaces import batched_apply


def receptive_arrays(ambient_by_variant: dict[str, torch.Tensor], codes: torch.Tensor,
                     D: torch.Tensor, bias: torch.Tensor, vals: torch.Tensor) -> dict:
    """PCA(3) coords per ambient variant + argmax atom and recon error vs raw ambient."""
    from sklearn.decomposition import PCA
    argmax = codes.argmax(-1)
    err = (bias + codes @ D.T - ambient_by_variant["raw"]).norm(dim=-1)
    out = {"argmax_atom": argmax.numpy().astype(np.int32),
           "recon_err": err.numpy().astype(np.float16),
           "values": vals.numpy(),
           "n_distinct_argmax": np.int64(len(argmax.unique()))}
    for variant, ambient in ambient_by_variant.items():
        coords = PCA(n_components=3, random_state=0).fit_transform(ambient.numpy())
        out[f"coords_{variant}"] = coords.astype(np.float16)
    return out


def selftest():
    torch.manual_seed(0)
    n, d, F = 200, 8, 16
    D, bias = torch.randn(d, F), torch.randn(d)
    atom = torch.randint(0, F, (n,))
    codes = torch.zeros(n, F)
    codes[torch.arange(n), atom] = 1.0
    ambient = bias + codes @ D.T  # exact reconstruction by construction
    vals = torch.randn(n)

    arrays = receptive_arrays({"raw": ambient, "centered": ambient - ambient.mean(0)},
                              codes, D, bias, vals)
    assert arrays["coords_raw"].shape == (n, 3)
    assert arrays["coords_centered"].shape == (n, 3)
    assert np.isfinite(arrays["coords_raw"]).all()
    assert (arrays["argmax_atom"] == atom.numpy()).all()
    assert arrays["n_distinct_argmax"] == len(atom.unique())
    assert arrays["recon_err"].max() < 1e-2, "exact recon should have ~0 error"

    shifted = receptive_arrays({"raw": ambient + 1.0}, codes, D, bias, vals)
    assert shifted["recon_err"].min() > 1.0, "shifted ambient should have visible error"
    print("receptive_fields selftest passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    parser.add_argument("--concepts", default="", help="comma-separated filter")
    parser.add_argument("--cap", type=int, default=20000)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    device = select_device()
    torch.set_grad_enabled(False)
    out_dir = results_dir("manifold_capture")
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_spaces = [(n, e) for n, e in spaces() if e is not None]
    if args.spaces:
        sae_spaces = [(n, e) for n, e in sae_spaces if n in args.spaces.split(",")]
    todo = concepts()
    if args.concepts:
        todo = [c for c in todo if c.name in args.concepts.split(",")]

    raw_fn = encode_fn("raw", None, device)
    corpora = {}
    for corpus in sorted({c.corpus for c in todo}):
        data = load_corpus(corpus)
        rows = subsample(len(data["acts"]), args.cap)
        ambient = batched_apply(raw_fn, data["acts"][rows], device)
        centered = center_by_t(ambient, data["t"][rows], data["legal"][rows])
        corpora[corpus] = (data, rows, ambient, centered)

    for name, entry in sae_spaces:
        pending = [c for c in todo
            if not (out_dir / f"receptive_{c.name}_{name}.npz").exists()]
        if len(pending) < len(todo):
            print(f"{name}: skipped {len(todo) - len(pending)} existing")
        if not pending:
            continue
        D, bias = decoder(entry)
        enc = encode_fn(name, entry, device)
        codes_by_corpus = {}
        for concept in pending:
            data, rows, ambient, centered = corpora[concept.corpus]
            if concept.corpus not in codes_by_corpus:
                codes = batched_apply(enc, data["acts"][rows], device)
                check_codes(name, codes)
                codes_by_corpus[concept.corpus] = codes
            variants = {"raw": ambient}
            if "centered" in concept.variants:
                variants["centered"] = centered
            arrays = receptive_arrays(variants, codes_by_corpus[concept.corpus],
                                      D, bias, concept_values(concept, data)[rows])
            f = out_dir / f"receptive_{concept.name}_{name}.npz"
            np.savez(f, **arrays)
            print(f"wrote {f.name} (n={len(rows)}, "
                  f"{arrays['n_distinct_argmax']} distinct argmax atoms)")


if __name__ == "__main__":
    main()
