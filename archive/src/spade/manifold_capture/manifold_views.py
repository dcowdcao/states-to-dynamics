"""Manifold views per concept (paper Figs 2, 5, 11, 12 + raw-vs-latent PCA).

Per concept: seeded value-balanced view subsample, ambient PCA(3) per variant,
each SAE's top-k partial reconstructions projected into that fitted basis, and
a PCA(3) of each SAE's own codes (SpaDE latent geometry vs ambient geometry is
a headline comparison). One npz per concept in results/manifold_capture/;
resumable (skips existing npz). No figures here — make_figures.py renders.

Usage: python -m spade.manifold_capture.manifold_views [--spaces a,b] [--concepts x,y] [--n_view 8000]
"""

import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA

from spade.common.device import select_device
from spade.common.paths import results_dir
from spade.manifold_capture import concepts, spaces
from spade.spaces import batched_apply

TOP_KS = (1, 2, 4, 8, 16)
MAX_VALUE_BINS = 58   # one balance group per distinct value below this, else equal-width
N_CONT_BINS = 40
MIN_RECON_R2 = 0.2    # low bar on purpose: SpaDE's VE~0.43 (spaces.LOW_VE) must still pass


def value_groups(vals: torch.Tensor, kind: str) -> torch.Tensor:
    """Balance-group id per row: class for categorical, value bin for continuous."""
    if kind == "categorical":
        return vals
    uniq = vals.unique()
    if len(uniq) <= MAX_VALUE_BINS:
        return torch.searchsorted(uniq, vals.contiguous())
    lo, hi = vals.min(), vals.max()
    return ((vals - lo) / (hi - lo) * N_CONT_BINS).long().clamp(max=N_CONT_BINS - 1)


def balanced_view(vals: torch.Tensor, kind: str, n: int, g: torch.Generator) -> torch.Tensor:
    """Seeded value-balanced row indices, sorted (spirit of spade.spaces isomap_subsample)."""
    gids = value_groups(vals, kind)
    groups = [(gids == gid).nonzero().flatten() for gid in gids.unique().tolist()]
    cap = n // len(groups)
    picks = [idx[torch.randperm(len(idx), generator=g)[:cap]] for idx in groups]
    return torch.cat(picks).sort().values


def view_arrays(ambient: torch.Tensor, t: torch.Tensor, legal: torch.Tensor,
                variants: tuple[str, ...],
                sae_items: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict:
    """All per-concept npz arrays from ambient acts + per-space (name, codes, D, bias)."""
    offset = (ambient - concepts.center_by_t(ambient, t, legal)
              if "centered" in variants else None)
    arrays, pcas, coords = {}, {}, {}
    for variant in variants:
        amb = (ambient if variant == "raw" else ambient - offset).numpy()
        pca = PCA(n_components=3, random_state=0)
        coords[variant] = pca.fit_transform(amb)
        pcas[variant] = pca
        arrays[f"pca3_{variant}"] = coords[variant].astype(np.float16)
        arrays[f"evr_{variant}"] = pca.explained_variance_ratio_.astype(np.float32)

    for name, codes, D, bias in sae_items:
        recons = {0: bias + codes @ D.T}  # 0 = full recon (all atoms)
        for k in TOP_KS:
            kk = min(k, codes.shape[1])
            topi = codes.abs().topk(kk, dim=1).indices
            masked = torch.zeros_like(codes).scatter_(1, topi, codes.gather(1, topi))
            recons[k] = bias + masked @ D.T
        for variant in variants:
            shift = 0 if variant == "raw" else offset
            proj_full = pcas[variant].transform((recons[0] - shift).numpy())
            c = coords[variant]
            r2 = 1 - ((proj_full - c) ** 2).sum() / ((c - c.mean(0)) ** 2).sum()
            arrays[f"r2_{name}_{variant}"] = np.float32(r2)
            if variant == "raw":
                low = " (LOW_VE)" if name in spaces.LOW_VE else ""
                assert r2 > MIN_RECON_R2, (
                    f"{name}{low}: full-recon R2 {r2:.3f} <= {MIN_RECON_R2} in fitted "
                    f"basis — basis/mean misalignment?")
            for k in TOP_KS:
                proj = pcas[variant].transform((recons[k] - shift).numpy())
                arrays[f"recon_{name}_{variant}_k{k}"] = proj.astype(np.float16)
            z = codes if variant == "raw" else concepts.center_by_t(codes, t, legal)
            lp = PCA(n_components=3, random_state=0)
            arrays[f"latent_pca3_{name}_{variant}"] = lp.fit_transform(z.numpy()).astype(np.float16)
            arrays[f"latent_evr_{name}_{variant}"] = lp.explained_variance_ratio_.astype(np.float32)
    return arrays


def selftest():
    torch.manual_seed(0)
    n, d = 1200, 16
    theta = torch.linspace(0, 4 * torch.pi, n)
    helix = torch.stack([theta.cos(), theta.sin(), theta / (4 * torch.pi)], 1)
    q, _ = torch.linalg.qr(torch.randn(d, 3))
    ambient = helix @ q.T + 0.01 * torch.randn(n, d)
    t = (theta / (4 * torch.pi) * 10).long().clamp(max=9)
    legal = torch.ones(n, dtype=torch.bool)
    p, _ = torch.linalg.qr(torch.randn(d, d))
    codes = torch.cat([(ambient @ p).relu(), (-(ambient @ p)).relu()], 1)
    D = torch.cat([p, -p], 1)  # exact autoencoder: codes @ D.T == ambient

    arrays = view_arrays(ambient, t, legal, ("raw", "centered"),
                         [("fake", codes, D, torch.zeros(d))])
    assert arrays["pca3_raw"].shape == (n, 3)
    assert arrays["evr_raw"].shape == (3,) and arrays["evr_raw"].sum() > 0.95
    for variant in ("raw", "centered"):
        for k in TOP_KS:
            assert arrays[f"recon_fake_{variant}_k{k}"].shape == (n, 3)
        assert arrays[f"latent_pca3_fake_{variant}"].shape == (n, 3)
        assert arrays[f"latent_evr_fake_{variant}"].shape == (3,)
    assert arrays["r2_fake_raw"] > 0.99, arrays["r2_fake_raw"]
    # each row has exactly d=16 nonzero codes, so k=16 partial recon is the exact recon
    gap = np.abs(arrays["recon_fake_raw_k16"].astype(np.float32) -
                 arrays["pca3_raw"].astype(np.float32)).max()
    assert gap < 0.05, f"k=16 recon coords off ambient coords by {gap:.4f}"

    g = torch.Generator().manual_seed(0)
    idx = balanced_view(theta, "continuous", 400, g)
    assert len(idx) <= 400 and len(idx.unique()) == len(idx)
    print(f"selftest ok: {len(arrays)} arrays, r2_raw={float(arrays['r2_fake_raw']):.4f}, "
          f"k16 coord gap={gap:.5f}, balanced view {len(idx)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default="", help="comma-separated SAE-space filter")
    parser.add_argument("--concepts", default="", help="comma-separated concept filter")
    parser.add_argument("--n_view", type=int, default=8000)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    device = select_device()
    out_dir = results_dir("manifold_capture")
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_spaces = [(n, e) for n, e in spaces.spaces() if e is not None]
    if args.spaces:
        sae_spaces = [(n, e) for n, e in sae_spaces if n in args.spaces.split(",")]
    concept_list = concepts.concepts()
    if args.concepts:
        concept_list = [c for c in concept_list if c.name in args.concepts.split(",")]

    loaded = (None, None)
    for concept in concept_list:
        out = out_dir / f"views_{concept.name}.npz"
        if out.exists():
            print(f"skip {concept.name} (exists)")
            continue
        if loaded[0] != concept.corpus:
            loaded = (concept.corpus, concepts.load_corpus(concept.corpus))
        data = loaded[1]
        vals = concepts.values(concept, data)
        g = torch.Generator().manual_seed(concepts.SUBSAMPLE_SEED)
        idx = balanced_view(vals, concept.kind, args.n_view, g)
        acts, t, legal = data["acts"][idx], data["t"][idx], data["legal"][idx]
        ambient = batched_apply(spaces.encode_fn("raw", None, device), acts, device)
        sae_items = []
        for name, entry in sae_spaces:
            codes = batched_apply(spaces.encode_fn(name, entry, device), acts, device)
            spaces.check_codes(name, codes)
            D, bias = spaces.decoder(entry)
            sae_items.append((name, codes, D, bias))
            print(f"{concept.name}: {name} encoded")
        arrays = view_arrays(ambient, t, legal, concept.variants, sae_items)
        arrays |= {"values": vals[idx].numpy(), "t": t.numpy(), "idx": idx.numpy()}
        np.savez(out, **arrays)
        print(f"wrote {out.name} ({len(idx)} rows)")


if __name__ == "__main__":
    main()
