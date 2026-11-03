"""Step 3 of the straightness study: trajectory geometry per representation space.

Spaces: token_embed + resid_post_0..7 (from extract_traj.py) and the 8 seed-42
SAE families encoding the layer-5 trajectory. Each space is computed in two
variants: raw, and centered (minus the mean legal train-split trajectory, since
move index t dominates activation geometry; SAE centering happens in latent
space). Position 0 (attention sink) is dropped, so trajectories cover positions
1..58: d [N,57], theta [N,56]. The raw variant also stores per-position norms
for diagnostics. Resumable: spaces with existing npz are skipped.

Usage: python -m spade.straightness.compute_geometry [--spaces resid_post_5,sae_TopKTrainer-othello]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.spaces import EXCLUDED_SAE
from spade.straightness.geometry import temporal_geometry, train_mask

FILE_SPACES = ["token_embed"] + [f"resid_post_{i}" for i in range(8)]

GAME_BATCH = 64
MAX_NAN_FRAC_DENSE = 0.01


def checks(space: str, variant: str, d: torch.Tensor, theta: torch.Tensor) -> None:
    assert torch.isfinite(d).all() and (d >= 0).all()
    valid = theta[~theta.isnan()]
    assert (valid >= 0).all() and (valid <= 180).all()
    nan_frac = theta.isnan().float().mean().item()
    assert space.startswith("sae_") or nan_frac < MAX_NAN_FRAC_DENSE, \
        f"{space}/{variant}: {nan_frac:.1%} undefined angles in a dense space"
    print(f"{space}/{variant}: mean_d {d.mean():.2f}  mean_theta {valid.mean():.1f}deg"
          f"  nan_frac {nan_frac:.2%}")


def save(out_dir: Path, space: str, variants: dict[str, tuple], norm: torch.Tensor,
         l0: torch.Tensor | None = None) -> None:
    """`norm` is the per-position norm in the space's own representation, so for
    SAE spaces it is the latent norm; `l0` is the per-position active-code count."""
    for variant, (d, theta) in variants.items():
        checks(space, variant, d, theta)
        arrays = {"d": d.numpy().astype(np.float32), "theta": theta.numpy().astype(np.float32)}
        if variant == "raw":
            arrays["norm"] = norm.numpy().astype(np.float32)
            if l0 is not None:
                arrays["l0"] = l0.numpy().astype(np.float32)
        np.savez(out_dir / f"{space}_{variant}.npz", **arrays)
    print(f"wrote {space}")


def sae_encoder(entry, device: str):
    from spade.evaluation.karvonen_adapter import (
        StandardizedDictionary, load_karvonen_sae, load_standardization_params)
    ae, _ = load_karvonen_sae(str(entry.ckpt_dir), device=device)
    params = load_standardization_params(str(entry.ckpt_dir), device=device)
    return StandardizedDictionary(
        ae, torch.tensor(params["mean"], device=device), params["scale"]).to(device).encode


def encode_batched(enc, Z: torch.Tensor, idx: torch.Tensor, device: str):
    """Yields (batch_idx, latents [b, T, dict]) over games in `idx`."""
    T = Z.shape[1]
    for start in range(0, len(idx), GAME_BATCH):
        batch_idx = idx[start:start + GAME_BATCH]
        flat = Z[batch_idx].float().reshape(-1, Z.shape[2]).to(device)
        with torch.no_grad():
            lat = enc(flat)
        yield batch_idx, lat.reshape(len(batch_idx), T, -1)


def sae_geometry(enc, Z: torch.Tensor, legal_train: torch.Tensor, device: str):
    mean_lat, count = None, 0
    for _, lat in encode_batched(enc, Z, legal_train.nonzero().flatten(), device):
        s = lat.sum(0)
        mean_lat = s if mean_lat is None else mean_lat + s
        count += lat.shape[0]
    mean_lat /= count

    parts = {v: ([], []) for v in ["raw", "centered"]}
    norms, l0s = [], []
    for _, lat in encode_batched(enc, Z, torch.arange(len(Z)), device):
        norms.append(lat.norm(dim=-1).cpu())
        l0s.append((lat != 0).float().sum(-1).cpu())
        for variant, latv in [("raw", lat), ("centered", lat - mean_lat)]:
            d, theta = temporal_geometry(latv)
            parts[variant][0].append(d.cpu())
            parts[variant][1].append(theta.cpu())
    norm, l0 = torch.cat(norms), torch.cat(l0s)
    assert 1 <= l0.mean() <= 4096, f"implausible mean L0 {l0.mean():.0f}"
    print(f"mean L0 {l0.mean():.0f}, mean latent norm {norm.mean():.3f}")
    variants = {v: (torch.cat(ds), torch.cat(ts)) for v, (ds, ts) in parts.items()}
    return variants, norm, l0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", default="", help="comma-separated filter (smoke tests)")
    args = parser.parse_args()

    from spade.saes.registry import seed42_entries

    device = select_device()
    base = study_dir("straightness")
    cls = torch.load(base / "games" / "games.pt", weights_only=True)["cls"]
    legal_train = (cls == 0) & train_mask(len(cls))

    out_dir = base / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = args.spaces.split(",") if args.spaces else None
    spaces = [(s, None) for s in FILE_SPACES]
    # the SAE registry asserts every seed-42 checkpoint, so open it only when a filter needs one
    if keep is None or set(keep) - set(FILE_SPACES):
        spaces += [(f"sae_{e.config}", e) for e in seed42_entries()
                   if e.config not in EXCLUDED_SAE]
    if keep:
        spaces = [s for s in spaces if s[0] in keep]

    for space, entry in spaces:
        if all((out_dir / f"{space}_{v}.npz").exists() for v in ["raw", "centered"]):
            print(f"skip {space} (exists)")
            continue
        if entry is None:
            Z = torch.load(base / "traj" / f"{space}.pt", weights_only=True).float()[:, 1:, :]
            mean_traj = Z[legal_train].mean(0)
            variants = {v: temporal_geometry(Zv)
                for v, Zv in [("raw", Z), ("centered", Z - mean_traj)]}
            save(out_dir, space, variants, Z.norm(dim=-1))
        else:
            Z = torch.load(base / "traj" / "resid_post_5.pt", weights_only=True)[:, 1:, :]
            variants, norm, l0 = sae_geometry(
                sae_encoder(entry, device), Z, legal_train, device)
            save(out_dir, space, variants, norm, l0)


if __name__ == "__main__":
    main()
