"""
Find inactive SpaDE prototypes in a trained checkpoint.

Loads the SpaDE encoder, runs OthelloGPT layer-5 on a fresh activation pool,
applies the saved standardization (mean, scale), and counts per-prototype
firings under sparsemax(-lambda * ||x - Ae||^2). Reports:
  - exact-dead count (fired 0 times)
  - rare-fire counts at <1e-5, <1e-4, <1e-3 activation rates
  - fraction of total mass each prototype carries
  - bottom-N prototype IDs

Usage:
    python -m spade.diagnostics.inactive_spade_prototypes \
        --checkpoint checkpoints/main/trainer_5/ae.pt \
        --std-params checkpoints/main/standardization_params.json \
        --n-tokens 60000
"""

import argparse
import json
import os
from pathlib import Path

import torch as t
from datasets import load_dataset
from transformer_lens import HookedTransformer

from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.saes.spade import SpaDEAutoEncoder


def load_spade(checkpoint_path: str, device: str) -> SpaDEAutoEncoder:
    state = t.load(checkpoint_path, map_location="cpu")
    dict_size, activation_dim = state["Ae"].shape
    ae = SpaDEAutoEncoder(activation_dim=activation_dim, dict_size=dict_size)
    ae.load_state_dict(state)
    ae.eval()
    ae.to(device)
    return ae


def stream_tokens(dataset_name: str, ctx_len: int, n_games: int):
    ds = load_dataset(dataset_name, split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i >= n_games:
            break
        out.append(ex["tokens"][:ctx_len])
    return t.tensor(out, dtype=t.long)  # [n_games, ctx_len]


@t.no_grad()
def collect_activations(model, tokens: t.Tensor, layer: int, batch_size: int, device: str):
    hook_name = f"blocks.{layer}.hook_resid_post"
    chunks = []
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i : i + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        h = cache[hook_name].reshape(-1, model.cfg.d_model).cpu()
        chunks.append(h)
        del cache
    return t.cat(chunks, dim=0)  # [n_games * ctx_len, d_model]


@t.no_grad()
def count_firings(ae: SpaDEAutoEncoder, x_std: t.Tensor, batch_size: int, device: str):
    n, d = x_std.shape
    K = ae.dict_size
    fire_counts = t.zeros(K, dtype=t.long)
    sum_mass = t.zeros(K, dtype=t.float64)  # sum of f-values (mass), for total contribution
    total = 0
    for i in range(0, n, batch_size):
        xb = x_std[i : i + batch_size].to(device)
        f = ae.encode(xb).float().cpu()  # sparsemax outputs
        fire_counts += (f > 0).sum(dim=0)
        sum_mass += f.sum(dim=0).double()
        total += xb.shape[0]
    return fire_counts, sum_mass, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="checkpoints/main/trainer_5/ae.pt",
    )
    p.add_argument(
        "--std-params",
        default="checkpoints/main/standardization_params.json",
    )
    p.add_argument("--model-name", default="Baidicoot/Othello-GPT-Transformer-Lens")
    p.add_argument("--dataset", default="taufeeque/othellogpt")
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--ctx-len", type=int, default=59)
    p.add_argument("--n-tokens", type=int, default=60_000)
    p.add_argument("--llm-batch-size", type=int, default=32)
    p.add_argument("--enc-batch-size", type=int, default=8192)
    p.add_argument("--bottom-n", type=int, default=20)
    p.add_argument("--save-counts", default=None)
    args = p.parse_args()

    root = repo_root()
    ckpt = root / args.checkpoint if not os.path.isabs(args.checkpoint) else Path(args.checkpoint)
    sp = root / args.std_params if not os.path.isabs(args.std_params) else Path(args.std_params)

    device = select_device()
    print(f"device={device}  ckpt={ckpt}  std={sp}")

    print("Loading SpaDE checkpoint...")
    ae = load_spade(str(ckpt), device)
    print(f"  Ae shape: {tuple(ae.Ae.shape)}  lambda={ae.lambda_val.item():.6f}")

    with open(sp) as f:
        sp_json = json.load(f)
    mean = t.tensor(sp_json["mean"], dtype=t.float32)
    scale = float(sp_json["scale"])
    d_model = int(sp_json["d_model"])
    print(f"  mean.shape={tuple(mean.shape)}  scale={scale:.6f}  d_model={d_model}")

    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name)
    model.to(device).eval()

    n_games = (args.n_tokens + args.ctx_len - 1) // args.ctx_len
    print(f"Streaming {n_games} games of {args.ctx_len} tokens (~{n_games * args.ctx_len} activations)...")
    tokens = stream_tokens(args.dataset, args.ctx_len, n_games)
    print(f"  tokens shape: {tuple(tokens.shape)}")

    print("Running OthelloGPT to collect layer-5 activations...")
    h_raw = collect_activations(model, tokens, args.layer, args.llm_batch_size, device)
    print(f"  raw activations shape: {tuple(h_raw.shape)}")

    h_std = scale * (h_raw - mean)
    print(f"  standardized: mean_norm={h_std.norm(dim=-1).mean().item():.4f}  "
          f"max_norm={h_std.norm(dim=-1).max().item():.4f}")

    print("Counting prototype firings under sparsemax...")
    fire_counts, sum_mass, total = count_firings(ae, h_std, args.enc_batch_size, device)
    print(f"  total tokens scored: {total}")

    rates = fire_counts.float() / total
    mass_frac = (sum_mass / sum_mass.sum()).float()
    K = ae.dict_size

    n_zero = int((fire_counts == 0).sum())
    n_lt_1e5 = int(((rates > 0) & (rates < 1e-5)).sum())
    n_lt_1e4 = int((rates < 1e-4).sum())
    n_lt_1e3 = int((rates < 1e-3).sum())
    n_lt_1e2 = int((rates < 1e-2).sum())

    print()
    print("=== Per-prototype activation rates ===")
    print(f"  total prototypes:                       {K}")
    print(f"  exactly dead (0 firings on {total} tokens): {n_zero}  ({100*n_zero/K:.2f}%)")
    print(f"  rate < 1e-5 (positive but rare):        {n_lt_1e5}")
    print(f"  rate < 1e-4 (cumulative incl. dead):    {n_lt_1e4}  ({100*n_lt_1e4/K:.2f}%)")
    print(f"  rate < 1e-3 (cumulative):               {n_lt_1e3}  ({100*n_lt_1e3/K:.2f}%)")
    print(f"  rate < 1e-2 (cumulative):               {n_lt_1e2}  ({100*n_lt_1e2/K:.2f}%)")

    print()
    print("=== Mass-weighted view (sum of sparsemax outputs) ===")
    sorted_mass, _ = mass_frac.sort()
    cum = sorted_mass.cumsum(0)
    for thresh in [0.01, 0.05, 0.10, 0.50]:
        n_carry = int((cum < thresh).sum())
        print(f"  bottom prototypes carrying < {thresh*100:>5.1f}% of total mass: {n_carry}")

    print()
    print(f"=== Bottom-{args.bottom_n} prototypes by activation rate ===")
    bottom_idx = rates.argsort()[: args.bottom_n]
    print(f"  {'rank':>4} {'id':>6} {'fires':>8} {'rate':>10} {'mass_frac':>10} {'norm(Ae_i)':>11}")
    norms = ae.Ae.detach().cpu().norm(dim=-1)
    for r, i in enumerate(bottom_idx.tolist()):
        print(
            f"  {r:>4} {i:>6} {int(fire_counts[i]):>8} "
            f"{rates[i].item():>10.2e} {mass_frac[i].item():>10.2e} "
            f"{norms[i].item():>11.4f}"
        )

    print()
    print(f"=== Top-{args.bottom_n} prototypes by activation rate (for reference) ===")
    top_idx = rates.argsort(descending=True)[: args.bottom_n]
    print(f"  {'rank':>4} {'id':>6} {'fires':>8} {'rate':>10} {'mass_frac':>10} {'norm(Ae_i)':>11}")
    for r, i in enumerate(top_idx.tolist()):
        print(
            f"  {r:>4} {i:>6} {int(fire_counts[i]):>8} "
            f"{rates[i].item():>10.2e} {mass_frac[i].item():>10.2e} "
            f"{norms[i].item():>11.4f}"
        )

    print()
    print(f"=== Prototype-norm summary (Ae rows) ===")
    print(f"  norm mean={norms.mean().item():.4f}  median={norms.median().item():.4f}  "
          f"min={norms.min().item():.4f}  max={norms.max().item():.4f}")
    print(f"  data norm   mean={h_std.norm(dim=-1).mean().item():.4f}  "
          f"max={h_std.norm(dim=-1).max().item():.4f}")

    if args.save_counts:
        out_path = root / args.save_counts if not os.path.isabs(args.save_counts) else Path(args.save_counts)
        t.save(
            {
                "fire_counts": fire_counts,
                "sum_mass": sum_mass,
                "total": total,
                "rates": rates,
                "mass_frac": mass_frac,
                "Ae_norms": norms,
                "checkpoint": str(ckpt),
                "n_tokens_scored": total,
            },
            out_path,
        )
        print(f"\nSaved per-prototype data to {out_path}")


if __name__ == "__main__":
    main()
