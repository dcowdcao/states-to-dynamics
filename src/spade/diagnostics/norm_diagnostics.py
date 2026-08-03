#!/usr/bin/env python3
"""
Diagnostic norm plots: activation norms and SpaDE weight norms,
comparing randn init vs data-sampled init.

Generates a 2x2 figure:
  (1,1): Post-standardization activation norms (histogram)
  (1,2): SpaDE Ae row norms — randn init (histogram)
  (2,1): SpaDE Ae row norms — data-sampled init (histogram)
  (2,2): Overlay of all three distributions (density-normalized)

The overlay panel (2,2) is the key diagnostic: data-sampled init norms
should match the activation norm distribution, while randn init norms
cluster at sqrt(d) ~ 22.6, well above the activation bulk.

Usage:
    python -m spade.diagnostics.norm_diagnostics --output_dir figures
    python -m spade.diagnostics.norm_diagnostics --output_dir figures --n_batches 50 --device cpu
"""

import argparse
import os

import torch as t
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spade.common.paths import add_karvonen_to_path, repo_root

add_karvonen_to_path()

from circuits.utils import (
    othello_hf_dataset_to_generator,
    get_model,
    get_submodule,
    SubmoduleType,
)
from circuits.dictionary_learning.buffer import NNsightActivationBuffer
from spade.saes.spade import SpaDEAutoEncoder

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})


def collect_activation_norms(buffer, n_batches):
    """Collect per-sample L2 norms from the activation buffer."""
    norms = []
    for i, act in enumerate(buffer):
        if i >= n_batches:
            break
        # act: [batch_size, activation_dim]
        norms.append(act.norm(dim=-1).cpu())
    return t.cat(norms).numpy()


def collect_raw_and_standardized(buffer, n_batches, d_model):
    """Collect raw activations, compute standardization, return both norms
    and the standardized tensor (for data-init sampling)."""
    raw_batches = []
    for i, act in enumerate(buffer):
        if i >= n_batches:
            break
        raw_batches.append(act.cpu())

    # Raw norms
    all_raw = t.cat(raw_batches, dim=0)
    raw_norms = all_raw.norm(dim=-1).numpy()

    # Standardization: act' = scale * (act - mean)
    mean = all_raw.mean(dim=0)
    centered = all_raw - mean
    mean_sq_norm = centered.pow(2).sum(dim=-1).mean().item()
    scale = (d_model / mean_sq_norm) ** 0.5
    standardized = scale * centered
    std_norms = standardized.norm(dim=-1).numpy()

    print(f"  Raw activations: mean norm = {np.mean(raw_norms):.2f}, "
          f"std = {np.std(raw_norms):.2f}")
    print(f"  Standardized:    mean norm = {np.mean(std_norms):.2f}, "
          f"std = {np.std(std_norms):.2f}")
    print(f"  Expected post-std norm ~ sqrt({d_model}) = {np.sqrt(d_model):.2f}")
    print(f"  Standardization scale = {scale:.4f}")

    return raw_norms, std_norms, standardized, mean, scale


def compute_spade_weight_norms(activation_dim, dict_size, seed=42):
    """Initialize SpaDEAutoEncoder with randn and compute weight norms."""
    t.manual_seed(seed)
    ae = SpaDEAutoEncoder(activation_dim, dict_size)

    # Ae: [dict_size, activation_dim] — row norms
    ae_row_norms = ae.Ae.data.norm(dim=1).cpu().numpy()
    # Ad: [activation_dim, dict_size] — column norms (= Ae row norms at init)
    ad_col_norms = ae.Ad.data.norm(dim=0).cpu().numpy()

    print(f"  Ae row norms: mean = {np.mean(ae_row_norms):.2f}, "
          f"std = {np.std(ae_row_norms):.2f}")
    print(f"  Ad col norms: mean = {np.mean(ad_col_norms):.2f}, "
          f"std = {np.std(ad_col_norms):.2f}")
    print(f"  Expected norm ~ sqrt({activation_dim}) = {np.sqrt(activation_dim):.2f}")

    return ae_row_norms, ad_col_norms


def compute_data_init_norms(standardized_acts, dict_size, seed=42):
    """Sample dict_size standardized activations (mimicking data-sampled init)
    and return their norms — these are what SpaDE's Ae rows look like after
    data-init."""
    t.manual_seed(seed)
    n = standardized_acts.shape[0]
    idx = t.randperm(n)[:dict_size]
    init_data = standardized_acts[idx]  # [dict_size, activation_dim]
    norms = init_data.norm(dim=-1).cpu().numpy()
    print(f"  Data-init Ae row norms: mean = {np.mean(norms):.2f}, "
          f"std = {np.std(norms):.2f}")
    return norms


def plot_histogram(ax, data, title, xlabel, color, ref_line=None, ref_label=None):
    """Plot a histogram on the given axes (log-log scale)."""
    # Use log-spaced bins; clip to strictly positive values
    pos = data[data > 0]
    lo = max(pos.min(), 1e-12)
    hi = pos.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    ax.hist(pos, bins=bins, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count (log)")

    # Stats annotation
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean = {mean_val:.2f}")

    if ref_line is not None:
        ax.axvline(ref_line, color="green", linestyle=":", linewidth=1.2,
                   label=ref_label or f"Ref = {ref_line:.2f}")

    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic norm plots for SpaDE SAE")
    parser.add_argument("--output_dir", type=str, default=os.path.join(str(repo_root()), "figures"),
                        help="Directory to save the figure")
    parser.add_argument("--n_batches", type=int, default=100,
                        help="Number of activation batches to collect (default: 100)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Constants (matching the training entrypoints) ----
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    dataset_name = "taufeeque/othellogpt"
    context_length = 59
    layer = 5
    activation_dim = 512
    dict_size = 16 * activation_dim  # 8192
    sae_batch_size = 8192

    if args.device is None:
        if t.cuda.is_available():
            device = "cuda:0"
        elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: Load model and create activation buffer ----
    print("Loading OthelloGPT model...")
    data = othello_hf_dataset_to_generator(
        dataset_name, context_length=context_length, split="train", streaming=True
    )
    model = get_model(model_name, device)
    submodule = get_submodule(model_name, layer, model, SubmoduleType.resid_post)

    activation_buffer = NNsightActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=1e3,
        ctx_len=context_length,
        refresh_batch_size=64,
        out_batch_size=sae_batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
    )

    # ---- Step 2: Collect raw and standardized activation norms ----
    print(f"\nCollecting {args.n_batches} batches of activations...")
    raw_norms, std_norms, std_acts, std_mean, std_scale = collect_raw_and_standardized(
        activation_buffer, args.n_batches, activation_dim
    )
    total_samples = len(raw_norms)
    print(f"  Total samples: {total_samples:,}")

    # ---- Step 3: Compute SpaDE weight norms at init ----
    print(f"\nComputing SpaDE weight norms at init (seed={args.seed})...")
    ae_norms, ad_norms = compute_spade_weight_norms(activation_dim, dict_size, seed=args.seed)

    # ---- Step 3b: Compute data-sampled init norms ----
    print(f"\nComputing data-sampled init norms (seed={args.seed})...")
    data_init_norms = compute_data_init_norms(std_acts, dict_size, seed=args.seed)

    # ---- Step 4: Create 2x2 figure ----
    print("\nGenerating 2x2 diagnostic figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Activation Norms & SpaDE Weight Norms: randn vs Data-Sampled Init",
        fontsize=14, fontweight="bold", y=0.98,
    )

    sqrt_d = np.sqrt(activation_dim)

    # (1,1) Post-standardization activation norms (the regime SpaDE trains in)
    plot_histogram(
        axes[0, 0],
        std_norms,
        "Post-Standardization: Activation Norms",
        r"$\|x'\|_2$ where $x' = s(x - \mu)$",
        color="#4e79a7",
        ref_line=sqrt_d,
        ref_label=f"sqrt(d) = {sqrt_d:.1f}",
    )

    # (1,2) SpaDE randn init weight norms
    plot_histogram(
        axes[0, 1],
        ae_norms,
        "SpaDE Ae Row Norms: randn Init",
        r"$\|Ae_j\|_2$ (row norms)",
        color="#f28e2b",
        ref_line=sqrt_d,
        ref_label=f"sqrt(d) = {sqrt_d:.1f}",
    )

    # (2,1) Data-sampled init weight norms
    plot_histogram(
        axes[1, 0],
        data_init_norms,
        "SpaDE Ae Row Norms: Data-Sampled Init",
        r"$\|Ae_j\|_2$ (row norms)",
        color="#59a14f",
        ref_line=sqrt_d,
        ref_label=f"sqrt(d) = {sqrt_d:.1f}",
    )

    # (2,2) Overlay: all three distributions on one axis
    ax = axes[1, 1]
    # Use shared log-spaced bins across all three
    all_data = np.concatenate([std_norms, ae_norms, data_init_norms])
    pos = all_data[all_data > 0]
    lo, hi = max(pos.min(), 1e-12), pos.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)

    ax.hist(std_norms, bins=bins, color="#4e79a7", alpha=0.5, edgecolor="black",
            linewidth=0.2, label=f"Activations (mean={np.mean(std_norms):.1f})", density=True)
    ax.hist(ae_norms, bins=bins, color="#f28e2b", alpha=0.5, edgecolor="black",
            linewidth=0.2, label=f"randn init (mean={np.mean(ae_norms):.1f})", density=True)
    ax.hist(data_init_norms, bins=bins, color="#59a14f", alpha=0.5, edgecolor="black",
            linewidth=0.2, label=f"Data-sampled init (mean={np.mean(data_init_norms):.1f})", density=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Overlay: Activations vs randn vs Data-Sampled Init", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"$L_2$ norm")
    ax.set_ylabel("Density (log)")
    ax.axvline(sqrt_d, color="green", linestyle=":", linewidth=1.2, label=f"sqrt(d) = {sqrt_d:.1f}")
    ax.legend(fontsize=8)

    # Match x-axis ranges across all 4 panels
    all_xlims = [axes[r, c].get_xlim() for r in range(2) for c in range(2)]
    x_min = max(min(xl[0] for xl in all_xlims), 1e-3)
    x_max = max(xl[1] for xl in all_xlims)
    for r in range(2):
        for c in range(2):
            axes[r, c].set_xlim(x_min, x_max)

    # Add text annotation with key stats
    stats_text = (
        f"Samples: {total_samples:,}  |  d_model: {activation_dim}  |  "
        f"dict_size: {dict_size}  |  std_scale: {std_scale:.4f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # ---- Save ----
    out_path = os.path.join(args.output_dir, "diagnostic_norm_histograms.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")

    # ---- Print summary for quick reference ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Post-std activation norms: mean={np.mean(std_norms):.2f}, std={np.std(std_norms):.2f}")
    print(f"SpaDE Ae (randn init):     mean={np.mean(ae_norms):.2f}, std={np.std(ae_norms):.2f}")
    print(f"SpaDE Ae (data-init):      mean={np.mean(data_init_norms):.2f}, std={np.std(data_init_norms):.2f}")
    print(f"sqrt(d_model) = {sqrt_d:.2f}")
    print(f"Standardization scale = {std_scale:.4f}")
    print(f"\nKey insight: randn init norms cluster at ~{sqrt_d:.0f} (sqrt(d)),")
    print(f"  while data-sampled init norms match the activation distribution.")


if __name__ == "__main__":
    main()
