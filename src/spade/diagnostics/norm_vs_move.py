"""
Plot per-token L2 norm of OthelloGPT layer-5 residual-stream activations
versus move position in the game.

Shows three things on one figure:

  Top:    full 59-position curve on a log y-scale, so the BOS attention
          sink at position 0 (norm ~300) and the rest (norm ~22) coexist
          in one frame. Mean line + 5-95 percentile band. Faint per-game
          trajectories overlaid in the background.

  Mid:    zoomed view of positions 1-58 only, linear y-scale, so any
          actual move-position drift is readable. Mean + percentile band
          + faint trajectories.

  Bot:    same zoomed view but for STANDARDIZED activations (the space
          the SAEs train in), to confirm the same conclusion holds after
          the StandardizedBuffer.calibrate() rescaling.

Also prints per-position summary stats so we have numbers, not just
pixels.
"""

import os
import argparse

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from spade.common.config import MODEL_NAME, DATASET, LAYER, CTX_LEN, D_MODEL
from spade.common.paths import add_karvonen_to_path, repo_root

add_karvonen_to_path()

from circuits.utils import (
    get_model,
    get_submodule,
    othello_hf_dataset_to_generator,
    SubmoduleType,
)


def stream_activations(model, submodule, gen, n_games, batch_size, device):
    n_emitted = 0
    while n_emitted < n_games:
        batch_tokens = []
        while len(batch_tokens) < batch_size and n_emitted + len(batch_tokens) < n_games:
            try:
                seq = next(gen)
            except StopIteration:
                break
            if len(seq) < CTX_LEN:
                continue
            batch_tokens.append(list(seq[:CTX_LEN]))
        if not batch_tokens:
            return
        token_tensor = t.tensor(batch_tokens, dtype=t.long, device=device)
        with t.no_grad():
            with model.trace(token_tensor, scan=False, validate=False):
                hidden = submodule.output.save()
        h = hidden.value
        if isinstance(h, tuple):
            h = h[0]
        h = h.detach().to("cpu", dtype=t.float32)
        n_emitted += h.shape[0]
        yield h


def calibrate_standardization(activations_list):
    """Compute (mean[D_MODEL], scale scalar) the same way StandardizedBuffer
    does: zero-mean per-dim, then a single scale s such that E[||s(x-mu)||^2]
    = D_MODEL."""
    flat = t.cat([h.reshape(-1, D_MODEL) for h in activations_list], dim=0).to(t.float64)
    mean = flat.mean(dim=0)
    diff_sq = (flat - mean).pow(2).sum(dim=1)
    mean_sq_norm = diff_sq.mean().item()
    scale = float((D_MODEL / mean_sq_norm) ** 0.5)
    return mean.to(t.float32), scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=500)
    parser.add_argument("--n_traj", type=int, default=80,
                        help="Number of individual game trajectories to overlay.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(str(repo_root()), "figures", "activation_norms"),
    )
    parser.add_argument("--name", type=str, default="plot5_norm_vs_move",
                        help="Output filename stem.")
    parser.add_argument("--exclude_pos0", action="store_true",
                        help="Drop the BOS / attention-sink token (position 0) "
                             "entirely. With this flag the figure is a "
                             "two-panel zoom on positions 1-58 only and all "
                             "stats are computed over t>=1.")
    args = parser.parse_args()

    if args.device is None:
        if t.cuda.is_available():
            args.device = "cuda"
        elif t.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = t.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model on {device}...", flush=True)
    model = get_model(MODEL_NAME, device)
    submodule = get_submodule(MODEL_NAME, LAYER, model, SubmoduleType.resid_post)

    print(f"Streaming {args.n_games} games...", flush=True)
    gen = othello_hf_dataset_to_generator(
        DATASET, context_length=CTX_LEN, split="train", streaming=True
    )

    cached = []
    n_seen = 0
    for h in stream_activations(model, submodule, gen, args.n_games, args.batch_size, device):
        cached.append(h)
        n_seen += h.shape[0]
        print(f"  {n_seen}/{args.n_games} games", flush=True)
    if not cached:
        raise RuntimeError("No games streamed.")

    H = t.cat(cached, dim=0)[:args.n_games]               # [N, 59, 512]
    N = H.shape[0]
    print(f"Got {N} games × {CTX_LEN} positions × {D_MODEL} dims.", flush=True)

    # Raw norms
    norm_raw = H.norm(dim=-1).numpy()                      # [N, 59]

    # Standardized norms
    print("Calibrating standardization...", flush=True)
    mean_vec, scale = calibrate_standardization(cached)
    print(f"  scale = {scale:.6f}", flush=True)
    H_std = scale * (H - mean_vec)                         # [N, 59, 512]
    norm_std = H_std.norm(dim=-1).numpy()

    # ---- Per-position summary stats ----
    def pct(x, q):
        return np.percentile(x, q, axis=0)
    summary = {
        "raw":  {"mean": norm_raw.mean(0), "median": np.median(norm_raw, 0),
                 "p05":  pct(norm_raw, 5), "p25": pct(norm_raw, 25),
                 "p75":  pct(norm_raw, 75), "p95": pct(norm_raw, 95),
                 "min":  norm_raw.min(0),  "max": norm_raw.max(0)},
        "std":  {"mean": norm_std.mean(0), "median": np.median(norm_std, 0),
                 "p05":  pct(norm_std, 5), "p25": pct(norm_std, 25),
                 "p75":  pct(norm_std, 75), "p95": pct(norm_std, 95),
                 "min":  norm_std.min(0),  "max": norm_std.max(0)},
    }
    print("\nPer-position raw L2 norm (mean) by move t:")
    sample_positions = [0, 1, 2, 5, 10, 20, 30, 40, 50, 58]
    if args.exclude_pos0:
        sample_positions = sample_positions[1:]
    for t_pos in sample_positions:
        print(f"  t={t_pos:2d}: mean={summary['raw']['mean'][t_pos]:8.3f}   "
              f"p5={summary['raw']['p05'][t_pos]:8.3f}   "
              f"p95={summary['raw']['p95'][t_pos]:8.3f}   "
              f"max={summary['raw']['max'][t_pos]:8.3f}")
    print("\nMove-position drift (positions 1..58 only):")
    bulk_raw = summary['raw']['mean'][1:]
    bulk_std = summary['std']['mean'][1:]
    print(f"  raw:          min={bulk_raw.min():.3f}  max={bulk_raw.max():.3f}  "
          f"max/min={bulk_raw.max()/bulk_raw.min():.4f}x")
    print(f"  standardized: min={bulk_std.min():.3f}  max={bulk_std.max():.3f}  "
          f"max/min={bulk_std.max()/bulk_std.min():.4f}x")
    # Linear regression of mean norm over positions 1..58
    x = np.arange(1, CTX_LEN)
    slope_raw, intercept_raw = np.polyfit(x, bulk_raw, 1)
    slope_std, intercept_std = np.polyfit(x, bulk_std, 1)
    pct_per_move_raw = slope_raw / bulk_raw.mean() * 100
    pct_per_move_std = slope_std / bulk_std.mean() * 100
    print(f"  raw mean trend: {slope_raw:+.5f} per move  "
          f"({pct_per_move_raw:+.4f}% / move)")
    print(f"  std mean trend: {slope_std:+.5f} per move  "
          f"({pct_per_move_std:+.4f}% / move)")

    # ---- Plot ----
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 200,
    })
    positions = np.arange(CTX_LEN)
    n_traj_plot = min(args.n_traj, N)

    if args.exclude_pos0:
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        axes = [None] + list(axes)  # pad so panel B/C indices stay [1], [2]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=False)

        # Panel A: full view, log y
        ax = axes[0]
        for i in range(n_traj_plot):
            ax.plot(positions, norm_raw[i], color="steelblue", alpha=0.10, lw=0.6)
        ax.fill_between(positions, summary['raw']['p05'], summary['raw']['p95'],
                        color="C0", alpha=0.20, label="5–95 percentile")
        ax.fill_between(positions, summary['raw']['p25'], summary['raw']['p75'],
                        color="C0", alpha=0.35, label="25–75 percentile")
        ax.plot(positions, summary['raw']['mean'], color="C0", lw=2, label="mean")
        ax.plot(positions, summary['raw']['max'], color="crimson", lw=1, ls="--",
                label="max over games")
        ax.axhline(np.sqrt(D_MODEL), color="grey", ls=":", lw=1,
                   label=fr"$\sqrt{{d_{{model}}}}={np.sqrt(D_MODEL):.2f}$")
        ax.set_yscale("log")
        ax.set_xlabel("token position in game (move number)")
        ax.set_ylabel(r"$\|x_t\|_2$  (raw, layer 5)")
        ax.set_title(
            f"OthelloGPT layer-5 residual-stream L2 norm vs move position  "
            f"(N={N} games, log y-scale, full view)"
        )
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9)

    # Panel B: zoomed, positions 1-58, linear y
    ax = axes[1]
    pos_b = positions[1:]
    for i in range(n_traj_plot):
        ax.plot(pos_b, norm_raw[i, 1:], color="steelblue", alpha=0.10, lw=0.6)
    ax.fill_between(pos_b, summary['raw']['p05'][1:], summary['raw']['p95'][1:],
                    color="C0", alpha=0.20, label="5–95 percentile")
    ax.fill_between(pos_b, summary['raw']['p25'][1:], summary['raw']['p75'][1:],
                    color="C0", alpha=0.35, label="25–75 percentile")
    ax.plot(pos_b, summary['raw']['mean'][1:], color="C0", lw=2, label="mean")
    # Linear fit on the mean trajectory
    fit_y = slope_raw * pos_b + intercept_raw
    ax.plot(pos_b, fit_y, color="black", lw=1, ls="--",
            label=f"linear fit: {slope_raw:+.4f}/move ({pct_per_move_raw:+.3f}%/move)")
    ax.axhline(np.sqrt(D_MODEL), color="grey", ls=":", lw=1,
               label=fr"$\sqrt{{d_{{model}}}}={np.sqrt(D_MODEL):.2f}$")
    ax.set_xlabel("token position in game (move number)")
    ax.set_ylabel(r"$\|x_t\|_2$  (raw, layer 5)")
    if args.exclude_pos0:
        ax.set_title(
            f"OthelloGPT layer-5 raw L2 norm vs move position "
            f"(positions 1–58, BOS sink dropped, N={N} games)  "
            f"— per-move drift {pct_per_move_raw:+.3f}% / move"
        )
    else:
        ax.set_title(
            f"Zoom: positions 1–58 only, linear y-scale  "
            f"(per-move drift {pct_per_move_raw:+.3f}% per move)"
        )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)

    # Panel C: zoomed, positions 1-58, standardized
    ax = axes[2]
    for i in range(n_traj_plot):
        ax.plot(pos_b, norm_std[i, 1:], color="indianred", alpha=0.10, lw=0.6)
    ax.fill_between(pos_b, summary['std']['p05'][1:], summary['std']['p95'][1:],
                    color="C3", alpha=0.20, label="5–95 percentile")
    ax.fill_between(pos_b, summary['std']['p25'][1:], summary['std']['p75'][1:],
                    color="C3", alpha=0.35, label="25–75 percentile")
    ax.plot(pos_b, summary['std']['mean'][1:], color="C3", lw=2, label="mean")
    fit_y_s = slope_std * pos_b + intercept_std
    ax.plot(pos_b, fit_y_s, color="black", lw=1, ls="--",
            label=f"linear fit: {slope_std:+.5f}/move ({pct_per_move_std:+.3f}%/move)")
    ax.axhline(np.sqrt(D_MODEL), color="grey", ls=":", lw=1,
               label=fr"$\sqrt{{d_{{model}}}}={np.sqrt(D_MODEL):.2f}$")
    ax.set_xlabel("token position in game (move number)")
    ax.set_ylabel(r"$\|x'_t\|_2$  (standardized, layer 5)")
    if args.exclude_pos0:
        ax.set_title(
            f"Standardized L2 norm vs move position "
            f"(positions 1–58, scale={scale:.4f})  "
            f"— per-move drift {pct_per_move_std:+.3f}% / move"
        )
    else:
        ax.set_title(
            f"Zoom: positions 1–58, STANDARDIZED activations  "
            f"(scale = {scale:.4f}; per-move drift {pct_per_move_std:+.3f}%/move)"
        )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)

    suptitle_extra = "  (BOS / position-0 attention sink dropped)" if args.exclude_pos0 else ""
    fig.suptitle(
        f"Activation norms over game move — OthelloGPT layer 5"
        f"{suptitle_extra}\n({MODEL_NAME})",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = os.path.join(args.output_dir, f"{args.name}.png")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved figure -> {out_path}")

    # Save the per-position summary as CSV for later reference
    csv_path = os.path.join(args.output_dir, f"{args.name}_summary.csv")
    with open(csv_path, "w") as f:
        f.write("position,raw_mean,raw_median,raw_p05,raw_p25,raw_p75,raw_p95,"
                "raw_min,raw_max,std_mean,std_median,std_p05,std_p25,std_p75,"
                "std_p95,std_min,std_max\n")
        for k in range(CTX_LEN):
            r = summary['raw']; s = summary['std']
            f.write(f"{k},{r['mean'][k]:.6f},{r['median'][k]:.6f},"
                    f"{r['p05'][k]:.6f},{r['p25'][k]:.6f},{r['p75'][k]:.6f},"
                    f"{r['p95'][k]:.6f},{r['min'][k]:.6f},{r['max'][k]:.6f},"
                    f"{s['mean'][k]:.6f},{s['median'][k]:.6f},"
                    f"{s['p05'][k]:.6f},{s['p25'][k]:.6f},{s['p75'][k]:.6f},"
                    f"{s['p95'][k]:.6f},{s['min'][k]:.6f},{s['max'][k]:.6f}\n")
    print(f"Saved per-position summary CSV -> {csv_path}")


if __name__ == "__main__":
    main()
