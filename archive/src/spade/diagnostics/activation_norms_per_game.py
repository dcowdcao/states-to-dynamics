"""
OthelloGPT activation-norm + SAE-MSE diagnostic plots.

Plot 1: per-token L2 norm trajectories for n_traj sample games (color = move).
Plot 2: cross-game mean squared norm E[||x_t||^2] vs. token position t,
        +/- 1 std band, computed over n_pop games.

If --sae_path is provided (default: spade_w4096_g0.2_s42), additionally:

Plot 3: per-token SAE reconstruction MSE ||x_t - x_hat_t||^2  trajectories
        for n_traj games (color = move). Matches the wandb 'mse_loss'
        per-token quantity (sum over the 512 feature dims).
Plot 4: scatter of MSE vs norm over all (game, move) tokens for the
        first n_traj games, color = move.

All extracted at blocks.5.hook_resid_post (the layer the SAEs train on).
The MSE is computed on STANDARDIZED activations to match the SAE's training
distribution.  Standardization stats (mean/scale) are recomputed locally
from a calibration pool, identical to StandardizedBuffer.calibrate() in
spade/training/core.py.
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
from spade.saes.spade import SpaDEAutoEncoder


DEFAULT_SAE_PATH = os.path.join(
    str(repo_root()),
    "archive",
    "runs",
    "Run_3_Results",
    "runpod_autoencoders",
    "spade_w4096_g0.2_s42",
)


# -----------------------------------------------------------------------------
# Activation streaming
# -----------------------------------------------------------------------------

def _make_gen():
    return othello_hf_dataset_to_generator(
        DATASET, context_length=CTX_LEN, split="train", streaming=True
    )


def stream_activations(model, submodule, gen, n_games, batch_size, device):
    """Generator yielding raw residual-stream activation batches of shape
    [B, CTX_LEN, D_MODEL] on CPU float32, until n_games full-length games
    have been emitted (or gen is exhausted)."""
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
        h = h.detach().to("cpu", dtype=t.float32)  # [B, 59, 512]
        n_emitted += h.shape[0]
        yield h


# -----------------------------------------------------------------------------
# Standardization (mirrors StandardizedBuffer.calibrate)
# -----------------------------------------------------------------------------

def calibrate_standardization(model, submodule, n_games, batch_size, device):
    """Compute (mean [D_MODEL], scale scalar) over a calibration pool of
    n_games games, identical to StandardizedBuffer.calibrate() in
    spade/training/core.py — except over a fixed pool of games rather
    than over n_calibration_batches buffer batches.
    """
    print(f"Calibrating standardization over {n_games} games...", flush=True)
    gen = _make_gen()
    sum_act = t.zeros(D_MODEL, dtype=t.float64)
    n_tokens = 0
    cached = []  # keep activations so we don't re-stream for variance

    n_seen = 0
    for h in stream_activations(model, submodule, gen, n_games, batch_size, device):
        flat = h.reshape(-1, D_MODEL).to(t.float64)  # [B*59, 512]
        sum_act += flat.sum(dim=0)
        n_tokens += flat.shape[0]
        cached.append(flat)
        n_seen += h.shape[0]
        if n_seen % (batch_size * 8) < batch_size:
            print(f"  calibration: {n_seen}/{n_games} games", flush=True)

    if n_tokens == 0:
        raise RuntimeError("Calibration pool is empty.")
    mean = (sum_act / n_tokens).to(t.float32)  # [D_MODEL]

    sum_sq = 0.0
    for flat in cached:
        diff = flat.to(t.float32) - mean
        sum_sq += diff.pow(2).sum().item()
    mean_sq_norm = sum_sq / n_tokens
    scale = float((D_MODEL / mean_sq_norm) ** 0.5)

    print(f"  Mean norm:        {mean.norm().item():.4f}")
    print(f"  E[||x-mean||^2]:  {mean_sq_norm:.4f}")
    print(f"  Scale factor:     {scale:.6f}")
    print(f"  After standardization: E[||x'||^2] ~= {D_MODEL}")
    return mean, scale


# -----------------------------------------------------------------------------
# SAE loading (SpaDE only — this diagnostic targets the SpaDE pathology)
# -----------------------------------------------------------------------------

def load_spade_sae(sae_path, device):
    """Load a SpaDE SAE checkpoint directory (containing ae.pt + config.json)."""
    ae_pt = os.path.join(sae_path, "ae.pt")
    if not os.path.isfile(ae_pt):
        raise FileNotFoundError(f"No ae.pt at {sae_path}")
    print(f"Loading SpaDE SAE from {sae_path} ...", flush=True)
    ae = SpaDEAutoEncoder.from_pretrained(ae_pt, device=device)
    ae.eval()
    return ae


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def _scatter_per_token(ax, values, n_games, ctx_len, ylabel, title, cmap="viridis"):
    """Helper: scatter values [n_games, ctx_len] with x=game idx, color=position."""
    arr = values.numpy()
    game_idx = np.repeat(np.arange(n_games), ctx_len)
    positions = np.tile(np.arange(ctx_len), n_games)
    sc = ax.scatter(
        game_idx, arr.reshape(-1),
        c=positions, cmap=cmap, s=8, alpha=0.55, linewidths=0,
    )
    ax.set_xlabel("game index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    return sc


def plot_norm_trajectories(norms_traj, out_path):
    n_games, ctx_len = norms_traj.shape
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = _scatter_per_token(
        ax, norms_traj, n_games, ctx_len,
        ylabel=r"$\|x_t\|_2$  (residual stream, layer 5, raw)",
        title=f"OthelloGPT residual-stream norm at layer {LAYER} "
              f"across {n_games} games (color = move in game)",
    )
    fig.colorbar(sc, ax=ax).set_label("token position in game")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved Plot 1 -> {out_path}")


def plot_population_curve(mean, std, out_path, ylabel, title, hline=None, hline_label=None):
    positions = np.arange(mean.shape[0])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, mean, color="C0", lw=2, label=ylabel)
    ax.fill_between(positions, mean - std, mean + std, color="C0", alpha=0.2,
                    label=r"$\pm 1$ std")
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", lw=1, label=hline_label)
    ax.set_xlabel("token position in game")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_mse_trajectories(mse_traj, out_path):
    n_games, ctx_len = mse_traj.shape
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = _scatter_per_token(
        ax, mse_traj, n_games, ctx_len,
        ylabel=r"$\|x_t - \hat x_t\|^2$  (SpaDE recon MSE, sum over 512 dims)",
        title=f"SpaDE reconstruction MSE per move at layer {LAYER} "
              f"across {n_games} games (color = move in game)",
    )
    fig.colorbar(sc, ax=ax).set_label("token position in game")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved Plot 3 -> {out_path}")


def plot_mse_vs_norm_scatter(norms_traj, mse_traj, out_path):
    n_games, ctx_len = norms_traj.shape
    positions = np.tile(np.arange(ctx_len), n_games)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        norms_traj.numpy().reshape(-1),
        mse_traj.numpy().reshape(-1),
        c=positions, cmap="viridis", s=10, alpha=0.6, linewidths=0,
    )
    fig.colorbar(sc, ax=ax).set_label("token position in game")
    ax.set_xlabel(r"$\|x_t\|_2$  (standardized activation norm)")
    ax.set_ylabel(r"$\|x_t - \hat x_t\|^2$  (SpaDE recon MSE)")
    ax.set_title(
        f"SpaDE reconstruction MSE vs. activation norm "
        f"(per-token, {n_games} games, layer {LAYER})"
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"Saved Plot 4 -> {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_traj", "--n_games_traj", type=int, default=100,
                        dest="n_traj")
    parser.add_argument("--n_pop", "--n_games_pop", type=int, default=2000,
                        dest="n_pop")
    parser.add_argument("--calib_games", type=int, default=None,
                        help="Games for standardization calibration "
                             "(default: same as --n_pop)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str,
                        default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sae_path", type=str, default=DEFAULT_SAE_PATH,
        help="SpaDE SAE checkpoint dir (containing ae.pt + config.json). "
             "Pass empty string to skip MSE plots.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(str(repo_root()), "figures", "activation_norms"),
    )
    args = parser.parse_args()
    if args.calib_games is None:
        args.calib_games = args.n_pop

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 200,
    })

    device = t.device(args.device)
    print(f"Loading model on {device}...", flush=True)
    model = get_model(MODEL_NAME, device)
    submodule = get_submodule(MODEL_NAME, LAYER, model, SubmoduleType.resid_post)

    use_sae = bool(args.sae_path) and os.path.isdir(args.sae_path)
    sae = load_spade_sae(args.sae_path, device) if use_sae else None
    if not use_sae:
        print("(no SAE checkpoint found; skipping MSE plots)", flush=True)

    # Calibration (only needed if we have an SAE)
    if use_sae:
        mean_vec, scale = calibrate_standardization(
            model, submodule, args.calib_games, args.batch_size, device,
        )
        mean_vec_d = mean_vec.to(device)
    else:
        mean_vec_d, scale = None, None

    # ---- Streaming pass for population stats + trajectory subset ----
    n_total = max(args.n_traj, args.n_pop)
    print(f"Streaming {n_total} games at {MODEL_NAME} layer {LAYER}...", flush=True)
    gen = _make_gen()

    # Per-position accumulators (raw norms, always)
    sum_norm_sq_raw = t.zeros(CTX_LEN, dtype=t.float64)
    sum_norm_sq_raw_2 = t.zeros(CTX_LEN, dtype=t.float64)
    n_pop_seen = 0

    # Trajectory caches
    norms_traj_raw = t.empty(args.n_traj, CTX_LEN)            # raw norms (Plot 1)
    norms_traj_std = t.empty(args.n_traj, CTX_LEN) if use_sae else None  # standardized norms (Plot 4 x-axis)
    mse_traj = t.empty(args.n_traj, CTX_LEN) if use_sae else None
    n_traj_seen = 0

    for h in stream_activations(model, submodule, gen, n_total, args.batch_size, device):
        B = h.shape[0]

        # Raw per-token L2 norm (Plot 1 / Plot 2 use these)
        norm_raw = h.norm(dim=-1)             # [B, 59]
        if n_pop_seen < args.n_pop:
            take = min(B, args.n_pop - n_pop_seen)
            sq = norm_raw[:take].pow(2).to(t.float64)
            sum_norm_sq_raw += sq.sum(dim=0)
            sum_norm_sq_raw_2 += sq.pow(2).sum(dim=0)
            n_pop_seen += take

        # Trajectory cache (raw norms, Plot 1)
        if n_traj_seen < args.n_traj:
            take = min(B, args.n_traj - n_traj_seen)
            norms_traj_raw[n_traj_seen : n_traj_seen + take] = norm_raw[:take]

        # SAE branch — only runs while the trajectory cache is filling,
        # since Plots 3 & 4 only use the first n_traj games and there's no
        # population MSE accumulator to keep up to date.
        if use_sae and n_traj_seen < args.n_traj:
            take = min(B, args.n_traj - n_traj_seen)
            h_take = h[:take].to(device)
            x_std = scale * (h_take - mean_vec_d)               # [take, 59, 512]
            x_flat = x_std.reshape(-1, D_MODEL)
            with t.no_grad():
                x_hat_flat = sae(x_flat)
            x_hat = x_hat_flat.reshape(take, CTX_LEN, D_MODEL)
            mse = (x_std - x_hat).pow(2).sum(dim=-1).cpu()       # [take, 59]
            norm_std = x_std.norm(dim=-1).cpu()                  # [take, 59]

            norms_traj_std[n_traj_seen : n_traj_seen + take] = norm_std
            mse_traj[n_traj_seen : n_traj_seen + take] = mse

        if n_traj_seen < args.n_traj:
            n_traj_seen += min(B, args.n_traj - n_traj_seen)
        print(f"  streamed {min(n_pop_seen, args.n_pop)}/{args.n_pop} pop, "
              f"{n_traj_seen}/{args.n_traj} traj", flush=True)

    if n_pop_seen == 0:
        raise RuntimeError("No games streamed.")

    # ---- Reduce population accumulators ----
    mean_norm_sq_raw = (sum_norm_sq_raw / n_pop_seen).to(t.float32).numpy()
    var_raw = (sum_norm_sq_raw_2 / n_pop_seen - (sum_norm_sq_raw / n_pop_seen) ** 2)
    std_norm_sq_raw = var_raw.clamp(min=0).sqrt().to(t.float32).numpy()

    # ---- Plots 1 & 2 ----
    plot_norm_trajectories(
        norms_traj_raw[:n_traj_seen],
        os.path.join(args.output_dir, "plot1_norm_trajectories.png"),
    )
    plot_population_curve(
        mean_norm_sq_raw, std_norm_sq_raw,
        os.path.join(args.output_dir, "plot2_mean_sq_norm_vs_position.png"),
        ylabel=r"$E[\|x_t\|^2]$  (raw)",
        title=f"OthelloGPT residual-stream squared norm vs. move "
              f"(layer {LAYER}, N={n_pop_seen} games)",
        hline=D_MODEL,
        hline_label=f"d_model = {D_MODEL} (standardization target)",
    )
    print(
        f"\nE[||x_t||^2] (raw) range over t: "
        f"min={mean_norm_sq_raw.min():.2f}  max={mean_norm_sq_raw.max():.2f}  "
        f"d_model={D_MODEL}"
    )

    # ---- Plots 3 & 4 (SAE branch) ----
    if use_sae:
        plot_mse_trajectories(
            mse_traj[:n_traj_seen],
            os.path.join(args.output_dir, "plot3_mse_trajectories.png"),
        )
        plot_mse_vs_norm_scatter(
            norms_traj_std[:n_traj_seen],
            mse_traj[:n_traj_seen],
            os.path.join(args.output_dir, "plot4_mse_vs_norm_scatter.png"),
        )
        traj_mean_mse = mse_traj[:n_traj_seen].mean().item()
        print(
            f"\nMean MSE over the {n_traj_seen} trajectory games: "
            f"{traj_mean_mse:.4f}  (comparable to wandb 'mse_loss')"
        )


if __name__ == "__main__":
    main()
