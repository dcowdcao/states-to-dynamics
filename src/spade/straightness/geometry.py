"""Trajectory geometry, ported verbatim from ReStraV (arXiv:2507.00583).

For a trajectory Z of per-position activations: step length d_i is the L2 norm
of the displacement z_{i+1}-z_i; turning angle theta_i is the angle in degrees
between consecutive displacements. A zero-length step makes the angle
undefined -> NaN (possible in sparse SAE codes); moments are nan-aware.
"""

import torch
import torch.nn.functional as F

N_RAW_D, N_RAW_THETA = 7, 6  # ReStraV's raw leading features

# Column names of restrav_features, in order. The layout is the contract for
# anything that reads a single component back out (loadings, ablations).
RESTRAV_FEATURE_NAMES = (
    [f"d{i + 1}" for i in range(N_RAW_D)]
    + [f"θ{i + 1}" for i in range(N_RAW_THETA)]
    + [f"{m} d" for m in ("mean", "min", "max", "var")]
    + [f"{m} θ" for m in ("mean", "min", "max", "var")])


def temporal_geometry(Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T, D] -> d [B, T-1], theta [B, T-2] (degrees)."""
    delta = Z[:, 1:, :] - Z[:, :-1, :]
    d = delta.norm(dim=-1)
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    theta[(d[:, :-1] == 0) | (d[:, 1:] == 0)] = torch.nan
    return d, theta


def moment4(x: torch.Tensor) -> torch.Tensor:
    """[B, T] -> [B, 4]: nan-aware {mean, min, max, var}."""
    mu = x.nanmean(-1)
    mn = torch.nan_to_num(x, nan=torch.inf).amin(-1)
    mx = torch.nan_to_num(x, nan=-torch.inf).amax(-1)
    var = (x - mu.unsqueeze(-1)).square().nanmean(-1)
    return torch.stack([mu, mn, mx, var], dim=-1)


def restrav_features(d: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """The paper's 21-D vector: first-7 d, first-6 theta, moment4 of each."""
    return torch.cat([d[:, :N_RAW_D], theta[:, :N_RAW_THETA],
                      moment4(d), moment4(theta)], dim=1)


def train_mask(n: int) -> torch.Tensor:
    """Deterministic split shared by centering and classification: even index ->
    train. Classes are equal-size contiguous blocks, so parity stratifies."""
    return torch.arange(n) % 2 == 0
