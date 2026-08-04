"""The ReStraV descriptor over a window, and the windows a fork figure can use.

R^21 is a whole-trajectory descriptor: one point per trajectory. Turning it into
a curve over move index means choosing a window, and the window length is what
decides whether the vector is R^21 at all -- `restrav_features` slices d[:, :7]
and theta[:, :6], so a window of fewer than 8 sampled positions returns a
*shorter* vector with no error. `r21` asserts the width; everything here is
built so a short window is a crash, not a silently different feature space.
"""

import numpy as np
import torch

from spade.straightness.geometry import (
    N_RAW_D, N_RAW_THETA, restrav_features, temporal_geometry)
from spade.violation_divergence.divergence import WINDOW

R21_DIM = 21
MIN_WINDOW = N_RAW_D + 1  # 8 sampled positions: 7 step lengths, 6 turning angles

# Sampling rule of the fork figure (analysis/violation_divergence/game_trajectory):
# one mover per curve, at t0's parity, over the study's window.
STRIDE = 2
FIRST_POS, LAST_POS = 1, 58  # position 0 is the attention sink


def positions(t0: int) -> np.ndarray:
    p = np.arange(max(FIRST_POS, t0 + WINDOW[0]), min(LAST_POS, t0 + WINDOW[-1]) + 1)
    return p[(p - t0) % STRIDE == 0]


def r21(Z: np.ndarray) -> np.ndarray:
    """[n, W, D] -> [n, 21]. Crashes rather than returning a truncated vector."""
    d, theta = temporal_geometry(torch.from_numpy(np.ascontiguousarray(Z)).double())
    f = restrav_features(d, theta).numpy()
    assert f.shape[1] == R21_DIM, (
        f"{Z.shape[1]} sampled positions give R^{f.shape[1]}, not R^21; "
        f"{MIN_WINDOW} are needed to fill d1..d{N_RAW_D} and "
        f"theta1..theta{N_RAW_THETA}")
    return f


def expanding(F: np.ndarray, k0: int) -> tuple[np.ndarray, np.ndarray]:
    """Descriptor of sampled positions k0..k, for every k with a full window.

    ([n, n_points, 21], the k indices). Causal and anchored at the branch: the
    descriptor at k is "the trajectory since t0", so it never mixes in a ply
    from before the fork.
    """
    ks = np.arange(k0 + MIN_WINDOW - 1, F.shape[1])
    return np.stack([r21(F[:, k0:k + 1]) for k in ks], axis=1), ks


def sliding(F: np.ndarray, w: int, placement: str) -> tuple[np.ndarray, np.ndarray]:
    """Descriptor of a fixed-length window of `w` sampled positions.

    `placement` names which sampled position the descriptor is drawn at:
    "trailing" (the window's last position, causal), "centered", or "leading".
    """
    starts = np.arange(F.shape[1] - w + 1)
    offset = {"leading": 0, "centered": (w - 1) // 2, "trailing": w - 1}[placement]
    return np.stack([r21(F[:, s:s + w]) for s in starts], axis=1), starts + offset


def travel(F: np.ndarray, k0: int, ref: int, n_shuffle: int = 20):
    """Curve path length after its first point / its norm there: observed, null.

    The null recomputes the same expanding window over a random permutation of
    the post-t0 positions, so the trajectory's temporal order is destroyed and
    growing the window is the only thing left that can move the descriptor.
    Both are [n_branches - 1] over the non-reference branches.
    """
    keep = np.arange(len(F)) != ref

    def ratio(G):
        D, _ = expanding(G, k0)
        D = rescale(D - D[ref])[keep]  # the reference is 0 everywhere: no ratio to take
        return (np.linalg.norm(np.diff(D, axis=1), axis=-1).sum(1)
                / np.linalg.norm(D[:, 0], axis=-1))

    rng = np.random.default_rng(0)
    post = np.arange(k0, F.shape[1])
    null = np.stack([ratio(np.concatenate([F[:, :k0], F[:, rng.permutation(post)]], axis=1))
        for _ in range(n_shuffle)])
    return ratio(F), null.mean(0)


def rescale(D: np.ndarray) -> np.ndarray:
    """Unit-RMS per coordinate: a diagonal map, so zero displacement stays zero.

    d1..d7 are activation-norm units and theta1..theta6 are degrees, so an
    unscaled PCA of R^21 reports whichever block happens to be larger.
    """
    rms = np.sqrt((D.reshape(-1, D.shape[-1]) ** 2).mean(0))
    assert (rms > 0).all(), f"coordinates with no spread: {np.flatnonzero(rms == 0)}"
    return D / rms
