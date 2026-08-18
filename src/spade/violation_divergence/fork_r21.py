"""The 21-D curvature descriptor a fork is routed through, and the planes it is
drawn in.

`fork_plane` embeds the raw displacement D(t) = z_b(t) - z_ref(t), which lives in
a different space in every cell of the nine-space grid. Here every space's
displacement trajectory is first summarised by the same 21 named ReStraV
components (`spade.straightness.geometry`), so "mean theta" is the same quantity
in raw layer 5 and in CosineSpaDE and one plane can be applied to every cell.

Shared by the single-game grids in `analysis/violation_divergence/r21_shared_basis`
and the population sweep in `analysis/violation_divergence/r21_run_population`.
"""

import numpy as np
import torch

from spade.figures.branch_panels import plane_displacement
from spade.straightness.geometry import (
    RESTRAV_FEATURE_NAMES, restrav_features, temporal_geometry)

W = 8  # smallest window with a full 21-D descriptor (restrav_features slices d[:, :7])
MEAN_D = RESTRAV_FEATURE_NAMES.index("mean d")
MEAN_THETA = RESTRAV_FEATURE_NAMES.index("mean θ")
# The two halves of R^21, which disagree about how far an illegal branch runs.
# No component name carries both letters, so the split is exact.
BLOCKS = {q: [i for i, n in enumerate(RESTRAV_FEATURE_NAMES) if c in n]
          for q, c in (("d", "d"), ("theta", "θ"))}


def descriptors(F: np.ndarray) -> np.ndarray:
    """[n, T, width] -> [n, T-W+1, 21]; a point is the W samples ending at t."""
    x = torch.from_numpy(F).unfold(1, W, 1).permute(0, 1, 3, 2).flatten(0, 1)
    f = restrav_features(*temporal_geometry(x))
    assert f.shape[1] == len(RESTRAV_FEATURE_NAMES), f"width {f.shape[1]}, need 21"
    assert torch.isfinite(f).all(), "a zero-length step left an undefined angle"
    return f.reshape(F.shape[0], -1, f.shape[1]).numpy()


def fold_rollouts(A: np.ndarray, k: int) -> np.ndarray:
    """Mean over each branch's K rollouts; rows are `source_row * K + rollout`.

    Called on DESCRIPTORS, never on trajectories: phi is nonlinear, so a mean over
    the four paths is the curvature of a fictitious path none of them took.
    """
    return A if k == 1 else A.reshape(len(A) // k, k, *A.shape[1:]).mean(1)


def displacement21(F: np.ndarray, ref: int, k: int = 1) -> np.ndarray:
    """[n, nw, 21] descriptor displacement from the reference, in per-space SDs.

    The mean cancels in the difference, so z-scoring leaves the reference exactly
    at the origin and only rescales each named component. The rollout mean is
    taken here, between phi and the difference, so the SD is the SD of the
    plotted object.
    """
    phi = fold_rollouts(descriptors(F), k)
    sd = phi.reshape(-1, phi.shape[-1]).std(0)
    assert (sd > 0).all(), f"constant component: {np.array(RESTRAV_FEATURE_NAMES)[sd == 0]}"
    return (phi - phi[ref]) / sd


def top2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uncentred top-2 plane (basis [2, k], variance ratios [2]).

    Uncentred because zero displacement is a real point: centring would move the
    reference off the origin it is drawn at.
    """
    _, s, vt = np.linalg.svd(X.reshape(-1, X.shape[-1]), full_matrices=False)
    v = vt[:2]
    sign = np.sign(v[np.arange(2), np.abs(v).argmax(1)])
    return v * sign[:, None], (s ** 2 / (s ** 2).sum())[:2]


def named_basis() -> np.ndarray:
    return np.eye(len(RESTRAV_FEATURE_NAMES))[[MEAN_D, MEAN_THETA]]


def activation_plane(F: np.ndarray, ref: int, k: int = 1) -> np.ndarray:
    """The status-quo comparison: top-2 of the raw displacement, [n, T, 2].

    Everything here is linear, so the rollout mean can be taken on the
    trajectories; it is the same plane as folding after the projection.
    """
    F = fold_rollouts(F, k)
    D = F - F[ref]
    basis, _ = top2(D)
    return D @ basis.T


def run_ratio(P: np.ndarray, rows: np.ndarray, ctl: int, ref: int,
              anchors: np.ndarray, t0: int) -> float:
    """How far `rows`' mean path runs from the reference over the drawn anchors,
    divided by the control's run in the same field.

    A ratio and never a distance: the field is z-scored per space and, for the
    refit planes, fit per space too, so only "this branch against this space's own
    legal control" carries from one space to the next.
    """
    run = plane_displacement(np.stack([P[ref], P[rows].mean(0), P[ctl]]),
                             anchors, t0, 0)
    return float(run[1] / run[2])
