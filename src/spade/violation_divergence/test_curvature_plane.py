"""Where a mean may be taken in the R^21 plane, and where it may not.

A figure drawing one path per rule averages over member categories. In the
displacement plane the order does not matter; here it does, and these are the
two halves of why.

Usage: python -m spade.violation_divergence.test_curvature_plane
"""

import numpy as np

from spade.violation_divergence.curvature_plane import W, fit_plane, windowed


def paths(n: int = 4, T: int = 12, D: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, T, D)).astype(np.float32)


def test_phi_does_not_commute_with_the_mean_over_branches():
    Z = paths()
    per_member = windowed(Z, W)[:, W - 1:].mean(0)
    of_the_mean = windowed(Z.mean(0, keepdims=True), W)[0, W - 1:]
    assert np.isfinite(per_member).all() and np.isfinite(of_the_mean).all()
    assert not np.allclose(per_member, of_the_mean), \
        "phi came out linear: averaging trajectories first would then be harmless"


def test_the_plane_projection_commutes_with_the_mean():
    phi = windowed(paths(), W)[:, W - 1:]
    plane = fit_plane(phi.reshape(-1, phi.shape[-1]), centred=False)
    assert np.allclose(plane.project(phi).mean(0), plane.project(phi.mean(0)),
                       atol=1e-5), "the projection must be safe to average either side of"


def test_a_shared_prefix_gives_one_descriptor():
    Z = paths()
    Z[1:, :W] = Z[0, :W]
    phi = windowed(Z, W)
    assert np.allclose(phi[:, W - 1] - phi[0, W - 1], 0), \
        "a window lying inside the shared prefix must not depend on the branch"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
