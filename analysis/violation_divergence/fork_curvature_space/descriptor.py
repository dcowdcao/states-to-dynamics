"""The ReStraV curvature descriptor as the plane the branch fork is drawn in.

The incumbent fork figure (../game_trajectory) projects the displacement
D(t) = z_b(t) - z_ref(t) onto a PCA of D. Here the plane is built inside the
space the straightness study works in instead: step lengths d and turning angles
theta, summarised by src/spade/straightness/geometry.py.

R^21 is the destination. The descriptor is defined for a whole trajectory while
the fork needs a curve over move index, and the two ways out are the two
variants:

  windowed  descriptor over the W sampled positions ENDING at t, so the curve
            survives. Causal, so a window lying inside the shared prefix is
            identical across branches and "zero before t0" stays true.
  whole     one descriptor per branch over its post-branch trajectory. No move
            axis; this is what analysis/straightness/feature_pca plots, cut down
            to one game's nine branches.

Dimensional accounting, which is what constrains the windowed variant. A window
of w sampled positions gives w-1 step lengths and w-2 angles, so ReStraV's 7 raw
d and 6 raw theta need w >= 8, i.e. 14 plies of span at STRIDE 2. The figure's
analysis window only reaches r=-8, four sampled positions before the branch, so
inside it no full-width window is also branch-free. Reading context back to the
START OF THE GAME dissolves that: at t0=20 the pre-branch positions are
2,4,..,20, and a window ending at r=-2 spans positions 4..18, entirely prefix.
The cost is the first two plotted offsets, whose windows run off the board and
stay undefined. The descriptor at offset r still covers r-14..r, so the branch
enters 1 of the window's 7 steps at r=0 and does not fill it until r=12; that
smear is the price of R^21 and measure.py reports it.
w<8 does NOT give a shorter R^21 -- restrav_features slices d[:, :7] and
theta[:, :6], which silently return fewer columns -- so descriptor() asserts the
width and falls back to the named 8-D moment block instead.

Angles are undefined at a zero-length step (geometry.py); a window of all-zero
steps additionally sends moment4's min/max to +-inf. Both are mapped to NaN here
and counted by measure.py -- the displacement D is entirely that case for the
reference branch, which is why the figures descriptor the trajectory.
"""

from dataclasses import dataclass

import numpy as np
import torch

from spade.common.study import study_dir
from spade.figures.paper_style import sae_label
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.straightness.geometry import (
    N_RAW_D, N_RAW_THETA, RESTRAV_FEATURE_NAMES, moment4, restrav_features,
    temporal_geometry)
from spade.violation_divergence.compute_divergence import sae_encoder

STRIDE = 2               # one mover per curve, as in ../game_trajectory
WINDOW = (-8, 24)        # the study's analysis window, r = t - t0
FIRST_POS, LAST_POS = 1, 58  # position 0 is the attention sink
W = 8                    # smallest window filling all 21 coordinates; see the docstring
# The game the grids draw, picked by a sweep over all 1550 branch-complete games
# and shared with the sibling fork studies so the panels line up.
DEFAULT_GAME = 1235
RAW_SPACE = "raw layer 5 (reference space)"
# The two descriptor components the straightness study states its results in, and
# the one fixed plane that is the same named pair in every cell of the grid.
FIXED_AXES = ("mean d", "mean θ")


def positions(t0: int, back: int = 0) -> np.ndarray:
    """The study's window at t0's parity, extended `back` sampled positions
    earlier -- clipped at the board, so a short game simply yields fewer."""
    lo = max(FIRST_POS, t0 + WINDOW[0] - STRIDE * back)
    p = np.arange(lo, min(LAST_POS, t0 + WINDOW[1]) + 1)
    return p[(p - t0) % STRIDE == 0]


def names_of(k: int) -> list[str]:
    """Column names of a k-wide descriptor: ReStraV's 21, or its 8 moments."""
    return list(RESTRAV_FEATURE_NAMES if k == len(RESTRAV_FEATURE_NAMES)
                else RESTRAV_FEATURE_NAMES[N_RAW_D + N_RAW_THETA:])


def descriptor(Z: torch.Tensor) -> np.ndarray:
    """[n, T, D] -> [n, K]: ReStraV's R^21 when T fills its raw leading features,
    else the named 8-D moment block those 21 end with. Non-finite -> NaN."""
    d, theta = temporal_geometry(Z)
    if d.shape[1] >= N_RAW_D and theta.shape[1] >= N_RAW_THETA:
        f = restrav_features(d, theta)
        assert f.shape[1] == len(RESTRAV_FEATURE_NAMES), \
            f"restrav_features truncated to {f.shape[1]} columns, not R^21"
    else:
        f = torch.cat([moment4(d), moment4(theta)], dim=1)
    return np.where(np.isfinite(f.numpy()), f.numpy(), np.nan)


def windowed(Z: np.ndarray, w: int = W) -> np.ndarray:
    """[n, T, D] -> [n, T, K], the descriptor over the w positions ending at t.

    K is 21 for w >= 8 and 8 below it; the first w-1 positions are NaN.
    """
    z = torch.from_numpy(Z)
    out = None
    for t in range(w - 1, Z.shape[1]):
        f = descriptor(z[:, t - w + 1:t + 1])
        if out is None:
            out = np.full(Z.shape[:2] + (f.shape[1],), np.nan, np.float32)
        out[:, t] = f
    return out


def jackknife(Z: np.ndarray) -> np.ndarray:
    """[n, T, D] -> [T, n, K]: whole-trajectory descriptor, one position dropped.

    The spread over which position was dropped is the only honest uncertainty a
    single game supports for the one-point-per-branch variant.
    """
    return np.stack([descriptor(torch.from_numpy(np.delete(Z, t, axis=1)))
                     for t in range(Z.shape[1])])


def spaces() -> list[tuple[str, object]]:
    """(label, SAE entry or None) -- raw layer 5 first, then the seed-42 families."""
    return [(RAW_SPACE, None)] + [
        (sae_label(e.config.replace("Trainer-othello", "")), e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]


def usable(n: int | None = None, w: int = W) -> list[int]:
    """Games carrying every branch type whose first full-width window still ends
    before the branch, so at least one drawn point is branch-free and the prefix
    can be seen collapsing rather than asserted. Needs t0 >= 18 at w=8."""
    meta = torch.load(study_dir("divergence") / "branches" / "branches.pt",
                      weights_only=True)
    n_types = len(CATEGORIES) + 2  # 6 illegal + reference + 2 legal controls
    t0 = {int(g): int(meta["t0"][(meta["game_id"] == g).nonzero()[0]])
          for g in meta["game_id"].unique().tolist()
          if int((meta["game_id"] == g).sum()) == n_types}
    ids = [g for g, t in t0.items() if positions(t, w - 1)[w - 1] < t]
    return ids[:n]


def stream(gids: list[int], device: str = "cpu", w: int = W):
    """Each game's nine branches in all nine representation spaces, one at a time.

    A generator, not a list: forty games in nine 8192-wide spaces is 2 GB on top
    of the 1 GB activation file, and the login node kills it.
    """
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    # mmap: the 1 GB activation file is indexed nine rows at a time, and
    # materialising it alongside eight SAE checkpoints trips the memory cap.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    encoders = [(label, None if e is None else sae_encoder(e, device))
                for label, e in spaces()]
    for gid in gids:
        idx = (meta["game_id"] == gid).nonzero().flatten()
        t0 = int(meta["t0"][idx[0]])
        pos = positions(t0, w - 1)
        Z = acts[idx][:, pos, :].float().numpy()
        g = {"game": gid, "t0": t0, "pos": pos,
             "n_context": len(pos) - len(positions(t0)),
             "category": meta["category"][idx].numpy(),
             "is_control": meta["is_control"][idx].numpy()}
        for label, enc in encoders:
            if enc is None:
                g[label] = Z
                continue
            with torch.no_grad():
                f = enc(torch.tensor(Z.reshape(-1, Z.shape[2]), device=device))
            g[label] = f.reshape(len(idx), len(pos), -1).cpu().numpy()
        yield g


def figure_game(gid: int = DEFAULT_GAME, w: int = W, device: str = "cpu") -> dict:
    """The one game the grids draw, cached so re-rendering skips the 1 GB load."""
    cache = study_dir("divergence") / "fork_curvature" / f"game_{gid}_w{w}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] if z[k].ndim else z[k].item() for k in z.files}
    g = next(stream([gid], device, w))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **g)
    return g


@dataclass(frozen=True)
class Plane:
    """A fitted 2-D plane in descriptor space, carrying the scaling it was fit
    with, so held-out rows (jackknife replicates) land in the same coordinates."""
    basis: np.ndarray    # [2, K]
    scale: np.ndarray    # [K]
    centre: np.ndarray   # [K], zeros when the fit was uncentred
    var: np.ndarray      # [2] variance ratios
    lead: tuple[str, str]

    def project(self, F: np.ndarray) -> np.ndarray:
        """[..., K] -> [..., 2]; a row with an undefined component stays NaN, so a
        curve starts where its window becomes defined, not at a fabricated origin."""
        X = F / self.scale - self.centre
        out = np.full(F.shape[:-1] + (2,), np.nan)
        ok = np.isfinite(X).all(-1)
        out[ok] = X[ok] @ self.basis.T
        return out


def fit_plane(F: np.ndarray, centred: bool, align: np.ndarray | None = None) -> Plane:
    """Top-2 PCs of the column-scaled descriptor.

    Scaled, not raw covariance: the components mix degrees with activation-scale
    lengths whose absolute variances differ ~450x across dictionaries, so a raw
    PCA reports units, not structure (analysis/straightness/feature_pca reasons
    the same way). Centring is the caller's -- for a difference-from-reference
    descriptor zero is a real point, as in the incumbent figure's uncentred PCA
    of D, while for an absolute descriptor there is no origin to preserve.
    """
    ok = np.isfinite(F).all(1)
    sd = np.nanstd(F, axis=0)
    scale = np.where(sd > 0, sd, 1)
    X = F / scale
    centre = X[ok].mean(0) if centred else np.zeros(F.shape[1])
    _, s, vt = np.linalg.svd(X[ok] - centre, full_matrices=False)
    v = vt[:2]
    sign = (np.sign((v @ align.T).diagonal()) if align is not None
            else np.sign(v[np.arange(2), np.abs(v).argmax(1)]))
    basis = v * sign[:, None]
    names = names_of(F.shape[1])
    return Plane(basis, scale, centre, (s ** 2 / (s ** 2).sum())[:2],
                 tuple(names[i] for i in np.abs(basis).argmax(1)))
