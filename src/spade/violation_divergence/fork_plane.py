"""The displacement plane a fork figure is drawn in, and the spaces and
positions every fork figure is drawn from.

`positions` and `spaces` are the study's sampling rule and its nine-cell grid,
shared with the R^21 plane of curvature_plane.py; the embeddings below are the
displacement plane specifically.

Move index is the third axis of every fork panel, so the plane must NOT be a
PCA of raw activations -- move index dominates activation geometry in every
geometry study in this repo, so PC1 would re-encode the vertical axis and the
figure would collapse to a slanted ribbon. Callers embed the displacement
D(t) = z_branch(t) - z_ref(t) instead: the reference is then exactly the
vertical line at the origin and the spray at t0 is the whole content.

Positions are sampled every STRIDE=2 plies at t0's own parity. At every ply the
path is a sawtooth that hides everything else: the board code is mover-relative,
so the frame inverts each ply and consecutive activations alternate. That ripple
is real -- it is the odd/even split in cos(D(r), D(0)) -- but it is a property of
the representation's frame, not of the trajectory's shape.
"""

import numpy as np
import torch

from spade.figures.paper_style import sae_label
from spade.l0_operating_point.checkpoints import Matched, matched
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.compute_divergence import sae_encoder

ISOMAP_NEIGHBORS, ISOMAP_PRE_DIMS = 12, 50  # spade/rules/embed.py uses 15 on ~4k points
STRIDE = 2               # one mover per curve; see the module docstring
WINDOW = (-8, 24)        # the study's analysis window, r = t - t0
FIRST_POS, LAST_POS = 1, 58  # position 0 is the attention sink
RAW_SPACE = "raw layer 5 (reference space)"
# picks.csv names a pick by its trainer class. The canonical roster's panel order and short
# names are kept, so an L0-matched figure's cells sit where a reader already found them and
# only the dictionary in each cell has changed.
L0_FAMILY = {"PAnnealTrainer": "PAnneal", "StandardTrainer": "Standard",
             "GatedSAETrainer": "GatedSAE", "TrainerTopK": "TopK",
             "GatedAnnealTrainer": "GatedAnneal", "BatchTopKTrainer": "BatchTopK",
             "CosineSpaDETrainer": "CosineSpaDE", "SpaDETrainer": "SpaDE"}


def positions(t0: int, back: int = 0) -> np.ndarray:
    """The study's window at t0's parity, clipped to the usable board and extended
    `back` sampled positions earlier -- a trailing descriptor window needs context
    the study window does not, and a short game simply yields fewer."""
    lo = max(FIRST_POS, t0 + WINDOW[0] - STRIDE * back)
    p = np.arange(lo, min(LAST_POS, t0 + WINDOW[1]) + 1)
    return p[(p - t0) % STRIDE == 0]


def spaces(l0_matched: bool = False) -> list[tuple[str, object]]:
    """(label, SAE entry or None) per grid cell -- raw layer 5 leads, then one
    seed-42 SAE family each (FIGURE_GUIDELINES 4.6).

    `l0_matched` swaps the eight SAE slots for `l0_operating_point`'s picks: one
    checkpoint per type, each the nearest-L0 member of its own grid to the shared
    target, so a difference between two cells is the architecture and not its
    sparsity. Raw layer 5 is not an SAE and stays as the reference cell.
    """
    if l0_matched:
        picks = {m.trainer_class: m for m in matched()}
        assert set(picks) == set(L0_FAMILY), f"picks.csv holds {sorted(picks)}"
        return [(RAW_SPACE, None)] + [
            (f"{family} ($L_0$ {picks[cls].l0:.0f})", picks[cls])
            for cls, family in L0_FAMILY.items()]
    return [(RAW_SPACE, None)] + [
        (sae_label(e.config.replace("Trainer-othello", "")), e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]


def encoder(entry, device: str):
    """`entry`'s encode function; None is raw layer 5.

    An `l0_operating_point` pick carries its own input convention -- Karvonen's five were
    trained on the raw residual stream, ours on scale*(x - mean) -- so the pick decides how
    it is fed, not the caller.
    """
    if entry is None:
        return None
    return entry.encode(device) if isinstance(entry, Matched) else sae_encoder(entry, device)


def pca2(X: np.ndarray, basis: np.ndarray | None = None):
    """(scores [n, 2], variance ratio [2], basis) — pass `basis` to reuse a fit.

    Uncentred, because zero displacement is a real point and not an arbitrary
    origin: mean-centring would move the reference off (0, 0) and separate it
    from the prefix line and the branch marker, which are drawn there.
    """
    if basis is None:
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        basis = Vt[:2]
        var = (S ** 2 / (S ** 2).sum())[:2]
    else:
        var = np.array([np.nan, np.nan])
    return X @ basis.T, var, basis


def isomap2(X: np.ndarray, fitted=None):
    """(scores [n, 2], nan var, fitted) — fit on nonzero rows, transform all.

    D is exactly zero for the reference at every position and for every branch
    before t0; those duplicate rows form a disconnected clique in the kNN graph.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    if fitted is None:
        keep = np.linalg.norm(X, axis=1) > 0
        pre = PCA(n_components=min(ISOMAP_PRE_DIMS, X.shape[1] - 1), random_state=0)
        iso = Isomap(n_neighbors=ISOMAP_NEIGHBORS, n_components=2, n_jobs=-1)
        iso.fit(pre.fit_transform(X[keep]))
        fitted = (pre, iso)
    pre, iso = fitted
    return iso.transform(pre.transform(X)), np.array([np.nan, np.nan]), fitted


EMBEDDINGS = {"pca": pca2, "isomap": isomap2}


def space_features(Z: np.ndarray, entry, device: str) -> np.ndarray:
    """[n, T, width] in one representation space; `entry` None is raw layer 5."""
    if entry is None:
        return Z
    n, T, D = Z.shape
    enc = encoder(entry, device)
    with torch.no_grad():
        f = enc(torch.tensor(Z.reshape(-1, D), device=device))
    return f.reshape(n, T, -1).cpu().numpy()
