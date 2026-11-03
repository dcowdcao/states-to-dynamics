"""Can the fork figure's plane be a PCA of R^21 instead of raw displacement?

R^21 describes a whole trajectory; the fork figure needs a curve over move
index. This script does the accounting that decides whether the two are
compatible on this window, and prints every number quoted in REPORT.md:

  1. what `restrav_features` returns for a short window (the truncation trap)
  2. how many curve points survive, and how much of t0 is smeared, per window
  3. what actually moves along an expanding window anchored at t0
  4. illegal-vs-control separation in each candidate plane, against the raw
     displacement PCA the figure uses today, under one shared effect size

Usage: python analysis/violation_divergence/fork_expanding_curvature/feasibility.py
"""

from itertools import combinations

import numpy as np
import torch

from spade.straightness.geometry import (
    RESTRAV_FEATURE_NAMES, restrav_features, temporal_geometry)

from cache_features import load
from window import (
    MIN_WINDOW, R21_DIM, STRIDE, branch_groups, expanding, rescale, sliding, travel)

RAW_BLOCK = slice(0, 13)     # d1..d7, theta1..theta6
MOMENT_BLOCK = slice(13, 21)  # mean/min/max/var of d and of theta


def pca2(X: np.ndarray):
    """Uncentred, as in the fork figure: zero displacement is a real point."""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T, (S ** 2 / (S ** 2).sum())[:2], Vt[:2]


def separation(P: np.ndarray, illegal: np.ndarray) -> float:
    """Centroid gap in pooled within-group per-axis SDs, on a 2-D plane."""
    a, b = P[illegal], P[~illegal]
    within = np.concatenate([((a - a.mean(0)) ** 2).sum(1),
                             ((b - b.mean(0)) ** 2).sum(1)])
    return np.linalg.norm(a.mean(0) - b.mean(0)) / np.sqrt(within.mean() / P.shape[1])


def width_of(F: np.ndarray, w: int) -> int:
    """What restrav_features actually returns for a window of w positions, 0 if it raises."""
    d, theta = temporal_geometry(torch.from_numpy(F[:, :w]).double())
    if theta.shape[1] == 0:
        return 0
    return restrav_features(d, theta).shape[1]


def truncation_trap(F: np.ndarray) -> None:
    print("\n1. what restrav_features returns, by window length W (sampled positions)\n")
    print("   W  plies  n_d  n_theta  returned width  raw columns present")
    for w in range(2, 12):
        d, theta = temporal_geometry(torch.from_numpy(F[:, :w]).double())
        head = f"  {w:2d}  {(w - 1) * STRIDE:5d}  {d.shape[1]:3d}  {theta.shape[1]:7d}"
        if theta.shape[1] == 0:
            print(f"{head}  IndexError from moment4: amin over an empty theta")
            continue
        names = (list(RESTRAV_FEATURE_NAMES[:7])[:d.shape[1]]
                 + list(RESTRAV_FEATURE_NAMES[7:13])[:theta.shape[1]])
        print(f"{head}  {width_of(F, w):14d}  {'+'.join(names)}+8 moments")
    print(f"\n   R^{R21_DIM} needs W >= {MIN_WINDOW} ({(MIN_WINDOW - 1) * STRIDE} plies at "
          f"stride {STRIDE}). Shorter windows return a shorter vector with no error,\n"
          "   and the column meanings shift: at W=5 column 4 is theta1, not d5.")


def window_accounting(F: np.ndarray, pos: np.ndarray, t0: int, ref: int) -> None:
    """Which sliding windows carry a branch, from the activations themselves.

    A window is contaminated iff it contains a sampled position whose activation
    differs across branches; the descriptor inherits that exactly, which the
    W = MIN_WINDOW check below verifies.
    """
    k0, n_pos = int(np.flatnonzero(pos == t0)[0]), len(pos)
    differs = (np.linalg.norm(F - F[ref], axis=-1).max(0) > 0)
    assert not differs[:k0].any() and differs[k0:].all(), "branches differ off t0"
    print(f"\n2. curve points and t0 smearing on this window "
          f"({n_pos} sampled positions, r = {pos[0] - t0}..{pos[-1] - t0}, t0 at index {k0})\n")
    for placement, off in (("trailing (causal)", lambda w: w - 1),
                           ("centered", lambda w: (w - 1) // 2)):
        print(f"   {placement}")
        print("     W  R^k  curve pts  first drawn r  pts free of the branch"
              "  first contaminated r  smearing (plies before t0)")
        for w in range(3, n_pos + 1):
            starts = np.arange(n_pos - w + 1)
            anchors = pos[starts + off(w)] - t0
            dirty = np.array([differs[s:s + w].any() for s in starts])
            first = anchors[dirty][0]
            print(f"    {w:2d} {width_of(F, w):4d}  {len(starts):9d}  {anchors[0]:13d}"
                  f"  {int((~dirty).sum()):22d}  {first:20d}  {max(0, -first):25d}")
        print()
    D, ks = sliding(F, MIN_WINDOW, "trailing")
    moved = np.linalg.norm(D - D[ref], axis=-1).max(0) > 0
    dirty = np.array([differs[s:s + MIN_WINDOW].any()
        for s in np.arange(n_pos - MIN_WINDOW + 1)])
    assert (moved == dirty).all(), "descriptor contamination != window contamination"
    print(f"   (verified at W={MIN_WINDOW}: the descriptor differs from the reference's on "
          "exactly the contaminated windows.)")
    print(f"   A curve point free of the branch needs the whole window before t0, i.e. "
          f"W <= {k0} for trailing\n   placement. At the W = {MIN_WINDOW} that R^21 requires "
          "there is none: every drawn point already\n   contains post-branch plies, so the "
          "figure loses both its zero-by-construction prefix and\n   its spray at t0.")


def expanding_accounting(g: dict, ref: int) -> None:
    pos, t0, F = g["pos"], g["t0"], g["F"][0]
    k0 = int(np.flatnonzero(pos == t0)[0])
    D, ks = expanding(F, k0)
    D = D - D[ref]
    print(f"\n3. expanding window anchored at t0: {len(ks)} curve points, "
          f"r = {pos[ks[0]] - t0}..{pos[ks[-1]] - t0}\n")
    print("   k  W  r   moves since previous point (||dD21||, raw block / moment block)")
    for i, k in enumerate(ks):
        w = k - k0 + 1
        if i == 0:
            print(f"  {k:2d} {w:2d} {pos[k] - t0:2d}   (first drawn point)")
            continue
        step = D[:, i] - D[:, i - 1]
        raw = np.linalg.norm(step[:, RAW_BLOCK], axis=1).max()
        mom = np.linalg.norm(step[:, MOMENT_BLOCK], axis=1).max()
        print(f"  {k:2d} {w:2d} {pos[k] - t0:2d}   {raw:.3e} / {mom:.3e}")
    frozen = np.abs(D[:, :, RAW_BLOCK] - D[:, :1, RAW_BLOCK]).max()
    print(f"\n   max change in d1..d7, theta1..theta6 along the curve: {frozen:.3e}"
          f"  (13/{R21_DIM} coordinates are frozen once W >= {MIN_WINDOW};"
          " the curve moves only in the 8 moments)")
    print("\n   path length after the first point / norm there, observed vs a time-shuffled"
          f"\n   null (the same {len(ks)} descriptors over a random permutation of the "
          "post-t0 positions,\n   so window growth is the only thing left):\n")
    print(f"   {'space':34s} {'observed':>10s} {'null':>10s}")
    groups = branch_groups(g["category"], g["is_control"])
    for name, Fs in zip(g["names"], g["F"]):
        obs, null = travel(Fs, groups, k0, ref)
        print(f"   {name:34s} {np.median(obs):10.2f} {np.median(null):10.2f}")


def permutation_p(P: np.ndarray, n_illegal: int) -> float:
    """One point per branch: enumerate every 6/2 relabelling of the 8 branches.

    The plane is fit without labels, so only the labels move. The floor is
    1/C(8,2) = 0.036 -- with two controls this cannot get below 'suggestive'.
    """
    obs = separation(P, np.arange(len(P)) < n_illegal)
    perms = [separation(P, ~np.isin(np.arange(len(P)), c))
        for c in combinations(range(len(P)), len(P) - n_illegal)]
    return float(np.mean(np.array(perms) >= obs))


def planes(g: dict, ref: int, illegal: np.ndarray) -> None:
    k0 = int(np.flatnonzero(g["pos"] == g["t0"])[0])
    keep = np.arange(len(g["category"])) != ref
    order = np.argsort(~illegal[keep])  # illegal first, so permutation_p can index
    print("\n4. illegal-vs-control separation: centroid gap in pooled within-group per-axis"
          "\n   SDs on each candidate plane. One game, 6 illegal and 2 control branches."
          "\n   (PC1+PC2 variance in parentheses; p = label-permutation null over all 28"
          "\n   relabellings, floor 0.036)\n")
    print(f"   {'space':34s} {'raw disp/pos':>14s} {'raw disp/branch':>21s} "
          f"{'R21 expanding':>15s} {'R21 one point':>21s}")
    for name, F in zip(g["names"], g["F"]):
        per_pos = (F - F[ref])[:, k0:]
        P, v, _ = pca2(per_pos.reshape(-1, F.shape[-1]))
        s_pos = separation(P.reshape(len(F), -1, 2)[keep].reshape(-1, 2),
                           np.repeat(illegal[keep], per_pos.shape[1]))

        P, v_rb, _ = pca2(per_pos.mean(1))
        s_rb, p_rb = separation(P[keep], illegal[keep]), permutation_p(P[keep][order], 6)

        D, _ = expanding(F, k0)
        D = rescale(D - D[ref])
        P, v_exp, _ = pca2(D.reshape(-1, R21_DIM))
        s_exp = separation(P.reshape(len(F), -1, 2)[keep].reshape(-1, 2),
                           np.repeat(illegal[keep], D.shape[1]))

        P, v_one, _ = pca2(rescale(D[:, -1]))
        s_one, p_one = separation(P[keep], illegal[keep]), permutation_p(P[keep][order], 6)
        print(f"   {name:34s} {s_pos:5.2f} ({v.sum():.2f}) {s_rb:5.2f} ({v_rb.sum():.2f}) "
              f"p={p_rb:.3f} {s_exp:5.2f} ({v_exp.sum():.2f}) {s_one:5.2f} "
              f"({v_one.sum():.2f}) p={p_one:.3f}")

    D, _ = expanding(g["F"][0], k0)
    _, _, basis = pca2((D - D[ref]).reshape(-1, R21_DIM))
    d_ish = np.array([i for i, n in enumerate(RESTRAV_FEATURE_NAMES) if "d" in n])
    share = (basis[0][d_ish] ** 2).sum()
    top = np.argsort(-np.abs(basis[0]))[:3]
    print(f"\n   Unscaled PCA of R^21 on {g['names'][0]} puts {share:.0%} of PC1's energy on "
          f"the step-length\n   coordinates, loading hardest on "
          f"{', '.join(RESTRAV_FEATURE_NAMES[i] for i in top)}: d is in activation-norm "
          "units and theta in\n   degrees, so an unscaled plane reports whichever block is "
          "larger. Every R21 plane above is\n   fit after rescaling each coordinate to unit "
          "RMS (diagonal, so zero displacement stays 0).")


def main():
    g = load()
    ref = int(np.flatnonzero((g["category"] == 0) & (g["is_control"] == 0))[0])
    illegal = g["category"] != 0
    print(f"game {g['game']}, t0 = {g['t0']}, positions {g['pos'][0]}-{g['pos'][-1]} "
          f"(stride {STRIDE}), {len(g['category'])} branches, {len(g['names'])} spaces")
    truncation_trap(g["F"][0])
    window_accounting(g["F"][0], g["pos"], g["t0"], ref)
    expanding_accounting(g, ref)
    planes(g, ref, illegal)


if __name__ == "__main__":
    main()
