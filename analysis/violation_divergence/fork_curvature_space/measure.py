"""The numbers that decide whether R^21 is a usable plane for the fork.

Five questions, one table each: how much of the descriptor is undefined in each
space; whether it belongs on the trajectory z_b or on the displacement
D = z_b - z_ref; whether the fork's two invariants (reference at the origin,
prefix collapsed) survive; how far the window smears the branch; and whether
illegal branches sit further from the two legal controls in a descriptor plane
than in the incumbent PCA of D.

Separation is ||mean_illegal - mean_control|| / pooled within-group RMS spread,
over 6 illegal and 2 legal-control branches of one game. Its raw size is not
comparable across planes -- the denominator differs -- so every plane is scored
against its OWN exact null: all 28 ways of calling two of the eight branches the
control group, of which the true labelling is one. Reported as the ratio to the
null mean (1.0 = the plane sees nothing) and as the true labelling's percentile
among the 28. The interval is a percentile bootstrap over games.

Usage: python analysis/violation_divergence/fork_curvature_space/measure.py
       [--n_games 40] [--w 8]
"""

import argparse
from itertools import combinations

import numpy as np
import torch

import descriptor as dsc
from spade.straightness.geometry import temporal_geometry

R_POST = 0     # separation is measured at offsets r >= this, i.e. after the branch
N_BOOT = 2000
SEED = 0
PLANES = ["PCA(D) 2-D", "full D", "Δφ PCA 2-D", "Δφ fixed", "full Δφ"]
WHOLE_PLANES = ["φ PCA 2-D", "φ fixed", "full φ"]
# The sharper question the fixed plane invites: do the two legal branches sit
# below all six illegal ones on a named axis? Exact chance is 1/C(8,2) = 0.036.
FLAGS = ["controls lowest on mean d", "controls lowest on mean θ"]
CHANCE = 1 / 28


def separation(G: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """||mean_a - mean_b|| / pooled within-group RMS spread, from the points' Gram
    matrix -- 28 relabellings of an 8192-d field then cost one 8x8 matrix, not 28
    passes over the field."""
    gaa, gbb, gab = G[np.ix_(a, a)], G[np.ix_(b, b)], G[np.ix_(a, b)]
    between = gaa.sum() / len(a) ** 2 - 2 * gab.sum() / (len(a) * len(b)) \
        + gbb.sum() / len(b) ** 2
    within = (np.trace(gaa) - gaa.sum() / len(a) + np.trace(gbb)
              - gbb.sum() / len(b)) / (len(a) + len(b) - 2)
    return float(np.sqrt(between / within))


def sep_and_rank(P: np.ndarray, illegal: np.ndarray, control: np.ndarray):
    """(separation, ratio to the null mean, percentile among the 28 splits).

    Points are centred before the Gram matrix: the fixed plane mixes degrees with
    step lengths ~10^3 smaller, and the uncentred cancellation drives the within
    term negative.
    """
    rows = np.concatenate([illegal, control])
    G = (P - P[rows].mean(0)) @ (P - P[rows].mean(0)).T
    true = separation(G, illegal, control)
    null = np.array([separation(G, np.setdiff1d(rows, c), np.array(c))
                     for c in combinations(rows, 2)])
    return true, true / null.mean(), float(np.mean(null <= true))


def pca_of_displacement(D: np.ndarray) -> np.ndarray:
    """The incumbent plane: uncentred top-2 PCA of D, pooled over the fork."""
    _, _, vt = np.linalg.svd(D.reshape(-1, D.shape[2]), full_matrices=False)
    return D @ vt[:2].T


def boot(x: np.ndarray) -> str:
    """mean [95% percentile bootstrap over games]; x is [n_games]."""
    rng = np.random.default_rng(SEED)
    m = x[rng.integers(len(x), size=(N_BOOT, len(x)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return f"{x.mean():5.2f} [{lo:.2f},{hi:.2f}]"


def scaled(F: np.ndarray) -> np.ndarray:
    """Column-scaled as `embed` scales, so a norm weights degrees and lengths alike."""
    sd = np.nanstd(F.reshape(-1, F.shape[-1]), axis=0)
    return F / np.where(sd > 0, sd, 1)


def rise(dphi: np.ndarray, r: np.ndarray, keep: np.ndarray) -> float:
    """Median offset at which ||Δφ|| first passes half its per-branch maximum."""
    m = np.linalg.norm(scaled(dphi), axis=2)[keep]
    hit = m >= 0.5 * np.nanmax(m, axis=1, keepdims=True)
    return float(np.median([r[np.flatnonzero(h)[0]] for h in hit]))


def branch_rows(g: dict):
    cat, ctrl = g["category"], g["is_control"]
    return (int(np.flatnonzero((cat == 0) & (ctrl == 0))[0]),
            np.flatnonzero(cat > 0), np.flatnonzero(ctrl > 0))


def per_game(g: dict, label: str, w: int):
    """One game's separation in every candidate plane, windowed and whole."""
    ref, ill, con = branch_rows(g)
    c = int(g["n_context"])
    r = g["pos"][c:] - g["t0"]
    post = np.flatnonzero(r >= R_POST)

    Z, k = g[label], c + int(post[0])
    D = (Z - Z[ref])[:, c:]
    phi = dsc.windowed(Z, w)[:, c:]
    dphi = phi - phi[ref]
    flat = dphi.reshape(-1, dphi.shape[2])
    sc = dsc.fit_plane(flat, centred=False).project(flat)
    # Descriptor fields are column-scaled, as fit_plane scales them: the 21 mix
    # degrees with activation-scale lengths, and a Euclidean statistic on the raw
    # columns is a statistic about theta alone.
    zphi = dphi / np.where(np.nanstd(flat, 0) > 0, np.nanstd(flat, 0), 1)
    ax = [dsc.names_of(dphi.shape[2]).index(a) for a in dsc.FIXED_AXES]
    fields = [pca_of_displacement(D), D, sc.reshape(dphi.shape[:2] + (2,)),
              zphi[:, :, ax], zphi]
    windowed = {name: np.mean([sep_and_rank(P[:, t], ill, con) for t in post], axis=0)
                for name, P in zip(PLANES, fields)}

    whole = dsc.descriptor(torch.from_numpy(Z[:, k:]))
    wax = [dsc.names_of(whole.shape[1]).index(a) for a in dsc.FIXED_AXES]
    wsc = dsc.fit_plane(whole, centred=True).project(whole)
    zwhole = whole / np.where(whole.std(0) > 0, whole.std(0), 1)
    flags = {f: float(whole[con, i].max() < whole[ill, i].min())
             for f, i in zip(FLAGS, wax)}
    return (windowed, {name: np.array(sep_and_rank(P, ill, con)) for name, P in
                       zip(WHOLE_PLANES, [wsc, zwhole[:, wax], zwhole])}, flags)


def invariants(g: dict, w: int) -> None:
    ref, _, _ = branch_rows(g)
    c = int(g["n_context"])
    r = g["pos"][c:] - g["t0"]
    pre, post = np.flatnonzero(r < 0), np.flatnonzero(r >= 0)
    keep = np.arange(len(g["category"])) != ref
    names = dsc.names_of(dsc.descriptor(torch.from_numpy(g[dsc.RAW_SPACE][:, :w])).shape[1])

    defined = np.flatnonzero(np.isfinite(dsc.windowed(g[dsc.RAW_SPACE], w)[:, c:]
                                         ).all(-1).all(0))
    print(f"the figure game: {g['game']}, t0={g['t0']}, {len(r)} plotted positions "
          f"r={r[0]}..{r[-1]} at stride {dsc.STRIDE}, 9 branches; {c} sampled "
          f"positions of prefix context reach back to {g['pos'][0]}")
    print(f"window w={w} sampled positions = {dsc.STRIDE * (w - 1)} plies -> "
          f"{w - 1} step lengths, {w - 2} angles -> {len(names)}-D descriptor; "
          f"{len(defined)}/{len(r)} curve points defined, r={r[defined[0]]}..{r[-1]}, "
          f"of which {int((r[defined] < 0).sum())} are branch-free\n")

    print("1. undefined components (fraction of entries), per space")
    print(f"{'space':34} {'θ NaN z_b':>10} {'θ NaN D':>8} {'θ NaN D, r>=0':>14} "
          f"{'φ(D) NaN':>9} {'φ(D) NaN, ref':>14}")
    for label, _ in dsc.spaces():
        Z = g[label]
        D = Z - Z[ref]
        _, th_z = temporal_geometry(torch.from_numpy(Z))
        _, th_d = temporal_geometry(torch.from_numpy(D))
        phid = dsc.windowed(D, w)[:, c:]
        print(f"{label:34} {th_z.isnan().float().mean():10.3f} "
              f"{th_d.isnan().float().mean():8.3f} "
              f"{th_d[:, c:].isnan().float().mean():14.3f} "
              f"{np.isnan(phid).mean():9.3f} {np.isnan(phid[ref]).mean():14.3f}")

    print("\n2. the fork's invariants and the window's smear, under "
          "Δφ(t) = φ(z_b(t)) − φ(z_ref(t))")
    print(f"{'space':34} {'max|Δφ| at r<0':>15} {'max|Δφ_ref|':>12} "
          f"{'PC1/PC2 var':>12} {'PC1 leads on':>14} {'r at half-max':>14}")
    for label, _ in dsc.spaces():
        phi = dsc.windowed(g[label], w)[:, c:]
        dphi = phi - phi[ref]
        pl = dsc.fit_plane(dphi.reshape(-1, dphi.shape[2]), centred=False)
        print(f"{label:34} {np.nanmax(np.abs(dphi[:, pre])):15.2e} "
              f"{np.nanmax(np.abs(dphi[ref, post])):12.2e} "
              f"{pl.var[0]:5.0%}/{pl.var[1]:<6.0%} {pl.lead[0]:>14} "
              f"{rise(dphi, r, keep):14.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=40)
    parser.add_argument("--w", type=int, default=dsc.W)
    parser.add_argument("--game", type=int, default=dsc.DEFAULT_GAME)
    args = parser.parse_args()

    win = {label: {k: [] for k in PLANES} for label, _ in dsc.spaces()}
    who = {label: {k: [] for k in WHOLE_PLANES} for label, _ in dsc.spaces()}
    flg = {label: {k: [] for k in FLAGS} for label, _ in dsc.spaces()}
    n = 0
    invariants(dsc.figure_game(args.game, args.w), args.w)
    for g in dsc.stream(dsc.usable(args.n_games, args.w), w=args.w):
        for label, _ in dsc.spaces():
            a, b, c = per_game(g, label, args.w)
            for k in PLANES:
                win[label][k].append(a[k])
            for k in WHOLE_PLANES:
                who[label][k].append(b[k])
            for k in FLAGS:
                flg[label][k].append(c[k])
        n += 1
    win = {l: {k: np.stack(v) for k, v in d.items()} for l, d in win.items()}
    who = {l: {k: np.stack(v) for k, v in d.items()} for l, d in who.items()}
    flg = {l: {k: np.array(v) for k, v in d.items()} for l, d in flg.items()}
    print(f"\n3. windowed variant: separation of the 6 illegal branches from the 2 "
          f"legal controls\n   as a RATIO TO ITS OWN PERMUTATION-NULL MEAN "
          f"(1.0 = the plane sees nothing), mean over r=0..24,\n   then mean "
          f"[95% bootstrap CI] over n={n} games")
    print(f"{'space':34} " + " ".join(f"{p:>19}" for p in PLANES))
    for label, _ in dsc.spaces():
        print(f"{label:34} " + " ".join(f"{boot(win[label][p][:, 1]):>19}"
                                        for p in PLANES))
    paired = np.concatenate([win[l]["Δφ fixed"][:, 1] - win[l]["PCA(D) 2-D"][:, 1]
                             for l, _ in dsc.spaces()])
    print(f"{'paired Δ, Δφ fixed − PCA(D)':34} {boot(paired)}   (n={n} games x 9 spaces)")

    print(f"\n4. whole variant: one 21-D point per branch over r=0..24, "
          f"same null ratio, n={n} games")
    print(f"{'space':34} " + " ".join(f"{p:>19}" for p in WHOLE_PLANES))
    for label, _ in dsc.spaces():
        print(f"{label:34} " + " ".join(f"{boot(who[label][p][:, 1]):>19}"
                                        for p in WHOLE_PLANES))

    print(f"\n5. every plane against its own null, pooled over 9 spaces x n={n} "
          f"games.\n   ratio 1.0 and percentile 0.5 both mean the plane sees nothing.")
    print(f"{'plane':22} {'raw separation':>19} {'ratio to null':>19} "
          f"{'null percentile':>19}")
    for table, planes in ((win, PLANES), (who, WHOLE_PLANES)):
        for p in planes:
            x = np.concatenate([table[l][p] for l, _ in dsc.spaces()])
            print(f"{p:22} {boot(x[:, 0]):>19} {boot(x[:, 1]):>19} "
                  f"{boot(x[:, 2]):>19}")

    print(f"\n6. whole variant, fixed axes: fraction of games in which BOTH legal "
          f"controls fall below\n   all six illegal branches on a named R^21 axis. "
          f"Exact chance {CHANCE:.3f}; n={n} games.")
    print(f"{'space':34} " + " ".join(f"{f:>28}" for f in FLAGS))
    for label, _ in dsc.spaces():
        print(f"{label:34} " + " ".join(f"{boot(flg[label][f]):>28}" for f in FLAGS))


if __name__ == "__main__":
    main()
