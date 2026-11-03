"""Step 2: does the supervised curvature axis mean anything off its training set?

Four questions, in the order that decides whether the figure may be drawn:

1. Does the probe generalize at all? Held-out AUROC on straightness pairs the fit
   never saw (odd source-game index), legal vs violation.
2. Does it transport across move index? The same fixed direction, applied to
   held-out pairs at windows ending at t0 + r. A direction that is chance at
   r > 0 cannot carry a curve over move index, whatever the figure looks like.
3. Does it fire on legality or on novelty? The divergence branches supply the
   test the straightness population cannot: two LEGAL control moves the probe
   never saw. If the axis separates illegal from those controls, it read
   legality; if it separates every branch from the reference equally, it read
   "a different move was played".
4. Is the DIRECTION doing the work, or just the magnitude? The same control test
   scored on PC1, on the descriptor's own named axes, on 100 random directions in
   R^21, and on the plain norm of the descriptor difference. A fitted axis has to
   beat a random line through the same space before "supervised" means anything.

5. Is any of it the SUFFIX rather than the move? --variant rollout re-runs all of
   the above on the copy of the study where every branch, reference included,
   samples its own continuation instead of replaying the source game's. The probe
   direction does not move -- it is fit on straightness pairs, which have no
   variant -- so the two runs differ only in the branches the axis is scored on.
   The r=0 window is a trailing 8 plies ending at t0 and the rollout copies
   tokens[:t0+1] verbatim, so r=0 is the same measurement in both variants by
   construction; `variant_agreement` measures the residual rather than trusting
   it, and what the rollout can actually change is r > 0.

6. Is the ordering an ARCHITECTURE ordering, or a sparsity one? The nine canonical
   spaces sit wherever their own defaults put them on the sparsity axis, and their L0s
   span 32.5 to 1,808 -- with the four that win this table the four sparsest.
   --l0-matched re-runs everything above over the eight `l0_operating_point` picks, one
   checkpoint per architecture at a shared L0. Nothing else moves: the axis is still fit
   per space on straightness pairs, in the pick's own space, off the `l0_*_raw.npz`
   geometry `analysis/straightness/l0_matched_profile.py` writes. It writes
   `evaluation_l0_matched.csv` and leaves the canonical table alone.

Usage: python analysis/violation_divergence/curvature_discriminant/evaluate.py
           [--variant {spliced,rollout}] [--l0-matched]
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from spade.common.study import variant_dir

from spade.straightness.geometry import RESTRAV_FEATURE_NAMES
from spade.violation_divergence.fork_plane import tag

from curvature_axis import (
    WINDOW_PLIES, CurvatureAxis, fit_axis, residual_plane, spaces, transported_auroc,
    window_features)
from branch_geometry import VARIANTS, PLOT_GAME, out_dir

OFFSETS = np.arange(-8, 25)   # the divergence study's own window, r = t - t0
FIT_OFFSET = 0                # the one window whose pair differs in a single token
PLOT_STRIDE = 2               # one mover per curve, as in the parent figure
N_BOOT, BOOT_SEED = 1000, 0
N_PERM = 1000                 # within-game label permutations for the aggregate null
N_DIRS = 100                  # random directions in R^21, the null for a FITTED axis


def meta(variant: str = "") -> dict:
    """Row-level meta: one row per branch (spliced) or per rollout of a branch."""
    m = dict(np.load(out_dir(variant) / "meta.npz"))
    assert PLOT_GAME in m["game_id"], f"plotted game {PLOT_GAME} not extracted"
    return m


def rollouts(m: dict) -> int:
    """Rows per branch: K for the rollout variant, 1 for spliced."""
    return int(m["rollout"].max()) + 1 if "rollout" in m else 1


def fold_rollouts(A: np.ndarray, k: int) -> np.ndarray:
    """Mean over each branch's K rollouts; rows are `source_row * K + rollout`.

    Applied to the DESCRIPTORS, never to d/theta or to a trajectory: phi is
    nonlinear, so a mean taken before it is the curvature of a path no rollout
    took. Everything after it -- the axis projection, the plane -- is linear and
    gives the same answer on either side, so this is the only place the choice
    matters and it is made once, here.
    """
    return A if k == 1 else A.reshape(len(A) // k, k, *A.shape[1:]).mean(1)


def branch_meta(m: dict) -> dict:
    """Row meta collapsed to one row per branch, to index the folded descriptors.

    Every column is constant across a branch's K rollouts except `row`, the
    branches.pt index, which runs source_row * K + rollout. It keeps the first
    rollout's value, which is the row `n_flipped` reads and which recovers the
    spliced row as row // K.
    """
    k = rollouts(m)
    if k == 1:
        return m
    out = {key: v for key, v in m.items() if key != "rollout"}
    assert np.array_equal(m["row"], np.repeat(m["row"][::k], k)
                          + np.tile(np.arange(k), len(m["t0"]) // k)), \
        "rows are not K contiguous rollouts per branch"
    for key, v in list(out.items()):
        if v.ndim and len(v) == len(m["t0"]):
            assert key == "row" or np.array_equal(v, np.repeat(v[::k], k)), \
                f"{key} differs across a branch's rollouts"
            out[key] = v[::k]
    return out


def descriptors(space: str, m: dict, variant: str = "") -> np.ndarray:
    """[n_rows, n_offsets, 21] windowed ReStraV descriptors, one row per rollout."""
    z = np.load(out_dir(variant) / f"{space}.npz")
    rows = np.arange(len(m["t0"]))
    return np.stack([window_features(z["d"], z["theta"], rows, m["t0"] + r)
        for r in OFFSETS], axis=1)


def reference_of(m: dict) -> np.ndarray:
    """Row index of each row's own game reference branch."""
    is_ref = (m["category"] == 0) & (m["is_control"] == 0)
    ref = {g: i for i, g in zip(np.flatnonzero(is_ref), m["game_id"][is_ref])}
    return np.array([ref[g] for g in m["game_id"]])


def deltas(F: np.ndarray, axis: CurvatureAxis, m: dict) -> np.ndarray:
    """[n_branches, n_offsets, 21] descriptor difference from the game's reference.

    `m` is the ROW meta; the K rollouts are averaged here, between phi and the
    difference, so the result is one row per branch and every caller downstream
    works in branch_meta(m)'s indexing.
    """
    F = fold_rollouts(np.where(np.isnan(F), axis.fill, F), rollouts(m))
    mb = branch_meta(m)
    return F - F[reference_of(mb)]


def boot_ci(y: np.ndarray, s: np.ndarray, game: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap over games -- the unit of variation is the game."""
    rng = np.random.default_rng(BOOT_SEED)
    games, inv = np.unique(game, return_inverse=True)
    by_game = [np.flatnonzero(inv == k) for k in range(len(games))]
    aucs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([by_game[k] for k in rng.integers(0, len(games), len(games))])
        if 0 < y[pick].sum() < len(pick):
            aucs.append(roc_auc_score(y[pick], s[pick]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def control_null(score_r0: np.ndarray, m: dict) -> tuple[float, float, float]:
    """Aggregate AUROC under within-game permutation of the illegal/control labels.

    Shuffling inside a game preserves every per-game nuisance -- t0, board, game
    length -- so what is left is the label. Returns (mean, 2.5%, 97.5%).
    """
    illegal, control = m["category"] != 0, m["is_control"] > 0
    keep = illegal | control
    y, g = illegal[keep].astype(float), m["game_id"][keep]
    s = score_r0[keep]
    rng = np.random.default_rng(BOOT_SEED)
    _, inv = np.unique(g, return_inverse=True)
    blocks = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    aucs = []
    for _ in range(N_PERM):
        yp = y.copy()
        for b in blocks:
            yp[b] = rng.permutation(y[b])
        aucs.append(roc_auc_score(yp, s))
    return float(np.mean(aucs)), *np.percentile(aucs, [2.5, 97.5])


def control_test(score: np.ndarray, m: dict) -> np.ndarray:
    """AUROC(illegal vs the two legal controls) at each offset. >0.5 = illegal higher."""
    illegal, control = m["category"] != 0, m["is_control"] > 0
    keep = illegal | control
    y = illegal[keep].astype(float)
    return np.array([roc_auc_score(y, score[keep, j]) for j in range(len(OFFSETS))])


def pc1(delta: np.ndarray, axis: CurvatureAxis) -> np.ndarray:
    """Unsupervised control: leading uncentred PC of the same z-scored differences.

    A component's sign is arbitrary, so it is pinned the way
    analysis/straightness/feature_pca does it -- largest loading positive -- and
    not by which sign scores better, which would smuggle the label back in.
    """
    z = (delta / axis.sd).reshape(-1, 21)
    _, _, vt = np.linalg.svd(z[np.linalg.norm(z, axis=1) > 0], full_matrices=False)
    v = vt[0] * np.sign(vt[0][np.abs(vt[0]).argmax()])
    return (delta / axis.sd) @ v


RULE_OF_CATEGORY = np.array([0, 1, 1, 2, 2, 2, 3])  # legal, R1, R1, R2, R2, R2, R3
RULE_NAMES = ["legal controls (null)", "R1 placement", "R2 bracketing", "R3 turn"]
CONTROL_BINS = ["legal control, flips 1 (null)", "legal control, flips $\\geq$4 (null)"]


def landing_groups(m: dict, split_controls: bool = False,
                   variant: str = "") -> dict[str, np.ndarray]:
    """The branch groups whose landing point is the honesty evidence.

    R1 is the doubly-held-out group: the straightness study injects only
    empty-square violations, so no occupied-square move is in the probe's
    training set at all. `split_controls` bins the legal controls by how many
    discs they flip, which is the confound the axis turns out to be reading.
    """
    control = m["is_control"] > 0
    if split_controls:
        flips = n_flipped(m, variant)
        out = {CONTROL_BINS[0]: control & (flips == 1),
               CONTROL_BINS[1]: control & (flips >= 4)}
    else:
        out = {RULE_NAMES[0]: control}
    for r in (1, 2, 3):
        out[RULE_NAMES[r]] = (m["category"] != 0) & (RULE_OF_CATEGORY[m["category"]] == r)
    return out


def _gap(sub: np.ndarray, group_a: np.ndarray) -> float:
    """Centroid distance between two branch groups, in pooled within-group SDs."""
    a = sub[group_a].reshape(-1, sub.shape[-1])
    b = sub[~group_a].reshape(-1, sub.shape[-1])
    return float(np.linalg.norm(a.mean(0) - b.mean(0)) /
                 np.sqrt(0.5 * (a.var(0) + b.var(0)).sum()))


def one_game_plane(coords: np.ndarray, m: dict, game: int):
    """(sub [8, n_pos, 2], is_illegal [8]) for the plotted game, reference dropped.

    Every 2nd ply at r >= 0, both plane coordinates -- the same slice the existing
    PCA-of-activations plane is scored on, so the two numbers compare.
    """
    idx = np.flatnonzero(m["game_id"] == game)
    branch = (m["category"][idx] != 0) | (m["is_control"][idx] > 0)
    keep = (OFFSETS >= 0) & (OFFSETS % PLOT_STRIDE == 0)
    return coords[idx][branch][:, keep], m["category"][idx][branch] != 0


def permutation_null(coords: np.ndarray, m: dict, game: int) -> dict:
    """The separation, divided by its own exact label-permutation null.

    A gap in SDs is inflated by its own denominator, so the number that means
    something is the ratio to what the same statistic gives on the same plane
    with the branch labels shuffled. One game has 6 illegal and 2 control
    branches, so the assignment has C(8,2) = 28 possibilities and the null is
    enumerated exactly -- the floor on the exact p is 1/28 = 0.036.

    Nothing is refit under the permutation, and nothing needs to be: axis 1 is
    fit on other games' labels and axis 2 on the whole evaluation population, so
    the plane does not move when this game's labels do. That is not true of a
    plane fit on this game -- there the null must refit, or it measures nothing.
    """
    from itertools import combinations
    sub, illegal = one_game_plane(coords, m, game)
    n, k = len(illegal), int((~illegal).sum())
    null = np.array([_gap(sub, ~np.isin(np.arange(n), c))
        for c in combinations(range(n), k)])
    obs = _gap(sub, illegal)
    return {"gap": obs, "null_mean": float(null.mean()),
            "ratio": obs / float(null.mean()),
            "p_exact": float((null >= obs).mean())}


def random_direction_null(delta: np.ndarray, axis: CurvatureAxis,
                          m: dict) -> np.ndarray:
    """[n_offsets] 95th percentile of the control test over random directions in R^21.

    The permutation null answers "is this grouping special"; this one answers the
    question a *fitted* axis actually raises, "is this DIRECTION special". The
    descriptor difference is simply larger for a branch that perturbed the
    trajectory more, so a random line through R^21 inherits much of the
    separation while knowing nothing. Scored as |AUROC - 0.5| + 0.5, since a
    random direction's sign is arbitrary.
    """
    illegal, control = m["category"] != 0, m["is_control"] > 0
    keep = illegal | control
    y = illegal[keep].astype(float)
    z = (delta / axis.sd)[keep]
    rng = np.random.default_rng(BOOT_SEED)
    w = rng.normal(size=(N_DIRS, 21))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    s = np.einsum("nrk,dk->dnr", z, w)
    aucs = np.array([[abs(roc_auc_score(y, s[d, :, j]) - 0.5)
        for j in range(len(OFFSETS))] for d in range(N_DIRS)])
    return 0.5 + np.percentile(aucs, 95, axis=0)


def magnitude_only(delta: np.ndarray, axis: CurvatureAxis) -> np.ndarray:
    """No direction at all: the norm of the z-scored descriptor difference."""
    return np.linalg.norm(delta / axis.sd, axis=-1)


def named_axes_plane(delta: np.ndarray, axis: CurvatureAxis) -> np.ndarray:
    """The unsupervised comparison: the descriptor's own (mean d, mean theta).

    Two fixed named columns of R^21, chosen a priori and fit to nothing, so they
    cost none of the 8 effective samples a fitted plane spends.
    """
    j = [RESTRAV_FEATURE_NAMES.index(name) for name in ("mean d", "mean θ")]
    return (delta / axis.sd)[..., j]


def parity_amplitude(curve: np.ndarray) -> float:
    """Mean |x(r) - (x(r-1)+x(r+1))/2|: the ply-alternating component of a curve."""
    return float(np.abs(curve[1:-1] - 0.5 * (curve[:-2] + curve[2:])).mean())


def n_flipped(m: dict, variant: str = "") -> np.ndarray:
    """Discs the branch move flips, -1 on illegal branches (they flip nothing)."""
    import torch
    meta_all = torch.load(variant_dir("divergence", variant=variant) / "branches"
                          / "branches.pt", weights_only=True)
    return meta_all["n_flipped"][m["row"]].numpy()


def flip_confound(score_r0: np.ndarray, m: dict, variant: str = "") -> float:
    """Spearman rho between axis 1 and flip count, among LEGAL control branches.

    The straightness study's violation is a no-flip placement on an empty square,
    and every legal branch here flips at least one disc, so "illegal vs legal
    control" is confounded with "flips 0 vs flips >= 1". Legality is constant in
    this correlation, so a large |rho| means the axis is a flip-count read-out
    wearing a legality label.
    """
    from scipy.stats import spearmanr
    sel = m["is_control"] > 0
    return float(spearmanr(score_r0[sel], n_flipped(m, variant)[sel]).statistic)


def variant_agreement(space: str, m: dict, variant: str) -> float:
    """Max |phi(rollout, K-mean) - phi(spliced)| over the windows ending at t <= t0.

    The descriptor window is TRAILING, so the one ending at t0 sees only
    tokens[:t0+1] -- which the rollout copies from the spliced branch verbatim.
    Every r <= 0 measurement, the fitted offset among them, is therefore the SAME
    measurement in both variants rather than a replication of it.

    The residual is not exactly zero and is not expected to be: the two trees were
    extracted in separate passes at different batch sizes and stored as fp16, so
    raw layer 5 lands at 1e-3 (fp16 eps on a step length of 0.65) and a sparse code
    amplifies it to 1e-1 where a feature sits on its threshold. It is returned
    rather than asserted away, because its size is the scale on which "identical"
    should be read -- and every r=0 AUROC does come out equal to three decimals.
    """
    k = rollouts(m)
    spliced = meta()
    assert np.array_equal(branch_meta(m)["row"] // k, spliced["row"]), \
        "the two variants' branch rows do not correspond"
    keep = OFFSETS <= 0
    a = fold_rollouts(descriptors(space, m, variant), k)[:, keep]
    b = descriptors(space, spliced)[:, keep]
    assert np.array_equal(np.isnan(a), np.isnan(b)), "undefined angles differ at r <= 0"
    return float(np.nanmax(np.abs(a - b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="spliced")
    parser.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                        help="score the eight l0_operating_point picks instead of the "
                             "seed-42 roster, so a difference between two spaces is the "
                             "architecture and not its sparsity")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    m = meta(variant)
    mb, k = branch_meta(m), rollouts(m)
    n_games = len(np.unique(mb["game_id"]))
    print(f"{n_games} evaluation games, {len(mb['t0'])} branches, "
          f"window {WINDOW_PLIES} plies, fit offset r={FIT_OFFSET}, "
          f"plotted game {PLOT_GAME}"
          + (f", {args.variant}: K={k} rollouts per branch, averaged after phi"
             if k > 1 else "") + "\n")

    r0 = int(np.flatnonzero(OFFSETS == 0)[0])
    illegal, control = mb["category"] != 0, mb["is_control"] > 0
    rows = []
    for space in spaces(args.l0_matched):
        axis = fit_axis(space, FIT_OFFSET, m["held_out_legal"])
        F = descriptors(space, m, variant)
        delta = deltas(F, axis, m)
        score = (delta / axis.sd) @ axis.w
        assert np.abs(score[:, OFFSETS < 0]).max() < 1e-6, \
            "branches differ before t0; the prefix is not shared"

        transport = transported_auroc(axis, [0, 2, 4, 8, 16, 24])
        ct = control_test(score, mb)
        ct_pc1 = control_test(pc1(delta, axis), mb)
        rand95 = random_direction_null(delta, axis, mb)
        ct_norm = control_test(magnitude_only(delta, axis), mb)
        lo, hi = boot_ci(illegal[illegal | control].astype(float),
                         score[illegal | control, r0], mb["game_id"][illegal | control])
        undiff = (fold_rollouts(np.where(np.isnan(F), axis.fill, F), k) / axis.sd) @ axis.w
        plane = residual_plane(delta.reshape(-1, 21), axis).reshape(*delta.shape[:2], 2)
        perm = permutation_null(plane, mb, PLOT_GAME)
        perm_named = permutation_null(named_axes_plane(delta, axis), mb, PLOT_GAME)
        null_mu, null_lo, null_hi = control_null(score[:, r0], mb)
        rows.append({
            "space": space, "auroc_straightness": axis.auroc,
            "gap_sd_one_game": perm["gap"], "gap_null_mean": perm["null_mean"],
            "gap_vs_null": perm["ratio"], "gap_p_exact": perm["p_exact"],
            "named_gap_vs_null": perm_named["ratio"], "named_p_exact": perm_named["p_exact"],
            "control_null_mean": null_mu,
            "control_null_lo": null_lo, "control_null_hi": null_hi,
            "n_train_pairs": axis.n_train, "n_test_pairs": axis.n_test,
            "transport": transport,
            "control_auroc_r0": ct[r0], "control_auroc_r0_lo": lo, "control_auroc_r0_hi": hi,
            "control_auroc_best": ct.max(), "control_auroc_best_r": int(OFFSETS[ct.argmax()]),
            "pc1_control_auroc_r0": ct_pc1[r0],
            "random_dir_p95_r0": rand95[r0], "norm_control_auroc_r0": ct_norm[r0],
            "mean_illegal_r0": score[illegal, r0].mean(),
            "mean_control_r0": score[control, r0].mean(),
            "gap_sd_r0": (score[illegal, r0].mean() - score[control, r0].mean()) /
                np.sqrt(0.5 * (score[illegal, r0].var() + score[control, r0].var())),
            "parity_ratio": parity_amplitude(score[illegal].mean(0)) /
                parity_amplitude(undiff[illegal].mean(0)),
            "landing": {name: float(score[sel, r0].mean())
                for name, sel in {**landing_groups(mb),
                                  **landing_groups(mb, True, variant)}.items()},
            "flip_rho_controls": flip_confound(score[:, r0], mb, variant),
            "r0_vs_spliced": 0.0 if k == 1 else variant_agreement(space, m, variant),
        })
        r = rows[-1]
        print(f"{space:40} probe {r['auroc_straightness']:.3f} | illegal-vs-control "
              f"{r['control_auroc_r0']:.3f} [{lo:.3f}, {hi:.3f}] vs perm null "
              f"{null_mu:.3f} [{null_lo:.3f}, {null_hi:.3f}] (PC1 "
              f"{r['pc1_control_auroc_r0']:.3f}) | flip rho "
              f"{r['flip_rho_controls']:+.2f}")
        print(f"{'':40}   direction check: random R^21 95th pct "
              f"{r['random_dir_p95_r0']:.3f}, norm alone {r['norm_control_auroc_r0']:.3f}, "
              f"PC1 {r['pc1_control_auroc_r0']:.3f}")
        print(f"{'':40}   one game {PLOT_GAME}: gap {r['gap_sd_one_game']:.2f} SD = "
              f"{r['gap_vs_null']:.2f}x its own permutation null (p={r['gap_p_exact']:.2f}); "
              f"named (mean d, mean θ) {r['named_gap_vs_null']:.2f}x "
              f"(p={r['named_p_exact']:.2f})")

    print("\nmean probe log-odds at r=0, branch minus reference "
          "(reference = 0 by construction):")
    for r in rows:
        print(f"  {r['space']:40} " +
              "  ".join(f"{k}:{r['landing'][k]:+.3f}" for k in r["landing"]))

    print("\ntransport of the r=0 direction to r (held-out straightness pairs):")
    for r in rows:
        print(f"  {r['space']:40} " +
              "  ".join(f"r={k}:{v:.3f}" for k, v in r["transport"].items()))
    print("\nparity: alternating amplitude of the differenced curve as a fraction "
          "of the undifferenced one (low = the mover-frame sawtooth cancels):")
    for r in rows:
        print(f"  {r['space']:40} {r['parity_ratio']:.3f}")
    if k > 1:
        print("\nmax |descriptor(rollout, K-mean) - descriptor(spliced)| over r <= 0 "
              "(the trailing window at r=0 sees only the shared prefix, so this is 0):")
        for r in rows:
            print(f"  {r['space']:40} {r['r0_vs_spliced']:.3e}")

    import pandas as pd
    out = out_dir(variant) / f"evaluation{tag(l0_matched=args.l0_matched)}.csv"
    flat = [{**{k: v for k, v in r.items() if k not in ("transport", "landing")},
             **{f"landing_{k}": v for k, v in r["landing"].items()},
             **{f"transport_r{k}": v for k, v in r["transport"].items()}} for r in rows]
    pd.DataFrame(flat).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
