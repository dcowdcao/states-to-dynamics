"""The supervised curvature axis: the LR-21 probe direction on a sliding window.

The straightness study's LR-21 probe classifies a *whole game* from one 21-D
ReStraV descriptor. A fork figure needs a value per move index, so the descriptor
here is computed on a trailing window of WINDOW_PLIES positions ending at t. The
window is trailing and not centred so that nothing moves before the branch: a
centred window would put the injection's effect at t < t0, which is a lie about
when the branches separate.

Dimensional accounting. A window of W sampled positions yields W-1 step lengths
and W-2 turning angles, and `restrav_features` slices d[:, :7] and theta[:, :6],
so a 21-D vector needs W >= 8 sampled positions. It does not raise below that --
it silently returns a shorter vector -- so `window_features` asserts the width.
W = 8 sampled positions is therefore forced, and it is also the best: held-out
AUROC falls monotonically with window length in every space (8 -> 0.72, 10 ->
0.64, 12 -> 0.58 in raw layer 5), because a violation perturbs a fixed handful of
d/theta entries and a longer window only dilutes them. At exactly 8 the four
moments are deterministic functions of the raw block, so the descriptor spans 13
dimensions, not 21; the redundant columns cost the fit nothing and keep the
vector the study's own object.

Sampling is every ply, so 8 sampled positions is 8 plies. The fork figure samples
every 2nd ply because the board code is mover-relative and the raw path is a
sawtooth -- at stride 2 the same window would be 16 plies out of a 17-position
figure, which is not a sliding window at all. The sawtooth is admissible here
because the plotted quantity is a *difference* of two descriptors at the same t,
and the frame flip is common to both, so it cancels. `check_parity_cancels` in
make_figures.py is the test of that claim, not an assertion of it.

The probe is fit on the straightness study's matched violation/source-legal
pairs, both windows ending at the same position t0 + offset. At offset 0 the two
windows differ in exactly one token -- the injected move, at the window's last
position -- which is the same contrast the divergence branches are built on.

Feature scaling follows spade.straightness.classify.lr_auc: NaN (undefined
angles in sparse codes) imputed with the training column mean, then z-scored on
the training split. The descriptor mixes degrees with activation-scale lengths
whose variances differ ~450x across dictionaries, so an unscaled discriminant
would be a units report, not a direction.
"""

from dataclasses import dataclass

import numpy as np
import torch

from spade.common.study import study_dir
from spade.l0_operating_point import roster
from spade.straightness.build_games import CLASSES
from spade.straightness.geometry import restrav_features

WINDOW_PLIES = 8
GEOM_DIR = study_dir("straightness") / "geometry"
RAW_SPACE = "resid_post_5"


def spaces(l0_matched: bool = False) -> list[str]:
    """Raw layer 5 first (the reference space), then every dictionary.

    The seed-42 roster sits wherever each family's default hyperparameters put it on the
    sparsity axis, so a difference between two of those cells confounds the architecture
    with its L0. `l0_matched` swaps in the eight `l0_operating_point` picks, one per family
    at the shared target, which is what removes that confound. Both rosters read the same
    two files per space -- the straightness `_raw` geometry the axis is fit on and the
    branch geometry it is scored on -- so only the names change.
    """
    if l0_matched:
        return [RAW_SPACE] + roster.SPACES
    return [RAW_SPACE] + sorted(
        f.name.removesuffix("_raw.npz") for f in GEOM_DIR.glob("sae_*_raw.npz"))


def window_features(d: np.ndarray, theta: np.ndarray, rows: np.ndarray,
                    t: np.ndarray) -> np.ndarray:
    """21-D descriptor of the trailing window ending at position t, per row.

    d/theta index positions 1..58: d[i] is the step from position i+1 to i+2 and
    theta[k] pivots at position k+2, so the window ending at t owns d[t-W:t-1]
    and theta[t-W:t-2]. Requires W <= t <= 58.
    """
    w = WINDOW_PLIES
    assert (t >= w).all() and (t <= 58).all(), "window runs off the trajectory"
    start = (t - w)[:, None]
    dw = d[rows[:, None], start + np.arange(w - 1)]
    tw = theta[rows[:, None], start + np.arange(w - 2)]
    f = restrav_features(torch.from_numpy(dw), torch.from_numpy(tw)).numpy()
    # restrav_features slices d[:, :7] and theta[:, :6] and returns a SHORTER
    # vector for a short window instead of raising. W=8 is the exact minimum.
    assert f.shape[1] == 21, f"descriptor is {f.shape[1]}-D, not 21-D (window too short)"
    return f


@dataclass(frozen=True)
class CurvatureAxis:
    """A fitted discriminant direction in 21-D curvature-descriptor space."""

    space: str
    w: np.ndarray        # [21], in z-scored feature units
    sd: np.ndarray       # [21], training-split scale
    fill: np.ndarray     # [21], training-split NaN imputation
    auroc: float         # held-out, at the offset the probe was fit on
    n_train: int         # matched pairs
    n_test: int

    def delta_score(self, F_branch: np.ndarray, F_ref: np.ndarray) -> np.ndarray:
        """Probe log-odds of `F_branch` minus that of `F_ref`, window by window.

        The paired form is what makes the axis usable over move index: both
        descriptors carry the same strong t-dependent curvature clock, and it
        cancels. The mean of the z-scoring cancels with it, so only `sd` is left.
        """
        return (self.delta_features(F_branch, F_ref) / self.sd) @ self.w

    def delta_features(self, F_branch: np.ndarray, F_ref: np.ndarray) -> np.ndarray:
        b = np.where(np.isnan(F_branch), self.fill, F_branch)
        r = np.where(np.isnan(F_ref), self.fill, F_ref)
        return b - r


def _pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(violation row, matched source-legal row, t0) for the straightness games."""
    g = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    n = len(g["cls"]) // len(CLASSES)
    block = slice(2 * n, 3 * n)
    return np.arange(2 * n, 3 * n), g["src_game"][block].numpy(), g["t0"][block].numpy()


def geometry(space: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(GEOM_DIR / f"{space}_raw.npz")
    return z["d"], z["theta"]


def fit_axis(space: str, offset: int, exclude: np.ndarray) -> CurvatureAxis:
    """Fit the discriminant on straightness violation/source-legal pairs whose
    source game index is EVEN and not in `exclude`, at windows ending at
    t0 + offset.

    Even/odd on the game index is `spade.straightness.geometry.train_mask`'s own
    split, and it carries most of the disjointness for free. `exclude` is the
    explicit list of straightness games the figure will be scored on, which is
    what actually enforces it: the plotted game is chosen for the figure, not for
    its parity, and its index can land on either side.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    viol, src, t0 = _pairs()
    t = t0 + offset
    ok = (t >= WINDOW_PLIES) & (t <= 58)
    d, theta = geometry(space)
    X = np.vstack([window_features(d, theta, src[ok], t[ok]),
                   window_features(d, theta, viol[ok], t[ok])])
    y = np.r_[np.zeros(ok.sum()), np.ones(ok.sum())]
    is_train = np.tile((src[ok] % 2 == 0) & ~np.isin(src[ok], exclude), 2)

    fill = np.nanmean(X[is_train], axis=0)
    X = np.where(np.isnan(X), fill, X)
    mu, sd = X[is_train].mean(0), X[is_train].std(0) + 1e-8
    Xz = (X - mu) / sd
    lr = LogisticRegression(max_iter=3000).fit(Xz[is_train], y[is_train])
    score = Xz @ lr.coef_[0]
    is_test = np.tile(src[ok] % 2 == 1, 2)
    return CurvatureAxis(
        space=space, w=lr.coef_[0], sd=sd, fill=fill,
        auroc=roc_auc_score(y[is_test], score[is_test]),
        n_train=int(is_train.sum() // 2), n_test=int(is_test.sum() // 2))


def transported_auroc(axis: CurvatureAxis, offsets: list[int]) -> dict[int, float]:
    """The fitted axis, applied to held-out straightness pairs at other offsets.

    This is the guard on the third axis of the fork figure: a direction fit at
    one move index is only allowed to carry a curve over move index if it still
    separates the classes at the other indices. Scored in the paired form the
    figure uses, violation minus its own source game, so it measures exactly the
    quantity the figure plots.
    """
    from sklearn.metrics import roc_auc_score

    viol, src, t0 = _pairs()
    d, theta = geometry(axis.space)
    out = {}
    for r in offsets:
        t = t0 + r
        ok = (t >= WINDOW_PLIES) & (t <= 58) & (src % 2 == 1)
        F = np.vstack([window_features(d, theta, src[ok], t[ok]),
                       window_features(d, theta, viol[ok], t[ok])])
        score = (np.where(np.isnan(F), axis.fill, F) / axis.sd) @ axis.w
        out[r] = roc_auc_score(np.r_[np.zeros(ok.sum()), np.ones(ok.sum())], score)
    return out


def residual_plane(delta: np.ndarray, axis: CurvatureAxis) -> np.ndarray:
    """[n, 21] -> [n, 2]: (probe log-odds, leading PC of the orthogonal complement).

    Axis 2 is unsupervised and refit per space per game, on the same z-scored
    differences, uncentred -- zero displacement in descriptor space is a real
    point (the reference branch), not an arbitrary origin.
    """
    z = delta / axis.sd
    u = axis.w / np.linalg.norm(axis.w)
    a1 = z @ axis.w
    resid = z - np.outer(z @ u, u)
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    return np.stack([a1, resid @ vt[0]], axis=1)
