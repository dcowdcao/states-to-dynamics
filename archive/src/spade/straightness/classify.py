"""Step 4 of the straightness study: separation metrics per space/variant/pair.

For every geometry npz and every legal-vs-OOD pair: AUROC of per-game mean
turning angle alone, AUROC of mean step length alone (norm-confound check),
and test AUROC of a logistic regression on the 21-D ReStraV feature vector.
Includes a legal-vs-legal null (must land near 0.5) and a violation-spike
localization check against the matched source legal games.

AUROC convention: score = metric of the OOD side, so >0.5 means the OOD class
is curvier/longer-stepped. Train/test is the shared even/odd-index split.

Usage: python -m spade.straightness.classify
"""

import numpy as np
import pandas as pd
import torch

from spade.common.paths import results_dir
from spade.common.study import study_dir
from spade.straightness.build_games import CLASSES
from spade.straightness.geometry import moment4, restrav_features, train_mask

NULL_PAIR = "null_legal_halves"
NULL_BAND = (0.42, 0.58)
BOOT_SEED = 0
SPIKE_WINDOW = 8  # relative positions [-SPIKE_WINDOW, SPIKE_WINDOW] around t0


def pairs(n: int, first_illegal: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    legal = torch.arange(n)
    out = {NULL_PAIR: (legal[: n // 2], legal[n // 2:])}
    for c in range(1, len(CLASSES)):
        out[f"legal_vs_{CLASSES[c]}"] = (legal, torch.arange(c * n, (c + 1) * n))
    rollout = torch.arange(n, 2 * n)
    legal_rollouts = rollout[first_illegal[rollout] == -1]
    if len(legal_rollouts) >= 50:
        out["legal_vs_rollout_legal_only"] = (legal, legal_rollouts)
    return out


def lr_auc(F21: np.ndarray, idx: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = F21[idx].copy()
    col_mean = np.nanmean(X[tr], axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    X = (X - mu) / sd
    lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    return roc_auc_score(y[te], lr.predict_proba(X[te])[:, 1])


def boot_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """Stratified percentile bootstrap over games (classes resampled separately)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = [roc_auc_score(y[i], s[i]) for i in
        (np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
         for _ in range(n_boot))]
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def pair_row(mean_theta, mean_d, F21, neg, pos, train) -> dict:
    from sklearn.metrics import roc_auc_score
    idx = torch.cat([neg, pos]).numpy()
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    is_train = train[idx].numpy()
    tr, te = np.where(is_train)[0], np.where(~is_train)[0]
    theta_te, d_te = mean_theta[idx[te]], mean_d[idx[te]]
    lo, hi = boot_ci(y[te], theta_te)
    d_lo, d_hi = boot_ci(y[te], d_te)
    return {
        "auc_mean_theta": roc_auc_score(y[te], theta_te),
        "auc_mean_theta_lo": lo,
        "auc_mean_theta_hi": hi,
        "auc_mean_d": roc_auc_score(y[te], d_te),
        "auc_mean_d_lo": d_lo,
        "auc_mean_d_hi": d_hi,
        "auc_lr21": lr_auc(F21, idx, y, tr, te),
        "n_test_neg": int((~is_train[y == 0]).sum()),
        "n_test_pos": int((~is_train[y == 1]).sum()),
    }


def check_spike(theta: np.ndarray, t0: torch.Tensor, src: torch.Tensor, n: int) -> None:
    """Violation curvature, relative to the matched source legal game, must
    peak at the injected move. theta index k pivots at position k+2."""
    viol = np.arange(2 * n, 3 * n)
    delta = theta[viol] - theta[src[viol].numpy()]
    rel_sum, rel_count = np.zeros(2 * SPIKE_WINDOW + 1), np.zeros(2 * SPIKE_WINDOW + 1)
    for row, t in zip(delta, t0[viol].numpy()):
        for r in range(-SPIKE_WINDOW, SPIKE_WINDOW + 1):
            k = t + r - 2
            if 0 <= k < row.shape[0] and not np.isnan(row[k]):
                rel_sum[r + SPIKE_WINDOW] += row[k]
                rel_count[r + SPIKE_WINDOW] += 1
    profile = rel_sum / rel_count
    # sign flips with depth (violations are CURVIER at layer 0, STRAIGHTER at
    # deep layers), so localization is tested on the magnitude
    peak = int(np.abs(profile).argmax()) - SPIKE_WINDOW
    assert -2 <= peak <= 2, f"violation |spike| at relative position {peak}, expected ~0"
    print(f"spike check ok: |peak| at t0{peak:+d}, height {profile[peak + SPIKE_WINDOW]:+.1f}deg")


def main():
    base = study_dir("straightness")
    data = torch.load(base / "games" / "games.pt", weights_only=True)
    n = len(data["cls"]) // len(CLASSES)
    train = train_mask(len(data["cls"]))
    pair_idx = pairs(n, data["first_illegal"])

    rows, stats = [], []
    for f in sorted((base / "geometry").glob("*_raw.npz")):
        space = f.name.removesuffix("_raw.npz")
        for variant in ["raw", "centered"]:
            g = np.load(base / "geometry" / f"{space}_{variant}.npz")
            d, theta = torch.from_numpy(g["d"]), torch.from_numpy(g["theta"])
            F21 = restrav_features(d, theta).numpy()
            mean_theta, mean_d = theta.nanmean(1).numpy(), d.mean(1).numpy()

            for name, (neg, pos) in pair_idx.items():
                rows.append({"space": space, "variant": variant, "pair": name,
                             **pair_row(mean_theta, mean_d, F21, neg, pos, train)})
            null = rows[-len(pair_idx)]
            if null["n_test_neg"] + null["n_test_pos"] >= 400:  # band calibrated for full runs
                assert NULL_BAND[0] <= null["auc_mean_theta"] <= NULL_BAND[1], \
                    f"{space}/{variant}: null AUROC {null['auc_mean_theta']:.3f} outside {NULL_BAND}"

            m_theta, m_d = moment4(theta), moment4(d)
            for c, cname in enumerate(CLASSES):
                block = slice(c * n, (c + 1) * n)
                stats.append({"space": space, "variant": variant, "cls": cname,
                              "mean_theta": m_theta[block, 0].mean().item(),
                              "var_theta": m_theta[block, 3].mean().item(),
                              "mean_d": m_d[block, 0].mean().item()})

            if space == "resid_post_5" and variant == "raw":
                check_spike(theta.numpy(), data["t0"], data["src_game"], n)

    out = results_dir("straightness")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "auroc.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "summary_stats.csv", index=False)
    print(f"wrote {len(rows)} auroc rows, {len(stats)} stat rows to {out}")


if __name__ == "__main__":
    main()
