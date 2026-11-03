"""The study's bootstrap over source games, and the count-based F1 it resamples.

Shared by `fit_probes` (held-out rows) and `readout` (the branch window) because both
report the same statistic over the same unit. One design per sample size, cached, so
that a difference between two spaces is a difference of the *same* draws -- which is what
makes `excess_macro_f1` and `macro_f1_minus_raw` paired rather than a difference of two
independent intervals.

F1 is a ratio of sums, so a resample re-pools each game's (tp, fp, fn) counts instead of
averaging a per-game statistic.
"""

from functools import lru_cache

import numpy as np

N_BOOT = 1000


@lru_cache(maxsize=None)
def resample_counts(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float32)


def boot_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean and percentile 95% CI over source games (axis 0), per column.

    Resampling as multinomial row counts rather than an index array: the same draw, but a
    [N_BOOT, n] @ [n, k] matmul instead of an [N_BOOT, n, k] gather.
    """
    valid = np.isfinite(values).astype(np.float32)
    counts = resample_counts(len(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ np.nan_to_num(values).astype(np.float32)) / (counts @ valid)
        mean = np.nansum(values, axis=0) / valid.sum(0)
        return (mean, *np.nanpercentile(draws, [2.5, 97.5], axis=0))


def resample(counts: np.ndarray) -> np.ndarray:
    """[N_BOOT, ...] re-pooled (tp, fp, fn) totals."""
    draws = resample_counts(len(counts))
    return (draws @ counts.reshape(len(counts), -1).astype(np.float32)
            ).reshape(N_BOOT, *counts.shape[1:])


def f1_of(counts: np.ndarray) -> np.ndarray:
    """Per-class F1 from [..., 3, C] of (tp, fp, fn); an absent class scores 0, which is
    upstream's `macro_f1_score` definition and `probe.macro_f1`'s."""
    tp, fp, fn = counts[..., 0, :], counts[..., 1, :], counts[..., 2, :]
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.where(denominator > 0, denominator, 1), 0.0)


def recall_of(counts: np.ndarray) -> np.ndarray:
    tp, fn = counts[..., 0, :], counts[..., 2, :]
    return np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)


def accuracy_of(counts: np.ndarray) -> np.ndarray:
    """Every square carries exactly one class, so the true totals are tp + fn."""
    tp, fn = counts[..., 0, :], counts[..., 2, :]
    return tp.sum(-1) / (tp + fn).sum(-1)


def tally(preds: np.ndarray, truth: np.ndarray, n_classes: int) -> np.ndarray:
    """(tp, fp, fn) per class over the squares axis: [..., 64] -> [..., 3, n_classes]."""
    out = np.empty((*preds.shape[:-1], 3, n_classes), dtype=np.int32)
    for c in range(n_classes):
        predicted, actual = preds == c, truth == c
        out[..., 0, c] = (predicted & actual).sum(-1)
        out[..., 1, c] = (predicted & ~actual).sum(-1)
        out[..., 2, c] = (~predicted & actual).sum(-1)
    return out


def by_game(values: np.ndarray, game: np.ndarray, n_games: int) -> np.ndarray:
    """Sum each game's rows. `game` is non-decreasing -- rows are grouped by game and no
    span splits one -- so this is a segmented reduction, not a scatter."""
    assert (np.diff(game) >= 0).all(), "rows are not grouped by game"
    present, start = np.unique(game, return_index=True)
    out = np.zeros((n_games, *values.shape[1:]), dtype=values.dtype)
    out[present] = np.add.reduceat(values, start, axis=0)
    return out


def per_game(values: np.ndarray, game: np.ndarray, n_games: int) -> np.ndarray:
    """Mean over each game's rows, NaN for games with none -- the bootstrap's unit."""
    total = by_game(values.astype(np.float64), game, n_games)
    count = by_game(np.ones((len(values), 1)), game, n_games)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def ci(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return lo, hi
