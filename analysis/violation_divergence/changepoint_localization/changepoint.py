"""A change-point score on one trajectory, with no counterfactual branch to lean on.

Every other read-out of this branch tree scores `phi_branch(t) - phi_ref(t)`, which needs
the game as it was actually played. Here a single trajectory is scored against what *legal
games do at the same move index*. That profile baseline is what removes the move-index
curvature clock the paired forms cancel by subtraction, and it absorbs the mover-relative
parity sawtooth for free, since parity is a function of the move index.

The score is one-sided on step length. Two-sided forms -- a Mahalanobis over the full 21-D
ReStraV descriptor, or any norm of a difference -- discard the sign, and then they fire
harder on legal human games than on violations (README).

Indexing follows the straightness study: `d[a]` is the step from position a+1 to a+2. The
move token at position c is first carried by the step arriving at c, i.e. `d[c-2]`, so a
candidate move index and its step index differ by CAND_OFFSET.
"""

import numpy as np
import torch

from spade.move_legality.probe import auroc as _auroc

CAND_OFFSET = 2   # candidate move index c <-> step index c - CAND_OFFSET
N_BOOT = 1000
N_SHIFT = 20      # circular shifts per row for the autocorrelation-preserving chance level


def profile(d_legal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-step-index mean and SD over a corpus of legal games."""
    mu, sd = np.nanmean(d_legal, 0), np.nanstd(d_legal, 0)
    assert (sd > 0).all(), "a step index with no variation across the legal corpus"
    return mu, sd


def drop(d: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """How much shorter than usual each step is, in SDs of that step index."""
    g = -(d - mu) / sd
    assert np.isfinite(g).all(), "a non-finite step length"
    return g


def candidates(width: int, m_max: int) -> np.ndarray:
    """Step indices scannable by every window up to m_max, so one chance level serves all."""
    return np.arange(m_max, width - m_max + 1)


def scan(g: np.ndarray, cand: np.ndarray, m: int, paired: bool) -> np.ndarray:
    """[n, width] -> [n, len(cand)], the score at each candidate.

    Unpaired is the mean drop over the m steps from the candidate. Paired subtracts the
    same mean over the m steps before it -- the MOSUM contrast, which cancels the
    game-level offset the unpaired form has to compete against, and whose equal-width
    windows leave `argmax` unbiased in position.
    """
    c = np.concatenate([np.zeros((len(g), 1)), g.cumsum(1)], 1)
    window = lambda a: (c[:, a + m] - c[:, a]) / m
    if paired:
        return np.sqrt(m / 2) * (window(cand) - window(cand - m))
    return np.sqrt(m) * window(cand)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based, ties averaged, never flipped -- `probe.auroc`, the repo's one rule."""
    scores = torch.from_numpy(np.concatenate([positive, negative]).astype(np.float64))
    mask = torch.zeros(len(scores), dtype=torch.bool)
    mask[:len(positive)] = True
    return _auroc(scores, mask)


def boot_ci(statistic, n_games: int, seed: int = 0) -> tuple[float, float]:
    """Percentile CI over resampled games. Games, never positions: the candidates of one
    game overlap in m-1 steps, so they are nowhere near independent draws."""
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.integers(n_games, size=n_games)) for _ in range(N_BOOT)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def located(s: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """The move index each row's score peaks at."""
    return cand[s.argmax(1)] + CAND_OFFSET


def shift_chance(s: np.ndarray, cand: np.ndarray, t0: np.ndarray, seed: int = 0) -> float:
    """Hit rate under the score's own autocorrelation.

    A flat 1/len(cand) is the chance level for a uniform argmax, and the argmax of an
    autocorrelated curve is not uniform. Rolling each row keeps its shape and peak width
    exactly and destroys only the alignment to t0, which is the null being tested.
    """
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(N_SHIFT):
        offset = rng.integers(1, s.shape[1], (len(s), 1))
        index = (np.arange(s.shape[1]) + offset) % s.shape[1]
        hits.append(located(np.take_along_axis(s, index, 1), cand) == t0)
    return float(np.mean(hits))
