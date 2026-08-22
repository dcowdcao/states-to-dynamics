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


def resamples(n_games: int, seed: int = 0) -> list[np.ndarray]:
    """Bootstrap draws over games, made once so two spaces can share them.

    Games, never positions: one game's candidates overlap in m-1 steps, so they are nowhere
    near independent draws. Sharing the draws is what makes a CI on a *difference* between
    two spaces correct rather than the two marginal CIs a reader would have to eyeball.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(n_games, size=n_games) for _ in range(N_BOOT)]


def ci(statistic, draws: list[np.ndarray]) -> tuple[float, float]:
    values = [statistic(i) for i in draws]
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def located(s: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """The move index each row's score peaks at."""
    return cand[s.argmax(1)] + CAND_OFFSET


# ---------------------------------------------------------------------------
# Thresholded detection: the same score read as a per-position flag rather than
# as one forced choice per game.
#
# `located` names exactly one move per sequence, so a legal game cannot come back
# clean and a game with two illegal moves cannot score above 1/2. A cut on the
# score emits zero, one or many flags. The score is already standardized per step
# index against the legal profile (`drop`), so one cut is meaningful across games.
#
# Accuracy is piecewise constant in the cut, with breakpoints only at observed
# scores, and it is not unimodal -- a grid straddles the optimum and a binary
# search converges on a local one. Sweeping the breakpoints is exact and costs one
# sort, the same sort the ROC needs, so the cut and the curve come out together.
# ---------------------------------------------------------------------------

FOLDS = 5

# tp, fp are cumulative counts down the descending score; P, N the class totals.
OBJECTIVES = {
    "balanced": lambda tp, fp, P, N: tp / P - fp / N,        # Youden's J
    "accuracy": lambda tp, fp, P, N: (tp + N - fp) / (P + N),
    "f1": lambda tp, fp, P, N: 2 * tp / (P + tp + fp),
}


def flags(s: np.ndarray, tau: float) -> np.ndarray:
    """Which candidates the detector calls illegal. `>=` matches `threshold`'s tie handling."""
    return s >= tau


def _cuts(score: np.ndarray, label: np.ndarray):
    """(tau, tp, fp, P, N) at the last index of every tie run, descending by score.

    Cutting inside a tie run is not a threshold anyone can apply, so only run ends
    are candidates and `>= tau` then flags exactly the rows counted at that index.
    """
    order = np.argsort(-score, kind="stable")
    s, hit = score[order], label[order]
    tp, fp = np.cumsum(hit), np.cumsum(~hit)
    end = np.append(s[:-1] != s[1:], True)
    return s[end], tp[end], fp[end], int(hit.sum()), int((~hit).sum())


def threshold(score: np.ndarray, label: np.ndarray, objective: str = "balanced") -> float:
    """The score cut maximizing `objective` over the pooled positions, exactly.

    "balanced" is (TPR + TNR)/2 up to a constant. Prevalence here is ~2%, so plain
    "accuracy" optimizes to a cut that flags almost nothing and scores ~0.98 by
    calling every move legal; it is reported to show that, not to be used.
    """
    tau, tp, fp, P, N = _cuts(score, label)
    assert P and N, "an objective needs both classes"
    return float(tau[np.argmax(OBJECTIVES[objective](tp, fp, P, N))])


def rates(predicted: np.ndarray, label: np.ndarray) -> dict:
    """TPR, TNR, their mean, and precision, from a boolean prediction."""
    tpr, tnr = float(predicted[label].mean()), float((~predicted[~label]).mean())
    hits = int((predicted & label).sum())
    return {"tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2,
            "precision": hits / max(int(predicted.sum()), 1)}


def folds(game_id: np.ndarray, k: int = FOLDS, seed: int = 0) -> np.ndarray:
    """Fold index per position, assigned by SOURCE game.

    By game and not by position: a game's arms share `tokens[:t0]` and its candidates
    overlap in m-1 steps, so splitting one game across folds leaks the fit into the
    held-out half and the out-of-fold number stops being out of fold.
    """
    ids = np.unique(game_id)
    order = np.random.default_rng(seed).permutation(len(ids))
    return (order % k)[np.searchsorted(ids, game_id)]


def out_of_fold(score, label, game_id, objective: str = "balanced", k: int = FOLDS,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(per-fold tau, pooled out-of-fold prediction).

    Fitting the cut and reading accuracy off the same positions is an upper bound, not
    an estimate. Each fold is predicted at a cut the other four chose.
    """
    fold = folds(game_id, k, seed)
    predicted, taus = np.zeros(len(score), dtype=bool), []
    for f in range(k):
        held = fold == f
        taus.append(threshold(score[~held], label[~held], objective))
        predicted[held] = flags(score[held], taus[-1])
    return np.array(taus), predicted


def by_game(game_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(positions sorted by game, count per game) -- the index a cluster bootstrap draws on."""
    order = np.argsort(game_id, kind="stable")
    return order, np.unique(game_id[order], return_counts=True)[1]


def take(order: np.ndarray, counts: np.ndarray, draw: np.ndarray) -> np.ndarray:
    """Position indices of the games in `draw`, with replacement: every position a game holds."""
    lengths = counts[draw]
    ends = np.cumsum(lengths)
    starts = np.append(0, np.cumsum(counts)[:-1])[draw] - np.append(0, ends[:-1])
    return order[np.repeat(starts, lengths) + np.arange(ends[-1] if len(ends) else 0)]
