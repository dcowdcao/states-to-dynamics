"""Self-tests for the thresholded-detection half of `changepoint.py`. No data, no GPU.

The exact-sweep claim is what these exist to pin: `threshold` says it finds the global
optimum, and the only honest check of that is a brute-force scan of every cut on a small
random case. The rest guard the two things that would silently invalidate an out-of-fold
number -- a fold that splits a game, and a tie run cut down the middle.

Usage: python analysis/violation_divergence/changepoint_localization/test_threshold.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    OBJECTIVES, by_game, flags, folds, out_of_fold, rates, take, threshold)


def _random_case(rng, n=200):
    label = rng.random(n) < 0.25
    return rng.normal(label * 1.5, 1.0), label


def _brute_force(score, label, objective):
    """Every distinct cut, scored directly from the confusion counts."""
    best = (-np.inf, None)
    for tau in np.unique(score):
        predicted = score >= tau
        tp = int((predicted & label).sum())
        fp = int((predicted & ~label).sum())
        value = OBJECTIVES[objective](tp, fp, int(label.sum()), int((~label).sum()))
        best = max(best, (float(value), float(tau)))
    return best


def test_threshold_matches_brute_force_on_every_objective():
    rng = np.random.default_rng(0)
    for trial in range(20):
        score, label = _random_case(rng)
        for objective in OBJECTIVES:
            tau = threshold(score, label, objective)
            value, _ = _brute_force(score, label, objective)
            predicted = flags(score, tau)
            got = OBJECTIVES[objective](int((predicted & label).sum()),
                                        int((predicted & ~label).sum()),
                                        int(label.sum()), int((~label).sum()))
            assert abs(got - value) < 1e-12, (trial, objective, got, value)


def test_ties_are_never_cut_in_half():
    """A tie run is one decision. Rounding the score to 1 dp makes runs ~20 deep."""
    rng = np.random.default_rng(1)
    score, label = _random_case(rng, 2000)
    score = np.round(score, 1)
    tau = threshold(score, label, "balanced")
    assert tau in set(score.tolist()), "cut is not an observed score"
    assert (flags(score, tau) == (score >= tau)).all()


def test_separable_score_is_read_perfectly():
    label = np.array([True] * 30 + [False] * 70)
    score = np.where(label, 5.0, -5.0)
    tau = threshold(score, label, "balanced")
    assert -5.0 < tau <= 5.0, tau
    assert rates(flags(score, tau), label)["balanced_accuracy"] == 1.0


def test_noise_lands_at_chance_and_out_of_fold_does_not_beat_in_sample():
    rng = np.random.default_rng(2)
    label = rng.random(3000) < 0.2
    score = rng.normal(size=3000)                    # carries nothing about label
    game = np.repeat(np.arange(300), 10)
    in_sample = rates(flags(score, threshold(score, label, "balanced")), label)
    _, predicted = out_of_fold(score, label, game, seed=0)
    out = rates(predicted, label)
    assert 0.45 < in_sample["balanced_accuracy"] < 0.60, in_sample
    assert out["balanced_accuracy"] <= in_sample["balanced_accuracy"] + 1e-12, (out, in_sample)


def test_a_fold_never_splits_a_game():
    game = np.repeat(np.arange(97), 13)
    fold = folds(game, k=5, seed=3)
    for g in np.unique(game):
        assert len(set(fold[game == g].tolist())) == 1, g
    assert set(fold.tolist()) == set(range(5)), "a fold came out empty"


def test_out_of_fold_predicts_every_position_exactly_once():
    rng = np.random.default_rng(4)
    label = rng.random(500) < 0.3
    score = rng.normal(label * 1.0, 1.0)
    game = np.repeat(np.arange(50), 10)
    taus, predicted = out_of_fold(score, label, game, seed=5)
    assert len(taus) == 5 and predicted.shape == score.shape
    # Every fold's rows are cut at that fold's own tau, and at no other.
    fold = folds(game, k=5, seed=5)
    for f, tau in enumerate(taus):
        assert (predicted[fold == f] == flags(score[fold == f], tau)).all(), f


def test_take_returns_every_position_of_every_drawn_game():
    game = np.array([2, 2, 2, 0, 0, 1])          # unsorted, unequal counts on purpose
    order, counts = by_game(game)
    assert counts.tolist() == [2, 1, 3]
    assert take(order, counts, np.array([0, 2])).tolist() == [3, 4, 0, 1, 2]
    twice = take(order, counts, np.array([1, 1]))
    assert twice.tolist() == [5, 5], "a game drawn twice must contribute twice"
    whole = take(order, counts, np.arange(3))
    assert sorted(whole.tolist()) == list(range(len(game)))


def test_take_preserves_the_class_balance_of_the_games_it_draws():
    rng = np.random.default_rng(6)
    game = np.repeat(np.arange(40), 7)
    label = np.repeat(rng.random(40) < 0.5, 7)   # a game is all-positive or all-negative
    order, counts = by_game(game)
    draw = rng.integers(40, size=40)
    rows = take(order, counts, draw)
    assert label[rows].sum() == 7 * label[np.searchsorted(np.arange(40), draw) * 7].sum()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
