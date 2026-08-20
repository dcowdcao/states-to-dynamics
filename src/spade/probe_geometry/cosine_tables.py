"""Cosines between OthelloGPT's board-state probe directions and its rule-legality ones.

The question: does "this square is illegal" reuse the direction the board probe
reads "there is a disc here" off? Both probes read blocks.5.hook_resid_post, and
probes.py turns each into gauge-invariant class contrasts, so a cosine between
them is a statement about the residual stream and not about a softmax offset.

The comparison has to be square-matched to mean anything. The committed rules
probes are square-agnostic — one direction, read wherever the candidate move was
spliced in — while every board direction belongs to one square. Cosines between
those two pool 60 candidate squares against one direction and land at the random
baseline by construction, which is why `square_matched_cosine.csv` (per-square
legal-vs-illegal refits against the board direction for the same square, with the
same direction at the other 59 squares as the null) is the lead table and
`cross_probe_cosine.csv` is the contrast, not the result.

Point estimates come from the committed probes wherever a committed probe exists;
the per-square rules arm has none and is ridge throughout. Intervals come from a
cluster bootstrap over games — games, not rows or squares, because rows within a
game share a board — resampling the two corpora independently and re-deriving
both arms each time. Committed and ridge do not coincide; bootstrap.json records
the per-family agreement, so the interval is read as the sampling noise of the
ridge estimator around its own centre.

`stats` is probes.ridge_weights accumulated in row chunks: a bootstrap resample of
the rules corpus is 700 MB in float64 and this machine cannot hold two of them, so
the resampled row set is indexed a chunk at a time and never materialized. On the same
rows it reproduces probes.refit_directions to 2e-16.

Usage: python src/probe_geometry/cosine_tables.py [--n-games N] [--n-boot 1000]
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from spade.common.paths import repo_root
from spade.probe_geometry.probes import (
    BOARD_FAMILIES,
    D_MODEL,
    Direction,
    balanced_binary,
    board_corpus,
    board_directions,
    board_probe,
    committed_directions,
    study_dir,
    one_hot,
    pairs_corpus,
    rows_of_games,
    rules_corpus,
    rules_directions,
    rules_proba,
    square_name,
    square_rules_directions,
    stack,
)

RESULTS = repo_root() / "analysis" / "probe_geometry" / "results"
MIN_FULL_BOOT = 100  # below this the intervals are a smoke test, not a result
CHUNK = 16384
LAMBDA_GRID = np.logspace(0, 5, 6)
SEED = 0
N_NULL_SHUFFLE = 100
N_NULL_RANDOM = 1000
RULE_ORDER = ("illegal", "R1", "R2", "R3")
PAIR_KINDS = (("colour", "colour"), ("occupancy", "occupancy"), ("colour", "occupancy"))
COLOUR_OFFDIAG_REFERENCE = 0.0184  # regression check on the canonical board probe

# The per-square arm is pinned rather than swept: held-out accuracy is 0.9252 here against
# 0.9269 at the grid argmax of 100 (probe_quality.csv), and 10 is what ROBUSTNESS.md and
# the figure were computed at. The matched cosine is shrinkage-sensitive, so one value has
# to hold across the study.
SQUARE_LAMBDA = 10.0
# Headline number: the committed board probe against the per-square legality refits.
# Full corpus only.
SQUARE_MATCHED_REFERENCE = {"colour": 0.1677, "occupancy": 0.3120}

# The board arm's ridge refits fit on the first 8000 of 10000 game-major rows; the cut
# falls exactly at game 160, so the eval rows are game-disjoint. The committed board probe
# saw none of these games — it was fit on 20000 Karvonen games.
BOARD_TRAIN_FRACTION = 0.8
COMMITTED_RULES_GAMES = 480  # fit_set/split.pt train split, which the committed rules probes saw


# ---------------------------------------------------------------- estimator

@dataclass(frozen=True)
class Ridge:
    weight: np.ndarray   # [d, k]
    mean_x: np.ndarray
    mean_y: np.ndarray

    def scores(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_x) @ self.weight + self.mean_y


@dataclass(frozen=True)
class Stats:
    """Sufficient statistics of a row set, so a lambda sweep is one pass and six solves."""
    gram: np.ndarray
    cross: np.ndarray
    mean_x: np.ndarray
    mean_y: np.ndarray
    n: int

    def solve(self, lam: float) -> Ridge:
        d = len(self.mean_x)
        a = self.gram - self.n * np.outer(self.mean_x, self.mean_x) + lam * np.eye(d)
        b = self.cross - self.n * np.outer(self.mean_x, self.mean_y)
        return Ridge(np.linalg.solve(a, b), self.mean_x, self.mean_y)


def stats(X: np.ndarray, Y: np.ndarray, rows: np.ndarray) -> Stats:
    """probes.ridge_weights' Gram over `rows` (which may repeat rows), chunk by chunk."""
    d, k = X.shape[1], Y.shape[1]
    gram, cross = np.zeros((d, d)), np.zeros((d, k))
    sx, sy = np.zeros(d), np.zeros(k)
    for a in range(0, len(rows), CHUNK):
        r = rows[a:a + CHUNK]
        x, y = X[r].astype(np.float64), Y[r]
        gram += x.T @ x
        cross += x.T @ y
        sx += x.sum(0)
        sy += y.sum(0)
    return Stats(gram, cross, sx / len(rows), sy / len(rows), len(rows))


# ---------------------------------------------------------------- arms

@dataclass(frozen=True)
class BoardArm:
    acts: np.ndarray     # [n, 512] float32
    labels: np.ndarray   # [n, 64] in {empty, theirs, mine}
    targets: np.ndarray  # [n, 192] one-hot, 64 squares x 3 classes
    game: np.ndarray


@dataclass(frozen=True)
class RulesArm:
    acts: np.ndarray
    rule: np.ndarray     # 0 legal, 1..3 the rule broken
    square: np.ndarray   # engine square of the candidate move
    item: np.ndarray     # one candidate move: the unit balanced_binary samples within
    y4: np.ndarray
    y2: np.ndarray
    game: np.ndarray


def board_arm(n_games: int | None) -> BoardArm:
    c = board_corpus()
    keep = np.flatnonzero(c["game"] < n_games) if n_games else slice(None)
    labels = c["labels"][keep]
    return BoardArm(c["acts"][keep].astype(np.float32), labels,
        np.concatenate([one_hot(labels[:, q], 3) for q in range(64)], 1), c["game"][keep])


def rules_arm(corpus: dict, n_games: int | None) -> RulesArm:
    acts = corpus["acts"].astype(np.float32)  # lossless: the stored tensors are float32
    keep = np.flatnonzero(corpus["game"] < n_games) if n_games else slice(None)
    rule = corpus["rule"][keep]
    return RulesArm(acts[keep], rule, corpus["square"][keep],
        corpus["game"][keep] * 64 + corpus["position"][keep],
        one_hot(rule, 4), one_hot((rule > 0).astype(int), 2), corpus["game"][keep])


def board_fit(arm: BoardArm, rows: np.ndarray, lam: float) -> Ridge:
    return stats(arm.acts, arm.targets, rows).solve(lam)


def rules_fits(arm: RulesArm, rows: np.ndarray, lam: float, seed: int) -> tuple[Ridge, Ridge]:
    binary = balanced_binary(arm.rule[rows], arm.item[rows], seed)
    return (stats(arm.acts, arm.y4, rows).solve(lam),
            stats(arm.acts, arm.y2, rows[binary]).solve(lam))


def square_directions(arm: RulesArm, rows: np.ndarray, lam: float) -> list[Direction]:
    return square_rules_directions(
        {"acts": arm.acts, "square": arm.square, "rule": arm.rule}, rows, lam)


def directions(fb: Ridge, f4: Ridge, f2: Ridge) -> list[Direction]:
    return board_directions(fb.weight.T.reshape(64, 3, D_MODEL)) + \
        rules_directions(f4.weight.T, f2.weight.T)


# ---------------------------------------------------------------- held-out accuracy

def games_of(arm, mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask[arm.game])


def board_split(arm: BoardArm) -> tuple[np.ndarray, np.ndarray]:
    cut = round(BOARD_TRAIN_FRACTION * (arm.game.max() + 1))
    return np.flatnonzero(arm.game < cut), np.flatnonzero(arm.game >= cut)


def rules_split(arm: RulesArm) -> tuple[np.ndarray, np.ndarray]:
    """The fit set's own grouped split, so the committed probe's test games are used as such."""
    split = torch.load(study_dir("move_legality") / "fit_set" / "split.pt",
                       weights_only=True).numpy()
    return games_of(arm, split == 0), games_of(arm, split == 2)


def grouped_split(arm, seed: int) -> tuple[np.ndarray, np.ndarray]:
    games = np.unique(arm.game)
    train = np.random.default_rng(seed).permutation(games)[:round(0.8 * len(games))]
    mask = np.zeros(games.max() + 1, bool)
    mask[train] = True
    return games_of(arm, mask), games_of(arm, ~mask)


def board_accuracy(arm: BoardArm, fit: Ridge, rows: np.ndarray) -> float:
    pred = fit.scores(arm.acts[rows]).reshape(len(rows), 64, 3).argmax(-1)
    return float((pred == arm.labels[rows]).mean())


def board_majority(arm: BoardArm, train: np.ndarray, test: np.ndarray) -> float:
    cls = [np.bincount(arm.labels[train, q], minlength=3).argmax() for q in range(64)]
    return float((arm.labels[test] == np.array(cls)).mean())


def majority(y_train: np.ndarray, y_test: np.ndarray, k: int) -> float:
    return float((y_test == np.bincount(y_train, minlength=k).argmax()).mean())


def metrics(pred: np.ndarray, y: np.ndarray, k: int) -> dict:
    """Accuracy, plus what a 4-class probe hides: every rule direction is W[c] - W[legal]."""
    out = {"accuracy": float((pred == y).mean())}
    if k != 4:
        return out
    f1 = [2 * ((pred == c) & (y == c)).sum() / ((pred == c).sum() + (y == c).sum()) for c in range(k)]
    return out | {"macro_f1": float(np.mean(f1)), "legal_recall": float((pred[y == 0] == 0).mean()),
                  "legal_as_R1": int((pred[y == 0] == 1).sum()), "legal_rows": int((y == 0).sum())}


def board_quality(arm: BoardArm) -> list[dict]:
    train, test = board_split(arm)
    base = dict(probe="board", corpus="arena_200", n_train_games=len(np.unique(arm.game[train])),
        n_eval_games=len(np.unique(arm.game[test])), n_eval_rows=len(test) * 64,
        majority_baseline=board_majority(arm, train, test))
    # The committed probe is 64 x 3 logits of the raw activation, with no intercept.
    committed = Ridge(board_probe().reshape(64 * 3, D_MODEL).T, np.zeros(D_MODEL), np.zeros(64 * 3))
    fit = stats(arm.acts, arm.targets, train)
    return [dict(base, estimator="committed", lam=None,
                 accuracy=board_accuracy(arm, committed, test))] + \
        [dict(base, estimator="ridge", lam=lam, accuracy=board_accuracy(arm, fit.solve(lam), test))
            for lam in LAMBDA_GRID]


def rules_quality(arm: RulesArm, corpus: str, train: np.ndarray, test: np.ndarray) -> list[dict]:
    binary = lambda rows: (arm.rule[rows] > 0).astype(int)
    b_train = train[balanced_binary(arm.rule[train], arm.item[train], SEED)]
    b_test = test[balanced_binary(arm.rule[test], arm.item[test], SEED)]
    out = []
    for k, tr, te, y_tr, y_te, targets in (
            (4, train, test, arm.rule[train], arm.rule[test], arm.y4),
            (2, b_train, b_test, binary(b_train), binary(b_test), arm.y2)):
        base = dict(probe=f"rules{k}", corpus=corpus, n_eval_rows=len(te),
            n_train_games=len(np.unique(arm.game[tr])),
            n_eval_games=len(np.unique(arm.game[te])), majority_baseline=majority(y_tr, y_te, k))
        out.append(dict(base, estimator="committed", lam=None, n_train_games=COMMITTED_RULES_GAMES,
            **metrics(rules_proba(k, arm.acts[te]).argmax(1), y_te, k)))
        fit, X = stats(arm.acts, targets, tr), arm.acts[te]
        out += [dict(base, estimator="ridge", lam=lam,
            **metrics(fit.solve(lam).scores(X).argmax(1), y_te, k)) for lam in LAMBDA_GRID]
    return out


def square_quality(arm: RulesArm, corpus: str, train: np.ndarray, test: np.ndarray) -> list[dict]:
    """Legal-vs-illegal, one probe per candidate square, pooled over squares to score."""
    y = (arm.rule > 0).astype(int)
    fits = [(stats(arm.acts, arm.y2, train[arm.square[train] == q]),
             test[arm.square[test] == q]) for q in np.unique(arm.square)]
    base = dict(probe="rules_square", corpus=corpus, n_eval_rows=len(test),
        n_train_games=len(np.unique(arm.game[train])),
        n_eval_games=len(np.unique(arm.game[test])), majority_baseline=majority(y[train], y[test], 2))
    return [dict(base, estimator="ridge", lam=lam, accuracy=float(np.concatenate(
        [f.solve(lam).scores(arm.acts[te]).argmax(1) == y[te] for f, te in fits]).mean()))
            for lam in LAMBDA_GRID]


def best_lambda(quality: pd.DataFrame, corpus: str, probes: tuple[str, ...]) -> float:
    """One shrinkage per arm, so a cross-arm cosine is not comparing two shrinkages."""
    rows = quality[(quality.estimator == "ridge") & (quality.corpus == corpus)
                   & quality.probe.isin(probes)]
    return float(rows.groupby("lam").accuracy.mean().idxmax())


# ---------------------------------------------------------------- cosine statistics

@dataclass(frozen=True)
class Layout:
    family: np.ndarray
    name: np.ndarray
    square: np.ndarray   # -1 for a square-agnostic rules direction

    def of(self, family: str) -> np.ndarray:
        return np.flatnonzero(self.family == family)

    def at(self, name: str) -> int:
        return int(np.flatnonzero(self.name == name)[0])


def layout(dirs: list[Direction]) -> Layout:
    return Layout(np.array([d.family for d in dirs]), np.array([d.name for d in dirs]),
                  np.array([-1 if d.square is None else d.square for d in dirs]))


def chebyshev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(a // 8 - b // 8), np.abs(a % 8 - b % 8))


@dataclass(frozen=True)
class Matched:
    """Per-square rules direction against the board direction for its own square."""
    square: np.ndarray   # the rules squares, in direction order
    matched: np.ndarray  # cos with the board direction at the same square
    other: np.ndarray    # cos with every board direction at a different square
    excess: np.ndarray   # |matched| minus the mean |other| for that same rules direction


def square_matched(A: np.ndarray, ix: Layout) -> dict[str, Matched]:
    rules = ix.of("rules_square")
    if not len(rules):
        return {}
    out = {}
    for f in BOARD_FAMILIES:
        board = ix.of(f)
        m = A[board] @ A[rules].T
        at = {q: i for i, q in enumerate(ix.square[board])}
        own = np.array([at[q] for q in ix.square[rules]])
        matched = m[own, np.arange(len(rules))]
        same = np.zeros(m.shape, bool)
        same[own, np.arange(len(rules))] = True
        out[f] = Matched(ix.square[rules], matched, m[~same],
            np.abs(matched) - (np.abs(m).sum(0) - np.abs(matched)) / (len(board) - 1))
    return out


def cross_cosines(A: np.ndarray, ix: Layout) -> dict[tuple[str, str], np.ndarray]:
    return {(f, r): A[ix.of(f)] @ A[ix.at(r)] for f in BOARD_FAMILIES for r in RULE_ORDER}


def board_cosines(A: np.ndarray, ix: Layout) -> dict[tuple[str, str], np.ndarray]:
    """Within-board cosines by family pair and by how far apart the two squares are.

    Same-square is a group only across families: within one family it is a
    direction against itself, which the i<j mask drops.
    """
    out = {}
    for fa, fb in PAIR_KINDS:
        ia, ib = ix.of(fa), ix.of(fb)
        m = A[ia] @ A[ib].T
        dist = chebyshev(ix.square[ia][:, None], ix.square[ib][None, :])
        keep = np.triu(np.ones(m.shape, bool), 1) if fa == fb else np.ones(m.shape, bool)
        groups = {"same_square": dist == 0, "different_square": dist > 0,
            **{f"chebyshev_{k}": dist == k for k in range(1, 8)}}
        out |= {(f"{fa}-{fb}", rel): m[keep & g] for rel, g in groups.items() if (keep & g).any()}
    return out


def rules_cosines(A: np.ndarray, ix: Layout) -> dict[tuple[str, str], float]:
    return {(a, b): float(A[ix.at(a)] @ A[ix.at(b)]) for a in RULE_ORDER for b in RULE_ORDER}


def statistics(A: np.ndarray, ix: Layout) -> dict[str, float]:
    """The scalars the bootstrap tracks, flattened so replicates stack into a frame."""
    out = {}
    for f, m in square_matched(A, ix).items():
        out[f"square|{f}|matched|mean_cos"] = m.matched.mean()
        out[f"square|{f}|matched|mean_abs_cos"] = np.abs(m.matched).mean()
        out[f"square|{f}|other|mean_cos"] = m.other.mean()
        out[f"square|{f}|other|mean_abs_cos"] = np.abs(m.other).mean()
        out[f"square|{f}|excess|mean_abs_cos"] = m.excess.mean()
    for (f, r), c in cross_cosines(A, ix).items():
        out[f"cross|{f}|{r}|mean_cos"] = c.mean()
        out[f"cross|{f}|{r}|mean_abs_cos"] = np.abs(c).mean()
    for (kind, rel), c in board_cosines(A, ix).items():
        out[f"board|{kind}|{rel}|mean_cos"] = c.mean()
    for (a, b), c in rules_cosines(A, ix).items():
        out[f"rules|{a}|{b}|cos"] = c
    return out


# ---------------------------------------------------------------- bootstrap and nulls

def bootstrap(board: BoardArm, rules: RulesArm, ix: Layout, replicate, n_boot: int) -> pd.DataFrame:
    """Games resampled with replacement, independently per arm; both arms refit each time."""
    rng = np.random.default_rng(SEED)
    bg, rg = np.unique(board.game), np.unique(rules.game)
    reps = []
    for i in range(n_boot):
        dirs = replicate(rows_of_games(board.game, rng.choice(bg, len(bg))),
                         rows_of_games(rules.game, rng.choice(rg, len(rg))), i)
        assert [d.name for d in dirs] == list(ix.name), "a resample changed the direction set"
        reps.append(statistics(stack(dirs), ix))
        if (i + 1) % 25 == 0:
            print(f"  bootstrap {i + 1}/{n_boot}", flush=True)
    return pd.DataFrame(reps)


def random_null(A: np.ndarray, ix: Layout) -> dict[str, tuple[float, float]]:
    """Each rule direction replaced by a random unit vector: the analytic chance line, measured."""
    v = np.random.default_rng(SEED).standard_normal((N_NULL_RANDOM, D_MODEL))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = {}
    for f in BOARD_FAMILIES:
        c = np.abs(A[ix.of(f)] @ v.T).mean(0)
        out[f] = (float(c.mean()), float(np.percentile(c, 97.5)))
    return out


def shuffle_null(A: np.ndarray, ix: Layout, arm: RulesArm, lam: float,
                 n: int) -> dict[tuple[str, str], tuple[float, float]]:
    """Rule labels permuted across rows, rules arm refit: what a probe of nothing aligns to."""
    rng = np.random.default_rng(SEED)
    rows = np.arange(len(arm.acts))
    draws = {(f, r): [] for f in BOARD_FAMILIES for r in RULE_ORDER}
    for i in range(n):
        rule = rng.permutation(arm.rule)
        fake = RulesArm(arm.acts, rule, arm.square, arm.item, one_hot(rule, 4),
                        one_hot((rule > 0).astype(int), 2), arm.game)
        f4, f2 = rules_fits(fake, rows, lam, SEED + i)
        vecs = {d.name: d.vec for d in rules_directions(f4.weight.T, f2.weight.T)}
        for f in BOARD_FAMILIES:
            for r in RULE_ORDER:
                draws[(f, r)].append(np.abs(A[ix.of(f)] @ vecs[r]).mean())
        if (i + 1) % 25 == 0:
            print(f"  shuffle null {i + 1}/{n}", flush=True)
    return {k: (float(np.mean(d)), float(np.percentile(d, 97.5))) for k, d in draws.items()}


# ---------------------------------------------------------------- tables

def ci(boot: pd.DataFrame, key: str) -> tuple[float, float]:
    return tuple(np.percentile(boot[key], [2.5, 97.5]))


def square_table(A: np.ndarray, ix: Layout, boot: pd.DataFrame, refit: dict[str, float],
                 cross: pd.DataFrame, rows_per_square: np.ndarray) -> pd.DataFrame:
    out = []
    for f, m in square_matched(A, ix).items():
        pooled = cross[(cross.board_family == f) & (cross.rule_direction == "illegal")]
        out.append(dict(board_family=f, n_squares=len(m.matched),
            rows_per_square_min=int(rows_per_square.min()),
            rows_per_square_max=int(rows_per_square.max()),
            matched_mean_cos=m.matched.mean(),
            **dict(zip(("matched_cos_lo", "matched_cos_hi"),
                       ci(boot, f"square|{f}|matched|mean_cos"))),
            matched_mean_abs_cos=np.abs(m.matched).mean(),
            **dict(zip(("matched_abs_lo", "matched_abs_hi"),
                       ci(boot, f"square|{f}|matched|mean_abs_cos"))),
            matched_median_abs_cos=np.median(np.abs(m.matched)),
            matched_min_abs_cos=np.abs(m.matched).min(),
            matched_max_abs_cos=np.abs(m.matched).max(),
            argmax_square=square_name(m.square[np.abs(m.matched).argmax()]),
            n_positive=int((m.matched > 0).sum()),
            other_mean_cos=m.other.mean(), other_mean_abs_cos=np.abs(m.other).mean(),
            **dict(zip(("other_abs_lo", "other_abs_hi"),
                       ci(boot, f"square|{f}|other|mean_abs_cos"))),
            excess_mean_abs_cos=m.excess.mean(),
            **dict(zip(("excess_lo", "excess_hi"), ci(boot, f"square|{f}|excess|mean_abs_cos"))),
            pooled_illegal_mean_abs_cos=float(pooled.mean_abs_cos.iloc[0]),
            refit_matched_mean_abs_cos=refit[f"square|{f}|matched|mean_abs_cos"],
            refit_other_mean_abs_cos=refit[f"square|{f}|other|mean_abs_cos"],
            refit_excess_mean_abs_cos=refit[f"square|{f}|excess|mean_abs_cos"]))
    return pd.DataFrame(out)


def cross_table(A: np.ndarray, ix: Layout, boot: pd.DataFrame, refit: dict[str, float],
                shuffle: dict, rand: dict) -> pd.DataFrame:
    out = []
    for (f, r), c in cross_cosines(A, ix).items():
        lo, hi = ci(boot, f"cross|{f}|{r}|mean_abs_cos")
        squares = ix.square[ix.of(f)]
        out.append(dict(board_family=f, rule_direction=r, n_squares=len(c),
            mean_cos=c.mean(), mean_abs_cos=np.abs(c).mean(), abs_cos_lo=lo, abs_cos_hi=hi,
            median_abs_cos=np.median(np.abs(c)), max_abs_cos=np.abs(c).max(),
            argmax_square=square_name(squares[np.abs(c).argmax()]),
            null_mean_abs_cos=shuffle[(f, r)][0], null_hi=shuffle[(f, r)][1],
            null_random_mean_abs_cos=rand[f][0], null_random_hi=rand[f][1],
            refit_mean_abs_cos=refit[f"cross|{f}|{r}|mean_abs_cos"]))
    return pd.DataFrame(out)


def within_board_table(A: np.ndarray, ix: Layout, boot: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (kind, rel), c in board_cosines(A, ix).items():
        lo, hi = ci(boot, f"board|{kind}|{rel}|mean_cos")
        out.append(dict(pair_kind=kind, square_relation=rel, n_pairs=len(c),
            mean_cos=c.mean(), mean_abs_cos=np.abs(c).mean(), median_cos=np.median(c),
            q25_cos=np.percentile(c, 25), q75_cos=np.percentile(c, 75),
            mean_ci_lo=lo, mean_ci_hi=hi))
    return pd.DataFrame(out)


def within_rules_table(A: np.ndarray, ix: Layout, boot: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (a, b), c in rules_cosines(A, ix).items():
        lo, hi = ci(boot, f"rules|{a}|{b}|cos")
        out.append(dict(direction_a=a, direction_b=b, cos=c, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out)


def gram_table(dirs: list[Direction], A: np.ndarray) -> pd.DataFrame:
    i, j = np.triu_indices(len(dirs), 1)
    sq = lambda d: "" if d.square is None else square_name(d.square)
    return pd.DataFrame({
        "family_a": [dirs[a].family for a in i],
        "name_a": [dirs[a].name for a in i],
        "square_a": [sq(dirs[a]) for a in i],
        "family_b": [dirs[b].family for b in j],
        "name_b": [dirs[b].name for b in j],
        "square_b": [sq(dirs[b]) for b in j],
        "cos": (A @ A.T)[i, j]})


def md_table(df: pd.DataFrame) -> str:
    cell = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    return "\n".join([
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
        *["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]])


def tables_md(square: pd.DataFrame, cross: pd.DataFrame, board: pd.DataFrame,
              rules: pd.DataFrame, repl: pd.DataFrame, quality: pd.DataFrame,
              n_boot: int) -> str:
    sq = square[["board_family", "n_squares", "matched_mean_cos", "matched_mean_abs_cos",
                 "matched_abs_lo", "matched_abs_hi", "other_mean_abs_cos", "excess_mean_abs_cos",
                 "excess_lo", "excess_hi", "pooled_illegal_mean_abs_cos"]]
    strong = square.loc[square.excess_mean_abs_cos.idxmax()]
    weak = square.loc[square.excess_mean_abs_cos.idxmin()]
    pooled = cross[["board_family", "rule_direction", "mean_cos", "mean_abs_cos", "abs_cos_lo",
                    "abs_cos_hi", "max_abs_cos", "argmax_square", "null_mean_abs_cos",
                    "null_random_mean_abs_cos"]]
    top = cross.loc[cross.mean_abs_cos.idxmax()]
    main = board[board.square_relation.isin(("same_square", "different_square"))]
    same = main[(main.pair_kind == "colour-occupancy")
                & (main.square_relation == "same_square")].iloc[0]
    wide = rules.pivot(index="direction_a", columns="direction_b", values="cos")
    r4 = quality[(quality.probe == "rules4") & (quality.estimator == "committed")].iloc[0]
    b = quality[(quality.probe == "board") & (quality.estimator == "committed")].iloc[0]
    r_repl = repl.loc[repl.mean_abs_cos.idxmax()]
    return f"""# Probe-geometry cosine tables

Point estimates are the committed probes, except the per-square rules arm, which has no
committed probe and is ridge at lambda 10 throughout.

Where each interval comes from, so no column is an unlabelled mixture:

- every `_lo`/`_hi` column in every CSV here: a {n_boot}-resample game-level cluster
  bootstrap, percentile, both corpora resampled independently and both arms re-derived by
  ridge each resample. Games are the unit; squares and rows are not independent.
- the intervals quoted in the table 1 reading below: the merged 100-replicate red-team
  bootstrap of the same statistic, `robustness_bootstrap.csv`, cited not recomputed.
- everything with no `_lo`/`_hi` column (medians, IQRs, min/max, argmax, `n_*`,
  `refit_*`, `pooled_*`): a bare point estimate, no interval.

Point estimate and interval come from different estimators — committed probe against ridge
refit, agreeing at median cos 0.75 (rules4) to 0.88 (colour) per family, in bootstrap.json.
Several intervals therefore do not contain their own point estimate. That gap is estimator
disagreement plus the attenuation below, not a computation error; the `refit_*` columns give
the all-ridge value the intervals are actually centred on.

Random unit vectors in 512-d have E|cos| = 0.0353, but see the covariance-matched floor in
table 2.

## 1. Square-matched: the rules direction for a square vs the board direction for that square

The committed rules probes are square-agnostic — one direction, read wherever the candidate
move was spliced in — so a cosine against a square-specific board direction pools 60
candidate squares and cannot come out large. Here the legal-vs-illegal probe is refit per
candidate square. The null is that same rules direction against the board directions of the
other squares: it holds probe, corpus and shrinkage fixed and varies only the square. Every
pair is matched on `Direction.square`, never on list position — `board_directions` returns
64 colour but only 60 occupancy directions, so position and engine square differ above
square 27.

{md_table(sq)}

Reading: {strong.board_family} at the matched square reaches mean |cos| =
{strong.matched_mean_abs_cos:.4f} against {strong.other_mean_abs_cos:.4f} at the other 59
squares, and {weak.board_family} {weak.matched_mean_abs_cos:.4f} against
{weak.other_mean_abs_cos:.4f}. {int(strong.n_positive)}/{int(strong.n_squares)} and
{int(weak.n_positive)}/{int(weak.n_squares)} squares are positive, so the signed means
({strong.matched_mean_cos:+.4f}, {weak.matched_mean_cos:+.4f}) track the unsigned ones —
a consistent direction, not an alignment of arbitrary sign. Both other-square nulls sit at
the 0.0353 random-vector scale, so a square's legality direction overlaps that square's
board directions and nothing else's.

The merged 100-replicate bootstrap of the same statistic (`robustness_bootstrap.csv`) gives
matched colour [0.1450, 0.1573], matched occupancy [0.3012, 0.3192], off-square colour
[0.0191, 0.0208], off-square occupancy [0.0171, 0.0179], with its own caveat, carried
verbatim: the resample mean sits below the point estimate on every matched row because a
resample carries ~63% distinct games and this statistic attenuates with noise, so the
interval is spread, not location. The same applies to the columns above.

Four qualifications on this table, all from the merged analyses in `ROBUSTNESS.md`:

- The effect is **R1-only**. Legal-vs-R1 per square gives colour +0.1802 / occupancy
  +0.3866 (positive at 97% / 93% of squares); legal-vs-R2 gives +0.064 / +0.008 and
  legal-vs-R3 +0.008 / +0.031, both at the 0.0353 floor. Not a sample-size artifact: R3 has
  more rows per square at the median than R1.
- Per-move-index centering leaves colour at +0.161 but drops occupancy from +0.359 to
  +0.077, so the occupancy half is largely move-index-mediated.
- 21.2% of each per-square legality direction lies in the 2-d board subspace at its own
  square (isotropic null 0.39%), but the full 124-direction board basis captures only 26.4%
  against a 24.2% chance level: the board-relatedness is local to the square, not a general
  overlap with the board code.
- The lambda is pinned at 10 rather than swept (probe_quality.csv: 0.9252 held-out against
  0.9269 at the grid argmax of 100). The matched cosine is shrinkage-sensitive and the two
  families' ordering reverses by lambda 1e3, so read the colour/occupancy gap as
  shrinkage-dependent and the sign and the matched-vs-other contrast as the robust part.

## 2. Pooled: board direction vs the square-agnostic rule directions

`null_mean_abs_cos` is the rules arm refit on permuted labels; `null_random_mean_abs_cos`
replaces the rule direction with a random unit vector.

{md_table(pooled)}

Reading: the largest cell is {top.board_family} vs {top.rule_direction} at mean |cos| =
{top.mean_abs_cos:.4f}, with signed means running from {cross.mean_cos.min():+.4f} to
{cross.mean_cos.max():+.4f}. The honest floor for a pooled number is not the isotropic
0.0353 but the covariance-matched null of 0.084-0.110 reported in `ROBUSTNESS.md`, against
which these cells are *more* orthogonal than chance rather than at it. Either way the
comparison to make is with table 1: the same probe family, measured square by square, is
{strong.matched_mean_abs_cos / top.mean_abs_cos:.0f}x higher. Read this table as the cost of
pooling over 60 squares, not as evidence that legality and board state are unrelated.

## 3. Within-board: colour vs occupancy

{md_table(main[["pair_kind", "square_relation", "n_pairs", "mean_cos", "mean_abs_cos",
                "median_cos", "q25_cos", "q75_cos", "mean_ci_lo", "mean_ci_hi"]])}

Reading: at one square, "which disc" and "is there a disc" sit at cos =
{same.mean_cos:.4f} (median {same.median_cos:.4f}, IQR {same.q25_cos:.4f} to
{same.q75_cos:.4f}), at the 0.0353 random-vector scale — the two board axes are close to
independent readouts of the same square. The CI is on the mean; per-Chebyshev-distance rows
are in within_board_cosine.csv.

## 4. Within-rules

{md_table(wide.reset_index())}

Reading: the four square-agnostic rule directions are far from orthogonal to each other, so
they are not four independent readouts. illegal.R1 = {float(wide.loc["R1", "illegal"]):.3f}
is a fact about the geometry and not about a confused anchor: every rule direction is
W[c] - W[legal], and the committed 4-class probe recovers `legal` at recall
{r4.legal_recall:.4f}, macro-F1 {r4.macro_f1:.4f}, with {int(r4.legal_as_R1)} of
{int(r4.legal_rows)} true-legal rows in the split.pt test games predicted R1. This is the
table where the two estimators disagree most: the ridge intervals sit far from the
committed cosines they accompany (R1.R2 is
{float(wide.loc["R1", "R2"]):.3f} committed against a ridge interval of
{float(rules[(rules.direction_a == "R1") & (rules.direction_b == "R2")].ci_lo.iloc[0]):.3f}
to {float(rules[(rules.direction_a == "R1") & (rules.direction_b == "R2")].ci_hi.iloc[0]):.3f}),
so treat the inter-rule geometry as estimator-dependent and read the intervals as width only.

## 5. Replication on the injected-move pairs corpus

Rules directions refit on the 4000-prefix pairs corpus, board arm unchanged: the pooled
maximum is mean |cos| = {r_repl.mean_abs_cos:.4f} ({r_repl.board_family} vs
{r_repl.rule_direction}) against a shuffled-label null of {r_repl.null_mean_abs_cos:.4f}, so
the pooled table replicates on an independent corpus. The per-square arm is not replicated
here: 28189 rows over 60 squares is ~470 rows per square against 512 dimensions.

## Probes

probe_quality.csv carries held-out-GAME accuracy for every probe above, over the whole
lambda grid. The committed board probe is {b.accuracy:.4f} against a
{b.majority_baseline:.4f} per-square majority baseline, and all 200 ARENA games are held out
for it: it was fit on 20000 Karvonen games. The 8000/2000 row cut that splits this arm falls
exactly at game 160, so the ridge refit's eval games are disjoint too. The committed 4-class
rules probe is {r4.accuracy:.4f} on the split.pt test games with macro-F1 {r4.macro_f1:.4f}
and legal-class recall {r4.legal_recall:.4f}, scored through `load_space("raw")` — the
feature map it was fit on.
"""


# ---------------------------------------------------------------- entrypoint

def check_square(matched: dict[str, Matched], full_corpus: bool) -> None:
    """The reference is a full-corpus, lam=10 number, so a capped run cannot be checked."""
    if not full_corpus:
        return
    for f, ref in SQUARE_MATCHED_REFERENCE.items():
        got = matched[f].matched.mean()
        assert abs(got - ref) < 0.01, f"{f} matched mean {got:.4f}, expected {ref}"


def check(dirs: list[Direction], A: np.ndarray) -> float:
    assert len(dirs) == 128, len(dirs)
    assert np.allclose(np.linalg.norm(A, axis=1), 1), "directions are not unit norm"
    assert np.isfinite(A).all(), "NaN in the direction matrix"
    offdiag = board_cosines(A, layout(dirs))[("colour-colour", "different_square")].mean()
    assert abs(offdiag - COLOUR_OFFDIAG_REFERENCE) < 1e-3, offdiag
    return float(offdiag)


def agreement(committed: list[Direction], refit: list[Direction]) -> dict[str, float]:
    cos = np.array([float(a.vec @ b.vec) for a, b in zip(committed, refit)])
    fam = np.array([d.family for d in committed])
    return {f: float(np.median(cos[fam == f])) for f in np.unique(fam)}


def results_dir(n_games: int | None, n_boot: int) -> Path:
    """Only a run over every game with real resampling owns the committed filenames.

    A reduced run writing `square_matched_cosine.csv` once got reviewed as if it
    were the full result, and its shrunken cosine was chased as a bug.
    """
    if n_games is None and n_boot >= MIN_FULL_BOOT:
        return RESULTS
    return RESULTS / f"partial_games{n_games or 'all'}_boot{n_boot}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=None, help="cap games per corpus (default: all)")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    out = results_dir(args.n_games, args.n_boot)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    board = board_arm(args.n_games)
    rules = rules_arm(rules_corpus(), args.n_games)
    pairs = rules_arm(pairs_corpus(), args.n_games)
    for name, arm in (("board", board), ("rules", rules), ("pairs", pairs)):
        print(f"{name}: {len(arm.acts)} rows, {len(np.unique(arm.game))} games")

    rules_train, rules_test = rules_split(rules)
    quality = pd.DataFrame(board_quality(board)
        + rules_quality(rules, "fit_set", rules_train, rules_test)
        + square_quality(rules, "fit_set", rules_train, rules_test)
        + rules_quality(pairs, "pairs", *grouped_split(pairs, SEED)))
    lam = {("arena_200", "board"): best_lambda(quality, "arena_200", ("board",))}
    for probe in ("rules4", "rules2"):
        lam[("fit_set", probe)] = best_lambda(quality, "fit_set", ("rules4", "rules2"))
        lam[("pairs", probe)] = best_lambda(quality, "pairs", ("rules4", "rules2"))
    lam[("fit_set", "rules_square")] = SQUARE_LAMBDA
    quality["chosen"] = [r.estimator == "ridge" and r.lam == lam[(r.corpus, r.probe)]
        for r in quality.itertuples()]
    lam_board, lam_rules = lam[("arena_200", "board")], lam[("fit_set", "rules4")]
    lam_square, lam_pairs = SQUARE_LAMBDA, lam[("pairs", "rules4")]
    print(f"lambda: board {lam_board:g}, rules {lam_rules:g}, "
          f"rules_square {lam_square:g}, pairs {lam_pairs:g}")

    committed = committed_directions()
    offdiag = check(committed, stack(committed))
    print(f"colour-colour off-diagonal mean cos = {offdiag:.4f}")

    all_rows = lambda arm: np.arange(len(arm.acts))
    square = square_directions(rules, all_rows(rules), lam_square)
    dirs = committed + square
    A, ix = stack(dirs), layout(dirs)
    refit = directions(board_fit(board, all_rows(board), lam_board),
                       *rules_fits(rules, all_rows(rules), lam_rules, SEED)) + square
    agree = agreement(committed, refit[:len(committed)])
    print("committed vs ridge, median cos:", {k: round(v, 3) for k, v in agree.items()})

    def replicate(brows, rrows, i):
        return directions(board_fit(board, brows, lam_board),
            *rules_fits(rules, rrows, lam_rules, SEED + i)) + \
            square_directions(rules, rrows, lam_square)

    boot = bootstrap(board, rules, ix, replicate, args.n_boot)
    rand = random_null(A, ix)
    shuffle = shuffle_null(A, ix, rules, lam_rules, N_NULL_SHUFFLE)
    check_square(square_matched(A, ix), args.n_games is None)
    refit_stats = statistics(stack(refit), ix)
    cross = cross_table(A, ix, boot, refit_stats, shuffle, rand)
    sq_table = square_table(A, ix, boot, refit_stats, cross,
                            np.bincount(rules.square)[np.unique(rules.square)])

    # Replication: only the rules corpus changes, so the board arm stays committed.
    pairs_ix = layout(committed)
    pairs_fits = rules_fits(pairs, all_rows(pairs), lam_pairs, SEED)
    pairs_dirs = [d for d in committed if d.family in BOARD_FAMILIES] + \
        rules_directions(pairs_fits[0].weight.T, pairs_fits[1].weight.T)
    pairs_refit = directions(board_fit(board, all_rows(board), lam_board), *pairs_fits)

    def pairs_replicate(brows, rrows, i):
        return directions(board_fit(board, brows, lam_board),
                          *rules_fits(pairs, rrows, lam_pairs, SEED + i))

    pairs_boot = bootstrap(board, pairs, pairs_ix, pairs_replicate, args.n_boot)
    repl = cross_table(stack(pairs_dirs), pairs_ix, pairs_boot,
        statistics(stack(pairs_refit), pairs_ix),
        shuffle_null(stack(committed), pairs_ix, pairs, lam_pairs, N_NULL_SHUFFLE), rand)

    board_tab = within_board_table(A, ix, boot)
    rules_tab = within_rules_table(A, ix, boot)
    write = lambda df, name: df.to_csv(out / name, index=False)
    write(gram_table(dirs, A), "direction_cosines.csv")
    write(sq_table, "square_matched_cosine.csv")
    write(cross, "cross_probe_cosine.csv")
    write(board_tab, "within_board_cosine.csv")
    write(rules_tab, "within_rules_cosine.csv")
    write(quality, "probe_quality.csv")
    write(repl, "replication_pairs_corpus.csv")
    (out / "TABLES.md").write_text(
        tables_md(sq_table, cross, board_tab, rules_tab, repl, quality, args.n_boot))
    (out / "bootstrap.json").write_text(json.dumps({
        "resampling_unit": "game", "n_boot": args.n_boot, "seed": SEED,
        "n_games": {"board": len(np.unique(board.game)), "rules": len(np.unique(rules.game)),
                    "pairs": len(np.unique(pairs.game))},
        "n_rows": {"board": len(board.acts), "rules": len(rules.acts), "pairs": len(pairs.acts)},
        "n_rules_squares": len(square),
        "lambda": {"board": lam_board, "rules": lam_rules, "rules_square": lam_square,
                   "pairs": lam_pairs},
        "lambda_grid": LAMBDA_GRID.tolist(),
        "split": {"board": "games 0..159 of 200 for the ridge refit; the committed probe "
                           "saw none of these games",
                  "rules": "fit_set/split.pt, train=0 test=2",
                  "pairs": f"random 80/20 over prefixes, seed {SEED}"},
        "committed_vs_ridge_median_cos": agree,
        "square_arm": "ridge only: there is no committed per-square rules probe",
        "ridge_fit_rows": "all games of each corpus; the committed rules probes saw 480",
        "null": {"shuffle_refits": N_NULL_SHUFFLE, "random_vectors": N_NULL_RANDOM,
                 "square_matched": "same rules direction at the other squares",
                 "analytic_random_abs_cos": float(np.sqrt(2 / (np.pi * D_MODEL)))},
        "colour_colour_offdiag_mean_cos": offdiag,
        "runtime_s": round(time.time() - started, 1)}, indent=2))

    print(sq_table[["board_family", "matched_mean_cos", "matched_mean_abs_cos", "other_mean_abs_cos",
        "excess_mean_abs_cos", "excess_lo", "excess_hi", "n_positive"]].to_string(index=False))
    print(cross[["board_family", "rule_direction", "mean_cos", "mean_abs_cos", "abs_cos_lo",
        "abs_cos_hi", "max_abs_cos", "argmax_square", "null_mean_abs_cos"]].to_string(index=False))
    print(board_tab[board_tab.square_relation.isin(("same_square", "different_square"))][
        ["pair_kind", "square_relation", "n_pairs", "mean_cos", "median_cos"]].to_string(index=False))
    print(rules_tab[rules_tab.direction_a < rules_tab.direction_b].to_string(index=False))
    print(quality[quality.chosen | quality.estimator.eq("committed")].to_string(index=False))
    print(f"wrote {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
