"""Red team for the probe-geometry claims at blocks.5.hook_resid_post.

Two claims, and the contrast between them is the finding.

POOLED: the square-agnostic rules directions (the committed 4-class and 2-class probes,
one direction read wherever the candidate move was spliced) sit at mean |cos| 0.016-0.029
against every board direction. A single global "illegal" direction cannot be a combination
of per-square board contrasts, so that near-orthogonality is close to forced.

SQUARE-MATCHED: legality refit per candidate square (`probes.square_rules_directions`)
aligns with the board directions at its OWN square and not at any other. That is the
claim worth attacking, so every threat below is measured against it, and against the
pooled number for contrast.

The threats: estimator, token identity, move index, corpus, split-half power, the choice
of null, and what the board basis actually spans. Two deflationary readings get their own
measurement: that the alignment is only R1 ("target square occupied"), which is the
occupancy contrast almost by definition, and that it is per-square class imbalance
leaking through the fit.

Usage: python src/probe_geometry/robustness.py
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from spade.common.paths import add_karvonen_to_path, repo_root, results_dir
from spade.probe_geometry.probes import (
    BOARD_FAMILIES,
    D_MODEL,
    RULE_CLASSES,
    RULES_FAMILIES,
    board_corpus,
    board_directions,
    board_probe,
    committed_directions,
    engine_square,
    study_dir,
    one_hot,
    pairs_corpus,
    refit_directions,
    ridge_weights,
    rows_of_games,
    rules_corpus,
    rules_proba,
    square_name,
    square_rules_directions,
    stack,
    unit,
)

add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, to_string

RESULTS = repo_root() / "analysis" / "probe_geometry" / "results"
PAPER_RESULTS = results_dir("fig02_probe_geometry_and_detection")   # robustness_cells.csv (A.4)
LAM_GRID = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
LAM = 1e3          # pooled arms: matched shrinkage for the board and square-agnostic fits
SQUARE_LAM = 1e1   # per-square fits see ~2.8k rows, not 176k; swept in threat 1
N_BOOT = 100       # game-level bootstrap replicates
N_SPLITS = 10      # random half-splits behind each reliability estimate
N_DRAWS = 10       # redraws behind each sampled or label-shuffled null
N_NULL = 2000      # random directions behind each null distribution
SUBSPACE_RANKS = (1, 2, 4, 8, 16, 32, 59)
MIN_RULE_ROWS = 200  # a square enters the per-rule split only with this many rows of that rule
TRAIN_FRACTION = 0.8
SEED = 0
RANDOM_FLOOR = float(np.sqrt(2 / (np.pi * D_MODEL)))
NONE = "-"


# ---------------------------------------------------------------- geometry

def arms(directions: list) -> tuple[list, list, list]:
    """Board directions, square-agnostic rules directions, per-square rules directions."""
    return ([d for d in directions if d.family in BOARD_FAMILIES],
            [d for d in directions if d.family in ("rules4", "rules2")],
            [d for d in directions if d.family == "rules_square"])


def span(directions: list) -> np.ndarray:
    """[rank, 512] orthonormal basis of the directions' span."""
    _, s, vt = np.linalg.svd(stack(directions), full_matrices=False)
    return vt[s > s[0] * 1e-10]


def captured(basis: np.ndarray, vec: np.ndarray) -> float:
    """Fraction of a unit vector's squared norm inside an orthonormal basis' span."""
    return float(np.square(basis @ vec).sum())


def cell(variant: str, statistic: str, family: str, rule: str, n: int, value) -> dict:
    return {"variant": variant, "statistic": statistic, "board_family": family,
            "rule": rule, "n": int(n), "value": float(value)}


def matched_cos(board: list, square: list, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Cosines of each per-square legality direction with `family` at its own square, and elsewhere."""
    group = [d for d in board if d.family == family]
    index = {d.square: i for i, d in enumerate(group)}
    cos = stack(square) @ stack(group).T
    own = np.zeros(cos.shape, dtype=bool)
    own[np.arange(len(square)), [index[d.square] for d in square]] = True
    return cos[own], cos[~own]


def matched_board(board: list, family: str, square: int) -> np.ndarray:
    return next(d.vec for d in board if d.family == family and d.square == square)


def pooled_cells(board: list, rules: list, variant: str) -> list[dict]:
    cos = stack(board) @ stack(rules).T
    rows = []
    for j, rule in enumerate(rules):
        for family in BOARD_FAMILIES:
            keep = np.array([d.family == family for d in board])
            c = np.abs(cos[keep, j])
            rows += [cell(variant, "mean_abs_cos", family, rule.name, keep.sum(), c.mean()),
                     cell(variant, "max_abs_cos", family, rule.name, keep.sum(), c.max())]
    return rows


def square_cells(board: list, square: list, variant: str) -> list[dict]:
    rows = []
    for family in BOARD_FAMILIES:
        own, other = matched_cos(board, square, family)
        rows += [cell(variant, "matched_mean_cos", family, "rules_square", len(own), own.mean()),
                 cell(variant, "matched_mean_abs_cos", family, "rules_square", len(own),
                      np.abs(own).mean()),
                 cell(variant, "matched_min_cos", family, "rules_square", len(own), own.min()),
                 cell(variant, "matched_positive_fraction", family, "rules_square", len(own),
                      (own > 0).mean()),
                 cell(variant, "other_mean_cos", family, "rules_square", len(other), other.mean()),
                 cell(variant, "other_mean_abs_cos", family, "rules_square", len(other),
                      np.abs(other).mean())]
    return rows


def capture_cells(board: list, rules: list, square: list, variant: str) -> list[dict]:
    """How much of a rules direction the board basis spans: all 124, or the matched pair."""
    basis = span(board)
    rows = [cell(variant, "captured_by_board_basis", NONE, d.name, len(basis),
                 captured(basis, d.vec)) for d in rules + square]
    pair = np.array([captured(span([b for b in board if b.square == d.square]), d.vec)
                     for d in square])
    rows.append(cell(variant, "captured_by_matched_pair_mean", NONE, "rules_square",
                     len(square), pair.mean()))
    return rows


def summarize(directions: list, variant: str) -> list[dict]:
    board, rules, square = arms(directions)
    return (pooled_cells(board, rules, variant) + square_cells(board, square, variant)
            + capture_cells(board, rules, square, variant))


def project_out(directions: list, basis: np.ndarray) -> list:
    return [replace(d, vec=unit(d.vec - basis.T @ (basis @ d.vec))) for d in directions]


def covariance(rows: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Feature covariance without materializing a centered copy of a 176k-row corpus."""
    mean = rows.mean(0, dtype=np.float64)
    out = np.zeros((rows.shape[1], rows.shape[1]))
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk].astype(np.float64) - mean
        out += block.T @ block
    return out / (len(rows) - 1)


def principal(rows: np.ndarray) -> np.ndarray:
    """Orthonormal principal directions of `rows` about their mean, most variance first."""
    return np.linalg.eigh(covariance(rows))[1].T[::-1]


def random_like(directions: list, rng: np.random.Generator, root: np.ndarray) -> list:
    """The same family and square layout, every direction redrawn as `root` times white noise."""
    draws = rng.standard_normal((len(directions), D_MODEL)) @ root.T
    return [replace(d, vec=unit(v)) for d, v in zip(directions, draws)]


# ---------------------------------------------------------------- fits

def compact(corpus: dict) -> dict:
    """Activations in float32.

    This box caps the user at 8 GB shared with other jobs, and a float64 corpus plus the
    copies `ridge_weights` makes to center it does not fit. Every fitted direction still
    comes out in float64 (the solve promotes), and agrees with the float64 corpus to
    cos >= 0.999998 across the whole lambda grid.
    """
    return {**corpus, "acts": corpus["acts"].astype(np.float32)}


def all_rows(corpus: dict) -> np.ndarray:
    return np.arange(len(corpus["acts"]))


def game_rows(corpus: dict, games: np.ndarray) -> np.ndarray:
    return rows_of_games(corpus["game"], games)


def split_games(corpus: dict, fraction: float, rng: np.random.Generator) -> tuple:
    games = rng.permutation(np.unique(corpus["game"]))
    cut = int(round(fraction * len(games)))
    return games[:cut], games[cut:]


def fit(board: dict, rules: dict, board_rows: np.ndarray, rules_rows: np.ndarray,
        lam: float = LAM, square_lam: float = SQUARE_LAM, seed: int = SEED) -> list:
    """Every direction of one variant: both pooled arms and the 60 per-square legality fits."""
    return (refit_directions(board, rules, board_rows, rules_rows, lam, lam, seed)
            + square_rules_directions(rules, rules_rows, square_lam))


def fit_board(board: dict, rows: np.ndarray, lam: float) -> list:
    """The board arm alone — `refit_directions` without the 176k-row pooled rules fit.

    The bootstrap needs the board and per-square arms and nothing else, and the pooled fit
    it would otherwise drag along costs 700 MB of working set this box does not have.
    """
    targets = np.concatenate([one_hot(board["labels"][rows, q], 3) for q in range(64)], 1)
    return board_directions(ridge_weights(board["acts"][rows], targets,
                                          lam).T.reshape(64, 3, D_MODEL))


def fit_all(board: dict, rules: dict, lam: float = LAM, square_lam: float = SQUARE_LAM,
            seed: int = SEED) -> list:
    """The whole-corpus fit, with the pooled arm indexed by slices.

    Fancy-indexing all 176k rows hands `ridge_weights` a 700 MB copy that it then copies
    again to center; a slice is a view, and the two copies do not both fit in this box.
    """
    return (refit_directions(board, rules, slice(None), slice(None), lam, lam, seed)
            + square_rules_directions(rules, all_rows(rules), square_lam))


def board_accuracy(board: dict, train: np.ndarray, test: np.ndarray, lam: float) -> float:
    """Mean over the 64 squares of 3-way argmax accuracy on held-out games."""
    target = np.concatenate([one_hot(board["labels"][train, q], 3) for q in range(64)], 1)
    w = ridge_weights(board["acts"][train], target, lam)
    scores = ((board["acts"][test] - board["acts"][train].mean(0)) @ w
              + target.mean(0)).reshape(-1, 64, 3)
    return float((scores.argmax(-1) == board["labels"][test]).mean())


def square_accuracy(rules: dict, train: np.ndarray, test: np.ndarray, lam: float) -> float:
    """Legal-vs-illegal accuracy of the per-square fits, pooled over held-out rows."""
    correct = 0
    for q in np.unique(rules["square"][train]):
        at, held = train[rules["square"][train] == q], test[rules["square"][test] == q]
        target = one_hot((rules["rule"][at] > 0).astype(int), 2)
        w = ridge_weights(rules["acts"][at], target, lam)
        scores = (rules["acts"][held] - rules["acts"][at].mean(0)) @ w + target.mean(0)
        correct += int((scores.argmax(1) == (rules["rule"][held] > 0)).sum())
    return correct / len(test)


# ---------------------------------------------------------------- threat 1: estimator

def estimator_sweep(board: dict, rules: dict) -> list[dict]:
    rng = np.random.default_rng(SEED)
    b_train, b_test = split_games(board, TRAIN_FRACTION, rng)
    r_train, r_test = split_games(rules, TRAIN_FRACTION, rng)
    rows = []
    for lam in LAM_GRID:
        variant = f"ridge_lam{lam:.0e}"
        rows += summarize(fit_all(board, rules, lam=lam, square_lam=lam), variant)
        rows.append(cell(variant, "test_accuracy_board", NONE, NONE, len(game_rows(board, b_test)),
                         board_accuracy(board, game_rows(board, b_train),
                                        game_rows(board, b_test), lam)))
        rows.append(cell(variant, "test_accuracy_square", NONE, NONE, len(game_rows(rules, r_test)),
                         square_accuracy(rules, game_rows(rules, r_train),
                                         game_rows(rules, r_test), lam)))
        print(f"  {variant} done", flush=True)
    return rows


# ---------------------------------------------------------------- threat 2: token identity

def square_means(rules: dict) -> np.ndarray:
    """[60, 512] mean activation per candidate square: the empirical token subspace."""
    return np.stack([rules["acts"][rules["square"] == q].mean(0)
                     for q in np.unique(rules["square"])])


def embedding_by_square() -> dict[int, np.ndarray]:
    """W_E of each move token, keyed by the engine square it plays. Token 0 is pass."""
    from spade.common.config import MODEL_NAME  # only this step needs the model
    from spade.common.net import force_ipv4
    from transformer_lens import HookedTransformer

    force_ipv4()
    w = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu").W_E[1:].detach()
    return dict(zip(engine_square(np.arange(1, 61)).tolist(), w.double().numpy()))


def token_threat(directions: list, rules: dict, embedding: dict) -> tuple[list, list]:
    """Global subspace projections, plus the sharper per-square W_E control."""
    board, _, square = arms(directions)
    energy, cells = [], []

    rows = np.stack([embedding[q] for q in sorted(embedding)])
    subspaces = {"W_E": principal(rows), "activation_square_means": principal(square_means(rules))}
    for source, basis in subspaces.items():
        for k in SUBSPACE_RANKS:
            top = basis[:k]
            for family in BOARD_FAMILIES + RULES_FAMILIES:
                group = [d for d in directions if d.family == family]
                e = np.array([captured(top, d.vec) for d in group])
                energy += [{"source": source, "rank": k, "family": family, "n": len(group),
                            "statistic": stat, "value": float(value)}
                           for stat, value in (("mean_energy", e.mean()),
                                               ("max_energy", e.max()))]
            cells += summarize(project_out(directions, top), f"token_out_{source}_k{k}")

    for family in BOARD_FAMILIES + ("rules_square",):
        group = [d for d in directions if d.family == family and d.square in embedding]
        e = np.array([abs(d.vec @ unit(embedding[d.square])) for d in group])
        energy.append({"source": "W_E_own_square", "rank": 1, "family": family, "n": len(group),
                       "statistic": "mean_abs_cos", "value": float(e.mean())})

    own = {q: unit(v) for q, v in embedding.items()}
    stripped = [replace(d, vec=unit(d.vec - own[d.square] * (own[d.square] @ d.vec)))
                for d in board + square if d.square in own]
    cells += square_cells(arms(stripped)[0], arms(stripped)[2], "token_out_own_square_W_E")
    return energy, cells


# ---------------------------------------------------------------- threat 3: move index

def legal_means_by_position(rules: dict) -> np.ndarray:
    """[n_move_indices, 512] mean legal-class activation at each move index."""
    legal = rules["rule"] == 0
    return np.stack([rules["acts"][legal & (rules["position"] == p)].mean(0)
                     for p in np.unique(rules["position"])])


def shift_by_position(rules: dict, means: np.ndarray, sign: int) -> None:
    """Add or remove each move index's legal-class mean, in place.

    In place because the corpus is 700 MB and a second copy of it does not fit
    alongside the ridge fit's own working set; `position_threat` undoes the shift.
    """
    for i, p in enumerate(np.unique(rules["position"])):
        at = rules["position"] == p
        rules["acts"][at] += sign * means[i]


def position_threat(board: dict, rules: dict, directions: list) -> list[dict]:
    means = legal_means_by_position(rules)
    shift_by_position(rules, means, -1)
    try:
        centered = fit_all(board, rules)
    finally:
        shift_by_position(rules, means, +1)
    rows = summarize(arms(directions)[0] + arms(centered)[1] + arms(centered)[2],
                     "position_centered")

    pc1 = principal(means)[0]
    for family in BOARD_FAMILIES + RULES_FAMILIES:
        group = [d for d in directions if d.family == family]
        rows.append(cell("position_pc1", "abs_cos_position_pc1", family, NONE, len(group),
                         np.abs(stack(group) @ pc1).mean()))
    return rows


# ---------------------------------------------------------------- threat 4: corpus

def replay_labels(games: np.ndarray) -> np.ndarray:
    """[n_games, n_moves, 64] mover-relative {0 empty, 1 theirs, 2 mine} after each move.

    Perspective comes from the engine's `next_hand_color`, never from move-index parity:
    a forfeited hand flips the mover without consuming a move.
    """
    out = np.zeros(games.shape + (64,), dtype=np.int64)
    for g, game in enumerate(games):
        board = OthelloBoardState()
        for p, token in enumerate(game):
            board.umpire(to_string(int(token)))
            relative = (board.state * board.next_hand_color).reshape(64)
            out[g, p] = np.where(relative == 0, 0, np.where(relative > 0, 2, 1))
    return out


def board_arm_on_rules(rules: dict) -> dict:
    """A board-probe corpus out of the rules corpus' legal rows, whose spliced move is played."""
    games = torch.load(study_dir("move_legality") / "fit_set" / "tokens.pt",
                       weights_only=True).numpy().astype(np.int64)
    labels = replay_labels(games)
    rows = np.flatnonzero(rules["rule"] == 0)
    assert (rules["square"][rows]
            == engine_square(games[rules["game"][rows], rules["position"][rows]])).all()
    return {"acts": rules["acts"][rows],
            "labels": labels[rules["game"][rows], rules["position"][rows]],
            "game": rules["game"][rows]}


def four_class_confusion(rules: dict) -> list[dict]:
    """The committed 4-class probe on its own test split: `legal` recall is the caveat."""
    split = torch.load(study_dir("move_legality") / "fit_set" / "split.pt",
                       weights_only=True).numpy()
    test = np.flatnonzero(split[rules["game"]] == 2)
    pred = rules_proba(4, rules["acts"][test]).argmax(1)
    truth = rules["rule"][test]
    return [{"statistic": "committed_4class_legal_predicted_as", "family": NONE, "name": name,
             "n": int((truth == 0).sum()), "value": float((pred[truth == 0] == c).mean())}
            for c, name in enumerate(RULE_CLASSES)]


def transfer_rows(matched: dict, arena_dirs: list, cross_dirs: list) -> list[dict]:
    """Does the board arm survive the corpus change, and does the committed probe read it?"""
    scores = np.einsum("nd,qcd->nqc", matched["acts"], board_probe())
    train, test = split_games(matched, TRAIN_FRACTION, np.random.default_rng(SEED))
    rows = [
        {"statistic": "committed_board_accuracy_on_rules_corpus", "family": NONE, "name": NONE,
         "n": len(matched["acts"]),
         "value": float((scores.argmax(-1) == matched["labels"]).mean())},
        {"statistic": "ridge_board_accuracy_on_rules_corpus", "family": NONE, "name": NONE,
         "n": len(game_rows(matched, test)),
         "value": board_accuracy(matched, game_rows(matched, train),
                                 game_rows(matched, test), LAM)},
    ]
    a, m = arms(arena_dirs)[0], arms(cross_dirs)[0]
    assert [d.name for d in a] == [d.name for d in m]
    return rows + [{"statistic": "cos_arena_vs_rules_corpus_board_direction", "family": x.family,
                    "name": x.name, "n": 1, "value": float(x.vec @ y.vec)}
                   for x, y in zip(a, m)]


# ---------------------------------------------------------------- threat 5: power

def reliability(board: dict, rules: dict, directions: list) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(SEED)
    cos = []
    for i in range(N_SPLITS):
        b0, b1 = split_games(board, 0.5, rng)
        r0, r1 = split_games(rules, 0.5, rng)
        first = fit(board, rules, game_rows(board, b0), game_rows(rules, r0), seed=i)
        second = fit(board, rules, game_rows(board, b1), game_rows(rules, r1), seed=i)
        assert [d.name for d in first] == [d.name for d in second] == [d.name for d in directions]
        cos.append([float(x.vec @ y.vec) for x, y in zip(first, second)])
        print(f"  split {i + 1}/{N_SPLITS}", flush=True)

    half = np.array(cos).mean(0)
    sd = np.array(cos).std(0, ddof=1)
    full = 2 * half / (1 + half)   # Spearman-Brown: reliability of a whole-corpus fit
    per_direction = [{"family": d.family, "name": d.name, "n_splits": N_SPLITS,
                      "split_half_cos": float(h), "split_half_sd": float(s),
                      "reliability_spearman_brown": float(f)}
                     for d, h, s, f in zip(directions, half, sd, full)]
    assert full.min() > 0, "a direction does not reproduce itself; the correction is undefined"

    rel = dict(zip([d.name for d in directions], full))
    board_arm, pooled, square = arms(directions)
    cells = []
    for family in BOARD_FAMILIES:
        group = [d for d in board_arm if d.family == family]
        own = np.array([d.vec @ matched_board(board_arm, family, d.square) for d in square])
        scale = np.sqrt([rel[d.name] * rel[f"{family}@{square_name(d.square)}"] for d in square])
        cells += [cell("attenuation_corrected", "matched_mean_cos", family, "rules_square",
                       len(own), (own / scale).mean()),
                  cell("attenuation_corrected", "matched_mean_abs_cos", family, "rules_square",
                       len(own), np.abs(own / scale).mean())]
        for d in pooled:
            c = np.abs((stack(group) @ d.vec)
                       / np.sqrt([rel[b.name] * rel[d.name] for b in group]))
            cells.append(cell("attenuation_corrected", "mean_abs_cos", family, d.name,
                              len(group), c.mean()))
    return per_direction, cells


# ---------------------------------------------------------------- threat 6: nulls

def shuffle_within_squares(rules: dict, rng: np.random.Generator) -> dict:
    """Legal/illegal labels permuted inside each candidate square: same marginals, no signal."""
    key = rules["square"]
    grouped = np.argsort(key, kind="stable")
    scrambled = np.lexsort((rng.random(len(key)), key))
    rule = np.empty_like(rules["rule"])
    rule[scrambled] = rules["rule"][grouped]
    return {**rules, "rule": rule}


def null_threat(board: dict, rules: dict, directions: list) -> list[dict]:
    rng = np.random.default_rng(SEED)
    board_arm, pooled, square = arms(directions)
    root = np.linalg.cholesky(covariance(rules["acts"]) + 1e-9 * np.eye(D_MODEL))
    rows = [cell("analytic_random", "mean_abs_cos", NONE, NONE, D_MODEL, RANDOM_FLOOR)]
    for name, draw in (("sampled_random", np.eye(D_MODEL)), ("covariance_matched", root)):
        for _ in range(N_DRAWS):
            rows += summarize(random_like(directions, rng, draw), name)
    for i in range(N_DRAWS):
        shuffled = fit_all(board, shuffle_within_squares(rules, rng), seed=i)
        rows += summarize(board_arm + arms(shuffled)[1] + arms(shuffled)[2],
                          "shuffled_rules_labels")
        permuted = {**board, "labels": board["labels"][rng.permutation(len(board["labels"]))]}
        rows += summarize(arms(fit_all(permuted, rules, seed=i))[0] + pooled + square,
                          "shuffled_board_labels")
        print(f"  null draw {i + 1}/{N_DRAWS}", flush=True)
    return rows


# ---------------------------------------------------------------- threat 7: capture nulls

def capture_nulls(directions: list, rules: dict) -> list[dict]:
    """What the board basis spans of a rules direction, against what it spans of nothing."""
    board_arm, pooled, square = arms(directions)
    basis = span(board_arm)
    rank = len(basis)
    rng = np.random.default_rng(SEED)
    root = np.linalg.cholesky(covariance(rules["acts"]) + 1e-9 * np.eye(D_MODEL))
    pcs = principal(rules["acts"])[:rank]

    rows = [{"statistic": "board_basis_rank", "name": NONE, "n": len(board_arm),
             "value": float(rank), "lo": np.nan, "hi": np.nan},
            {"statistic": "analytic_null_captured_full_basis", "name": NONE, "n": D_MODEL,
             "value": rank / D_MODEL, "lo": np.nan, "hi": np.nan},
            {"statistic": "analytic_null_captured_matched_pair", "name": NONE, "n": D_MODEL,
             "value": 2 / D_MODEL, "lo": np.nan, "hi": np.nan}]
    for name, draw in (("isotropic", np.eye(D_MODEL)), ("covariance_matched", root)):
        v = rng.standard_normal((N_NULL, D_MODEL)) @ draw.T
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        e = np.square(basis @ v.T).sum(0)
        rows.append({"statistic": f"sampled_null_captured_full_basis_{name}", "name": NONE,
                     "n": N_NULL, "value": float(e.mean()),
                     "lo": float(np.quantile(e, 0.025)), "hi": float(np.quantile(e, 0.975))})
        pair = np.concatenate([np.square(span([b for b in board_arm if b.square == d.square])
                                         @ v.T).sum(0) for d in square])
        rows.append({"statistic": f"sampled_null_captured_matched_pair_{name}", "name": NONE,
                     "n": len(pair), "value": float(pair.mean()),
                     "lo": float(np.quantile(pair, 0.025)), "hi": float(np.quantile(pair, 0.975))})
    rows += [{"statistic": "captured_by_top_activation_pcs", "name": d.name, "n": rank,
              "value": captured(pcs, d.vec), "lo": np.nan, "hi": np.nan}
             for d in pooled + square]
    rows += [{"statistic": "captured_by_matched_pair", "name": d.name, "n": 2,
              "value": captured(span([b for b in board_arm if b.square == d.square]), d.vec),
              "lo": np.nan, "hi": np.nan} for d in square]
    return rows


# ---------------------------------------------------------------- deflationary readings

def square_stats(rules: dict, keep: np.ndarray) -> dict[int, tuple[int, float]]:
    """Rows and legal-class share per candidate square, over `keep`.

    The legal share is what the class-imbalance reading of the alignment would key on, so
    it travels next to every per-square cosine.
    """
    square, rule = rules["square"][keep], rules["rule"][keep]
    return {int(q): (int((square == q).sum()), float((rule[square == q] == 0).mean()))
            for q in np.unique(square)}


def square_detail(board_arm: list, square: list, stats: dict, variant: str) -> list[dict]:
    return [{"variant": variant, "board_family": family, "square": d.square,
             "square_name": square_name(d.square), "n_rows": stats[d.square][0],
             "legal_fraction": stats[d.square][1],
             "cos_matched": float(d.vec @ matched_board(board_arm, family, d.square))}
            for d in square for family in BOARD_FAMILIES]


def rule_split(board_arm: list, rules: dict) -> tuple[list[dict], list[dict]]:
    """Is the per-square alignment only R1, which is the occupancy contrast by definition?"""
    cells, detail = [], []
    for r, name in enumerate(RULE_CLASSES):
        if r == 0:
            continue
        counts = square_stats(rules, np.flatnonzero(rules["rule"] == r))
        keep = [q for q, (n, _) in counts.items() if n >= MIN_RULE_ROWS]
        rows = np.flatnonzero(np.isin(rules["square"], keep)
                              & ((rules["rule"] == 0) | (rules["rule"] == r)))
        square = square_rules_directions(rules, rows, SQUARE_LAM)
        variant = f"square_legal_vs_{name}"
        cells += square_cells(board_arm, square, variant)
        detail += square_detail(board_arm, square, square_stats(rules, rows), variant)
        print(f"  legal vs {name}: {len(square)} squares", flush=True)
    return cells, detail


def balanced_square_rows(rules: dict, rows: np.ndarray, seed: int) -> np.ndarray:
    """Equal legal and illegal counts inside every candidate square."""
    rng = np.random.default_rng(seed)
    keep = []
    for q in np.unique(rules["square"][rows]):
        at = rows[rules["square"][rows] == q]
        legal, illegal = at[rules["rule"][at] == 0], at[rules["rule"][at] > 0]
        n = min(len(legal), len(illegal))
        keep += [rng.choice(legal, n, replace=False), rng.choice(illegal, n, replace=False)]
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------- bootstrap

def bootstrap(board: dict, rules: dict, variant: str, same_games: bool) -> pd.DataFrame:
    """Game-level percentile interval on the square-matched claim.

    Resamples games, refits the board arm and the 60 per-square legality directions, and
    reports every statistic they support. The pooled square-agnostic arms are not in here:
    they carry point estimates only (see robustness_cells.csv).
    """
    def square_summary(board_rows: np.ndarray, rules_rows: np.ndarray) -> list[dict]:
        b = fit_board(board, board_rows, LAM)
        q = square_rules_directions(rules, rules_rows, SQUARE_LAM)
        return square_cells(b, q, variant) + capture_cells(b, [], q, variant)

    rng = np.random.default_rng(SEED)
    board_games, rules_games = np.unique(board["game"]), np.unique(rules["game"])
    draws = []
    for i in range(N_BOOT):
        b = rng.choice(board_games, len(board_games), replace=True)
        r = b if same_games else rng.choice(rules_games, len(rules_games), replace=True)
        draws += square_summary(game_rows(board, b), game_rows(rules, r))
        if (i + 1) % 10 == 0:
            print(f"  {variant} bootstrap {i + 1}/{N_BOOT}", flush=True)
    keys = ["variant", "statistic", "board_family", "rule"]
    grouped = pd.DataFrame(draws).groupby(keys)["value"]
    out = pd.DataFrame({"boot_mean": grouped.mean(), "lo": grouped.quantile(0.025),
                        "hi": grouped.quantile(0.975)}).reset_index()
    out["n_boot"] = N_BOOT
    point = pd.DataFrame(square_summary(all_rows(board), all_rows(rules)))
    return out.merge(point.rename(columns={"value": "point"}), on=keys)


# ---------------------------------------------------------------- entrypoint

def write(rows, name: str, directory: Path = RESULTS) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(directory / name, index=False)
    print(f"wrote {len(frame)} rows to {directory / name}", flush=True)


def main() -> None:
    board, rules = compact(board_corpus()), compact(rules_corpus())
    print(f"board corpus {board['acts'].shape}, rules corpus {rules['acts'].shape}", flush=True)
    stats = square_stats(rules, all_rows(rules))
    square = square_rules_directions(rules, all_rows(rules), SQUARE_LAM)
    committed = committed_directions() + square
    matched = fit_all(board, rules)
    board_committed, board_ridge = arms(committed)[0], arms(matched)[0]
    cells = summarize(committed, "committed") + summarize(matched, "ridge_matched")
    detail = (square_detail(board_committed, square, stats, "committed")
              + square_detail(board_ridge, square, stats, "ridge_matched"))

    print("threat 1: estimator", flush=True)
    cells += estimator_sweep(board, rules)

    print("threat 2: token identity", flush=True)
    energy, token_cells = token_threat(matched, rules, embedding_by_square())
    write(energy, "robustness_token_energy.csv")
    cells += token_cells

    print("threat 3: move index", flush=True)
    cells += position_threat(board, rules, matched)

    print("threat 4: corpus", flush=True)
    one_corpus = board_arm_on_rules(rules)
    cross = fit_all(one_corpus, rules)
    cells += summarize(cross, "cross_corpus_ridge")
    detail += square_detail(arms(cross)[0], arms(cross)[2], stats, "cross_corpus_ridge")
    write(transfer_rows(one_corpus, matched, cross) + four_class_confusion(rules),
          "robustness_corpus_transfer.csv")
    pairs = compact(pairs_corpus())
    independent = square_rules_directions(pairs, all_rows(pairs), SQUARE_LAM)
    cells += square_cells(board_ridge, independent, "independent_pairs_corpus")
    detail += square_detail(board_ridge, independent, square_stats(pairs, all_rows(pairs)),
                            "independent_pairs_corpus")
    del pairs

    print("threat 5: power", flush=True)
    per_direction, corrected = reliability(board, rules, matched)
    write(per_direction, "robustness_reliability.csv")
    cells += corrected

    print("threat 6: nulls", flush=True)
    cells += null_threat(board, rules, matched)

    print("threat 7: captured fraction", flush=True)
    write(capture_nulls(matched, rules), "robustness_captured.csv")

    print("deflationary readings", flush=True)
    rule_cells, rule_detail = rule_split(board_ridge, rules)
    cells += rule_cells
    detail += rule_detail
    balanced_rows = balanced_square_rows(rules, all_rows(rules), SEED)
    balanced = square_rules_directions(rules, balanced_rows, SQUARE_LAM)
    cells += square_cells(board_ridge, balanced, "square_balanced")
    detail += square_detail(board_ridge, balanced, square_stats(rules, balanced_rows),
                            "square_balanced")

    write(cells, "robustness_cells.csv", PAPER_RESULTS)
    write(detail, "robustness_square_detail.csv")

    print("bootstraps", flush=True)
    write(pd.concat([bootstrap(board, rules, "ridge_matched", same_games=False),
                     bootstrap(one_corpus, rules, "cross_corpus_ridge", same_games=True)]),
          "robustness_bootstrap.csv")


if __name__ == "__main__":
    main()
