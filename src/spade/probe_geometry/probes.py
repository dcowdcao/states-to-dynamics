"""Board-state and rule-legality probe directions in OthelloGPT's layer-5 residual space.

Both probes read the same 512-d space (blocks.5.hook_resid_post of
common.config.MODEL_NAME), so their weights are directly comparable — but only
as class *contrasts*. A k-way softmax is unchanged by adding one vector to all k
class weights, so a single class weight does not name a direction; a difference
of two does. The committed board probe carries a large unidentified common
component (‖mean_c W_q‖ ≈ 1.1 against ‖mine − theirs‖ ≈ 8.5), so this is not a
formality. Every direction below is a contrast, unit-normalized.

Board contrasts, per square, span the gauge-invariant space of a 3-way
{empty, theirs, mine} head, and are the two axes of bracket_flip.probe_space's
`coords`: `colour` = mine − theirs (whose disc) and `occupancy` =
(mine + theirs)/2 − empty (whether there is one). mine − empty and
theirs − empty are occupancy ± colour/2.

The rules probes standardize their inputs, so their weights live in z-space; a
direction in activation space is W_c / scale.

Primary directions come from the committed probes. `refit_directions` re-derives
both arms by ridge regression onto centered one-hot targets — closed form, so
cosine_tables.py can refit inside a game-level bootstrap, and one shrinkage for
both arms, which a cosine between them is otherwise sensitive to.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


from spade.common.paths import add_karvonen_to_path, repo_root
from spade.othello.taxonomy import CATEGORIES
from spade.common.study import study_dir

add_karvonen_to_path()

from circuits.othello_engine_utils import stoi_indices  # vocab token i+1 <-> engine square

D_MODEL = 512
COLS = "ABCDEFGH"
BOARD_CLASSES = ("empty", "theirs", "mine")
RULE_CLASSES = ("legal", "R1", "R2", "R3")
BOARD_FAMILIES = ("colour", "occupancy")
RULES_FAMILIES = ("rules4", "rules2", "rules_square")

# illegal_taxonomy.CATEGORIES -> the rule each category violates (docs/othello_rules.md).
RULE_OF_CATEGORY = np.array([0, 1, 1, 2, 2, 2, 3])
assert len(RULE_OF_CATEGORY) == len(CATEGORIES)

BOARD_DIR = Path(os.environ.get(
    "SPADE_LINEAR_PROBE_ACTIVATIONS",
    repo_root() / "analysis" / "linear_probe" / "activations"))
BOARD_GAME_LEN = 50  # rows are game-major over moves 5..54

# R7 seeds these four and no move empties a square, so "empty" never occurs
# there and the occupancy contrast has no support. Colour still does.
NEVER_EMPTY = (8 * 3 + 3, 8 * 3 + 4, 8 * 4 + 3, 8 * 4 + 4)


@dataclass(frozen=True)
class Direction:
    family: str          # one of BOARD_FAMILIES + RULES_FAMILIES
    name: str            # "colour@D2", "R1", "illegal"
    vec: np.ndarray      # [512], unit norm
    square: int | None = None   # engine square 8*row + col, None for a rules direction


def square_name(square: int) -> str:
    return f"{COLS[square % 8]}{square // 8}"


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def stack(directions: list[Direction]) -> np.ndarray:
    return np.stack([d.vec for d in directions])


# ---------------------------------------------------------------- estimator

def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k, dtype=np.float64)[y]


def ridge_weights(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """[d, k] weights of centered X onto centered targets Y, ridge-penalized."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)


# ---------------------------------------------------------------- committed probes

def board_probe() -> np.ndarray:
    """[64, 3, 512] committed board-state probe, engine-square order."""
    w = torch.load(BOARD_DIR / "linear_probe.pt", weights_only=True).double().numpy()
    return w.reshape(D_MODEL, 64, 3).transpose(1, 2, 0)


def rules_probe(n_classes: int) -> dict:
    """Committed move-legality probe, weights mapped out of z-space into activation space."""
    p = torch.load(study_dir("move_legality") / "probes" / f"raw_{n_classes}class.pt",
                   weights_only=False)
    return {"weight": p["weight"].double().numpy() / p["scale"].double().numpy(),
            "classes": tuple(p["classes"]), "l2": p["l2"]}


# ---------------------------------------------------------------- corpora

def board_corpus() -> dict[str, np.ndarray]:
    """Layer-5 activations and mover-relative board labels, 200 ARENA games."""
    acts = torch.load(BOARD_DIR / "primary_layer_activations.pt", weights_only=True)
    boards = torch.load(BOARD_DIR / "board_states.pt", weights_only=True)
    return {"acts": acts.double().numpy(),
            "labels": boards.reshape(len(boards), 64).numpy(),
            "game": np.arange(len(acts)) // BOARD_GAME_LEN}


def rules_corpus() -> dict[str, np.ndarray]:
    """The move-legality fit set: one row per spliced candidate move, 800 games."""
    d = study_dir("move_legality")
    load = lambda name: torch.load(d / "fit_set" / f"{name}.pt", weights_only=True).numpy()
    return {"acts": torch.load(d / "acts" / "resid_post_5.pt", weights_only=True).double().numpy(),
            "rule": load("label").astype(np.int64),
            "category": load("category").astype(np.int64),
            "position": load("position").astype(np.int64),
            "square": engine_square(load("token").astype(np.int64)),
            "game": load("game_id").astype(np.int64)}


def pairs_corpus() -> dict[str, np.ndarray]:
    """The rules study's injected-move pairs: an independent corpus, 4000 games."""
    items = torch.load(study_dir("rules") / "pairs" / "items.pt", weights_only=True)
    category = items["category"].numpy()
    return {"acts": torch.load(study_dir("rules") / "acts" / "acts_final.pt", weights_only=True).double().numpy(),
            "rule": RULE_OF_CATEGORY[category],
            "category": category,
            "position": items["t"].numpy(),
            "square": engine_square(items["injected_token"].numpy()),
            "game": items["prefix_id"].numpy()}


def engine_square(token: np.ndarray) -> np.ndarray:
    """Vocab token of a move -> engine square 8*row + col. Token 0 is pass and never occurs."""
    return np.array(stoi_indices)[token - 1]


def rows_of_games(game: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Rows belonging to `keep` games, repeating a game's rows if it repeats."""
    order = np.argsort(game, kind="stable")
    lo = np.searchsorted(game[order], keep, side="left")
    hi = np.searchsorted(game[order], keep, side="right")
    return np.concatenate([order[a:b] for a, b in zip(lo, hi)])


def balanced_binary(rule: np.ndarray, item: np.ndarray, seed: int) -> np.ndarray:
    """Legal rows plus one illegal row per item — the 2-class probe's own balance rule.

    Averaging an item's illegal rows instead would hand the illegal class
    1/sqrt(3) of its real spread (FIGURE_GUIDELINES 3).
    """
    rng = np.random.default_rng(seed)
    illegal = np.flatnonzero(rule > 0)
    order = np.argsort(item[illegal], kind="stable")
    bounds = np.flatnonzero(np.diff(item[illegal][order], prepend=-1, append=-1))
    picked = [illegal[order[a + rng.integers(b - a)]] for a, b in zip(bounds[:-1], bounds[1:])]
    return np.sort(np.concatenate([np.flatnonzero(rule == 0), np.array(picked, dtype=int)]))


# ---------------------------------------------------------------- directions

def board_directions(w: np.ndarray) -> list[Direction]:
    """[64, 3, 512] probe weights -> colour and occupancy contrasts per square."""
    empty, theirs, mine = (w[:, i] for i in range(3))
    rows = {"colour": mine - theirs, "occupancy": (mine + theirs) / 2 - empty}
    support = {"colour": range(64),
               "occupancy": [q for q in range(64) if q not in NEVER_EMPTY]}
    return [Direction(family, f"{family}@{square_name(q)}", unit(rows[family][q]), q)
        for family in BOARD_FAMILIES
        for q in support[family]]


def rules_directions(w4: np.ndarray, w2: np.ndarray) -> list[Direction]:
    """Legal-referenced contrasts of the 4-class rule probe and the 2-class legality probe."""
    return [Direction("rules4", name, unit(w4[i] - w4[0]))
        for i, name in enumerate(RULE_CLASSES) if i > 0] + \
        [Direction("rules2", "illegal", unit(w2[1] - w2[0]))]


def committed_directions() -> list[Direction]:
    """The primary direction set: both arms straight off the committed probes."""
    return board_directions(board_probe()) + rules_directions(
        rules_probe(4)["weight"], rules_probe(2)["weight"])


def refit_directions(board: dict, rules: dict, board_rows: np.ndarray,
                     rules_rows: np.ndarray, lam_board: float, lam_rules: float,
                     seed: int = 0) -> list[Direction]:
    """Both arms re-derived by ridge on the given rows — the bootstrap's estimator."""
    targets = np.concatenate([one_hot(board["labels"][board_rows, q], 3) for q in range(64)], 1)
    wb = ridge_weights(board["acts"][board_rows], targets, lam_board).T.reshape(64, 3, D_MODEL)

    rule = rules["rule"][rules_rows]
    w4 = ridge_weights(rules["acts"][rules_rows], one_hot(rule, 4), lam_rules).T
    item = rules["game"][rules_rows] * 64 + rules["position"][rules_rows]
    binary = balanced_binary(rule, item, seed)
    w2 = ridge_weights(rules["acts"][rules_rows][binary],
                       one_hot((rule[binary] > 0).astype(int), 2), lam_rules).T
    return board_directions(wb) + rules_directions(w4, w2)


def square_rules_directions(rules: dict, rows: np.ndarray, lam: float) -> list[Direction]:
    """Legal-vs-illegal refit separately on each candidate square.

    The committed rules probes are square-agnostic — one direction read off the
    residual wherever the candidate move was spliced in — while every board
    direction is square-specific. Pooling over squares is what makes the two
    families look orthogonal, so the comparison the board arm can actually be
    matched against is this one, square by square.
    """
    square, rule = rules["square"][rows], (rules["rule"][rows] > 0).astype(int)
    out = []
    for q in np.unique(square):
        at = rows[square == q]
        w = ridge_weights(rules["acts"][at], one_hot(rule[square == q], 2), lam).T
        out.append(Direction("rules_square", f"illegal@{square_name(q)}", unit(w[1] - w[0]), q))
    return out
