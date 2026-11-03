"""The legal/illegal move detector, as the four figures of this study read it.

One score, one operating point, one game, shared so the panels cannot disagree with each
other. Raw layer 5 is the default and what four of the five panels draw: they are about what
the detector does, not about which dictionary does it best. `figure_auroc_l0_matched.py` is
the exception and passes a `space`, which is what `geometry_path` and `profile_path` are for.

The score is `changepoint`'s: step length standardized per move index against legal games,
then the paired MOSUM contrast over `M` steps either side of a candidate. `tau` is refit on
THIS population with the same exact sweep -- the committed 0.869 was fitted on the 575 games
that study selected, and this one holds every eligible game, so reusing the number would quote
a cut fitted elsewhere.

`ARMS` cuts the tree the way `move_legality.localize.arms` does, so a rule arm is one branch
per game drawn at a stated seed rather than a mean over its member categories.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from spade.common.paths import results_dir
from spade.common.study import study_dir
from spade.l0_operating_point import roster
from spade.move_legality import localize
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_OF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "changepoint_localization"))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    CAND_OFFSET, auroc, by_game, candidates, ci, drop, profile, resamples, scan, take,
    threshold)

DIR = study_dir("divergence") / "illegal_move_detector"
FIGURES = Path(__file__).resolve().parent / "figures"
EVAL_GAMES = results_dir("shared") / "eval_games.csv"
EXAMPLE_RESULTS = results_dir("fig03_detector_example")   # Fig 3: tables + figure_game.json
SAE_RESULTS = results_dir("fig04_detector_sae")
M, PAIRED = 4, True                     # the window changepoint_localization selected
SEED = 42
N_LEGAL = 2000                          # the straightness legal block
RULES = ["R1", "R2", "R3"]
ARMS = ["illegal", *RULES]
# Games per class in the population figures. 500 over the 25 scored positions is 20 games
# each, and the thinnest position holds 22, so the quota is affordable everywhere. The corpus
# ceiling under an equal quota is 550; the round number costs one interval width in the third
# decimal and buys a figure whose n a caption can state without a footnote.
N_EVAL = 500

# One game for the board panel and the trace beside it. Picked by `select_game.py`: all three
# rules have a branch at its t0, t0 sits far enough inside the band that +-5 plies stay on the
# board, and raw's score peaks at t0 on all three rule arms. Asserted at load, so a corpus
# change fails loudly rather than drawing a game the caption no longer describes.
FIGURE_GAME = None                      # set by select_game.py, written into figure_game.json
FIGURE_GAME_JSON = EXAMPLE_RESULTS / "figure_game.json"


def band() -> np.ndarray:
    """Candidate step indices: every position an M-step window can score, moves 6-55.

    The ablation passes its M_MAX = 12 here instead, so that one chance level serves every
    window size it compares. This study never varies the window -- it scores M = 4 and nothing
    else -- and that margin cost it 16 positions at each end, which is why its t0 floor used to
    sit at 14. Scoring what M can actually reach is what lets these panels scan the same
    [10, 34] as `move_legality.detection_auroc`'s bars. The price is that this candidate set is
    no longer the ablation's, so neither its chance level nor a peak position is comparable
    with that study's position for position.
    """
    return candidates(load()["d"].shape[1], M)


REFERENCE = "raw"


def geometry_path(space: str):
    if space == REFERENCE:
        return DIR / "d.npz"
    return DIR / (f"d_{space}.npz" if space in roster.SPACES else f"d_sae_{space}.npz")


def profile_path(space: str):
    """The legal-game geometry `legal_profile` fits "normal at this move index" on.

    An L0-matched pick has only the legal block, which is all this reads:
    `analysis/straightness/l0_matched_profile.py` writes it, `compute_geometry.py` does not.
    """
    geometry = study_dir("straightness") / "geometry"
    if space in roster.SPACES:
        return geometry / f"{space}_legal.npz"
    stem = "resid_post_5" if space == REFERENCE else f"sae_{space}"
    return geometry / f"{stem}_raw.npz"


def load(space: str = REFERENCE) -> dict:
    """d [n, 57] in one space and the branch metadata, from extract_geometry.py."""
    meta = dict(np.load(DIR / "meta.npz"))
    path = geometry_path(space)
    flag = "--l0-matched" if space in roster.SPACES else f"--spaces {space}"
    assert path.exists(), f"no geometry for {space}: run extract_geometry.py {flag}"
    d = np.load(path)["d"].astype(np.float64)
    assert len(d) == len(meta["game_id"]), \
        f"{path.name} holds {len(d)} rows against meta's {len(meta['game_id'])}: re-extract it"
    return {"d": d, "space": space, **meta}


def legal_profile(held_out: np.ndarray, space: str = REFERENCE) -> tuple[np.ndarray, np.ndarray]:
    """Per-move-index mean and SD of raw step length over straightness legal games.

    Minus the ones that ARE scored games: `held_out_legal` names the straightness legal index
    of every game in this corpus, and a trajectory contributing to the baseline it is scored
    against would flatter every number here. Same rule as `changepoint_localization.measure`.
    """
    d = np.load(profile_path(space))["d"]
    keep = np.setdiff1d(np.arange(N_LEGAL), held_out[held_out >= 0])
    return profile(d[keep].astype(np.float64))


def score(data: dict, cand: np.ndarray) -> np.ndarray:
    """[n_rows, len(cand)] detector score: standardized step-length drop, paired MOSUM."""
    mu, sd = legal_profile(data["held_out_legal"], data.get("space", REFERENCE))
    return scan(drop(data["d"], mu, sd), cand, M, PAIRED)


def arms(data: dict, seed: int = SEED) -> dict[str, np.ndarray]:
    """Row indices per arm, one branch per game, aligned across arms by game."""
    meta = {k: torch.from_numpy(data[k]) for k in ("game_id", "t0", "category", "is_control")}
    return {k: v.numpy() for k, v in localize.arms(meta, seed).items()}


def stratified(data: dict, sel: dict, n_games: int = N_EVAL) -> np.ndarray:
    """Positions into every arm's index array: the same number of games at every t0.

    The tree is not balanced on move index -- `extend_branches` forked 1,666 of its games at
    t0 = 20 and about 30 at each other value -- so a prefix of game ids is 46% one position,
    and move index is the strongest nuisance this tree has. An equal quota per t0 makes the
    sample flat across the range, the way `move_legality.detection_auroc` deals its groups.

    `n_games` rarely divides the number of positions -- 200 over 21 does not -- so the lowest
    `n_games % len(values)` positions take one game more than the rest. That is a spread of one
    game, against the 46% imbalance the unstratified prefix carried. Within a position the
    games are the first by sorted game_id: a stated prefix that reproduces without a seed.
    """
    game, t0 = data["game_id"][sel["reference"]], data["t0"][sel["reference"]]
    values = np.unique(t0)
    quota, extra = divmod(n_games, len(values))
    assert quota + bool(extra) <= min(int((t0 == v).sum()) for v in values), \
        f"n = {n_games} over {len(values)} positions needs more games than the thinnest holds"
    keep = [np.flatnonzero(t0 == v)[np.argsort(game[t0 == v], kind="stable")]
            [:quota + (i < extra)] for i, v in enumerate(values)]
    return np.sort(np.concatenate(keep))


def assert_eval_set(game: np.ndarray, t0: np.ndarray) -> int:
    """Fail unless these are exactly the games results/shared/eval_games.csv names. Returns n.

    `figure_auroc.write_eval_games` writes that file out of `stratified`; every figure read
    beside those bars -- here and in `changepoint_localization` -- checks its own draw against
    it rather than trusting that two calls of `stratified` agree.
    """
    want = np.loadtxt(EVAL_GAMES, delimiter=",", skiprows=1, dtype=int)
    order = np.argsort(game, kind="stable")
    assert np.array_equal(game[order], want[:, 0]) and np.array_equal(t0[order], want[:, 1]), \
        f"{len(game)} games, not the {len(want)} of results/shared/eval_games.csv"
    return len(want)


def at_t0(s: np.ndarray, cand: np.ndarray, t0: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The score of each row at its own t0 -- the value the histograms bin and AUROC ranks."""
    where = np.searchsorted(cand + CAND_OFFSET, t0[rows])
    assert (cand[where] + CAND_OFFSET == t0[rows]).all(), "a t0 is outside the scanned band"
    return s[rows, where]


def operating_point(s: np.ndarray, cand: np.ndarray, data: dict, sel: dict) -> float:
    """tau refit on this population, same balanced-accuracy sweep as the parent study.

    Pooled over every scored position of the illegal arm and the three legal arms, labelled
    1 only at t0, cut at t0 for every arm alike because ground truth stops there.
    """
    t0 = data["t0"][sel["reference"]]
    keep = (cand + CAND_OFFSET)[None, :] <= t0[:, None]
    blocks = ["illegal", "reference", "control", "control2"]
    pooled = np.concatenate([s[sel[a]][keep] for a in blocks])
    label = np.concatenate(
        [((cand + CAND_OFFSET)[None, :] == t0[:, None])[keep]]
        + [np.zeros(int(keep.sum()), dtype=bool)] * 3)
    return threshold(pooled, label)


def rule_of_row(category: np.ndarray) -> np.ndarray:
    """R1 / R2 / R3 per row, "" for the legal branches."""
    return np.array(["" if c == LEGAL else RULE_OF[CATEGORIES[c]].split()[0]
                     for c in category])
