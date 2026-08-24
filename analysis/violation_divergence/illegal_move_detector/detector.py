"""The legal/illegal move detector, as the four figures of this study read it.

One score, one operating point, one game, shared so the panels cannot disagree with each
other. Everything here is raw layer 5: the panels are about what the detector does, not about
which dictionary does it best, which is `changepoint_localization`'s question.

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

from spade.common.study import study_dir
from spade.move_legality import localize
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_OF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "changepoint_localization"))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    CAND_OFFSET, auroc, by_game, candidates, ci, drop, profile, resamples, scan, take,
    threshold)

DIR = study_dir("divergence") / "illegal_move_detector"
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
M, PAIRED, M_MAX = 4, True, 12          # the window changepoint_localization selected
SEED = 42
N_LEGAL = 2000                          # the straightness legal block
RULES = ["R1", "R2", "R3"]
ARMS = ["illegal", *RULES]

# One game for the board panel and the trace beside it. Picked by `select_game.py`: all three
# rules have a branch at its t0, t0 sits far enough inside the band that +-5 plies stay on the
# board, and raw's score peaks at t0 on all three rule arms. Asserted at load, so a corpus
# change fails loudly rather than drawing a game the caption no longer describes.
FIGURE_GAME = None                      # set by select_game.py, written into figure_game.json


def band() -> np.ndarray:
    """Candidate step indices, the same 33-ply window the ablation scans."""
    return candidates(load()["d"].shape[1], M_MAX)


def load() -> dict:
    """d [n, 57] and the branch metadata, from extract_geometry.py."""
    meta = dict(np.load(DIR / "meta.npz"))
    return {"d": np.load(DIR / "d.npz")["d"].astype(np.float64), **meta}


def legal_profile(held_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-move-index mean and SD of raw step length over straightness legal games.

    Minus the ones that ARE scored games: `held_out_legal` names the straightness legal index
    of every game in this corpus, and a trajectory contributing to the baseline it is scored
    against would flatter every number here. Same rule as `changepoint_localization.measure`.
    """
    d = np.load(study_dir("straightness") / "geometry" / "resid_post_5_raw.npz")["d"]
    keep = np.setdiff1d(np.arange(N_LEGAL), held_out[held_out >= 0])
    return profile(d[keep].astype(np.float64))


def score(data: dict, cand: np.ndarray) -> np.ndarray:
    """[n_rows, len(cand)] detector score: standardized step-length drop, paired MOSUM."""
    mu, sd = legal_profile(data["held_out_legal"])
    return scan(drop(data["d"], mu, sd), cand, M, PAIRED)


def arms(data: dict, seed: int = SEED) -> dict[str, np.ndarray]:
    """Row indices per arm, one branch per game, aligned across arms by game."""
    meta = {k: torch.from_numpy(data[k]) for k in ("game_id", "t0", "category", "is_control")}
    return {k: v.numpy() for k, v in localize.arms(meta, seed).items()}


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
