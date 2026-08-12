"""What the probe's per-position output says about *where* the illegal move is.

Three statistics, all read off the [n, 59] score the read-out writes per space, and all
with a cluster bootstrap over source games rather than over rows -- a game's branches
share tokens[:t0] and are not independent draws.

  peri_event    the score at t0 + r, the event-locked curve of FIGURE_GUIDELINES 4.2
  localization  does argmax over t in [1, t0] land on t0 (position 0 is the sink)
  detection     AUROC of the score at t0, illegal branch against its own legal control

`arms` names the contrasts the figures draw, so `localization.py` and `space_comparison.py`
cut the branch tree the same way rather than each deciding what "the illegal arm" is.

`localization` and `detection` stop at t0 because ground truth does. A branch keeps the
source game's tokens[t0+1:], legal on the reference's board and on no other, and after an
illegal move no board exists for the engine to judge them on. Reading the argmax over the
whole 59 positions would be scoring the probe against labels nobody has. The legal control
branch of the same game at the same t0 supplies what the truncation costs: its hit rate is
the false-alarm rate of exactly the same statistic, matched on game, on t0 and on prefix.
"""

import numpy as np
import torch
from torch import Tensor

from spade.move_legality.fit_set import SINK
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_OF
from spade.violation_divergence.branches import T0_MAX, T0_MIN, role

# The window stops AT the event. Past t0 a branch replays the reference's suffix onto a board
# that no longer exists, so no move there has a legality label to plot against -- and the
# probe never saw such a prefix in training either (every fit-set row's tokens[:t] is a legal
# game), so its output there is extrapolation, not a judgement. r = -9 is what t0 >= 10
# allows once position 0, the attention sink, is dropped.
R_MAX = 0
WINDOW = tuple(range(SINK + 1 - T0_MIN, R_MAX + 1))
assert T0_MAX + R_MAX < 59, "the window leaves the board at the latest fork"

N_BOOT = 1000
BOOT_CHUNK = 100
RULES = ["R1", "R2", "R3"]


def roles(meta: dict) -> np.ndarray:
    """`branches.role` per row: reference, control_legal, control_legal_2, or a category."""
    return np.array([role(int(c), int(k))
                     for c, k in zip(meta["category"], meta["is_control"])])


def rules(meta: dict) -> np.ndarray:
    """R1 / R2 / R3 per row, and "" for the three legal branches."""
    return np.array(["" if int(c) == LEGAL else RULE_OF[CATEGORIES[int(c)]].split()[0]
                     for c in meta["category"]])


def peri_event(score: Tensor, t0: Tensor, window=WINDOW) -> Tensor:
    """[n, len(window)]: the score at t0 + r. Every r stays on the board by construction."""
    pos = t0.unsqueeze(1) + torch.tensor(list(window))
    assert pos.min() >= 0 and pos.max() < score.shape[1], "window leaves the sequence"
    return score.gather(1, pos)


def localization(score: Tensor, t0: Tensor) -> tuple[Tensor, Tensor]:
    """(hit, rank) per row over the ground-truth region t in [0, t0].

    hit is argmax == t0; rank is t0's 1-based rank by descending score, so 1 is a hit and
    t0 is last. Ties break toward the earlier position, which is the conservative side.

    The sink is excluded at both ends. Position 0 holds no board and the probe is not fit
    there, so its score is a per-space constant; leaving it in the argmax cost BatchTopK
    every single hit while its AUROC at t0 stayed at 0.99.
    """
    positions = torch.arange(score.shape[1]).unsqueeze(0)
    masked = score.masked_fill((positions > t0.unsqueeze(1)) | (positions <= SINK), -torch.inf)
    at_t0 = score.gather(1, t0.unsqueeze(1))
    rank = 1 + (masked > at_t0).sum(1)
    return masked.argmax(1) == t0, rank


def cluster_bootstrap(values: Tensor, game_id: Tensor, n_boot: int = N_BOOT,
                      seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Mean over rows and a 95% percentile CI resampling *games* with replacement.

    values [n, k] -> three [k] tensors. Resampling rows would treat a game's branches as
    independent and give an interval several times too tight.
    """
    ids, inverse = game_id.unique(return_inverse=True)
    sums = torch.zeros(len(ids), values.shape[1]).index_add_(0, inverse, values.float())
    counts = torch.bincount(inverse, minlength=len(ids)).float()
    generator = torch.Generator().manual_seed(seed)
    boot = torch.cat([
        (sums[draw].sum(1) / counts[draw].sum(1, keepdim=True))
        for draw in torch.randint(len(ids), (n_boot, len(ids)), generator=generator)
        .split(BOOT_CHUNK)])
    return values.float().mean(0), boot.quantile(0.025, dim=0), boot.quantile(0.975, dim=0)


def one_per_game(mask: Tensor, game_id: Tensor, seed: int = 0) -> Tensor:
    """One row index per game among `mask`, drawn uniformly at `seed`.

    A rule can have three member categories and therefore three branches in one game.
    Pooling all of them and calling the result a per-game statistic gives the rule ~1/sqrt(3)
    of an item's real spread (FIGURE_GUIDELINES 3); one draw per game is the estimator the
    control also has.
    """
    rows = mask.nonzero().flatten()
    draw = torch.rand(len(rows), generator=torch.Generator().manual_seed(seed),
                      dtype=torch.float64)
    order = rows[(game_id[rows].double() + draw).argsort()]
    first = torch.ones(len(order), dtype=torch.bool)
    first[1:] = game_id[order][1:] != game_id[order][:-1]
    return order[first]


def detection(score: Tensor, meta: dict, rows: Tensor, control_rows: Tensor,
              n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """AUROC at t0 of `rows` against `control_rows`, one aligned pair per game.

    Paired on the game: an illegal branch is scored against a legal move at the same
    position of the same prefix, so the contrast is the move and neither the game nor the
    move index. The bootstrap resamples games, which resamples both arms together.

    Every resample goes through the same tie-averaged `probe.auroc` as the point estimate.
    A batched rank trick is ~30x faster and was tried; it cannot average ties, and the
    no-probe baseline saturates 1 - P(played) at 1.0 on most illegal branches, so it is
    exactly the arm that would have been scored by a different estimator.
    """
    from spade.move_legality.probe import auroc

    at_t0 = score.gather(1, meta["t0"].unsqueeze(1)).squeeze(1)
    assert torch.equal(meta["game_id"][rows], meta["game_id"][control_rows]), "arms not paired"
    positive, negative = at_t0[rows], at_t0[control_rows]
    labels = torch.cat([torch.ones(len(rows)), torch.zeros(len(rows))])

    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(rows), (n_boot, len(rows)), generator=generator)
    boot = torch.tensor([auroc(torch.cat([positive[d], negative[d]]), labels) for d in draws])
    return {"auroc": auroc(torch.cat([positive, negative]), labels),
            "lo": float(boot.quantile(0.025)), "hi": float(boot.quantile(0.975)),
            "n_games": len(rows)}


def arms(meta: dict, seed: int = 0) -> dict[str, Tensor]:
    """One row per game per contrast: the reference, the legal control, the pooled illegal
    branch, and one branch per rule. The rule arms draw once per game (`one_per_game`), so
    every arm carries the same estimator as the control it is scored against."""
    role_of, rule_of, game = roles(meta), rules(meta), meta["game_id"]
    out = {"reference": torch.from_numpy(role_of == "reference").nonzero().flatten(),
           "control": torch.from_numpy(role_of == "control_legal").nonzero().flatten(),
           "control2": torch.from_numpy(role_of == "control_legal_2").nonzero().flatten(),
           "illegal": one_per_game((meta["category"] != LEGAL), game, seed)}
    out.update({r: one_per_game(torch.from_numpy(rule_of == r), game, seed) for r in RULES})
    for name, rows in out.items():
        assert len(rows) == len(game[rows].unique()), f"{name}: two rows from one game"
    return out


def matched_control(meta: dict, rows: Tensor, control: Tensor) -> Tensor:
    """`control` cut and ordered to the games `rows` covers. R3's single member category is
    absent from a few games, so the arms are not always the same 3,923."""
    index = torch.full((int(meta["game_id"].max()) + 1,), -1, dtype=torch.long)
    index[meta["game_id"][control]] = control
    matched = index[meta["game_id"][rows]]
    assert (matched >= 0).all(), "a game with an illegal branch has no legal control"
    return matched


def curve(score: Tensor, meta: dict, rows: Tensor, seed: int = 0) -> tuple:
    """Event-locked mean over `rows` with its 95% cluster-bootstrap band."""
    values = peri_event(score, meta["t0"], WINDOW)[rows]
    return cluster_bootstrap(values, meta["game_id"][rows], seed=seed)


def hit_rate(score: Tensor, meta: dict, rows: Tensor, seed: int = 0) -> dict:
    """Pooled localization rate with a cluster-bootstrap CI over games, and t0's median rank."""
    hit, rank = localization(score, meta["t0"])
    pooled = cluster_bootstrap(hit[rows].float().unsqueeze(1), meta["game_id"][rows], seed=seed)
    return {"pooled": tuple(float(v[0]) for v in pooled),
            "median_rank": float(rank[rows].float().median())}


def scalar_rows(space: str, kind: str, score: Tensor, meta: dict, selection: dict,
                names: list[str], seed: int = 0) -> list[dict]:
    """One tidy row per contrast: localization rate, t0's rank, and the paired AUROC.

    The control arm is scored against `control_legal_2`, the second legal branch every game
    carries. Pairing it with itself would print a tautological 0.5 where the reader needs an
    empirical legal-vs-legal null -- which is the reason that second control exists
    (violation_divergence.branches).
    """
    out = []
    for name in names:
        rows = selection[name]
        null = selection["control2" if name == "control" else "control"]
        hits = hit_rate(score, meta, rows, seed)
        paired = detection(score, meta, rows, matched_control(meta, rows, null), seed=seed)
        out.append({"space": space, "score": kind, "arm": name, "n_games": len(rows),
                    "hit_rate": round(hits["pooled"][0], 5),
                    "hit_lo": round(hits["pooled"][1], 5),
                    "hit_hi": round(hits["pooled"][2], 5),
                    "median_rank_of_t0": hits["median_rank"],
                    "auroc_at_t0": round(paired["auroc"], 5),
                    "auroc_lo": round(paired["lo"], 5),
                    "auroc_hi": round(paired["hi"], 5)})
    return out


def read(space: str, kind: str) -> Tensor:
    """The read-out tensor for a space: `p_illegal` [n, 59] or `proba4` [n, 59, 4]."""
    from spade.common.study import study_dir
    return torch.load(study_dir("move_legality") / "readout" / f"{kind}_{space}.pt",
                      map_location="cpu", weights_only=True)


def branches() -> dict:
    from spade.common.study import study_dir
    return torch.load(study_dir("divergence") / "branches" / "branches.pt",
                      map_location="cpu", weights_only=True)
