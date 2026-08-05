"""Which game should a three-rule fork figure be drawn on, judged in CosineSpaDE?

A figure whose violation condition is one game and whose curves are the three
constraint rules -- R1 placement (2 categories), R2 bracketing (3), R3 turn (1) --
wants the game whose three *rule* trajectories are furthest apart relative to the
spread of the members each one averages over. `score_games.py` answers the
category-level version of that question in raw layer 5; this answers the
rule-level version in the **CosineSpaDE latent space** (registry default,
gamma = 1.0), which is the space the figure is drawn in. Scored in the full 8192-d
latent space, never in a 2-D plane: the plane is refit per figure, so a criterion
computed in it selects the game that flatters one projection.

Two things the rule level changes, either of which returns a plausible wrong
answer if got backwards:

- **Per member, then average.** The encode is nonlinear, so a rule's mean is the
  mean of its members' *encoded* displacements, never the encoding of their mean
  activation. Same trap `test_curvature_plane.py` pins for phi.
- **R3 has one member**, so it contributes no within-rule spread. The pooled
  within-rule radius is estimated from R1 and R2 alone, on dof = 1 + 2 + 0 = 3.
  That is a finite number and not a statement about R3; a per-rule normalisation
  would divide by zero here.

## The criterion, fixed before the sweep ran

**rule_separation = B_min / W.** B_min is the *smallest* of the three distances
between rule-mean displacement trajectories; W is the pooled within-rule RMS
radius above. Three curves are legible only if the closest two are apart, so the
weakest pair decides -- a mean over the three pairs is passed by a game where R3
flies off and R1, R2 coincide. Reported alongside are its two scale-free halves
(B_min and W over the RMS rule-mean norm), the mean-pair version, the study's
`rule_groups.spread_ratio`, and the category-level criterion of `score_games.py`
recomputed in this space so the two selections can be compared.

**perm_p** and **null_draw** are the honest question -- whether the rules separate
at all here. The six illegal branches are re-dealt into groups of (2, 3, 1) in all
60 distinct ways, of which the true labelling is one; `perm_p` is the share scoring
at or above it (floor 1/60 = 0.017), and `null_draw` is `rule_separation` under one
of the 59 wrong labellings, drawn per game from `default_rng(game)`. `null_draw` is
the population an argmax over 1550 games has to beat, since it is one draw per game
exactly as the observed column is.

| key | definition | units |
|---|---|---|
| rule_separation | B_min / W | pooled within-rule SDs |
| between_min | B_min / RMS \\|mu_rule\\| | dimensionless |
| within_spread | W / RMS \\|mu_rule\\| | dimensionless |
| between_mean | mean over the 3 rule pairs / RMS \\|mu_rule\\| | [0, sqrt(2)] |
| r1_r2, r1_r3, r2_r3 | the pairs themselves, same normalisation | [0, sqrt(2)] |
| member_spread_ratio | within-rule / between-rule mean member distance | 1 = no grouping |
| category_differentiation | score_games.py's S3, in this space | [0, sqrt(2)] |

Displacements are D(t) = f(z_branch(t)) - f(z_ref(t)) over the study window at
t0's parity restricted to r >= 0, stacked per branch into one vector.

Usage: python analysis/violation_divergence/game_selection/score_rules_cosine_spade.py
"""

import csv
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.taxonomy import RULE_MEMBERS
from spade.saes.registry import seed42_entries
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.fork_plane import positions
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

SPACE = "CosineSpaDETrainer-othello"   # the registry default, gamma = 1.0
LAYER = 5
OUT_DIR = Path(__file__).resolve().parent / "results"
STEM = "rule_cosine_spade"             # sibling sweeps score other spaces
ANCHORS = {"FIGURE_GAME": FIGURE_GAME, "game_selection S3 argmax": 662}
RULE_SIZES = [len(m) for m in RULE_MEMBERS.values()]


@dataclass(frozen=True)
class Scores:
    game: int
    t0: int
    rule_separation: float
    between_min: float
    within_spread: float
    between_mean: float
    r1_r2: float
    r1_r3: float
    r2_r3: float
    member_spread_ratio: float
    perm_p: float
    null_draw: float
    category_differentiation: float


def relabelings(sizes: list[int]) -> list[list[list[int]]]:
    """Every distinct deal of the illegal branches into groups of `sizes` -- 60 of
    them for (2, 3, 1), one being the true rule labelling."""
    cut = np.cumsum([0, *sizes])
    deals = {tuple(tuple(sorted(p[a:b])) for a, b in zip(cut, cut[1:])): None
        for p in permutations(range(sum(sizes)))}
    return [[list(g) for g in d] for d in deals]


RELABELINGS = relabelings(RULE_SIZES)
TRUE_LABELS = [list(range(a, b)) for a, b in
               zip(np.cumsum([0, *RULE_SIZES]), np.cumsum(RULE_SIZES))]
WRONG_LABELS = [i for i in range(len(RELABELINGS)) if RELABELINGS[i] != TRUE_LABELS]


def rule_scores(G: np.ndarray, groups: list[list[int]]) -> tuple[np.ndarray, float, float]:
    """(the 3 rule-pair distances, W, RMS |mu|) from the illegal branches' Gram matrix.

    Everything the criterion needs is an inner product of the six stacked
    displacements, so the 60 relabelings cost one 6x6 matrix between them.
    """
    M = np.zeros((len(groups), len(G)))
    for k, g in enumerate(groups):
        M[k, g] = 1 / len(g)
    C = M @ G @ M.T
    norm2 = np.diag(C)
    iu = np.triu_indices(len(groups), 1)
    between = np.sqrt(np.maximum(norm2[:, None] + norm2[None, :] - 2 * C, 0)[iu])
    dof = sum(len(g) - 1 for g in groups)
    assert dof, "every rule has one member: within-rule spread is not estimable"
    ss = sum(G[b, b] - 2 * M[k] @ G[b] + C[k, k]
        for k, g in enumerate(groups) for b in g)
    return between, float(np.sqrt(ss / dof)), float(np.sqrt(norm2.mean()))


def score(F: np.ndarray, g: Game) -> Scores:
    """F is [9, n_positions, width] in one space, already restricted to t >= t0."""
    D = F - F[reference_row(g.rows)]
    groups = rule_rows(g.rows)
    assert [len(v) for v in groups.values()] == RULE_SIZES, \
        f"game {g.game}: rule sizes {[len(v) for v in groups.values()]}, not branch-complete"
    X = D[[i for v in groups.values() for i in v]].reshape(sum(RULE_SIZES), -1)
    G = X @ X.T

    pairs, w, rms = rule_scores(G, TRUE_LABELS)
    observed = pairs.min() / w
    null = np.array([d_pairs.min() / d_w for d_pairs, d_w, _ in
        (rule_scores(G, d) for d in RELABELINGS)])
    norm2 = np.diag(G)
    member = np.sqrt(np.maximum(norm2[:, None] + norm2[None, :] - 2 * G, 0))
    iu = np.triu_indices(len(G), 1)
    return Scores(g.game, g.t0, observed, pairs.min() / rms, w / rms,
                  pairs.mean() / rms, *(pairs / rms).tolist(),
                  spread_ratio(D, g.rows), float((null >= observed).mean()),
                  float(null[np.random.default_rng(g.game).choice(WRONG_LABELS)]),
                  float(member[iu].mean() / np.sqrt(norm2.mean())))


def features(acts: torch.Tensor, g: Game, enc, device: str) -> np.ndarray:
    pos = positions(g.t0)
    Z = acts[g.idx][:, pos[pos >= g.t0], :].float()
    if enc is None:
        return Z.numpy()
    with torch.no_grad():
        n, T, d = Z.shape
        return enc(Z.reshape(-1, d).to(device)).reshape(n, T, -1).cpu().numpy()


def sweep(meta: dict, acts: torch.Tensor, games: list[int], enc, device: str
          ) -> list[Scores]:
    out = []
    for i, game in enumerate(games, 1):
        g = load_game(meta, game)
        out.append(score(features(acts, g, enc, device), g))
        if i % 250 == 0:
            print(f"    {i}/{len(games)} games")
    return out


def ranks(values: list[float]) -> list[int]:
    """1-based descending rank, in the order the values came in."""
    out = [0] * len(values)
    for rank, i in enumerate(sorted(range(len(values)), key=lambda i: -values[i]), 1):
        out[i] = rank
    return out


def rows(cosine: list[Scores], raw: list[Scores]) -> list[dict]:
    """One row per game, ranked in CosineSpaDE, carrying raw layer 5 as reference."""
    rank, raw_rank = (ranks([s.rule_separation for s in cosine]),
                      ranks([s.rule_separation for s in raw]))
    out = [{**asdict(c), "rank": rank[i],
            "raw_l5_rule_separation": raw[i].rule_separation,
            "raw_l5_rank": raw_rank[i]} for i, c in enumerate(cosine)]
    return sorted(out, key=lambda r: r["rank"])


def write_csv(path: Path, table: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return path


KEYS = ["game", "t0", "rule_separation", "between_min", "within_spread",
        "r1_r2", "r1_r3", "r2_r3", "member_spread_ratio", "perm_p",
        "category_differentiation", "rank", "raw_l5_rank"]
POPULATION = [k for k in KEYS if k not in ("game", "t0", "rank", "raw_l5_rank")] + ["null_draw"]
WIDTH = {"game": 5, "t0": 3, "rank": 5, "raw_l5_rank": 7}


def table(table_rows: list[dict], n: int) -> str:
    head = " ".join(f"{k:>{WIDTH.get(k, 12)}}" for k in KEYS)
    body = [" ".join(f"{r[k]:>{WIDTH[k]}}" if k in WIDTH else f"{r[k]:>12.4f}"
        for k in KEYS) for r in table_rows[:n]]
    return "\n".join([head, "-" * len(head), *body])


def summary(table_rows: list[dict], cosine: list[Scores], raw: list[Scores]) -> str:
    n = len(table_rows)
    by_game = {r["game"]: r for r in table_rows}
    p = np.array([r["perm_p"] for r in table_rows])
    lines = [
        f"n = {n} branch-complete games, scored in {SPACE} (gamma=1.0), full latent width,",
        f"layer {LAYER} displacements over r >= 0 at t0's parity. Criterion: argmax "
        f"rule_separation = B_min / W.",
        "",
        "SELECTED " + table([table_rows[0]], 1).splitlines()[-1],
        "",
        "anchors:",
    ]
    for name, game in ANCHORS.items():
        lines.append(f"  {name:<24} " + table([by_game[game]], 1).splitlines()[-1])
    lines += ["", "top 20 by rule_separation:", table(table_rows, 20), ""]

    lines.append("population spread over all complete games:")
    for key in POPULATION:
        v = np.array([r[key] for r in table_rows])
        lines.append(f"  {key:<26} min {v.min():.4f}  q25 {np.percentile(v, 25):.4f}  "
                     f"median {np.median(v):.4f}  q75 {np.percentile(v, 75):.4f}  "
                     f"max {v.max():.4f}")

    lines += ["", f"label-permutation null ({len(RELABELINGS)} distinct deals of the six "
                  f"illegal branches into (2, 3, 1); floor p = {1 / len(RELABELINGS):.3f}):",
              f"  median p {np.median(p):.3f} (0.508 if the rule labels carry nothing)",
              f"  p <= 0.05 in {(p <= 0.05).mean():.1%} of games (5.0% under the null); "
              f"p at the floor in {(p <= 1 / len(RELABELINGS) + 1e-9).mean():.1%} "
              f"({1 / len(RELABELINGS):.1%} under the null)",
              f"  members travel closer to their own rule than to another in "
              f"{(np.array([r['member_spread_ratio'] for r in table_rows]) < 1).mean():.1%} of games",
              ""]

    obs = np.array([r["rule_separation"] for r in table_rows])
    null = np.array([r["null_draw"] for r in table_rows])
    lines += [
        "the argmax against a matched null (one wrong labelling per game, so the two "
        "columns are\nthe same 1550 draws and their maxima are comparable):",
        f"  observed   median {np.median(obs):.4f}  q75 {np.percentile(obs, 75):.4f}  "
        f"max {obs.max():.4f} (game {table_rows[0]['game']})",
        f"  wrong labels median {np.median(null):.4f}  q75 {np.percentile(null, 75):.4f}  "
        f"max {null.max():.4f} (game {table_rows[int(np.argmax(null))]['game']})",
        f"  the selected game beats {(null < obs.max()).mean():.2%} of the null draws; "
        f"{(null >= obs.max()).sum()} of {len(null)} wrong labellings reach it",
        ""]

    cos_sep = [s.rule_separation for s in cosine]
    lines += [
        "rank agreement (Spearman over all complete games):",
        f"  rule_separation, CosineSpaDE vs raw layer 5   "
        f"{spearmanr(cos_sep, [s.rule_separation for s in raw]).statistic:.3f}",
        f"  rule_separation vs category_differentiation   "
        f"{spearmanr(cos_sep, [s.category_differentiation for s in cosine]).statistic:.3f}"
        "   (both in CosineSpaDE)",
        f"  rule_separation vs member_spread_ratio        "
        f"{spearmanr(cos_sep, [s.member_spread_ratio for s in cosine]).statistic:.3f}",
        "",
    ]
    t0s = np.array([r["t0"] for r in table_rows])
    lines.append(f"t0: min {t0s.min()}, median {int(np.median(t0s))}, max {t0s.max()}; "
                 f"selected t0={table_rows[0]['t0']} at percentile "
                 f"{100 * (t0s <= table_rows[0]['t0']).mean():.1f}.")
    return "\n".join(lines) + "\n"


def main():
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / f"resid_post_{LAYER}.pt", weights_only=True)
    games = complete_games(meta)
    print(f"{len(games)} complete games, device {device}")

    entry = next(e for e in seed42_entries() if e.config == SPACE)
    print(f"  scoring in {SPACE} from {entry.ckpt_dir}")
    cosine = sweep(meta, acts, games, sae_encoder(entry, device), device)
    print("  scoring in raw layer 5 (reference)")
    raw = sweep(meta, acts, games, None, device)

    table_rows = rows(cosine, raw)
    print(f"wrote {write_csv(OUT_DIR / f'{STEM}_ranking.csv', table_rows)}")
    text = summary(table_rows, cosine, raw)
    (OUT_DIR / f"{STEM}_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
