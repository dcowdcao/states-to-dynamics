"""Which game should a three-rule fork figure be drawn on, in SpaDE latents?

`score_games.py` picked game 1235 by the mutual separation of its six *category*
trajectories in raw layer 5. A figure that draws one mean per constraint rule --
R1 placement (2 members), R2 bracketing (3), R3 turn (1) -- asks a different
question, and asks it in the space the paper argues for, so the ranking is
recomputed here rather than inherited.

Two things the rule level forces that the category level did not:

- **A rule mean is a mean over members, taken last.** Every per-branch quantity
  below -- a member's displacement trajectory, its norm, its deviation from its
  rule's mean -- is formed per member and only then averaged. The displacement
  field is linear, so the mean itself is order-free; `test_curvature_plane.py`
  pins the case where it is not.
- **R3 has one member, so it has no within-rule spread.** The denominator is a
  *pooled* within-rule RMS radius with dof = sum(n_r - 1) = 3. R3 contributes
  zero to the sum and zero to the dof, which is an explicit weight of zero and
  not a silent inf.

## The criterion, fixed before any ranking was looked at

**sep_min: the closest pair of rule means, in pooled within-rule SDs.** A reader
has to tell all three lines apart, so the bottleneck pair is what "legible"
means for this figure; `sep_mean` would hide a collapsed pair behind a distant
third. It is a ratio, so it survives the per-game plane refit that normalises
magnitude out of the drawing. `cone_*` is the same numerator over the RMS rule-mean
norm -- direction only, the rule-level analogue of `score_games.py`'s S3, in the
same [0, sqrt(2)] units -- and is reported, not used to decide.

**This criterion is not independent of a claim the study makes**, unlike S3.
`rule_groups.spread_ratio` (within-rule over between-rule mean pairwise distance,
median 0.945 over these same games) is a reported result, and argmax sep_min
leans towards argmin spread_ratio. The correlation is measured and the selected
game's spread_ratio percentile is printed so a caption can disclose it.

**The null is exact.** Six illegal branches split 2/3/1 admits 60 partitions;
`null_beats` is the fraction of the other 59 that the true R1/R2/R3 grouping
beats on sep_min. Near 0.5 the taxonomy is carrying nothing and the three lines
are only as separated as any arbitrary 2/3/1 split of the same six trajectories.

Usage: python analysis/violation_divergence/game_selection/score_rules.py
"""

import csv
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.taxonomy import RULE_MEMBERS
from spade.saes.registry import seed42_entries
from spade.violation_divergence.branches import (
    FIGURE_GAME, complete_games, load_game, reference_row)
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.fork_plane import RAW_SPACE, positions
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

OUT_DIR = Path(__file__).resolve().parent / "results"
PRIOR_RANKING = OUT_DIR / "ranking.csv"   # score_games.py's raw-L5 category sweep
# The seed-42 distance-SpaDE checkpoint every study uses: SpaDETrainer-othello is
# the degenerate gamma=1e-2 one, excluded repo-wide in spade.spaces.
SPADE_CONFIG = "SpaDE-lam1e-03-gam1e-04"
SPADE_SPACE = "SpaDE (lam=1e-3, gam=1e-4)"
PRIOR_PICK = 662          # score_games.py's S3 argmax, before the criterion changed
CRITERIA = ["sep_min", "sep_mean", "cone_min", "cone_mean"]
RECOMMENDED = "sep_min"

RULES = list(RULE_MEMBERS)
SIZES = [len(m) for m in RULE_MEMBERS.values()]
DOF = sum(n - 1 for n in SIZES)
PAIRS = [f"{RULES[i][:2]}_{RULES[j][:2]}" for i, j in zip(*np.triu_indices(len(RULES), 1))]


def blocks(sizes: list[int]) -> list[list[int]]:
    ends = np.cumsum(sizes)
    return [list(range(e - n, e)) for e, n in zip(ends, sizes)]


def partitions(sizes: list[int]) -> list[list[list[int]]]:
    """Every split of range(sum(sizes)) into blocks of exactly these sizes."""
    out = []
    for a in itertools.combinations(range(sum(sizes)), sizes[0]):
        rest = [i for i in range(sum(sizes)) if i not in a]
        for b in itertools.combinations(rest, sizes[1]):
            out.append([list(a), list(b), [i for i in rest if i not in b]])
    return out


TRUE_BLOCKS = blocks(SIZES)
NULL_BLOCKS = [p for p in partitions(SIZES) if p != TRUE_BLOCKS]


@dataclass(frozen=True)
class Scores:
    game: int
    t0: int
    sep_min: float
    sep_mean: float
    cone_min: float
    cone_mean: float
    sep_R1_R2: float
    sep_R1_R3: float
    sep_R2_R3: float
    spread_ratio: float
    null_beats: float
    bottleneck: str


def separations(G: np.ndarray, parts: list[list[int]]) -> tuple[np.ndarray, float, float]:
    """(pairwise distances between block means, pooled within-block RMS radius,
    RMS block-mean norm), from the branches' Gram matrix alone -- which is what
    makes the 60-partition null free."""
    W = np.zeros((len(parts), len(G)))
    for k, part in enumerate(parts):
        W[k, part] = 1 / len(part)
    M = W @ G @ W.T
    d2 = np.diag(M)[:, None] + np.diag(M)[None, :] - 2 * M
    within = sum(G[p, p].sum() - len(p) * M[k, k] for k, p in enumerate(parts))
    return (np.sqrt(np.maximum(d2[np.triu_indices(len(parts), 1)], 0)),
            np.sqrt(max(within, 0) / DOF), np.sqrt(np.diag(M).mean()))


def criterion(G: np.ndarray, parts: list[list[int]]) -> float:
    d, pooled, _ = separations(G, parts)
    return float(d.min() / pooled)


def score(F: np.ndarray, rows: list[dict], game: int, t0: int) -> Scores:
    """F is [9, 13, width] in one space; the 13 positions are r = 0, 2, ..., 24."""
    groups = rule_rows(rows)
    assert [len(g) for g in groups.values()] == SIZES, f"game {game}: {groups}"
    D = F - F[reference_row(rows)]
    idx = [i for g in groups.values() for i in g]
    X = D[idx].reshape(len(idx), -1).astype(np.float64)
    G = X @ X.T

    d, pooled, rms = separations(G, TRUE_BLOCKS)
    true = float(d.min() / pooled)
    beats = float(np.mean([true > criterion(G, p) for p in NULL_BLOCKS]))
    return Scores(game, t0, true, float(d.mean() / pooled), float(d.min() / rms),
                  float(d.mean() / rms), *(float(v / pooled) for v in d),
                  spread_ratio(D, rows), beats, PAIRS[int(d.argmin())])


def features(acts: torch.Tensor, g, enc, device: str) -> np.ndarray:
    pos = positions(g.t0)
    Z = acts[g.idx][:, pos[pos >= g.t0], :].float().numpy()
    if enc is None:
        return Z
    n, T, d = Z.shape
    with torch.no_grad():
        f = enc(torch.tensor(Z.reshape(-1, d), device=device))
    return f.reshape(n, T, -1).cpu().numpy()


def sweep(meta: dict, acts: torch.Tensor, games: list[int], enc, device: str) -> list[Scores]:
    out = []
    for n, game in enumerate(games, 1):
        g = load_game(meta, game)
        out.append(score(features(acts, g, enc, device), g.rows, game, g.t0))
        if n % 250 == 0:
            print(f"  {n}/{len(games)}", flush=True)
    return out


def ranked(scores: list[Scores]) -> list[dict]:
    rows = [asdict(s) for s in scores]
    for key in CRITERIA:
        for rank, i in enumerate(sorted(range(len(rows)), key=lambda i: -rows[i][key]), 1):
            rows[i][f"rank_{key}"] = rank
    return sorted(rows, key=lambda r: r[f"rank_{RECOMMENDED}"])


def prior_s1() -> dict[int, tuple[float, int]]:
    """game -> (S1 illegal-vs-control, its rank among the 1550), from score_games.py.

    A disclosure column, never a selector. `FIGURE_GAME` was picked under an
    at-or-above-median-S1 guard, because argmax separation alone landed on a game
    at the 5.8th percentile of the study's own scientific contrast; a rule-level
    argmax that quietly fails the same guard would reintroduce that problem.
    """
    with PRIOR_RANKING.open() as f:
        return {int(r["game"]): (float(r["s1_illegal_vs_control"]), int(r["rank_s1"]))
                for r in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def anchor(rows: list[dict], game: int, label: str) -> str:
    r = next(x for x in rows if x["game"] == game)
    n = len(rows)
    return (f"{label} game {game}, t0={r['t0']}  "
            + "  ".join(f"{k}={r[k]:.4f} (rank {r['rank_' + k]}/{n})" for k in CRITERIA)
            + f"  spread_ratio={r['spread_ratio']:.4f}  null_beats={r['null_beats']:.2f}"
            + f"  S1={r['s1']:.4f} (rank {r['rank_s1']}/{n})")


def table(rows: list[dict], n: int) -> str:
    keys = ["game", "t0", *CRITERIA, "spread_ratio", "null_beats", "bottleneck", "rank_s1"]
    head = " | ".join(f"{k:>13}" for k in keys)
    body = [" | ".join(f"{r[k]:>13.4f}" if isinstance(r[k], float) else f"{r[k]:>13}"
                       for k in keys) for r in rows[:n]]
    return "\n".join([head, "-" * len(head), *body])


def space_report(space: str, rows: list[dict], width: int) -> str:
    n = len(rows)
    pick, sr = rows[0], np.array([r["spread_ratio"] for r in rows])
    beats = np.array([r["null_beats"] for r in rows])
    lines = [
        f"=== {space} (width {width}) ===",
        anchor(rows, pick["game"], "SELECTED  "),
        anchor(rows, FIGURE_GAME, "FIGURE_GAME"),
        anchor(rows, PRIOR_PICK, "PRIOR PICK"),
        "",
        f"selected game's spread_ratio {pick['spread_ratio']:.4f} is at percentile "
        f"{100 * (sr < pick['spread_ratio']).mean():.1f} (lower is a tighter grouping; "
        f"population median {np.median(sr):.4f}, above 1 in {(sr > 1).mean():.0%})",
        f"selected game's S1 rank {pick['rank_s1']}/{n} (score_games.py, raw layer 5): "
        f"{'passes' if pick['rank_s1'] <= n // 2 else 'FAILS'} the at-or-above-median-S1 "
        "guard FIGURE_GAME was picked under",
        f"Spearman(sep_min, spread_ratio) = "
        f"{spearmanr([r['sep_min'] for r in rows], sr).statistic:.3f}  "
        "-- how far maximising the criterion also selects the grouping claim",
        "",
        f"null (60 shape-matched 2/3/1 partitions of the same six branches, exact): "
        f"the true R1/R2/R3 grouping beats a median of {np.median(beats):.2f} of the "
        f"other 59; it is the argmax in {(beats == 1).mean():.1%} of games and below "
        f"chance in {(beats < 0.5).mean():.1%}",
        "",
        "bottleneck rule pair (which two means are closest):",
    ]
    for pair in PAIRS:
        share = np.mean([r["bottleneck"] == pair for r in rows])
        v = np.array([r[f"sep_{pair}"] for r in rows])
        lines.append(f"  {pair:<8} closest in {share:>6.1%} of games; "
                     f"separation median {np.median(v):.3f} pooled SDs "
                     f"[IQR {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}]")
    lines += ["", "population spread over all complete games:"]
    for key in CRITERIA + ["spread_ratio", "null_beats"]:
        v = np.array([r[key] for r in rows])
        lines.append(f"  {key:<14} min {v.min():.4f}  q25 {np.percentile(v, 25):.4f}  "
                     f"median {np.median(v):.4f}  q75 {np.percentile(v, 75):.4f}  "
                     f"max {v.max():.4f}")
    return "\n".join(lines + ["", f"top 10 by {RECOMMENDED}:", table(rows, 10)])


def summary(by_space: dict[str, list[dict]], widths: dict[str, int]) -> str:
    n = len(by_space[SPADE_SPACE])
    head = [
        f"n = {n} branch-complete games. Field: D(t) = f_branch(t) - f_ref(t) over the "
        "13 positions r = 0, 2, ..., 24 at t0's parity, at full width, never a projection.",
        f"Recommendation space: {SPADE_SPACE}. Reference space: {RAW_SPACE}.",
        f"Rule means: {', '.join(f'{r} (n={s})' for r, s in zip(RULES, SIZES))}; each mean is "
        f"formed per member and averaged last. Pooled within-rule RMS radius has dof {DOF} "
        "(R3 contributes no within-rule term and no dof).",
        f"Criterion: {RECOMMENDED} = min over rule pairs ||M_i - M_j|| / pooled within-rule "
        "RMS radius, in pooled SDs. cone_* is the same numerator over RMS ||M_r||, in [0, sqrt(2)].",
        "",
    ]
    spade = by_space[SPADE_SPACE]
    raw = {r["game"]: r for r in by_space[RAW_SPACE]}
    order = [r["game"] for r in spade]
    rho = spearmanr([r["sep_min"] for r in spade],
                    [raw[g]["sep_min"] for g in order]).statistic
    tail = [
        "",
        f"space agreement: Spearman(sep_min in SpaDE, sep_min in raw layer 5) = {rho:.3f} "
        f"over all {n} games.",
        f"SpaDE's pick ranks {raw[spade[0]['game']]['rank_sep_min']}/{n} in raw layer 5.",
    ]
    return "\n".join(head + [space_report(s, rows, widths[s]) + "\n"
                             for s, rows in by_space.items()] + tail) + "\n"


def main():
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)
    games = complete_games(meta)
    print(f"{len(games)} complete games on {device}")

    entry = next(e for e in seed42_entries() if e.config == SPADE_CONFIG)
    encoders = {SPADE_SPACE: sae_encoder(entry, device), RAW_SPACE: None}
    s1 = prior_s1()
    by_space, widths = {}, {}
    for space, enc in encoders.items():
        print(f"scoring in {space}", flush=True)
        widths[space] = features(acts, load_game(meta, games[0]), enc, device).shape[-1]
        rows = ranked(sweep(meta, acts, games, enc, device))
        for r in rows:
            r["s1"], r["rank_s1"] = s1[r["game"]]
        by_space[space] = rows

    write_csv(OUT_DIR / "rule_ranking.csv",
              [{"space": s, **r} for s, rows in by_space.items() for r in rows])
    text = summary(by_space, widths)
    (OUT_DIR / "rule_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
