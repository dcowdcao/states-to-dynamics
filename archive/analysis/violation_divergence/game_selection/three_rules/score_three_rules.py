"""Which game can show all three constraint rules, in every space, on a legible board?

`score_games.py` asks which game separates the six illegal *categories* best, and
answers it in raw layer 5 alone. The figure being built asks something narrower and
wider at once: one game, drawn in all nine spaces, whose three *rule* means (R1
placement n=2, R2 bracketing n=3, R3 turn n=1) a reader can tell apart -- in every
cell of the grid, not in the best one -- and whose violating squares are legible on
the 8x8 board panel beside it. Three questions, in the order they bind:

1. **Feasibility.** A game can only show a rule it has a branch for. Reported as
   counts, not assumed.
2. **Cross-space agreement.** A game that separates the rules in one space and not
   in another is a bad pick for a figure whose point is comparing spaces, however
   high its peak. Candidates are ranked on their WORST space, not their best.
   Raw layer 5 costs no encoder, so it scores every complete game and nominates
   the candidates; those plus a random reference sample (the percentiles' null,
   and the check that raw-L5 nomination missed nothing) go through all nine.
3. **Board legibility.** Six squares that coincide, touch, cluster in a corner, or
   sit away from every disc do not read at the board panel's printed size
   (3.9 in / 8 cells = 35 pt per cell, `fork_and_board`).

## Definitions

Displacements D(t) = z_b(t) - z_ref(t) over the 13 sampled positions at r >= 0,
stacked per branch into X_b, in each space at full width -- never in a 2-D
projection, which is refit per panel. Per space:

| key | definition | units |
|---|---|---|
| sep | mean pairwise distance between the 3 rule means / RMS rule-mean norm | [0, sqrt(2)] |
| ratio | `rule_groups.spread_ratio`: within- over between-rule mean pairwise distance | dimensionless |

sep is what the figure asks a reader to see: three curves, apart. ratio is the
study's existing coherence statistic (median 0.945 over the 1550 games) and says
whether the mean is a fair summary of its members. Ranking is on sep.

**The rule mean is taken per member.** Here the field is the displacement, which is
linear, so member-then-mean and mean-then-member agree. In the R^21 curvature field
they do NOT -- phi is nonlinear -- and `check_r21_rules.py` is that field, with phi
applied per member. **R3 has one member**, so its within-rule spread is exactly 0
by construction, not NaN: a single member is its own mean. It contributes no pair
to `ratio`'s within-rule set, which is therefore R1's 1 pair and R2's 3.

Nothing here holds the population in more than one space at a time: the board pass
and the raw-L5 pass read one game out of the mmapped activations, keep its scalars
and drop it, and only the pool is stacked for the eight SAE spaces. The all-games
version needed more than a login node's 8 GB and was killed without printing.

Usage: python analysis/violation_divergence/game_selection/three_rules/score_three_rules.py
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.boards import replay
from spade.othello.taxonomy import CATEGORIES, RULE_MEMBERS, RULE_OF
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.curvature_plane import W
from spade.violation_divergence.fork_plane import positions, spaces
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

OUT_DIR = Path(__file__).resolve().parent / "results"
CHUNK = 16                # games encoded per call; 16*9*13 rows x 8192 codes is 61 MB
BOARD_CELL_PT = 35.1      # 3.9 in board / 8 cells, at 72 pt/in (fork_and_board)
POPULATION, POP_SEED = 300, 0   # the reference sample percentiles are taken against
POOL_TOP = 120            # raw-L5 candidates carried into the eight SAE spaces
LEGIBILITY = {"min_rule_gap": 1, "spread": 1, "min_context": 1, "detached_gap": -1}
INCUMBENT = FIGURE_GAME

# Every game any search has nominated, forced into the pool and printed as one
# table, so the searches are compared on one basis instead of each quoting its own.
CANDIDATES = {
    1558: "SpaDE argmax (spread_ratio 0.1st pct)",
    191: "CosineSpaDE argmax",
    659: "high in SpaDE and CosineSpaDE",
    1442: "SpaDE runner-up",
    706: "SpaDE robust alternative",
    1202: "SpaDE robust alternative",
    INCUMBENT: "incumbent FIGURE_GAME",
    662: "parent sub-study's pick",
}


@dataclass(frozen=True)
class Board:
    """The board panel's geometry at t0, in cells. Distances are Chebyshev: the
    panel draws 1x1 cells, so king-moves are what "next to" means on it."""
    min_rule_gap: int     # nearest pair of squares belonging to different rules
    n_touching: int       # different-rule pairs one cell apart
    n_glyph_pairs: int    # pairs offset (+-1, +-1) with dx == dy: a would-be disc
                          # and the next square's marker land 0.4 cells apart
    min_gap_to_null: int  # nearest illegal square to the reference or the control
    spread: float         # RMS distance of the 6 squares from their centroid
    quadrants: int        # of the four 4x4 board quadrants, how many are used
    min_context: int      # occupied 8-neighbours of the emptiest branch square,
                          # skipping empty_detached, whose 0 is its definition
    detached_gap: int     # empty_detached square to the nearest disc: how far off
                          # the one square with no context by construction sits


def separation(X: np.ndarray, groups: list[list[int]]) -> float:
    """Mean pairwise distance between the rule means, over their RMS norm.

    sqrt(2) is three equal-length mutually orthogonal rule means; 0 is three rules
    that displace identically. Scale-free, because the panel's plane is refit per
    space and only directions survive into the drawing.
    """
    M = np.stack([X[g].mean(0) for g in groups])
    return float(pdist(M).mean() / np.sqrt((M ** 2).sum(1).mean()))


def score_field(D: np.ndarray, rows: list[dict]) -> tuple[float, float]:
    """(sep, ratio) from a [n_branches, ..., dim] field already referenced to the
    played move. The caller owns the field, so the R^21 check can pass its own."""
    groups = rule_rows(rows)
    assert len(groups) == len(RULE_MEMBERS), f"{len(groups)} rules, not 3"
    idx = [i for g in groups.values() for i in g]
    X = D.reshape(len(D), -1)
    remap = {i: k for k, i in enumerate(idx)}
    return (separation(X[idx], [[remap[i] for i in g] for g in groups.values()]),
            spread_ratio(D, rows))


def score_space(F: np.ndarray, rows: list[dict], ref: int) -> tuple[float, float]:
    """(sep, ratio) for one game in one space. F is [9, n_pos, width]."""
    return score_field(F - F[ref], rows)


def within_rule_spread(F: np.ndarray, rows: list[dict], ref: int) -> dict[str, float]:
    """RMS distance from each rule's members to that rule's mean. R3 has one
    member, so it is exactly 0 -- the mean IS the member -- and never NaN."""
    X = (F - F[ref]).reshape(len(F), -1)
    out = {}
    for rule, members in rule_rows(rows).items():
        M = X[members]
        out[rule] = float(np.sqrt(((M - M.mean(0)) ** 2).sum(1).mean()))
    return out


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def board_geometry(board, rows: list[dict]) -> Board:
    """Where the branch squares land on the panel `branch_panels.board_panel` draws."""
    xy = {i: (r["square"] % 8, r["square"] // 8) for i, r in enumerate(rows)}
    illegal = [i for i, r in enumerate(rows) if r["category"] != 0]
    # control 2 is not drawn: fork_and_board keeps the reference and one control.
    nulls = [i for i, r in enumerate(rows)
             if r["category"] == 0 and r["is_control"] != 2]
    rule = {i: RULE_OF[CATEGORIES[rows[i]["category"]]] for i in illegal}
    pairs = [(i, j) for k, i in enumerate(illegal) for j in illegal[k + 1:]]

    cross = [chebyshev(xy[i], xy[j]) for i, j in pairs if rule[i] != rule[j]]
    P = np.array([xy[i] for i in illegal], float)
    quad = {(int(x) // 4, int(y) // 4) for x, y in P}
    discs = [(sq % 8, sq // 8) for sq in np.flatnonzero(board.state.flatten() != 0)]
    detached = next(i for i, r in enumerate(rows)
                    if CATEGORIES[r["category"]] == "empty_detached")
    context = [sum(chebyshev(xy[i], d) == 1 for d in discs)
               for i in illegal if i != detached]
    return Board(
        min_rule_gap=min(cross),
        n_touching=sum(d == 1 for d in cross),
        n_glyph_pairs=sum(1 for i, j in pairs
                          if xy[i][0] - xy[j][0] == xy[i][1] - xy[j][1]
                          and abs(xy[i][0] - xy[j][0]) == 1),
        min_gap_to_null=min(chebyshev(xy[i], xy[j]) for i in illegal for j in nulls),
        spread=float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())),
        quadrants=len(quad),
        min_context=min(context),
        detached_gap=min(chebyshev(xy[detached], d) for d in discs))


def branch_block(acts: torch.Tensor, g: Game) -> torch.Tensor:
    """One game's [9, 13, 512] at r >= 0, read out of the mmapped act tensor."""
    pos = positions(g.t0)
    return acts[g.idx][:, pos[pos >= g.t0], :].float()


def sweep(Z: torch.Tensor, rows_per_game: list[list[dict]], refs: list[int],
          device: str) -> np.ndarray:
    """[n_games, n_spaces, 2] of (sep, ratio), one encoder loaded at a time."""
    out = np.zeros((len(Z), len(spaces()), 2))
    for j, (label, entry) in enumerate(spaces()):
        enc = None if entry is None else sae_encoder(entry, device)
        for lo in range(0, len(Z), CHUNK):
            block = Z[lo:lo + CHUNK]
            if enc is None:
                F = block.numpy()
            else:
                with torch.no_grad():
                    f = enc(block.reshape(-1, block.shape[-1]).to(device))
                F = f.reshape(block.shape[:3] + (-1,)).cpu().numpy()
            for i in range(len(F)):
                out[lo + i, j] = score_space(F[i], rows_per_game[lo + i], refs[lo + i])
        print(f"  {label:<32} sep median {np.median(out[:, j, 0]):.4f}")
    return out


def percentiles(v: np.ndarray) -> np.ndarray:
    return 100 * (rankdata(v) - 1) / (len(v) - 1)


def legibility(rows: list[dict]) -> np.ndarray:
    """Mean percentile over LEGIBILITY's four board terms, the -1 ones reversed."""
    return np.mean([percentiles(np.array([r[key] for r in rows], float)) if sign > 0
                    else 100 - percentiles(np.array([r[key] for r in rows], float))
                    for key, sign in LEGIBILITY.items()], axis=0)


def feasibility(meta: dict) -> str:
    """Every game's rule availability, re-derived rather than assumed."""
    gid, cat, ctl = meta["game_id"].numpy(), meta["category"].numpy(), meta["is_control"].numpy()
    t0 = meta["t0"].numpy()
    games = np.unique(gid)
    per_game = {int(g): {CATEGORIES[c] for c, k in zip(cat[gid == g], ctl[gid == g])
                         if k == 0 and c != 0} for g in games}
    n_rules = {g: len({RULE_OF[c] for c in cats}) for g, cats in per_game.items()}
    complete = [g for g, cats in per_game.items() if len(cats) == len(CATEGORIES) - 1]
    t0_of = {int(g): int(t0[gid == g][0]) for g in games}
    tc = np.array([t0_of[g] for g in complete])
    lines = [
        f"{len(games)} source games, {len(complete)} branch-complete (all six illegal "
        "categories).",
        "Branch-completeness IS rule-completeness: a complete game carries all six "
        "categories, so",
        f"  all three rules are available in {sum(1 for g in complete if n_rules[g] == 3)}"
        f"/{len(complete)} of them.",
        f"Relaxing to one member per rule: {sum(1 for v in n_rules.values() if v == 3)}"
        f"/{len(games)} games admit R1, R2 and R3.",
        "Missing-category counts over all source games:",
    ]
    for c in CATEGORIES[1:]:
        n = sum(1 for cats in per_game.values() if c not in cats)
        lines.append(f"  {c:<18} ({RULE_OF[c]:<14}) missing in {n:>4} / {len(games)}")
    # An R^21 panel needs one full-width descriptor window that ends before the
    # branch; curvature_plane.usable is that test, evaluated here rather than
    # quoted as its docstring's t0 >= 18, which is one off for odd t0.
    r21 = [g for g in complete if positions(t0_of[g], W - 1)[W - 1] < t0_of[g]]
    lines += [
        f"t0 over the complete games: min {tc.min()}, median {int(np.median(tc))}, "
        f"max {tc.max()}.",
        f"An R^21 curvature panel needs a branch-free W={W} window: {len(r21)}"
        f"/{len(complete)} complete games qualify (t0 >= {min(t0_of[g] for g in r21)}).",
    ]
    return "\n".join(lines)


def summary(rows: list[dict], board_rows: list[dict], scored: np.ndarray, pop: list[int],
            feasible: str, within: dict[str, float]) -> str:
    """`rows` are the pool games, `board_rows` all complete games; `scored` is the
    pool's [n, n_spaces, 2] and `pop` indexes the reference sample inside it."""
    labels = [label for label, _ in spaces()]
    n, nb = len(rows), len(board_rows)
    by_game = {r["game"]: r for r in rows}

    def table(order: list[dict], keys: list[str], k: int = 12) -> str:
        head = " | ".join(f"{key:>14}" for key in keys)
        out = [head, "-" * len(head)]
        for r in order[:k]:
            out.append(" | ".join(
                f"{r[key]:>14.4f}" if isinstance(r[key], float) else f"{r[key]:>14}"
                for key in keys))
        return "\n".join(out)

    cands = [by_game[game] for game in CANDIDATES]
    lines = ["=" * 78, "1. FEASIBILITY", "=" * 78, feasible, "",
             "The games every search has nominated, on one basis. r21_window is 1 where a "
             f"full-width\nW={W} descriptor window still ends before the branch, which is "
             "what an R^21 panel needs:",
             table(cands, ["game", "t0", "t0_pct", "r21_window"], n),
             "  " + "\n  ".join(f"{game:>5}  {why}" for game, why in CANDIDATES.items()), "",
             "=" * 78, "2. CROSS-SPACE AGREEMENT", "=" * 78,
             f"sep = mean pairwise distance between the three rule means / RMS rule-mean "
             f"norm,\nper game in each of the {len(labels)} spaces.",
             f"{n} games carried into the SAE spaces: the top {POOL_TOP} of all {nb} "
             f"complete games\nin raw layer 5, the {len(pop)}-game reference sample "
             f"(seed {POP_SEED}), and the {len(CANDIDATES)} named candidates.",
             "pct is against the reference sample in that space, so pct_min reads: in its "
             "WORST\nspace, this game still separates the rules better than pct_min% of a "
             "random\nbranch-complete game. pct_max is the peak a single-space search "
             "returns.", ""]
    for j, label in enumerate(labels):
        v, r = scored[pop, j, 0], scored[pop, j, 1]
        lines.append(f"  {label:<32} sep median {np.median(v):.4f} "
                     f"[{np.percentile(v, 25):.4f}-{np.percentile(v, 75):.4f}]  "
                     f"ratio median {np.median(r):.4f}   (reference sample)")

    lines += ["", f"Spearman rho of sep between spaces (n = {len(pop)} reference games):",
              "  " + " " * 34 + "".join(f"{lab.split(' (')[0][:9]:>10}" for lab in labels)]
    for a, label in enumerate(labels):
        lines.append(f"  {label:<34}" + "".join(
            f"{spearmanr(scored[pop, a, 0], scored[pop, b, 0]).statistic:>10.3f}"
            for b in range(len(labels))))

    lines += ["", "Top 12 on consistency (pct_min):",
              table(sorted(rows, key=lambda r: -r["pct_min"]),
                    ["game", "t0", "pct_min", "pct_mean", "pct_max", "sep_min",
                     "sep_mean", "sep_cv"])]
    lines += ["", "Top 12 on peak (pct_max) -- what a single-space search returns:",
              table(sorted(rows, key=lambda r: -r["pct_max"]),
                    ["game", "t0", "pct_min", "pct_mean", "pct_max", "sep_min",
                     "sep_mean", "sep_cv"])]

    lines += ["", "The named candidates on consistency:",
              table(sorted(cands, key=lambda r: -r["pct_min"]),
                    ["game", "t0", "pct_min", "pct_mean", "pct_max", "sep_min",
                     "sep_mean", "sep_cv"], n)]

    lines += ["", "Per-space argmax of sep over the pool, and where that game sits "
                  "elsewhere:"]
    for j, label in enumerate(labels):
        r = rows[int(np.argmax(scored[:, j, 0]))]
        lines.append(f"  {label:<32} argmax game {r['game']:>5} (t0={r['t0']:>2})  "
                     f"pct_min {r['pct_min']:>5.1f}  pct_mean {r['pct_mean']:>5.1f}  "
                     f"pct_max {r['pct_max']:>5.1f}  legibility {r['legibility']:>5.1f}")
    best_pop = max((rows[i] for i in pop), key=lambda r: r["pct_min"])
    lines += [f"  Best pct_min inside the reference sample alone: game {best_pop['game']} "
              f"at {best_pop['pct_min']:.1f} -- if that beats the",
              "  pool's best, selecting candidates on raw layer 5 missed a game the SAE "
              "spaces liked."]

    lines += ["", "=" * 78, "3. BOARD LEGIBILITY", "=" * 78,
              f"Chebyshev distances in cells; the board panel prints at "
              f"{BOARD_CELL_PT:.0f} pt per cell.",
              "legibility = mean percentile of min_rule_gap, spread, min_context "
              "(up is better)\nand detached_gap (down is better), over all complete games. "
              "Argued, not measured.",
              ""]
    for key in LEGIBILITY:
        v = np.array([r[key] for r in board_rows], float)
        lines.append(f"  {key:<16} min {v.min():>5.2f}  median {np.median(v):>5.2f}  "
                     f"max {v.max():>5.2f}")
    lines += [f"  n_touching > 0 in {sum(1 for r in board_rows if r['n_touching']):>5} / {nb} "
              "games (two different-rule squares share an edge or corner)",
              f"  n_glyph_pairs > 0 in {sum(1 for r in board_rows if r['n_glyph_pairs']):>5}"
              f" / {nb} games (a would-be disc 0.4 cells from the next square's marker)",
              f"  the six squares fall in one board quadrant in "
              f"{sum(1 for r in board_rows if r['quadrants'] == 1):>5} / {nb} games"]

    lines += ["", "Top 12 on consistency, with their board:",
              table(sorted(rows, key=lambda r: -r["pct_min"]),
                    ["game", "t0", "t0_pct", "pct_min", "legibility", "min_rule_gap",
                     "n_touching", "spread", "min_context", "detached_gap"])]
    lines += ["", "Top 12 on consistency AND legibility (mean of the two percentiles):",
              table(sorted(rows, key=lambda r: -(r["pct_min"] + r["legibility"])),
                    ["game", "t0", "t0_pct", "pct_min", "pct_max", "legibility",
                     "min_rule_gap", "spread", "min_context", "detached_gap"])]

    lines += ["", "The named candidates on the board:",
              table(sorted(cands, key=lambda r: -r["legibility"]),
                    ["game", "t0", "legibility", "min_rule_gap", "n_touching",
                     "n_glyph_pairs", "spread", "quadrants", "min_context",
                     "detached_gap"], n)]

    inc = by_game[INCUMBENT]
    lines += ["", f"Incumbent FIGURE_GAME {INCUMBENT} (t0={inc['t0']}): "
                  f"pct_min {inc['pct_min']:.1f}, pct_mean {inc['pct_mean']:.1f}, "
                  f"pct_max {inc['pct_max']:.1f}, legibility {inc['legibility']:.1f}, "
                  f"min_rule_gap {inc['min_rule_gap']}, spread {inc['spread']:.2f}, "
                  f"min_context {inc['min_context']}, detached_gap {inc['detached_gap']}.",
              "Its within-rule spread in raw layer 5, RMS member-to-rule-mean distance:",
              "  " + "  ".join(f"{rule} {v:.1f} (n={len(m)})"
                               for (rule, v), m in zip(within.items(), RULE_MEMBERS.values())),
              "  R3's 0 is exact: one member is its own mean. It contributes no within-rule",
              "  pair, so `ratio`'s within set is R1's 1 pair and R2's 3.", ""]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)

    feasible = feasibility(meta)
    print(feasible)

    games = complete_games(meta)
    per_game = [load_game(meta, g) for g in games]
    refs = [reference_row(g.rows) for g in per_game]

    # One streaming pass answers both questions that need no dictionary: the board
    # geometry, and raw layer 5 -- which ranks the whole population and picks the
    # candidates the eight SAE spaces are worth spending an encode on. Only the
    # scalars survive the loop.
    t0_pct = percentiles(np.array([g.t0 for g in per_game], float))
    board_rows, raw = [], np.zeros(len(games))
    for i, g in enumerate(per_game):
        board_rows.append({"game": g.game, "t0": g.t0, "t0_pct": float(t0_pct[i]),
                           "r21_window": int(positions(g.t0, W - 1)[W - 1] < g.t0),
                           **asdict(board_geometry(replay(g.tokens[:g.t0]), g.rows))})
        raw[i] = score_space(branch_block(acts, g).numpy(), g.rows, refs[i])[0]
    for r, score in zip(board_rows, legibility(board_rows)):
        r["legibility"] = float(score)

    # A random sample rides along in the pool as the reference the percentiles are
    # taken against, and as the check that raw-L5 selection did not miss a game.
    rng = np.random.default_rng(POP_SEED)
    sample = set(rng.choice(len(games), POPULATION, replace=False).tolist())
    for game in CANDIDATES:
        assert game in games, f"candidate game {game} is not branch-complete"
    pool = sorted(sample | set(np.argsort(-raw)[:POOL_TOP].tolist())
                  | {games.index(game) for game in CANDIDATES})
    print(f"scoring {len(pool)} games in {len(spaces())} spaces")

    Z = torch.stack([branch_block(acts, per_game[i]) for i in pool])
    scored = sweep(Z, [per_game[i].rows for i in pool], [refs[i] for i in pool], device)
    pop = [k for k, i in enumerate(pool) if i in sample]
    sep = scored[:, :, 0]
    pct = np.stack([100 * (sep[pop, j][None, :] < sep[:, j][:, None]).mean(axis=1)
                    for j in range(sep.shape[1])], axis=1)

    by_game = {r["game"]: r for r in board_rows}
    rows = [{**by_game[games[i]],
             "in_sample": int(i in sample),
             "pct_min": float(pct[k].min()), "pct_mean": float(pct[k].mean()),
             "pct_max": float(pct[k].max()),
             "sep_min": float(sep[k].min()), "sep_mean": float(sep[k].mean()),
             "sep_cv": float(sep[k].std() / sep[k].mean()),
             "ratio_max": float(scored[k, :, 1].max()),
             **{f"sep_{j}": float(sep[k, j]) for j in range(sep.shape[1])}}
            for k, i in enumerate(pool)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "three_rules_board.csv",
              sorted(board_rows, key=lambda r: -r["legibility"]))
    write_csv(OUT_DIR / "three_rules_ranking.csv",
              sorted(rows, key=lambda r: -r["pct_min"]))
    (OUT_DIR / "three_rules_spaces.txt").write_text(
        "\n".join(f"sep_{j}\t{label}" for j, (label, _) in enumerate(spaces())) + "\n")

    i = games.index(INCUMBENT)
    within = within_rule_spread(branch_block(acts, per_game[i]).numpy(),
                                per_game[i].rows, refs[i])
    text = summary(rows, board_rows, scored, pop, feasible, within)
    (OUT_DIR / "three_rules_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
