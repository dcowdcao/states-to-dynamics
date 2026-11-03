"""Which game should the fork figure be drawn on?

`game_trajectory/make_figures.py` draws `complete[0]` -- the lowest game_id that
carries all nine branch types. That resolves to game 0, t0=10, the *minimum* t0
over the 1550 complete games (median 21): the least typical game available,
picked by list order and not by any criterion.

This sweeps every complete game of the tree's first 2000 (the 1550 B.2 selected
from; later appends are not candidates) and scores how distinct its nine branch
trajectories are, so the illustration can be drawn on the clearest example and
say so.

Two rules the scoring obeys, both of which would otherwise rig the figure:

- **Raw layer 5 only.** The grid's whole point is to compare nine representation
  spaces. Selecting with an SAE in the loop would tilt that comparison toward
  whichever dictionary drove the choice. Raw layer 5 is the common input to all
  nine and is the only defensible selector. The SAE spaces appear here once, in
  `transfer()`, purely as a check on whether the raw-L5 choice survives them.
- **Full R^512, not a 2-D plane.** The projection is being redesigned; a
  criterion computed on a top-2 PCA would select the game that flatters one
  particular plane. Every score below is computed on the full displacement
  vectors D(t) = z_branch(t) - z_ref(t).

## The criterion, fixed before any figure was looked at

**S3 (category differentiation) is the recommendation criterion.** Three reasons:

1. The six illegal categories are the colours the fork asks a reader to tell
   apart, so their mutual separation is what "readable" means for this figure.
2. It is scale-free. The plane is refit per game and per space, so overall
   divergence magnitude is normalised away by the drawing itself; only the
   *directions* survive into the picture. A criterion carrying magnitude (S2, S4)
   would re-select the largest-delta game for a property the figure cannot show.
3. It is not the study's scientific contrast. S1 is -- and the study's aggregate
   finding is that illegal branches displace layer 5 *less* than a legal
   alternative (ratio 0.978). Maximising S1 therefore picks the game that most
   contradicts the paper's own headline. That is a cherry-pick, and S3 cannot be
   one because no claim rests on it.

All four are computed and reported; the recommendation is argmax S3, and the
report states where that game sits on S1 so a reader can see it is not an S1
outlier.

## Definitions

Positions are the study window r in [-8, 24] at t0's parity, restricted to
r >= 0 (before t0 every displacement is exactly zero by construction). Stack each
branch's displacements over those 13 positions into one vector X_b in R^(13*512).

| key | definition | units |
|---|---|---|
| S1 | \\|mu_illegal - mu_control\\| / pooled within-group RMS radius | pooled SDs |
| S2 | mean pairwise \\|X_i - X_j\\| over all 9 branches / \\|Z_ref\\|_F | dimensionless |
| S3 | mean pairwise \\|X_i - X_j\\| over the 6 illegal / RMS \\|X_illegal\\| | in [0, sqrt(2)] |
| S4 | mean over illegal of delta_hat at r = 24 | dimensionless |

S3's ceiling is sqrt(2): six equal-length mutually orthogonal displacements.
Zero means the six illegal branches move identically.

Writes ranking.csv, transfer_check.csv, summary.txt and inputs.json (the sha256 of
branches.pt and the activations it scored) into a new `--out` directory. The committed
results/ are the provenance of B.2's choice and are never rewritten: the activations were
re-extracted after they were scored, which moves S3 by up to 6e-7.

Usage: python analysis/violation_divergence/game_selection/score_games.py [--out DIR]
"""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.violation_divergence.compute_divergence import sae_encoder

SELECTION_GAMES = 2000            # game_id below this: the tree B.2's game was chosen on
STRIDE, WINDOW = 2, (-8, 24)      # mirrors game_trajectory/make_figures.py, the
FIRST_POS, LAST_POS = 1, 58       # figure this selects for; keep the two in sync
CRITERIA = ["s1_illegal_vs_control", "s2_branch_distinctness",
            "s3_category_differentiation", "s4_delta_hat_end"]
RECOMMENDED = "s3_category_differentiation"
INCUMBENT = 0                     # the game complete[0] currently resolves to
TRANSFER_CONFIGS = ["StandardTrainer-othello", "TopKTrainer-othello",
                    "SpaDE-lam1e-03-gam1e-04"]
TRANSFER_SAMPLE, TRANSFER_SEED = 300, 0


@dataclass(frozen=True)
class Scores:
    game: int
    t0: int
    s1_illegal_vs_control: float
    s2_branch_distinctness: float
    s3_category_differentiation: float
    s4_delta_hat_end: float


def positions(t0: int) -> np.ndarray:
    p = np.arange(max(FIRST_POS, t0 + WINDOW[0]), min(LAST_POS, t0 + WINDOW[1]) + 1)
    return p[(p - t0) % STRIDE == 0]


def mean_pairwise(X: np.ndarray) -> float:
    return float(pdist(X).mean())


def score(F: np.ndarray, category: np.ndarray, is_control: np.ndarray,
          post: np.ndarray, t0: int, game: int) -> Scores:
    """F is [n_branches, n_positions, width] in one space; `post` selects r >= 0."""
    ref = int(np.flatnonzero((category == 0) & (is_control == 0))[0])
    D = (F - F[ref])[:, post, :]
    X = D.reshape(len(D), -1)
    ill, ctl = category > 0, is_control > 0
    ref_traj = F[ref][post]

    within = (((X[ill] - X[ill].mean(0)) ** 2).sum()
              + ((X[ctl] - X[ctl].mean(0)) ** 2).sum())
    pooled = np.sqrt(within / (ill.sum() + ctl.sum() - 2))
    s1 = float(np.linalg.norm(X[ill].mean(0) - X[ctl].mean(0)) / pooled)
    s2 = mean_pairwise(X) / float(np.linalg.norm(ref_traj))
    s3 = mean_pairwise(X[ill]) / float(np.sqrt((X[ill] ** 2).sum(1).mean()))
    s4 = float(np.mean(np.linalg.norm(D[ill][:, -1], axis=1)
                       / np.linalg.norm(ref_traj[-1])))
    return Scores(game, t0, s1, s2, s3, s4)


def game_rows(meta: dict, acts: torch.Tensor, game: int):
    """(features [9, T, 512], category, is_control, post mask, t0) for one game."""
    idx = (meta["game_id"] == game).nonzero().flatten()
    t0 = int(meta["t0"][idx[0]])
    pos = positions(t0)
    F = acts[idx][:, pos, :].float().numpy()
    return (F, meta["category"][idx].numpy(), meta["is_control"][idx].numpy(),
            pos >= t0, t0)


def encode(F: np.ndarray, enc, device: str) -> np.ndarray:
    n, T, d = F.shape
    with torch.no_grad():
        return enc(torch.tensor(F.reshape(-1, d), device=device)).reshape(n, T, -1).cpu().numpy()


def sweep(meta: dict, acts: torch.Tensor, games: list[int]) -> list[Scores]:
    return [score(*game_rows(meta, acts, g), g) for g in games]


def transfer(meta: dict, acts: torch.Tensor, games: list[int], device: str
             ) -> dict[str, list[Scores]]:
    by_config = {e.config: e for e in seed42_entries()}
    out = {}
    for config in TRANSFER_CONFIGS:
        enc = sae_encoder(by_config[config], device)
        rows = []
        for g in games:
            F, cat, ctl, post, t0 = game_rows(meta, acts, g)
            rows.append(score(encode(F, enc, device), cat, ctl, post, t0, g))
        out[config] = rows
        print(f"  scored {len(rows)} games in {config}")
    return out


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def ranked(scores: list[Scores]) -> list[dict]:
    """One row per game, with a 1-based descending rank under every criterion."""
    rows = [asdict(s) for s in scores]
    for key in CRITERIA:
        order = sorted(range(len(rows)), key=lambda i: -rows[i][key])
        for rank, i in enumerate(order, 1):
            rows[i][f"rank_{key.split('_')[0]}"] = rank
    return sorted(rows, key=lambda r: r[f"rank_{RECOMMENDED.split('_')[0]}"])


def table(rows: list[dict], keys: list[str], n: int) -> str:
    head = " | ".join(f"{k:>26}" if k in CRITERIA else f"{k:>6}" for k in keys)
    out = [head, "-" * len(head)]
    for r in rows[:n]:
        out.append(" | ".join(
            f"{r[k]:>26.4f}" if k in CRITERIA else f"{r[k]:>6}" for k in keys))
    return "\n".join(out)


def summary(rows: list[dict], t0s: np.ndarray, sae: dict[str, list[Scores]]) -> str:
    pick = rows[0]
    incumbent = next(r for r in rows if r["game"] == INCUMBENT)
    n = len(rows)
    lines = [
        f"n = {n} complete games (all nine branch types) with game_id < {SELECTION_GAMES}, "
        "scored in raw layer 5, full R^512.",
        f"recommendation criterion: {RECOMMENDED} (argmax).",
        "",
        f"SELECTED game {pick['game']}, t0={pick['t0']}  "
        f"S3={pick[RECOMMENDED]:.4f}  S1={pick['s1_illegal_vs_control']:.4f} "
        f"(rank {pick['rank_s1']}/{n})  S2={pick['s2_branch_distinctness']:.4f} "
        f"(rank {pick['rank_s2']}/{n})  S4={pick['s4_delta_hat_end']:.4f} "
        f"(rank {pick['rank_s4']}/{n})",
        f"INCUMBENT game {INCUMBENT}, t0={incumbent['t0']}  "
        + "  ".join(f"{k[:2].upper()}={incumbent[k]:.4f} (rank {incumbent['rank_' + k.split('_')[0]]}/{n})"
                    for k in CRITERIA),
        "",
        f"t0 of complete games: min {t0s.min()}, median {int(np.median(t0s))}, "
        f"max {t0s.max()}; selected t0={pick['t0']} is at percentile "
        f"{100 * (t0s <= pick['t0']).mean():.1f}.",
        "",
        "criterion distribution over all complete games:",
    ]
    for key in CRITERIA:
        v = np.array([r[key] for r in rows])
        lines.append(f"  {key:<28} min {v.min():.4f}  q25 {np.percentile(v, 25):.4f}  "
                     f"median {np.median(v):.4f}  q75 {np.percentile(v, 75):.4f}  "
                     f"max {v.max():.4f}")

    lines += ["", "Spearman rank correlation between criteria (n = %d games):" % n,
              "  " + " " * 28 + "".join(f"{k[:2].upper():>10}" for k in CRITERIA)]
    cols = {k: np.array([r[k] for r in rows]) for k in CRITERIA}
    for a in CRITERIA:
        lines.append(f"  {a:<28}" + "".join(
            f"{spearmanr(cols[a], cols[b]).statistic:>10.3f}" for b in CRITERIA))

    for key in CRITERIA:
        lines += ["", f"top 10 by {key}:",
                  table(sorted(rows, key=lambda r: -r[key]),
                        ["game", "t0", *CRITERIA], 10)]

    lines += ["", f"transfer check: S3 of the selected game re-scored in SAE spaces, "
                  f"within a random sample of {len(sae[TRANSFER_CONFIGS[0]])} games "
                  f"(seed {TRANSFER_SEED}, selected game and incumbent forced in)."]
    raw_s3 = {r["game"]: r[RECOMMENDED] for r in rows}
    for config, rs in sae.items():
        by_game = {s.game: getattr(s, RECOMMENDED) for s in rs}
        order = sorted(by_game, key=lambda g: -by_game[g])
        rho = spearmanr([raw_s3[g] for g in order],
                        [by_game[g] for g in order]).statistic
        lines.append(
            f"  {config:<28} selected game rank {order.index(pick['game']) + 1}/{len(order)} "
            f"(S3={by_game[pick['game']]:.4f}); incumbent rank "
            f"{order.index(INCUMBENT) + 1}/{len(order)}; "
            f"Spearman vs raw L5 rho={rho:.3f}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study_dir("divergence") / "game_selection",
                        help="a directory that does not exist yet")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; pass a new --out rather than overwrite a ranking")

    device = select_device()
    base = study_dir("divergence")
    inputs = {path: base / path for path in ("branches/branches.pt", "acts/resid_post_5.pt")}
    meta = torch.load(inputs["branches/branches.pt"], weights_only=True)
    acts = torch.load(inputs["acts/resid_post_5.pt"], weights_only=True)

    n_types = len(CATEGORIES) + 2
    complete = [g for g in meta["game_id"].unique().tolist()
        if g < SELECTION_GAMES
        and int((meta["game_id"] == g).sum()) == n_types]
    print(f"{len(complete)} complete games with game_id < {SELECTION_GAMES}")

    args.out.mkdir(parents=True)
    (args.out / "inputs.json").write_text(json.dumps(
        {"game_id_below": SELECTION_GAMES,
         "sha256": {name: sha256(path) for name, path in inputs.items()}}, indent=2))
    scores = sweep(meta, acts, complete)
    rows = ranked(scores)
    print(f"wrote {write_csv(args.out / 'ranking.csv', rows)}")

    pick = rows[0]["game"]
    rng = np.random.default_rng(TRANSFER_SEED)
    sample = sorted({pick, INCUMBENT}
                    | set(rng.choice(complete, TRANSFER_SAMPLE, replace=False).tolist()))
    print(f"transfer check on {len(sample)} games x {len(TRANSFER_CONFIGS)} SAE spaces")
    sae = transfer(meta, acts, sample, device)
    write_csv(args.out / "transfer_check.csv",
              [{"space": c, **asdict(s)} for c, rs in sae.items() for s in rs])

    t0s = np.array([r["t0"] for r in rows])
    text = summary(rows, t0s, sae)
    (args.out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
