"""Step 1: the (game, position) items the move-legality probe is fit and scored on.

An item is one source game and one move index t. It carries the move the game really
played -- legal by construction, since the source games replay end to end -- and one
substituted illegal move per constraint rule that has a candidate at t. Every row of an
item shares tokens[:t], so the four rows differ in exactly one token and the probe cannot
separate them by anything but that token and the board it lands on.

That is where the class balance comes from. `probe.balanced` keeps whole items, so a
split holds the same number of rows per class and the classes share their game and
move-index marginals exactly, with no reweighting and no subsampling that could correlate
with t. The 4-class set is the items where all three rules have a candidate; the 2-class
set is every item, with one illegal row drawn per item at a stated seed (never a mean over
the three -- that would give the illegal class 1/sqrt(3) of its real spread).

Positions run the whole board minus the sink, not the [5, 55) of
`spade.legality_probe.fit_set`: the read-out draws the probe at every position of a branch
sequence, and a probe never fit near the edges would be extrapolating exactly where the
curve is read. Position 0 is the attention sink -- a ~300-norm residual carrying no board,
dropped everywhere else in this repo (`straightness.extract_traj.check_sink`) -- and it is
dropped here too. Left in, it is not merely noisy: the probe emits a per-space *constant*
there, 1.0 in BatchTopK and 0.0 in raw, and one saturated position takes the argmax away
from t0 in every game.

Source games are disjoint by exact token match from the violation-divergence branch tree
the probe is read out on; `legality_probe.fit_set.branch_tokens` is that set.

Usage: python -m spade.move_legality.fit_set [--n_games 500]
"""

import argparse
import json
from collections import Counter
from random import Random

import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN, provenance
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe.fit_set import branch_tokens
from spade.othello.games import classify_prefix, collect_legal, stream_games
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_MEMBERS

SINK = 0                                         # position 0 is the attention sink
POSITIONS = range(SINK + 1, CTX_LEN)
RULES = list(RULE_MEMBERS)                       # R1, R2, R3 in taxonomy order
CLASSES = ("legal", *(r.split()[0] for r in RULES))
RULE_CATEGORIES = [[CATEGORIES.index(c) for c in RULE_MEMBERS[r]] for r in RULES]
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)

DTYPES = {
    "game_id": torch.int16,
    "position": torch.int16,
    "token": torch.int16,
    "category": torch.int8,
    "label": torch.int8,       # index into CLASSES: 0 legal, 1..3 the rule
}


def item_rows(game: list[int], t: int, cats: list[int], rng: Random) -> list[dict]:
    """The played legal move at t, plus one uniform draw from each rule with a candidate.

    Uniform over the rule's *tokens* rather than over its member categories: the class the
    probe is asked about is "an illegal move breaking rule R", and the categories inside a
    rule are not equally common on a board.
    """
    rows = [{"position": t, "token": game[t], "category": LEGAL, "label": 0}]
    for label, categories in enumerate(RULE_CATEGORIES, start=1):
        candidates = [i + 1 for i, c in enumerate(cats) if c in categories]
        if not candidates:
            continue
        token = rng.choice(candidates)
        rows.append({"position": t, "token": token,
                     "category": cats[token - 1], "label": label})
    return rows


def collect(games, n_games: int, forbidden: set[tuple[int, ...]],
            rng: Random, skips: Counter) -> tuple[list[list[int]], dict[str, list]]:
    tokens, rows = [], {k: [] for k in DTYPES}
    while len(tokens) < n_games:
        game = collect_legal(games, 1, skips)[0]
        if tuple(game) in forbidden:
            skips["in_readout_set"] += 1
            continue
        game_rows = []
        for t in POSITIONS:
            prefix = classify_prefix(game, t, skips)
            if prefix is None:
                continue
            game_rows += item_rows(game, t, prefix[1], rng)
        game_id = len(tokens)
        tokens.append(game)
        for row in game_rows:
            rows["game_id"].append(game_id)
            for key, value in row.items():
                rows[key].append(value)
        if len(tokens) % 100 == 0:
            print(f"{len(tokens)}/{n_games} games, {len(rows['position'])} rows", flush=True)
    return tokens, rows


def assign_splits(n_games: int, seed: int) -> torch.Tensor:
    """Split index per game. By game, so no position in a split shares a prefix with another."""
    order = torch.randperm(n_games, generator=torch.Generator().manual_seed(seed))
    cuts = np.cumsum([round(f * n_games) for f in SPLIT_FRACTIONS[:-1]])
    split = torch.empty(n_games, dtype=torch.int8)
    for i, part in enumerate(np.split(order.numpy(), cuts)):
        split[part] = i
    return split


def verify(fit: dict, tokens: torch.Tensor) -> None:
    """Re-derive every row's label from the saved tokens and the engine."""
    from circuits.othello_engine_utils import to_string

    from spade.othello.boards import replay
    from spade.othello.taxonomy import classify

    rng = Random(0)
    for row in rng.sample(range(len(fit["position"])), 2000):
        game = tokens[int(fit["game_id"][row])].tolist()
        t, token = int(fit["position"][row]), int(fit["token"][row])
        board = replay(game[:t])
        category = classify(board, to_string(token))
        assert category == int(fit["category"][row]), f"row {row}: category"
        label = 0 if category == LEGAL else 1 + next(
            i for i, cs in enumerate(RULE_CATEGORIES) if category in cs)
        assert label == int(fit["label"][row]), f"row {row}: label"
        if label == 0:
            assert token == game[t], f"row {row}: legal row is not the played move"

    keys = list(zip(fit["game_id"].tolist(), fit["position"].tolist(), fit["label"].tolist()))
    assert len(set(keys)) == len(keys), "an item carries two rows of one class"
    n_legal = int((fit["label"] == 0).sum())
    assert n_legal == len(set((g, p) for g, p, _ in keys)), "an item has no legal row"


def item_table(fit: dict, split: torch.Tensor) -> pd.DataFrame:
    """One row per (split, class): rows available, and the items where all rules exist."""
    label, game = fit["label"], fit["game_id"].long()
    key = game * CTX_LEN + fit["position"].long()
    complete = torch.bincount(key)[key] == len(CLASSES)
    rows = []
    for s, name in enumerate(SPLITS):
        in_split = split[game] == s
        for c, cls in enumerate(CLASSES):
            mask = in_split & (label == c)
            rows.append({"split": name, "class": cls, "n_rows": int(mask.sum()),
                         "n_rows_4class": int((mask & complete).sum()),
                         "n_games": len(game[in_split].unique())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    skips = Counter()
    tokens, rows = collect(stream_games(args.dataset_name, args.seed), args.n_games,
                           branch_tokens(), Random(args.seed), skips)

    fit = {k: torch.tensor(rows[k], dtype=d) for k, d in DTYPES.items()}
    games = torch.tensor(tokens, dtype=torch.int16)
    split = assign_splits(args.n_games, args.seed)
    assert games.shape == (args.n_games, CTX_LEN)
    verify(fit, games)

    out_dir = study_dir("move_legality") / "fit_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in {**fit, "tokens": games, "split": split}.items():
        torch.save(tensor, out_dir / f"{name}.pt")

    table = item_table(fit, split)
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "dataset": args.dataset_name,
        "n_games": args.n_games,
        "n_rows": len(fit["position"]),
        "classes": list(CLASSES),
        "split_fractions": list(SPLIT_FRACTIONS),
        "splits": {name: int((split == i).sum()) for i, name in enumerate(SPLITS)},
        "row_of": "residual at `position` of tokens[game_id] with `token` spliced in at it",
        "disjoint_from": str(study_dir("divergence") / "branches" / "branches.pt"),
        "skips": dict(skips),
        **provenance(),
    }, indent=2))

    results = repo_root() / "analysis" / "move_legality" / "results"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "fit_set_balance.csv", index=False)
    print(f"wrote {len(fit['position'])} rows from {args.n_games} games to {out_dir}")
    print(f"skips: {dict(skips)}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
