"""Step 1 of the violation-divergence study: branch one legal game at t0.

Every branch of a source game shares `tokens[:t0]` and `tokens[t0+1:]` and
differs only at t0 -- the played move (reference), *two* further legal moves
(control_legal, control_legal_2), and one move per available illegal category.
Two branches therefore differ in exactly one token, so a divergence downstream
is attributable to that token and to nothing else.

There are two legal controls because one gives no null. Every quantity measured
against control_legal -- the divergence ratio, and cos(delta_b, delta_control)
above all -- needs a scale for "what a legal alternative does", and only a
second legal branch supplies it: control_legal_2's own row is that null, since
it is scored against control_legal exactly as the illegal branches are. The
first pass of this study shipped with one control and could not interpret its
own direction result. See RED_TEAM.md F1.

t0 is round-robin over [10, 34] so the analysis window r in [-8, +24] fits
inside every 59-token game; a skipped game retries the same t0.

**Do not re-run this over a tree that already exists.** `game_id` is the
accepted-game index, so a rerun under any changed acceptance rule renumbers every
game, moving FIGURE_GAME and breaking the row alignment that acts/*.pt, the rollout
tree and the game-selection studies are all built on. To add games, append them with
extend_branches, which replays this draw first and proves the existing rows unchanged.

Usage: python -m spade.violation_divergence.branches [--n_prefixes 2000] [--smoke]
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from random import Random

import torch

from spade.common.config import CTX_LEN
from spade.common.study import study_dir
from spade.othello.boards import replay
from spade.othello.games import (  # imports force_ipv4() + karvonen path
    classify_prefix, collect_legal, first_illegal_pos, stream_games)
from spade.othello.taxonomy import (
    CATEGORIES, LEGAL, RULE_MEMBERS, RULE_OF, classify)
from spade.rules.flip_labels import flip_stats

from circuits.othello_engine_utils import to_string

T0_MIN, T0_MAX = 10, 34
SMOKE_GAMES = 50

DTYPES = {
    "tokens": torch.int16,
    "game_id": torch.long,
    "t0": torch.long,
    "category": torch.int8,
    "is_control": torch.int8,
    "branch_token": torch.long,
    "n_flipped": torch.long,
    "n_legal": torch.long,
}
N_CONTROLS = 2  # is_control 1 and 2; 2 exists to be the null for 1
CONTROL_NAMES = ["control_legal", "control_legal_2"]
ROLE_NAMES = ["reference", *CONTROL_NAMES, *CATEGORIES[1:]]


def role(category: int, is_control: int) -> str:
    if is_control:
        return CONTROL_NAMES[is_control - 1]
    return "reference" if category == LEGAL else CATEGORIES[category]


# The game every single-game figure draws. It is an argmax over the
# branch-complete games -- of the mutual separation of its six illegal
# displacement trajectories, in raw layer 5 at full width -- and not a draw from
# them, so every figure drawing it owes the reader that disclosure in its
# caption; the study READMEs carry it. Picked by
# analysis/violation_divergence/game_selection.
FIGURE_GAME = 1235


@dataclass(frozen=True)
class Game:
    """One source game's branch rows, read back out of branches.pt."""
    game: int
    t0: int
    idx: torch.Tensor          # row indices into branches.pt, for the act tensors
    rows: list[dict]
    tokens: list[int]


def load_game(meta: dict, game: int) -> Game:
    idx = (meta["game_id"] == game).nonzero().flatten()
    assert len(idx), f"game {game} has no branches"
    rows = [{"category": int(meta["category"][i]),
             "is_control": int(meta["is_control"][i]),
             "square": to_string(int(meta["branch_token"][i])),
             "n_legal": int(meta["n_legal"][i])} for i in idx]
    return Game(game, int(meta["t0"][idx[0]]), idx, rows,
                meta["tokens"][idx[0]].long().tolist())


def reference_row(rows: list[dict]) -> int:
    return next(i for i, r in enumerate(rows)
                if r["category"] == LEGAL and not r["is_control"])


def complete_games(meta: dict) -> list[int]:
    """Games carrying every branch type: the reference, both controls, all six.

    Counts distinct branch types rather than rows, because the rollout variant
    stores K rows per branch: a row count finds no complete game there at all.
    """
    n_types = len(CATEGORIES) + N_CONTROLS
    kind = meta["category"].long() * (N_CONTROLS + 1) + meta["is_control"].long()
    return [g for g in meta["game_id"].unique().tolist()
        if len(kind[meta["game_id"] == g].unique()) == n_types]


def rules_present(categories) -> set[str]:
    """Which of R1-R3 the given branch categories cover."""
    return {RULE_OF[CATEGORIES[int(c)]] for c in categories if int(c) != LEGAL}


def rule_complete_games(meta: dict) -> list[int]:
    """Games carrying at least one branch of every rule -- what a per-rule curve needs.

    Much weaker than `complete_games`, which wants all six categories. R3's only
    member `opponent_only` is the one a t0 can fail to offer at all, so it decides
    the membership; the ids are written out beside branches.pt by extend_branches
    and a figure should read that file rather than re-derive this.
    """
    return [g for g in meta["game_id"].unique().tolist()
        if len(rules_present(meta["category"][meta["game_id"] == g])) == len(RULE_MEMBERS)]


def pick_branches(reference: int, cats: list[int], rng: Random) -> list[tuple[int, int, int]] | None:
    """(token, category, is_control) per branch; None when t0 offers too few legal moves."""
    assert cats[reference - 1] == LEGAL, f"played token {reference} is not legal at t0"
    by_cat = {c: [i + 1 for i, ci in enumerate(cats) if ci == c] for c in range(len(CATEGORIES))}
    others = [t for t in by_cat[LEGAL] if t != reference]
    if len(others) < N_CONTROLS:
        return None
    controls = rng.sample(others, N_CONTROLS)
    return ([(reference, LEGAL, 0)]
        + [(tok, LEGAL, i + 1) for i, tok in enumerate(controls)]
        + [(rng.choice(by_cat[c]), c, 0) for c in range(1, len(CATEGORIES)) if by_cat[c]])


def t0_for(game_id: int) -> int:
    """Round-robin over [T0_MIN, T0_MAX] by accepted-game index."""
    return T0_MIN + (game_id % (T0_MAX - T0_MIN + 1))


def draw(games, rng: Random, t0: int, skips: Counter) -> list[dict] | None:
    """The branch rows of the next streamed source game at `t0`, None if it is skipped.

    One call consumes exactly one game from `games` and, when that game is usable,
    one block of draws from `rng`. Replaying the call sequence therefore replays
    the tensor, which is what lets extend_branches append without renumbering.
    """
    seq = collect_legal(games, 1, skips)[0]
    prefix = classify_prefix(seq, t0, skips)
    if prefix is None:
        return None
    board, cats = prefix
    picks = pick_branches(seq[t0], cats, rng)
    if picks is None:
        skips[f"fewer_than_{N_CONTROLS + 1}_legal"] += 1
        return None
    n_legal = sum(c == LEGAL for c in cats)
    return [{"tokens": seq[:t0] + [token] + seq[t0 + 1:],
             "t0": t0,
             "category": category,
             "is_control": is_control,
             "branch_token": token,
             "n_flipped": flip_stats(board.state, to_string(token), board.next_hand_color)[0]
                 if category == LEGAL else -1,
             "n_legal": n_legal}
        for token, category, is_control in picks]


def summary(items: dict[str, torch.Tensor]) -> dict:
    """The manifest fields that are re-derivable from the saved rows.

    The t0 histogram and no range: `extend_branches` appends blocks that all fork at
    one position, so [min, max] would read as a description of the tree and not be one.
    """
    is_reference = (items["category"] == LEGAL) & (items["is_control"] == 0)
    counts = Counter(role(int(c), int(k))
                     for c, k in zip(items["category"], items["is_control"]))
    t0_hist = Counter(items["t0"][is_reference].tolist())
    return {
        "n_games": len(items["game_id"].unique()),
        "n_branches": len(items["t0"]),
        "branch_counts": {name: counts[name] for name in ROLE_NAMES},
        "t0_histogram": {str(t0): n for t0, n in sorted(t0_hist.items())},
    }


def verify(items: dict[str, torch.Tensor]) -> None:
    """Every claim the study rests on, re-derived from the saved rows alone."""
    tokens = items["tokens"].long()
    assert tokens.shape[1] == CTX_LEN and (tokens > 0).all(), "padding or pass token in a branch"
    legal = items["category"] == LEGAL
    assert (items["n_flipped"][legal] > 0).all(), "a legal branch flips nothing"
    assert (items["n_flipped"][~legal] == -1).all(), "an illegal branch carries a flip count"
    assert (items["n_legal"] >= N_CONTROLS + 1).all(), \
        f"a game reached the rows with fewer than {N_CONTROLS + 1} legal moves"
    for game_id in items["game_id"].unique().tolist():
        idx = (items["game_id"] == game_id).nonzero().flatten().tolist()
        t0 = int(items["t0"][idx[0]])
        ref = [i for i in idx if legal[i] and items["is_control"][i] == 0]
        assert len(ref) == 1, f"game {game_id}: {len(ref)} reference rows"
        for c in range(1, N_CONTROLS + 1):
            n_c = sum(1 for i in idx if items["is_control"][i] == c and legal[i])
            assert n_c == 1, f"game {game_id}: {n_c} rows with is_control={c}"
        assert first_illegal_pos(tokens[ref[0]].tolist()) == -1, \
            f"game {game_id}: source game does not replay legally end to end"
        board = replay(tokens[ref[0], :t0].tolist())
        for i in idx:
            assert int(items["t0"][i]) == t0, f"game {game_id}: rows disagree on t0"
            assert torch.equal(tokens[i, :t0], tokens[ref[0], :t0]), f"game {game_id}: prefix differs"
            assert torch.equal(tokens[i, t0 + 1:], tokens[ref[0], t0 + 1:]), \
                f"game {game_id}: suffix differs"
            assert int(tokens[i, t0]) == int(items["branch_token"][i]), "branch_token is not at t0"
            assert classify(board, to_string(int(tokens[i, t0]))) == int(items["category"][i]), \
                f"game {game_id}: category disagrees with the engine at t0={t0}"
        assert len({int(items["branch_token"][i]) for i in idx}) == len(idx), \
            f"game {game_id}: two branches play the same token"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_prefixes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_GAMES} games into $DIR/smoke/")
    args = parser.parse_args()

    import datasets as hf_datasets
    n_games = SMOKE_GAMES if args.smoke else args.n_prefixes
    rng = Random(args.seed)
    games = stream_games(args.dataset_name, args.seed)
    skips = Counter()
    rows: dict[str, list] = {k: [] for k in DTYPES}

    game_id = 0
    while game_id < n_games:
        branches = draw(games, rng, t0_for(game_id), skips)
        if branches is None:
            continue
        for branch in branches:
            rows["game_id"].append(game_id)
            for key, value in branch.items():
                rows[key].append(value)
        game_id += 1
        if game_id % 250 == 0:
            print(f"{game_id}/{n_games} games, {len(rows['t0'])} branches")

    items = {k: torch.tensor(v, dtype=DTYPES[k]) for k, v in rows.items()}
    verify(items)

    out_dir = (study_dir("divergence") / "smoke" if args.smoke else study_dir("divergence")) / "branches"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(items, out_dir / "branches.pt")

    manifest = {
        "seed": args.seed,
        "dataset": args.dataset_name,
        "datasets_version": hf_datasets.__version__,
        **summary(items),
        "skips": dict(skips),
        "appended": [],  # one block per extend_branches run, in the order they were appended
        "categories": CATEGORIES,
        "note": "t0 assigned to accepted games; skips retry the same t0, so late-t0 "
                "games are conditioned on pass-free non-forfeit sources",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows['t0'])} branches from {n_games} games to {out_dir}")
    print(json.dumps({"branch_counts": manifest["branch_counts"], "skips": dict(skips)}, indent=2))


if __name__ == "__main__":
    main()
