"""Append source games to branches.pt at one fork position, renumbering nothing.

The per-rule population is measured at a single t0 so that where the fork happened
is not confounded with what was violated there, and the round-robin leaves only ~80
of the tree's games at any one position. This draws further games from the same
stream, forks all of them at `--t0`, and appends them with `game_id` continuing from
the last.

Rule completeness is a property of the board at t0, not of the game: a game that
admits all three rules at its own round-robin t0 need not admit them at t0 = 20. The
acceptance test is therefore applied at the position the game is actually forked on,
and the fraction of draws it rejects is a measured number (`n_rejected_rule_incomplete`).

**Append, never regenerate.** `game_id` is the accepted-game index, so re-running
branches.py under any changed acceptance rule renumbers every game. That would move
`FIGURE_GAME`, break the row alignment between branches.pt and acts/*.pt, break the
rollout tree's `row = source_row * K + k`, and invalidate the game-selection studies.
So the games already on disk are *replayed* -- same seed, same stream, each at its own
saved t0 and under the acceptance test it was drawn under -- and asserted equal to the
saved tensor column by column before anything is appended. That replay is also what
leaves the stream and the RNG where the previous run left them, which is what makes
the appended draw a continuation rather than a new sample.

The filter is applied to games in stream order and to nothing else: the first
qualifying games are taken, never the best ones, so the appended set is conditioned
on rule membership and on no measured outcome.

Writes `rule_complete_games.json` beside branches.pt -- the ids a per-rule figure
should read instead of re-deriving the membership rule.

Usage: python -m spade.violation_divergence.extend_branches --t0 20 [--target 2000]
"""

import argparse
import hashlib
import json
from collections import Counter
from random import Random

import torch

from spade.common.study import study_dir
from spade.othello.games import stream_games
from spade.othello.taxonomy import RULE_MEMBERS
from spade.violation_divergence.branches import (
    DTYPES, draw, rule_complete_games, rules_present, summary, verify)

REASON = ("the per-rule population averages games that all fork at one t0, so that the "
          "fork position is not confounded with the violation; the appended games "
          "continue the same stream and seed, fork at --t0, and are kept only if their "
          "branches cover every rule there")


def tensors(drawn: list[tuple[int, dict]]) -> dict[str, torch.Tensor]:
    """branches.pt's column layout, from (game_id, branch row) pairs."""
    columns = {"game_id": [game_id for game_id, _ in drawn]}
    columns |= {key: [branch[key] for _, branch in drawn] for key in DTYPES if key != "game_id"}
    return {key: torch.tensor(columns[key], dtype=DTYPES[key]) for key in DTYPES}


def digest(items: dict[str, torch.Tensor], n_rows: int) -> str:
    """sha256 over the first `n_rows` of every column, in the dict's own key order."""
    h = hashlib.sha256()
    for column in items.values():
        h.update(column[:n_rows].contiguous().numpy().tobytes())
    return h.hexdigest()


def at_fork(items: dict[str, torch.Tensor], t0: int) -> list[int]:
    """The rule-complete games forking at `t0`: the population a single-t0 curve draws."""
    at = dict(zip(items["game_id"].tolist(), items["t0"].tolist()))
    return [game for game in rule_complete_games(items) if at[game] == t0]


def accepted(branches: list[dict]) -> bool:
    """The append's acceptance test: the branches available at t0 cover R1, R2 and R3."""
    return len(rules_present(branch["category"] for branch in branches)) == len(RULE_MEMBERS)


def next_game(games, rng: Random, skips: Counter, t0: int, rule_filter: bool) -> list[dict]:
    """Draw from the stream until a game is usable at `t0` and passes the filter."""
    while True:
        branches = draw(games, rng, t0, skips)
        if branches is None:
            continue
        if rule_filter and not accepted(branches):
            skips["rule_incomplete"] += 1
            continue
        return branches


def replay(games, rng: Random, skips: Counter, t0s: list[int],
           filter_from: int) -> list[tuple[int, dict]]:
    """Redraw the games already on disk, which walks the stream back to where it stopped.

    Each game is redrawn at its own saved t0 and under the test it was accepted under:
    everything from `filter_from` on was appended by this script and so was filtered.
    """
    out = []
    for game_id, t0 in enumerate(t0s):
        branches = next_game(games, rng, skips, t0, game_id >= filter_from)
        out += [(game_id, branch) for branch in branches]
    return out


def extend(games, rng: Random, skips: Counter, first_id: int, n_new: int,
           t0: int) -> list[tuple[int, dict]]:
    """The next `n_new` games in stream order whose branches cover R1, R2 and R3 at `t0`."""
    out = []
    for game_id in range(first_id, first_id + n_new):
        out += [(game_id, branch) for branch in next_game(games, rng, skips, t0, True)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=int, required=True,
                        help="fork position of every appended game")
    parser.add_argument("--target", type=int, default=2000,
                        help="rule-complete games wanted forking at --t0 in total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    out_dir = study_dir("divergence") / "branches"
    path = out_dir / "branches.pt"
    saved = torch.load(path, weights_only=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    n_games, n_rows = len(saved["game_id"].unique()), len(saved["t0"])
    n_have = len(at_fork(saved, args.t0))
    n_new = args.target - n_have
    assert n_new > 0, f"{args.target} rule-complete games already fork at t0 = {args.t0}"
    print(f"{n_games} games / {n_rows} rows on disk, {n_have} rule-complete at "
          f"t0 = {args.t0}; drawing {n_new} more")
    print(f"branches.pt sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    before = digest(saved, n_rows)

    rng = Random(args.seed)
    games = stream_games(args.dataset_name, args.seed)
    skips = Counter()
    at = dict(zip(saved["game_id"].tolist(), saved["t0"].tolist()))
    blocks = manifest["appended"]
    replayed = tensors(replay(games, rng, skips, [at[g] for g in range(n_games)],
                              blocks[0]["first_game_id"] if blocks else n_games))
    for key, column in saved.items():
        assert torch.equal(replayed[key], column), f"the replay does not reproduce {key}"
    assert dict(skips) == manifest["skips"], \
        f"the replay skipped {dict(skips)}, the manifest recorded {manifest['skips']}"
    print("replay reproduces every saved column exactly; the stream is where it was left")

    rejected_by_replay = skips["rule_incomplete"]
    appended = extend(games, rng, skips, n_games, n_new, args.t0)
    items = {key: torch.cat([saved[key], column])
             for key, column in tensors(appended).items()}
    verify(items)
    assert digest(items, n_rows) == before, "an existing row changed"

    torch.save(items, path)
    reloaded = torch.load(path, weights_only=True)
    assert digest(reloaded, n_rows) == before, "an existing row changed on the round trip"
    print(f"existing {n_rows} rows byte-identical, sha256 {before}")

    complete = rule_complete_games(items)
    (out_dir / "rule_complete_games.json").write_text(json.dumps({
        "criterion": "the game's branches include at least one member of every constraint rule",
        "rule_members": RULE_MEMBERS,
        "n_games": len(complete),
        "game_ids": complete,
    }, indent=2))

    rejected = skips["rule_incomplete"] - rejected_by_replay
    manifest |= summary(items)
    manifest["n_rule_complete"] = len(complete)
    manifest["skips"] = dict(skips)
    blocks.append({
        "first_game_id": n_games,
        "n_appended": n_new,
        "t0": args.t0,
        "n_rule_complete_at_t0": len(at_fork(items, args.t0)),
        "n_rejected_rule_incomplete": rejected,
        "qualified_per_draw": round(n_new / (n_new + rejected), 4),
        "reason": REASON,
    })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(items['t0'])} rows over {manifest['n_games']} games, "
          f"{len(complete)} of them rule-complete")
    print(json.dumps({"branch_counts": manifest["branch_counts"],
                      "skips": manifest["skips"],
                      "appended": manifest["appended"][-1]}, indent=2))


if __name__ == "__main__":
    main()
