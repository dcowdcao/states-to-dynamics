"""Append rule-complete source games to branches.pt, renumbering nothing.

The 2000 games branches.py drew give per-rule curves at n = 2000 for R1 and R2 but
only 1997 for R3: `opponent_only` is R3's single member and three t0s offer no
candidate for it. This draws further games from the same stream, keeps the ones
whose branches cover all three rules, and appends them with `game_id` continuing
from the last.

**Append, never regenerate.** `game_id` is the accepted-game index, so re-running
branches.py under any changed acceptance rule renumbers every game. That would move
`FIGURE_GAME`, break the row alignment between branches.pt and acts/*.pt, break the
rollout tree's `row = source_row * K + k`, and invalidate the game-selection studies.
So the first `n` games are *replayed* -- the same seed, the same stream, the same
`draw` -- and asserted equal to the saved tensor column by column before anything is
appended. That replay is also what leaves the stream and the RNG at the position the
original run left them, which is what makes the appended draw a continuation rather
than a new sample.

The filter is applied to games in stream order and to nothing else: the first
qualifying games are taken, never the best ones, so the appended set is conditioned
on rule membership and on no measured outcome.

Writes `rule_complete_games.json` beside branches.pt -- the ids a per-rule figure
should read instead of re-deriving the membership rule.

Usage: python -m spade.violation_divergence.extend_branches [--target 2000]
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
    DTYPES, draw, rule_complete_games, rules_present, summary, t0_for, verify)

REASON = ("R3's only member opponent_only has no candidate at three of the original "
          "games' t0, leaving the many-game per-rule curves at n = 1997 for R3 against "
          "2000 for R1 and R2; the appended games continue the same stream and seed and "
          "are kept only if their branches cover every rule")


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


def replay(games, rng: Random, skips: Counter, n_games: int) -> list[tuple[int, dict]]:
    """Redraw the games already on disk, which walks the stream back to where it stopped."""
    out = []
    for game_id in range(n_games):
        branches = None
        while branches is None:
            branches = draw(games, rng, t0_for(game_id), skips)
        out += [(game_id, branch) for branch in branches]
    return out


def extend(games, rng: Random, skips: Counter, first_id: int, n_new: int) -> list[tuple[int, dict]]:
    """The next `n_new` games in stream order whose branches cover R1, R2 and R3."""
    out, game_id = [], first_id
    while game_id - first_id < n_new:
        branches = draw(games, rng, t0_for(game_id), skips)
        if branches is None:
            continue
        if len(rules_present(branch["category"] for branch in branches)) < len(RULE_MEMBERS):
            skips["rule_incomplete"] += 1
            continue
        out += [(game_id, branch) for branch in branches]
        game_id += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000,
                        help="rule-complete games wanted in the tree in total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    out_dir = study_dir("divergence") / "branches"
    path = out_dir / "branches.pt"
    saved = torch.load(path, weights_only=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    n_games, n_rows = len(saved["game_id"].unique()), len(saved["t0"])
    n_new = args.target - len(rule_complete_games(saved))
    assert n_new > 0, f"{args.target} rule-complete games are already in the tree"
    print(f"{n_games} games / {n_rows} rows on disk, "
          f"{args.target - n_new} rule-complete; drawing {n_new} more")
    print(f"branches.pt sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    before = digest(saved, n_rows)

    rng = Random(args.seed)
    games = stream_games(args.dataset_name, args.seed)
    skips = Counter()
    replayed = tensors(replay(games, rng, skips, n_games))
    for key, column in saved.items():
        assert torch.equal(replayed[key], column), f"the replay does not reproduce {key}"
    assert dict(skips) == manifest["skips"], \
        f"the replay skipped {dict(skips)}, the manifest recorded {manifest['skips']}"
    print("replay reproduces every saved column exactly; the stream is where it was left")

    appended = extend(games, rng, skips, n_games, n_new)
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

    manifest |= summary(items)
    manifest["n_rule_complete"] = len(complete)
    manifest["skips"] = dict(skips)
    manifest["appended"] = {
        "n_original": n_games,
        "n_appended": len(items["game_id"].unique()) - n_games,
        "first_game_id": n_games,
        "n_rejected_rule_incomplete": skips["rule_incomplete"],
        "reason": REASON,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(items['t0'])} rows over {manifest['n_games']} games, "
          f"{len(complete)} of them rule-complete")
    print(json.dumps({"branch_counts": manifest["branch_counts"],
                      "skips": manifest["skips"],
                      "appended": manifest["appended"]}, indent=2))


if __name__ == "__main__":
    main()
