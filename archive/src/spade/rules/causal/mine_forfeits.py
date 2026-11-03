"""R4 mining: natural forfeit minimal pairs from the Karvonen dataset.

A forfeit event: mover e plays move a, after which the opponent d has no
regular move anywhere, so e moves again — unmarked in the token stream. The
minimal pair keeps the shared prefix and swaps a for an alternative legal
move b of e's after which d does have moves:

  clean     = prefix + a   (next mover: e again — requires the forfeit
                            conclusion, a universal quantifier over the board)
  corrupted = prefix + b   (next mover: d, normal alternation)

Stored per event, all computed on the corrupted board B_b so they are
color-diagnostic there: E_diag = e's hypothetical regular moves minus d's,
D_diag = d's minus e's (both required nonempty), plus each board's true legal
set for the behavioral baseline.

Usage: python -m spade.rules.causal.mine_forfeits [--n_events 800] [--seed 123]
"""

import argparse
import json
from collections import Counter
from random import Random

import torch

from spade.common.config import CTX_LEN, provenance
from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path
from spade.common.study import study_dir
from spade.othello.boards import flip_set, regular_moves, replay, tokens_of

force_ipv4()
add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, stoi, to_string
from spade.othello.taxonomy import flips

J_MIN = 5


def event_from(game: list[int], j: int, board_before_a: OthelloBoardState,
               rng: Random, skips: Counter) -> dict | None:
    """Minimal pair for the forfeit at token index j (0-based), or None.
    `board_before_a` is the position before token j-1."""
    e = board_before_a.next_hand_color
    a_sq = to_string(game[j - 1])
    if not flips(board_before_a.state, a_sq, e):
        skips["prev_move_also_forfeit"] += 1
        return None
    candidates = []
    for b_sq in regular_moves(board_before_a.state, e) - {a_sq}:
        state_b = board_before_a.state.copy()
        for f in flip_set(state_b, b_sq, e):
            state_b[f // 8, f % 8] = e
        state_b[b_sq // 8, b_sq % 8] = e
        d_moves = regular_moves(state_b, -e)
        if not d_moves:
            continue  # b also forfeits d
        e_hyp = regular_moves(state_b, e)
        e_diag, d_diag = e_hyp - d_moves, d_moves - e_hyp
        if e_diag and d_diag:
            candidates.append((b_sq, d_moves, e_diag, d_diag))
    if not candidates:
        skips["no_alternative"] += 1
        return None
    b_sq, d_moves, e_diag, d_diag = rng.choice(candidates)

    board_a = replay(game[:j])
    l_a = regular_moves(board_a.state, e)  # d is forfeited: e moves again
    assert not regular_moves(board_a.state, -e), "not actually a forfeit"
    if not l_a:
        skips["game_over_after_a"] += 1
        return None
    return {
        "clean_seq": game[:j],
        "corr_seq": game[:j - 1] + [stoi[b_sq]],
        "j": j, "mover_e": e,
        "clean_legal": tokens_of(l_a),
        "corr_legal": tokens_of(d_moves),
        "e_diag": tokens_of(e_diag),
        "d_diag": tokens_of(d_diag),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_events", type=int, default=800)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    from datasets import load_dataset
    rng = Random(args.seed)
    ds = load_dataset(args.dataset_name, split="train", streaming=True)
    games = iter(ds.shuffle(seed=args.seed, buffer_size=10000))

    events, skips = [], Counter()
    n_games = 0
    while len(events) < args.n_events:
        game = next(games)["tokens"]
        n_games += 1
        if 0 in game:
            skips["pass_token_game"] += 1
            continue
        board = OthelloBoardState()
        for j, sq in enumerate(to_string(game)):
            if J_MIN <= j <= CTX_LEN and not flips(board.state, sq, board.next_hand_color):
                # rare: rebuild the pre-a board only when a forfeit is found
                event = event_from(game, j, replay(game[:j - 1]), rng, skips)
                if event:
                    events.append(event)
            board.umpire(sq)
        if n_games % 1000 == 0:
            print(f"{n_games} games scanned, {len(events)} events")

    out_dir = study_dir("rules") / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(events[:args.n_events], out_dir / "forfeit_pairs.pt")
    j_hist = Counter(e["j"] for e in events[:args.n_events])
    manifest = {"seed": args.seed, "dataset": args.dataset_name,
                "n_events": args.n_events, "n_games_scanned": n_games,
                "skips": dict(skips), "j_histogram": dict(sorted(j_hist.items())), **provenance()}
    (out_dir / "forfeit_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.n_events} events from {n_games} games; skips: {dict(skips)}")


if __name__ == "__main__":
    main()
