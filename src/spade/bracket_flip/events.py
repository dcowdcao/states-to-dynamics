"""Step 1 of the bracket-flip study: mine single-flip events and the probe-training set.

An event is a move that flips exactly one disc: for prefix P = tokens[:t], board
B = replay(P) and mover e, the played move m has flip_set(B, m, e) == [s].
Positions where a side forfeits at t or at t+1 are dropped -- a forfeit at t+1
leaves e to move again, which inverts the "s reads theirs before and after"
prediction the study is built on.

The probe-training set samples every position in [5, 55) of a disjoint slice of
the same stream and carries both label conventions, mover-relative and absolute.
Whose turn it is comes from engine replay: forfeits are unmarked in this corpus,
so move-number parity desynchronizes from the mover.

Usage: python -m spade.bracket_flip.events [--n_events 2000] [--n_probe_games 400]
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from random import Random

import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN
from spade.common.study import study_dir, variant_dir
from spade.othello.boards import flip_set, next_mover, rays_touch, regular_moves, replay
from spade.othello.games import T_MAX, T_MIN, stream_games  # imports force_ipv4() + karvonen path

from circuits.othello_engine_utils import OthelloBoardState, stoi, to_string

PROBE_T_MIN, PROBE_T_MAX = 5, 55  # positions of analysis/linear_probe/activations
EMPTY, PLACED, FLIPPED, MOVER_DISC, OPP_DISC = 0, 1, 2, 3, 4  # control_squares codes


@dataclass(frozen=True)
class Candidate:
    t: int
    mover: int  # +1 black, -1 white
    move: int  # vocab token
    move_square: int
    square_s: int
    legal: frozenset[int]  # regular moves for the mover at t
    before: np.ndarray
    after: np.ndarray


@dataclass(frozen=True)
class Control:
    move: int  # vocab token
    kind: str  # strict | loose
    flips: tuple[int, ...]


def usable(tokens: list[int], skips: Counter) -> bool:
    if len(tokens) < CTX_LEN:
        skips["short_game"] += 1
        return False
    if 0 in tokens:
        skips["pass_token"] += 1
        return False
    return True


def candidates(tokens: list[int], skips: Counter) -> list[Candidate]:
    board = OthelloBoardState()
    out = []
    for t, token in enumerate(tokens[: T_MAX + 1]):
        square = to_string(token)
        before, mover = board.state.copy(), board.next_hand_color
        board.umpire(square)
        if t < T_MIN:
            continue
        legal = regular_moves(before, mover)
        if not legal:
            skips["forfeit_at_t"] += 1
            continue
        if square not in legal:
            skips["move_not_regular"] += 1
            continue
        flipped = flip_set(before, square, mover)
        if len(flipped) != 1:
            skips["multi_flip"] += 1
            continue
        if not regular_moves(board.state, -mover):
            skips["forfeit_after"] += 1
            continue
        out.append(Candidate(t, mover, token, square, flipped[0], frozenset(legal),
                             before, board.state.copy()))
    return out


def controls(tokens: list[int], c: Candidate) -> list[Control]:
    """Other legal moves for the mover that leave s unflipped and do not forfeit the opponent.

    `strict` additionally keeps the placement and every disc it flips off s's
    eight rays, so the control branch cannot reach s along any line.
    """
    out = []
    for square in sorted(c.legal - {c.move_square}):
        flips = flip_set(c.before, square, c.mover)
        if c.square_s in flips:
            continue
        if not regular_moves(replay(tokens[: c.t] + [stoi[square]]).state, -c.mover):
            continue
        kind = "loose" if rays_touch(c.square_s, {square, *flips}) else "strict"
        out.append(Control(stoi[square], kind, tuple(sorted(flips))))
    return out


def verify(tokens: list[int], c: Candidate) -> None:
    """Every check the study rests on, re-derived from `replay` rather than the incremental walk."""
    assert c.t >= T_MIN, f"t={c.t} below {T_MIN}"
    assert np.array_equal(c.before, replay(tokens[: c.t]).state), "before != replay(P)"
    assert np.array_equal(c.after, replay(tokens[: c.t + 1]).state), "after != replay(P + m)"
    assert (c.before != c.after).sum() == 2, "board changed in other than two squares"
    mr, mc = c.move_square // 8, c.move_square % 8
    sr, sc = c.square_s // 8, c.square_s % 8
    assert c.before[mr, mc] == 0, "played square not empty before"
    assert c.after[mr, mc] == c.mover, "played square not the mover's colour after"
    assert c.before[sr, sc] == -c.mover, "s not the opponent's colour before"
    assert c.after[sr, sc] == c.mover, "s not the mover's colour after"


def square_codes(c: Candidate) -> str:
    before = c.before.reshape(64)
    codes = []
    for square in range(64):
        if square == c.move_square:
            codes.append(PLACED)
        elif square == c.square_s:
            codes.append(FLIPPED)
        elif before[square] == 0:
            codes.append(EMPTY)
        elif before[square] == c.mover:
            codes.append(MOVER_DISC)
        else:
            codes.append(OPP_DISC)
    return "".join(map(str, codes))


def event_row(tokens: list[int], game_id: int, c: Candidate, ctrl: Control) -> dict:
    return {
        "game_id": game_id, "t": c.t, "mover": c.mover, "move": c.move,
        "square_s": c.square_s, "flip_set": c.square_s,
        "control_move": ctrl.move, "control_kind": ctrl.kind,
        "control_flip_set": " ".join(map(str, ctrl.flips)),
        "n_legal": len(c.legal), "control_squares": square_codes(c),
        "prefix": " ".join(map(str, tokens[: c.t])),
    }


def mine_events(games, n_events: int, rng: Random, skips: Counter) -> tuple[list[dict], dict]:
    rows, n_games, n_candidates = [], 0, 0
    while len(rows) < n_events:
        tokens = next(games)["tokens"]
        n_games += 1
        if not usable(tokens, skips):
            continue
        found = candidates(tokens, skips)
        n_candidates += len(found)
        rng.shuffle(found)
        for c in found:
            options = controls(tokens, c)
            if not options:
                skips["no_control"] += 1
                continue
            strict = [o for o in options if o.kind == "strict"]
            verify(tokens, c)
            rows.append(event_row(tokens, n_games - 1, c, rng.choice(strict or options)))
            break
        if n_games % 500 == 0:
            print(f"{len(rows)}/{n_events} events, {n_games} games scanned")
    return rows, {"games_scanned": n_games, "candidate_events": n_candidates}


def probe_labels(tokens: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per position in [PROBE_T_MIN, PROBE_T_MAX): relative board, absolute board, mover next."""
    board = OthelloBoardState()
    relative, absolute, movers = [], [], []
    for t, token in enumerate(tokens[:PROBE_T_MAX]):
        board.umpire(to_string(token))
        if t < PROBE_T_MIN:
            continue
        mine = next_mover(board.state, board.next_hand_color)
        state = board.state
        relative.append(np.where(state == 0, 0, np.where(state == mine, 2, 1)))
        absolute.append(np.where(state == 0, 0, np.where(state == 1, 1, 2)))
        movers.append(mine)
    return np.stack(relative), np.stack(absolute), np.array(movers)


def build_probe_set(games, n_games: int, skips: Counter) -> dict[str, torch.Tensor]:
    seqs, relative, absolute, movers = [], [], [], []
    while len(seqs) < n_games:
        tokens = next(games)["tokens"]
        if not usable(tokens, skips):
            continue
        r, a, m = probe_labels(tokens)
        seqs.append(tokens[:CTX_LEN])
        relative.append(r)
        absolute.append(a)
        movers.append(m)
    n_pos = PROBE_T_MAX - PROBE_T_MIN
    return {
        "tokens": torch.tensor(seqs, dtype=torch.int16),
        "board_states_relative": torch.tensor(np.concatenate(relative), dtype=torch.long),
        "board_states_absolute": torch.tensor(np.concatenate(absolute), dtype=torch.long),
        "move_indices": torch.arange(PROBE_T_MIN, PROBE_T_MAX).repeat(n_games),
        "mover_next": torch.tensor(np.concatenate(movers), dtype=torch.int8),
        "game_id": torch.arange(n_games).repeat_interleave(n_pos),
    }


def parity_mismatch(probe: dict[str, torch.Tensor]) -> dict:
    """How often ARENA's `t % 2` rule names the wrong player to move (black=+1 at odd t)."""
    parity = torch.where(probe["move_indices"] % 2 == 0, -1, 1).to(torch.int8)
    wrong = parity != probe["mover_next"]
    by_game = wrong.reshape(probe["tokens"].shape[0], -1)
    return {
        "positions": int(wrong.numel()),
        "positions_wrong": int(wrong.sum()),
        "games": int(by_game.shape[0]),
        "games_with_any_wrong": int(by_game.any(1).sum()),
    }


def mine_only(games, args, skips: Counter) -> None:
    """A second event set, for a panel that needs a different t distribution than the study's.

    Same miner, same stream and same seed, so it extends the canonical corpus rather than
    replacing it: the first `n_events` of a larger draw are the smaller draw. Only the events
    are built -- the probe set the decoder is fitted on is shared and a bigger event corpus
    does not change it.
    """
    rows, yields = mine_events(games, args.n_events, Random(args.seed), skips)
    events = pd.DataFrame(rows)
    out_dir = variant_dir("bracket", variant=args.variant) / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "events.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(
        {"seed": args.seed, "dataset": args.dataset_name, "n_events": args.n_events,
         "variant": args.variant, "one_event_per_game": True,
         "t_histogram": events.t.value_counts().sort_index().to_dict(),
         "control_kind": events.control_kind.value_counts().to_dict(),
         **yields, "skips": dict(skips)}, indent=2))
    print(f"{len(events)} events -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_events", type=int, default=2000)
    parser.add_argument("--n_probe_games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    parser.add_argument("--variant", default="",
                        help="write the event set to $DIR/<variant>/events and skip the "
                             "probe set, which is shared and unchanged by a bigger corpus")
    args = parser.parse_args()

    games = stream_games(args.dataset_name, args.seed)
    probe_skips, event_skips = Counter(), Counter()

    if args.variant:
        mine_only(games, args, event_skips)
        return

    probe = build_probe_set(games, args.n_probe_games, probe_skips)
    mismatch = parity_mismatch(probe)
    probe_dir = study_dir("bracket") / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in probe.items():
        torch.save(tensor, probe_dir / f"{name}.pt")
    (probe_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "dataset": args.dataset_name,
        "n_games": args.n_probe_games,
        "n_positions": int(probe["move_indices"].shape[0]),
        "position_range": [PROBE_T_MIN, PROBE_T_MAX],
        "label_encoding": {
            "board_states_relative": {"0": "empty", "1": "theirs", "2": "mine"},
            "board_states_absolute": {"0": "empty", "1": "black", "2": "white"},
        },
        "mover_next": "engine-derived colour to move at t+1 (+1 black, -1 white)",
        "parity_mismatch": mismatch,
        "skips": dict(probe_skips),
    }, indent=2))
    print(f"probe set: {probe['move_indices'].shape[0]} positions from "
          f"{args.n_probe_games} games -> {probe_dir}")
    print(f"t % 2 names the wrong mover at {mismatch['positions_wrong']}/"
          f"{mismatch['positions']} positions, in {mismatch['games_with_any_wrong']}/"
          f"{mismatch['games']} games")

    rng = Random(args.seed)
    rows, yields = mine_events(games, args.n_events, rng, event_skips)
    events = pd.DataFrame(rows)
    assert len(events) == args.n_events
    assert events["t"].min() >= T_MIN and events["t"].max() <= T_MAX

    out_dir = variant_dir("bracket", variant=args.variant) / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "events.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "dataset": args.dataset_name,
        "n_events": len(events),
        "one_event_per_game": True,
        "t_range": [T_MIN, T_MAX],
        "t_histogram": {str(k): int(v) for k, v in sorted(events["t"].value_counts().items())},
        "control_kinds": {k: int(v) for k, v in events["control_kind"].value_counts().items()},
        "square_codes": {"empty": EMPTY, "placed": PLACED, "flipped": FLIPPED,
                         "mover_disc": MOVER_DISC, "opp_disc": OPP_DISC},
        "rejections": dict(event_skips),
        **yields,
        "candidates_per_1000_games": round(
            1000 * yields["candidate_events"] / yields["games_scanned"], 1),
    }, indent=2))
    print(f"wrote {len(events)} events from {yields['games_scanned']} games to {out_dir}")
    print(json.dumps({"rejections": dict(event_skips), **yields}, indent=2))


if __name__ == "__main__":
    main()
