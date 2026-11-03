"""Step 1 of the manifold-capture study: natural game positions + concept labels.

Streams legal games and replays them through the engine, recording per position
p in 1..58 (position 0 is the attention sink and is dropped everywhere):
  t          = p
  board      = mover-relative square states {0 empty, 1 theirs, 2 mine},
               "mine" = player to move next (linear-probe convention)
  n_flipped  = discs flipped by the move just played
  n_legal    = flip-yielding moves available to the player to move (mobility;
               the engine's forfeit fallback means this is 0 only at game end)
  disc_diff  = (#mine - #theirs) discs on the board

All label tensors are flattened to one row per position ([n_games * 58]) so
analyses index rows directly; game_id recovers game identity.

Usage: python -m spade.manifold_capture.build_corpus [--n_games 4000] [--seed 42]
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch

from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.othello.boards import regular_moves
from spade.othello.games import collect_legal, first_illegal_pos, stream_games

force_ipv4()

from circuits.othello_engine_utils import OthelloBoardState, to_string

from spade.common.config import CTX_LEN

POSITIONS = range(1, CTX_LEN)  # position 0 dropped (attention sink)


def replay_labels(tokens: list[int]) -> dict[str, np.ndarray]:
    """Per-position labels for one game, rows aligned with POSITIONS."""
    board = OthelloBoardState()
    rows = {k: [] for k in ["board", "n_flipped", "n_legal", "disc_diff"]}
    for p, tok in enumerate(tokens):
        before = board.state.copy()
        board.umpire(to_string(tok))
        if p == 0:
            continue
        mover = board.next_hand_color
        rel = np.zeros((8, 8), dtype=np.int8)
        rel[board.state == mover] = 2
        rel[board.state == -mover] = 1
        rows["board"].append(rel)
        rows["n_flipped"].append(int((board.state != before).sum()) - 1)
        rows["n_legal"].append(len(regular_moves(board.state, mover)))
        rows["disc_diff"].append(int((board.state == mover).sum())
                                 - int((board.state == -mover).sum()))
    return {k: np.array(v) for k, v in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="adamkarvonen/othello_45MB_games")
    args = parser.parse_args()

    skips = Counter()
    games = collect_legal(stream_games(args.dataset_name, args.seed), args.n_games, skips)
    for seq in games[:50]:
        assert first_illegal_pos(seq) == -1, "legal game fails engine replay"

    labels = [replay_labels(g) for g in games]
    n_pos = len(POSITIONS)
    data = {
        "tokens": torch.tensor(games, dtype=torch.int16),
        "game_id": torch.arange(args.n_games).repeat_interleave(n_pos),
        "t": torch.tensor(list(POSITIONS), dtype=torch.int16).repeat(args.n_games),
        "board": torch.from_numpy(np.concatenate([l["board"] for l in labels])),
        **{k: torch.tensor(np.concatenate([l[k] for l in labels]), dtype=torch.int8)
           for k in ["n_flipped", "n_legal", "disc_diff"]},
    }
    n = args.n_games * n_pos
    assert data["board"].shape == (n, 8, 8)
    assert (data["n_flipped"] >= 1).all(), "legal move flipped no discs"

    out_dir = study_dir("manifold") / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "corpus.pt")
    manifest = {
        "seed": args.seed,
        "dataset": args.dataset_name,
        "n_games": args.n_games,
        "n_rows": n,
        "positions": [POSITIONS.start, POSITIONS.stop - 1],
        "board_convention": "mover-relative {0 empty, 1 theirs, 2 mine}",
        "skips": dict(skips),
        "n_flipped_hist": dict(Counter(data["n_flipped"].tolist())),
        "n_legal_zero_rows": int((data["n_legal"] == 0).sum()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {n} rows ({args.n_games} games) to {out_dir}")


if __name__ == "__main__":
    main()
