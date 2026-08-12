"""Step 3b: `scan.py`, but over prefixes the model wrote for itself.

`scan.py` measures P(illegal move) at position t of a *corpus* game -- a legal prefix the
model was handed. This measures the same quantity at position t of a game the model
generated, which is the prefix distribution it is actually in once it starts playing. The
two differ only in where the legal prefix came from, so the by-rule figures they feed are
the same picture under two conditioning distributions.

**A rollout is only scannable while it is still legal.** After its first illegal move no
board exists -- `OthelloBoardState.umpire` asserts on an occupied square and on a move that
flips nothing -- so a game contributes rows for t < `first_illegal` and stops. The last row
it contributes is the position whose distribution the illegal move was drawn from, which is
the one worth having. n therefore falls along the move axis (~1000 games at t = 0 to ~885 at
t = 55) and every read-out of this table is conditional on survival: a hazard, not a
marginal. `scan.py`'s table is conditional on a legal prefix too, so the comparison holds;
what does not hold is reading either as "P(illegal at ply t)" unconditionally.

Positions where the mover forfeits are dropped one at a time rather than taking the whole
game out, as `positions.py` does: n already varies with t here, so a second whole-game rule
would only add a bias without buying a constant n.

The games are the first `N_GAMES` of `spade.othello.rollout_bank`, not a draw of this
study's own: one population, generated once, shared with every other rollout read-out.

Usage: python -m spade.generation_legality.rollout_scan
"""

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.generation_legality.positions import N_SQUARES, POSITIONS
from spade.generation_legality.scan import D_VOCAB, next_move_probs, table
from spade.othello.games import classify_prefix
from spade.othello.rollout_bank import load as load_bank

force_ipv4()

N_GAMES = 1000  # the prefix of the bank this study reads; see othello.rollout_bank


def alive_rows(tokens: torch.Tensor, first_illegal: np.ndarray,
               skips: Counter) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(row index into the flat (game, t) grid, game_id, t, cats) for every scannable row.

    A row (g, t) is scannable when tokens[g, :t+1] is legal end to end, which is
    t < first_illegal[g], and when the mover does not forfeit there.
    """
    rows, games, ts, cats = [], [], [], []
    limit = np.where(first_illegal < 0, CTX_LEN, first_illegal)
    for g, sequence in enumerate(tokens.tolist()):
        for i, t in enumerate(POSITIONS):
            if t >= limit[g]:
                skips["after_first_illegal"] += 1
                continue
            prefix = classify_prefix(sequence, t + 1, skips)
            if prefix is None:
                continue
            rows.append(g * len(POSITIONS) + i)
            games.append(g)
            ts.append(t)
            cats.append(prefix[1])
        if (g + 1) % 100 == 0:
            print(f"classified {g + 1}/{len(tokens)} rollouts", flush=True)
    return (np.array(rows), np.array(games, dtype=np.int32), np.array(ts, dtype=np.int16),
            np.array(cats, dtype=np.int8))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    from transformer_lens import HookedTransformer

    base = study_dir("generation") / "rollout"
    base.mkdir(parents=True, exist_ok=True)
    tokens, first = load_bank(N_GAMES)
    first_illegal = first.numpy()
    assert tokens.shape == (N_GAMES, CTX_LEN) and (tokens > 0).all()

    skips = Counter()
    rows, game_id, t, cats = alive_rows(tokens, first_illegal, skips)
    assert cats.shape == (len(rows), N_SQUARES), cats.shape
    assert (cats == 0).sum(1).min() > 0, "a scannable position has no legal square"

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()
    assert model.cfg.d_vocab == D_VOCAB, f"vocab is {model.cfg.d_vocab}, not {D_VOCAB}"

    probs = next_move_probs(model, tokens, device)[rows]
    frame = table(probs, cats, game_id, t)
    frame.to_csv(base / "positions.csv", index=False)

    alive = frame.groupby("t").size()
    (base / "scan_manifest.json").write_text(json.dumps({
        "n_games": len(first_illegal),
        "games": "the first N_GAMES of spade.othello.rollout_bank",
        "n_rows": len(frame),
        "position_range": [POSITIONS.start, POSITIONS.stop],
        "n_alive_first_t": int(alive.iloc[0]),
        "n_alive_last_t": int(alive.iloc[-1]),
        "row_of": "row (g, t) holds the model's next-move distribution at position t of its "
                  "OWN generated game g, and the taxonomy of the move at ply t+1",
        "conditional_on": "the rollout being legal through ply t; games leave the table at "
                          "their first illegal move and never return",
        "skips": dict(skips),
    }, indent=2))

    illegal = frame.p_R1 + frame.p_R2 + frame.p_R3
    print(f"wrote {len(frame)} rows from {len(first_illegal)} rollouts to "
          f"{base / 'positions.csv'}\nalive {int(alive.iloc[0])} -> {int(alive.iloc[-1])} "
          f"games over t\nmean p_illegal {illegal.mean():.3e}  median {illegal.median():.3e}\n"
          f"skips: {dict(skips)}")


if __name__ == "__main__":
    main()
