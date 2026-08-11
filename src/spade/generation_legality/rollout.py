"""Step 3: free-running games, and the ply at which the first illegal move appears.

The teacher-forced scan of `scan.py` never lets the model see its own output, so composing
its per-position mass into a whole-game legality is a statement about a prefix distribution
the model is never in once it starts generating. This arm is the measured version: whole
59-token games sampled at temperature 1.0 from a uniform-random legal opening, with the
engine asked where the first illegal move is.

`spade.straightness.build_games.sample_rollouts` is that sampler and is reused unchanged --
same temperature, same pass-token mask, same `first_illegal_pos` from
`spade.othello.games` -- so this arm and the straightness study's `rollout` class are the
same population under different seeds.

Usage: python -m spade.generation_legality.rollout [--n_games 1000]
"""

import argparse
import json

import pandas as pd
import torch

from spade.common.config import CTX_LEN, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.straightness.build_games import sample_rollouts

force_ipv4()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()

    seqs, first_illegal = sample_rollouts(model, device, args.n_games, args.seed)
    assert all(len(s) == CTX_LEN and 0 not in s for s in seqs)

    out_dir = study_dir("generation") / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": range(args.n_games), "first_illegal": first_illegal}) \
        .to_csv(out_dir / "rollouts.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "n_games": args.n_games, "temperature": 1.0,
        "n_tokens": CTX_LEN, "model": MODEL_NAME,
        "sampler": "spade.straightness.build_games.sample_rollouts",
        "first_illegal": "0-indexed ply of the first engine-illegal move, -1 if none",
    }, indent=2))

    clean = sum(1 for p in first_illegal if p < 0)
    print(f"wrote {args.n_games} rollouts to {out_dir}\n"
          f"fully legal {clean}/{args.n_games}")


if __name__ == "__main__":
    main()
