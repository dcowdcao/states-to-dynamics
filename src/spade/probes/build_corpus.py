"""Board-state probe corpus: OthelloGPT activations at every layer over adamkarvonen games.

One corpus for every probe in the project, so a probe's training set is a property
of the project rather than of whichever study fit it first. Stores the ABSOLUTE
board and the mover from engine replay, never `t % 2`: forfeits desync parity and
the mover-relative label is derived downstream.

Positions 0..57 are all kept. The window a probe trains on is the trainer's
choice, not the corpus's — the opening is near-forced and the endgame has no
empty squares, and a trainer that wants to exclude them should say so itself.

Usage: python src/probes/build_corpus.py [--n-games 10000] [--seed 42]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch


from spade.common.config import HF_REVISIONS, MODEL_NAME
from spade.common.paths import add_karvonen_to_path
from spade.probes.study_dir import probes_dir

add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, to_string

DATASET = "adamkarvonen/othello_45MB_games"
MOVES = 60          # every game in this corpus is a full game
KEEP = 58           # the model is fed tokens[:-1], so positions 0..57 carry a board
BATCH = 256
LAYERS = tuple(range(8))


def hook(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def games(n: int, seed: int) -> np.ndarray:
    """[n, 60] vocab ids, pass-free. Materialized before the model runs, because the
    streaming parquet reader faults during interpreter shutdown if left open."""
    from datasets import load_dataset
    stream = load_dataset(DATASET, split="train", streaming=True, revision=HF_REVISIONS[DATASET])
    it = iter(stream.shuffle(seed=seed, buffer_size=10000))
    out, skipped = [], 0
    while len(out) < n:
        tokens = next(it)["tokens"]
        if len(tokens) < MOVES or 0 in tokens:
            skipped += 1
            continue
        out.append(tokens[:MOVES])
    del it, stream
    print(f"streamed {n} games, skipped {skipped} (short or containing a pass)")
    return np.array(out, dtype=np.int16)


def boards(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute state after each move [n, KEEP, 8, 8] and the mover next to play [n, KEEP]."""
    n = len(tokens)
    state = np.zeros((n, KEEP, 8, 8), dtype=np.int8)
    mover = np.zeros((n, KEEP), dtype=np.int8)
    for g in range(n):
        board = OthelloBoardState()
        for i, square in enumerate(to_string(tokens[g].tolist())[:KEEP]):
            board.umpire(square)
            state[g, i] = board.state
            mover[g, i] = board.next_hand_color
    return state, mover


def activations(model, tokens: np.ndarray, device: str) -> dict[int, torch.Tensor]:
    """{layer: [n, KEEP, 512] fp16}. Every layer in one pass — the forward is the cost."""
    names = {hook(l) for l in LAYERS}
    acts = {l: torch.zeros(len(tokens), KEEP, 512, dtype=torch.float16) for l in LAYERS}
    for start in range(0, len(tokens), BATCH):
        batch = torch.from_numpy(tokens[start:start + BATCH, :-1]).long().to(device)
        _, cache = model.run_with_cache(batch, names_filter=lambda n: n in names)
        for l in LAYERS:
            acts[l][start:start + BATCH] = cache[hook(l)][:, :KEEP, :].half().cpu()
        del cache
        if start % (BATCH * 20) == 0:
            print(f"  activations {start + len(batch)}/{len(tokens)}", flush=True)
    return acts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = probes_dir() / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    tokens = games(args.n_games, args.seed)
    state, mover = boards(tokens)
    print(f"replayed boards: {state.shape}, mover balance {np.bincount((mover.ravel() + 1) // 2)}")

    torch.set_grad_enabled(False)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=args.device)
    acts = activations(model, tokens, args.device)

    torch.save({"state": torch.from_numpy(state), "mover": torch.from_numpy(mover),
                "tokens": torch.from_numpy(tokens), "dataset": DATASET,
                "seed": args.seed, "n_games": len(tokens), "positions": KEEP},
               out / "labels.pt")
    for layer, tensor in acts.items():
        torch.save(tensor, out / f"acts_layer{layer}.pt")
    print(f"wrote {out}: labels.pt + {len(acts)} layer tensors, each {tuple(acts[0].shape)}")


if __name__ == "__main__":
    main()
