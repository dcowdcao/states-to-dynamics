"""Step 2: one teacher-forced pass over the position set -> mass, and the move played.

Every game is 59 tokens, so a single forward pass gives the next-move distribution at every
position at once: `logits[:, t]` predicts the move at ply t+1, which is the ply
`positions.py` classified at row (g, t). Nothing is generated here -- the prefix is always
the real legal game -- so this measures what the model would emit at an on-distribution
position, not what a rollout does to itself.

The pass token (0) never occurs in this corpus and is **not** masked: its mass is a column
of the output, so a reader can see it rather than have it folded into `p_legal`.

`sampled_category` is ONE draw per position at temperature 1.0 under SAMPLE_SEED, not a
mean over draws (FIGURE_GUIDELINES 3): a per-position average over k draws would have
~1/sqrt(k) the spread of the greedy arm it is drawn beside.

Usage: python -m spade.generation_legality.scan
"""

import argparse

import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN, HF_REVISIONS, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.generation_legality.positions import N_SQUARES, POSITIONS
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_MEMBERS

force_ipv4()

BATCH_SIZE = 256
TEMPERATURE = 1.0
SAMPLE_SEED = 42
PASS_TOKEN = 0
D_VOCAB = 1 + N_SQUARES


def next_move_probs(model, tokens: torch.Tensor, device: str) -> torch.Tensor:
    """[n_games * len(POSITIONS), D_VOCAB], rows in (game, t) order."""
    out = torch.empty(len(tokens), len(POSITIONS), D_VOCAB)
    for start in range(0, len(tokens), BATCH_SIZE):
        batch = tokens[start : start + BATCH_SIZE].long().to(device)
        logits = model(batch)[:, list(POSITIONS)].float()
        out[start : start + BATCH_SIZE] = (logits / TEMPERATURE).softmax(-1).cpu()
        print(f"scanned {min(start + BATCH_SIZE, len(tokens))}/{len(tokens)}")
    return out.reshape(-1, D_VOCAB)


def played(probs: torch.Tensor, cats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Categories of the argmax move and of one temperature-1.0 draw per position."""
    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    names = np.array(CATEGORIES + ["pass"])
    rows = np.arange(len(cats))
    out = []
    for token in (probs.argmax(1), torch.multinomial(probs, 1, generator=generator)[:, 0]):
        token = token.numpy()
        square = cats[rows, np.maximum(token - 1, 0)]
        out.append(names[np.where(token == PASS_TOKEN, len(CATEGORIES), square)])
    return tuple(out)


def table(probs: torch.Tensor, cats: np.ndarray, game_id: np.ndarray,
          t: np.ndarray) -> pd.DataFrame:
    square_probs = probs[:, 1:].numpy().astype(np.float64)
    mass = {c: (square_probs * (cats == i)).sum(1) for i, c in enumerate(CATEGORIES)}
    count = {c: (cats == i).sum(1) for i, c in enumerate(CATEGORIES)}
    greedy, sampled = played(probs, cats)

    frame = pd.DataFrame({
        "game_id": game_id, "t": t,
        "n_legal": count[CATEGORIES[LEGAL]],
        "p_legal": mass[CATEGORIES[LEGAL]],
        "p_pass": probs[:, PASS_TOKEN].numpy().astype(np.float64),
        "greedy_category": greedy, "sampled_category": sampled,
    })
    for rule, members in RULE_MEMBERS.items():
        key = rule.split()[0]
        frame[f"p_{key}"] = sum(mass[c] for c in members)
        frame[f"n_{key}"] = sum(count[c] for c in members)
        frame[f"avail_{key}"] = frame[f"n_{key}"] / N_SQUARES
    for category in CATEGORIES[1:]:
        frame[f"p_{category}"] = mass[category]
        frame[f"n_{category}"] = count[category]

    rules = [r.split()[0] for r in RULE_MEMBERS]
    total = frame.p_legal + frame.p_pass + sum(frame[f"p_{r}"] for r in rules)
    assert np.allclose(total, 1.0, atol=1e-5), "category masses do not partition the softmax"
    assert (frame.n_legal + sum(frame[f"n_{r}"] for r in rules) == N_SQUARES).all()
    return frame


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    from transformer_lens import HookedTransformer

    base = study_dir("generation")
    tokens = torch.load(base / "positions" / "tokens.pt", weights_only=True)
    cats = torch.load(base / "positions" / "cats.pt", weights_only=True).numpy()
    game_id = torch.load(base / "positions" / "game_id.pt", weights_only=True).numpy()
    t = torch.load(base / "positions" / "t.pt", weights_only=True).numpy()
    assert tokens.shape[1] == CTX_LEN and (tokens > 0).all()
    assert len(cats) == len(tokens) * len(POSITIONS)

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()
    assert model.cfg.d_vocab == D_VOCAB, f"vocab is {model.cfg.d_vocab}, not {D_VOCAB}"

    frame = table(next_move_probs(model, tokens, device), cats, game_id, t)

    out_dir = base / "scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "positions.csv", index=False)
    illegal = frame.p_R1 + frame.p_R2 + frame.p_R3
    print(f"wrote {len(frame)} positions to {out_dir / 'positions.csv'}\n"
          f"mean p_illegal {illegal.mean():.3e}  median {illegal.median():.3e}  "
          f"max {illegal.max():.3e}\n"
          f"mean p_pass {frame.p_pass.mean():.3e}\n"
          f"greedy illegal {(frame.greedy_category != 'legal').sum()}/{len(frame)}  "
          f"sampled illegal {(frame.sampled_category != 'legal').sum()}/{len(frame)}")


if __name__ == "__main__":
    main()
