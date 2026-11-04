"""The canonical free-running rollout population: generated once, read by every study.

Two studies called `straightness.build_games.sample_rollouts` and each spent its own GPU
pass on it, at its own seed -- so a rollout number from one was not about the same games as
a rollout number from the other, and neither draw was reusable. This is that population,
once, under the data root, with a manifest that says what generated it.

**The bank extends by prefix, and that is what makes one bank enough.** `sample_rollouts`
draws in `ROLLOUT_BATCH` blocks from a single seeded generator, so block k depends only on
the seed and on every earlier block being full-size. A consumer asking for n takes the
bank's first n games; growing the bank appends without touching them, provided the old size
was a whole number of blocks. `build` enforces that and re-derives the overlap rather than
trusting it.

A consumer that needs a *different* population -- another seed, another temperature -- does
not belong here; it should say so and generate its own, because the point of the bank is
that "the rollouts" names one thing.

Usage: python -m spade.othello.rollout_bank [--n_games 1000]
"""

import argparse
import json

import torch

from spade.common.config import CTX_LEN, HF_REVISIONS, MODEL_NAME, provenance
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.straightness.build_games import ROLLOUT_BATCH, sample_rollouts

force_ipv4()

SEED = 42
TEMPERATURE = 1.0


def path():
    return study_dir("rollouts")


def load(n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens [n, 59] int16, first_illegal [n] long) -- the bank's first n games.

    `first_illegal` is the 0-indexed ply of the first engine-illegal move, -1 if none. A
    caller wanting more games than the bank holds gets a hard error naming the command that
    extends it, never a silently short population.
    """
    base = path()
    tokens = torch.load(base / "tokens.pt", weights_only=True)
    first_illegal = torch.load(base / "first_illegal.pt", weights_only=True)
    assert len(tokens) == len(first_illegal), "bank tensors disagree on size"
    if n is None:
        return tokens, first_illegal
    assert n <= len(tokens), (
        f"the bank holds {len(tokens)} games and {n} were asked for; extend it with "
        f"`python -m spade.othello.rollout_bank --n_games {n}`")
    return tokens[:n], first_illegal[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_games", type=int, default=1000)
    args = parser.parse_args()
    assert args.n_games % ROLLOUT_BATCH == 0, (
        f"the bank size must be a whole number of {ROLLOUT_BATCH}-game blocks, or the next "
        "extension will not reproduce the games already in it")

    from transformer_lens import HookedTransformer

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()

    seqs, first_illegal = sample_rollouts(model, device, args.n_games, SEED)
    assert all(len(s) == CTX_LEN and 0 not in s for s in seqs)
    tokens = torch.tensor(seqs, dtype=torch.int16)
    first = torch.tensor(first_illegal, dtype=torch.long)

    base = path()
    base.mkdir(parents=True, exist_ok=True)
    if (base / "tokens.pt").exists():
        previous, previous_first = load()
        keep = len(previous)
        assert keep <= len(tokens), f"the bank already holds {keep} games; this run drew fewer"
        assert torch.equal(tokens[:keep], previous) and torch.equal(first[:keep], previous_first), \
            "the redraw does not reproduce the games already in the bank"
        print(f"reproduced the existing {keep} games exactly")

    torch.save(tokens, base / "tokens.pt")
    torch.save(first, base / "first_illegal.pt")
    (base / "manifest.json").write_text(json.dumps({
        "seed": SEED, "n_games": args.n_games, "temperature": TEMPERATURE,
        "n_tokens": CTX_LEN, "model": MODEL_NAME, "batch": ROLLOUT_BATCH,
        "sampler": "spade.straightness.build_games.sample_rollouts",
        "opening": "uniform over the four legal first moves",
        "first_illegal": "0-indexed ply of the first engine-illegal move, -1 if none",
        "extends_by_prefix": "the first k games are the same for any bank size >= k, as long "
                             "as k is a multiple of `batch`",
        **provenance(),
    }, indent=2))
    clean = int((first < 0).sum())
    print(f"bank of {args.n_games} rollouts at {base}\nfully legal {clean}/{args.n_games}")


if __name__ == "__main__":
    main()
