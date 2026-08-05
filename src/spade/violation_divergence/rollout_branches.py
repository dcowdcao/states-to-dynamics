"""Step 1b of the violation-divergence study: resample every branch's continuation.

branches.py splices. A branch keeps the source game's `tokens[t0+1:]`, and that
suffix is a legal continuation of the *reference's* board alone: it survives for
the legal control in 76.8% of games (median 2 moves), and after an illegal branch
no board exists for it to be legal on. Every branch but the reference therefore
carries an out-of-distribution future, and the reference is the only sequence the
model was ever shown the like of.

Here `tokens[:t0]` and the branch token at t0 are kept and the remaining
`58 - t0` tokens are sampled from OthelloGPT at temperature 1.0, K=4 times per
branch. Only the pass token is masked -- the model's own move distribution,
illegal moves included, is the object of study. **The reference is regenerated on
the same footing as everything else**, which is the whole point: replaying its
real continuation would leave it the one in-distribution sequence and nothing
would have been fixed.

What this costs: single-token causal attribution. Two branches now differ at t0
*and* at every position after it, so a downstream displacement is the branch move
plus everything the model chose to do about it, not the branch move alone.
Averaging the K rollouts separates sampling variance from the branch effect; it
does not restore attribution. branches.pt stays the tensor for the single-token
claim, and this one cannot be substituted for it.

Seeds: the uniforms driving the sampler are drawn per (game_id, k) from
`Generator(seed * 1_000_003 + game_id * K + k)`, so a row's continuation depends on
its game and its rollout index and on nothing else -- not on batch composition, not
on which branches the game happened to yield. All branches of a game share the k-th
uniform stream, so their K rollouts are paired (common random numbers) and the
branch effect is read against a shared noise draw.

Rows: each branches.pt row repeated K times contiguously, `row = source_row * K + k`.
Everything but `tokens` carries over unchanged, `n_flipped` and `n_legal` included --
they describe the move at t0, not the suffix. One new column, `rollout` (0..K-1).

Usage: python -m spade.violation_divergence.rollout_branches [--smoke]
"""

import argparse
import json

import torch

from spade.common.config import CTX_LEN, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.othello.taxonomy import LEGAL
from spade.violation_divergence.branches import DTYPES

force_ipv4()

K_ROLLOUTS = 4
TEMPERATURE = 1.0
PASS_TOKEN = 0
BATCH_SIZE = 1024
SEED = 42

ROLLOUT_DTYPES = {**DTYPES, "rollout": torch.int8}


def uniforms(game_id: torch.Tensor, rollout: torch.Tensor, seed: int) -> torch.Tensor:
    """u[row, pos] ~ U(0,1), one stream per (game, rollout)."""
    pairs = [(int(g), int(k)) for g, k in zip(game_id, rollout)]
    streams = {}
    for game, k in set(pairs):
        generator = torch.Generator().manual_seed(seed * 1_000_003 + game * K_ROLLOUTS + k)
        streams[(game, k)] = torch.rand(CTX_LEN, generator=generator)
    return torch.stack([streams[p] for p in pairs])


def sample_suffix(model, tokens: torch.Tensor, t0: int, u: torch.Tensor, device: str) -> torch.Tensor:
    """Fill positions t0+1.. by sampling; inverse-CDF so the draw is u's, not the RNG's.

    No KV cache: transformer_lens builds the cache's attention mask from a tokenizer
    and OthelloGPT has none, so every step re-runs the whole prefix -- ~40x a cached
    step's compute, which for an 8-layer 512-wide model is still minutes on one H100.
    """
    seq = tokens.clone()
    for pos in range(t0 + 1, CTX_LEN):
        logits = model(seq[:, :pos].to(device))[:, -1].cpu()
        logits[:, PASS_TOKEN] = -torch.inf
        cdf = (logits / TEMPERATURE).softmax(-1).cumsum(-1)
        # right=True: torch.rand can return exactly 0.0, and only this side of the
        # boundary keeps that draw off the masked pass token.
        drawn = torch.searchsorted(cdf, u[:, pos, None].contiguous(), right=True).squeeze(1)
        seq[:, pos] = drawn.clamp(max=cdf.shape[1] - 1)
    return seq


def generate(model, items: dict[str, torch.Tensor], u: torch.Tensor, device: str) -> torch.Tensor:
    """Batch by t0 so one loop drives a whole batch; a game's rows share a t0."""
    tokens = items["tokens"].long()
    done = 0
    for t0 in items["t0"].unique().tolist():
        idx = (items["t0"] == t0).nonzero().flatten()
        for start in range(0, len(idx), BATCH_SIZE):
            rows = idx[start:start + BATCH_SIZE]
            tokens[rows] = sample_suffix(model, tokens[rows], t0, u[rows], device)
            done += len(rows)
            print(f"rolled out {done}/{len(tokens)} rows (t0={t0})")
    return tokens.to(DTYPES["tokens"])


def verify(items: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    """Re-derive the interface contract and the fix itself from the saved tensors alone."""
    tokens = items["tokens"].long()
    n_source = len(source["t0"])
    assert tokens.shape == (n_source * K_ROLLOUTS, CTX_LEN), "not K rows per source branch"
    assert (tokens > 0).all(), "padding or pass token in a rollout"
    assert torch.equal(items["rollout"].long(), torch.arange(len(tokens)) % K_ROLLOUTS), \
        "rollout is not row % K"

    keys = list(zip(source["game_id"].tolist(), source["category"].tolist(),
                    source["is_control"].tolist()))
    assert len(set(keys)) == n_source, "(game_id, category, is_control) does not name one branch"
    for key in DTYPES:
        if key != "tokens":
            assert torch.equal(items[key], source[key].repeat_interleave(K_ROLLOUTS)), \
                f"{key} does not carry over from the matching source row"

    t0 = source["t0"].repeat_interleave(K_ROLLOUTS)
    through_t0 = torch.arange(CTX_LEN) <= t0[:, None]
    replayed = source["tokens"].long().repeat_interleave(K_ROLLOUTS, 0)
    assert torch.equal(tokens[through_t0], replayed[through_t0]), \
        "prefix through t0 drifted from branches.pt"

    is_reference = (source["category"] == LEGAL) & (source["is_control"] == 0)
    n_distinct, n_identical, n_replayed = 0, 0, 0
    for i in range(n_source):
        after = int(source["t0"][i]) + 1
        rows = tokens[i * K_ROLLOUTS:(i + 1) * K_ROLLOUTS, after:]
        distinct = len({tuple(r.tolist()) for r in rows})
        n_distinct += distinct
        n_identical += distinct == 1
        if is_reference[i]:
            n_replayed += int((rows == source["tokens"][i, after:]).all(-1).sum())
    # a forced endgame can collide honestly; a constant seed collides everywhere
    assert n_identical < 0.01 * n_source, \
        f"{n_identical}/{n_source} branches drew {K_ROLLOUTS} identical rollouts"
    assert n_replayed == 0, \
        f"{n_replayed} reference rollouts replay the real game -- the reference was not regenerated"
    print(f"verify ok: {n_source} branches x {K_ROLLOUTS} rollouts, "
          f"{n_distinct / n_source:.3f} distinct suffixes per branch "
          f"({n_identical} all-identical), 0 references replayed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="read and write under $DIR/smoke/")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    base = study_dir("divergence") / "smoke" if args.smoke else study_dir("divergence")
    source = torch.load(base / "branches" / "branches.pt", weights_only=True)
    items = {k: v.repeat_interleave(K_ROLLOUTS, 0) for k, v in source.items()}
    items["rollout"] = (torch.arange(len(items["t0"])) % K_ROLLOUTS).to(ROLLOUT_DTYPES["rollout"])

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()
    u = uniforms(items["game_id"], items["rollout"], SEED)
    items["tokens"] = generate(model, items, u, device)

    assert list(items) == list(ROLLOUT_DTYPES), "saved keys are not the contracted keys"
    assert all(items[k].dtype == d for k, d in ROLLOUT_DTYPES.items()), "a column changed dtype"
    verify(items, source)

    out_dir = base / "rollout" / "branches"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(items, out_dir / "branches.pt")
    manifest = {
        "seed": SEED,
        "k_rollouts": K_ROLLOUTS,
        "temperature": TEMPERATURE,
        "masked_tokens": ["pass"],
        "source": str(base / "branches" / "branches.pt"),
        "n_source_branches": len(source["t0"]),
        "n_rows": len(items["t0"]),
        "note": "tokens[:t0+1] copied from the source branch, tokens[t0+1:] sampled from the "
                "model; the reference is sampled too, so branches differ at every position "
                "after t0 and single-token attribution does not hold here",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(items['t0'])} rollout rows to {out_dir}")


if __name__ == "__main__":
    main()
