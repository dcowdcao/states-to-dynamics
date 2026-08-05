"""Step 2 of the violation-divergence study: activations and logits for every branch.

One batched pass over the 59-token branch sequences saves resid_post for all 8
layers, the token-embedding trajectory (the input-side floor for every depth
claim) and the full next-move logits, rows aligned with branches.pt. Every
sequence is the same length, so no length bucketing is needed -- asserted, not
assumed.

Usage: python -m spade.violation_divergence.extract_acts [--smoke] [--variant rollout]
"""

import argparse

import torch

from spade.common.config import CTX_LEN, D_MODEL, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import variant_dir
from spade.straightness.extract_traj import check_sink
from spade.violation_divergence.branches import ROLE_NAMES, role

force_ipv4()

N_LAYERS = 8
HOOKS = [f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)]
BATCH_SIZE = 256


def spans(game_id: torch.Tensor, size: int) -> list[tuple[int, int]]:
    """Row spans of at most `size` that never split a game.

    Two branches of one game must go through the same batch shape or cuBLAS
    reduces their shared prefix in a different order and their pre-t0
    activations stop being bit-identical.
    """
    bounds = ([0] + (game_id[1:] != game_id[:-1]).nonzero().flatten().add(1).tolist()
        + [len(game_id)])
    out, lo = [], 0
    for prev, nxt in zip(bounds, bounds[1:]):
        if nxt - lo > size:
            out.append((lo, prev))
            lo = prev
    out.append((lo, len(game_id)))
    assert all(0 < b - a <= size for a, b in out), f"a game holds more than {size} branches"
    return out


def extract(model, tokens: torch.Tensor, game_id: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    n = tokens.shape[0]
    out = {f"resid_post_{i}": torch.empty(n, CTX_LEN, D_MODEL, dtype=torch.float16)
        for i in range(N_LAYERS)}
    out["token_embed"] = model.W_E[tokens.long()].cpu().half()
    out["logits"] = torch.empty(n, CTX_LEN, model.cfg.d_vocab, dtype=torch.float16)
    for k, (start, end) in enumerate(spans(game_id, BATCH_SIZE)):
        batch = tokens[start:end].long().to(device)
        logits, cache = model.run_with_cache(batch, names_filter=lambda name: name in HOOKS)
        for i, hook in enumerate(HOOKS):
            out[f"resid_post_{i}"][start:end] = cache[hook].cpu().half()
        out["logits"][start:end] = logits.cpu().half()
        if k % 4 == 0:
            print(f"extracted {end}/{n}")
    return out


def check_prefix_invariance(acts: torch.Tensor, items: dict[str, torch.Tensor]) -> None:
    """Every branch of a game agrees before t0, and differs at t0 iff its token does.

    Two rows agreeing at t0 is a bug in the spliced set, where a game's branch
    tokens are distinct, but is the expected state in the rollout set, where the
    same branch is resampled K times and only the continuation differs.
    """
    for game_id in items["game_id"].unique().tolist():
        idx = (items["game_id"] == game_id).nonzero().flatten()
        t0 = int(items["t0"][idx[0]])
        rows = acts[idx]
        assert (rows[:, :t0] == rows[0, :t0]).all(), \
            f"game {game_id}: layer-5 activations differ before t0={t0}"
        token = items["branch_token"][idx]
        assert torch.equal((rows[:, t0].unsqueeze(0) == rows[:, t0].unsqueeze(1)).all(-1),
                           token.unsqueeze(0) == token.unsqueeze(1)), \
            f"game {game_id}: activations at t0={t0} do not track the branch token"
    print(f"prefix invariance ok on all {len(items['game_id'].unique())} games")


def report_p_branch(items: dict[str, torch.Tensor], logits: torch.Tensor) -> None:
    """Model probability of each branch token, read at the position that predicts it."""
    predicted = logits[torch.arange(logits.shape[0]), items["t0"] - 1].float().softmax(-1)
    p = predicted.gather(1, items["branch_token"].unsqueeze(1)).squeeze(1)
    roles = [role(int(c), int(k)) for c, k in zip(items["category"], items["is_control"])]
    means = {}
    for name in ROLE_NAMES:
        mask = torch.tensor([r == name for r in roles])
        means[name] = p[mask].mean().item()
        print(f"mean p_branch  {name:18} n={int(mask.sum()):6}  {means[name]:.2e}")
    for name in ["occupied_own", "occupied_opp"]:
        assert means[name] < 1e-2 and means["reference"] > 10 * means[name], \
            f"model assigns implausible probability to {name} moves"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="read and write under $DIR/smoke/")
    parser.add_argument("--variant", default="", choices=["rollout"],
                        help="read and write under $DIR[/smoke]/rollout/, where every "
                             "branch carries its own model-sampled continuation")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    base = variant_dir("divergence", args.smoke, args.variant)
    items = torch.load(base / "branches" / "branches.pt", weights_only=True)
    tokens = items["tokens"]
    assert tokens.shape[1] == CTX_LEN and (tokens > 0).all(), "branches are not 59 nonzero tokens"

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()
    assert model.cfg.n_layers == N_LAYERS and model.cfg.d_model == D_MODEL

    out = extract(model, tokens, items["game_id"], device)
    check_sink(out["resid_post_5"])
    check_prefix_invariance(out["resid_post_5"], items)
    report_p_branch(items, out["logits"])

    out_dir = base / "acts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in out.items():
        assert tensor.shape[0] == tokens.shape[0] and tensor.isfinite().all()
        torch.save(tensor, out_dir / f"{name}.pt")
    print(f"wrote {len(out)} tensors for {tokens.shape[0]} branches to {out_dir}")


if __name__ == "__main__":
    main()
