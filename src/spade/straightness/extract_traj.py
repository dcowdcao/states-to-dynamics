"""Step 2 of the straightness study: full per-game activation trajectories.

One batched run_with_cache pass saves resid_post for all 8 layers plus the
token-embedding trajectory (the "pixel-space" control), each [N, 59, 512] fp16.
Verifies the position-0 attention sink at layer 5 and prefix invariance of
violation games against their source legal games.

Usage: python -m spade.straightness.extract_traj
"""

import torch

from spade.common.config import CTX_LEN, D_MODEL, HF_REVISIONS, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir

force_ipv4()

N_LAYERS = 8
HOOKS = [f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)]
BATCH_SIZE = 256


def extract(model, tokens: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    n = tokens.shape[0]
    out = {h: torch.empty(n, CTX_LEN, D_MODEL, dtype=torch.float16) for h in HOOKS}
    out["token_embed"] = model.W_E[tokens.long()].cpu().half()
    for start in range(0, n, BATCH_SIZE):
        batch = tokens[start:start + BATCH_SIZE].long().to(device)
        _, cache = model.run_with_cache(batch, names_filter=lambda name: name in HOOKS)
        for h in HOOKS:
            out[h][start:start + BATCH_SIZE] = cache[h].cpu().half()
        if (start // BATCH_SIZE) % 4 == 0:
            print(f"extracted {min(start + BATCH_SIZE, n)}/{n}")
    return out


def check_sink(acts5: torch.Tensor) -> None:
    norms = acts5.float().norm(dim=-1)
    sink, rest = norms[:, 0].mean().item(), norms[:, 1:].mean().item()
    assert sink > 3 * rest, f"position-0 sink missing: {sink:.1f} vs {rest:.1f}"
    print(f"sink check ok: pos0 norm {sink:.1f} vs rest {rest:.1f}")


def check_prefix_invariance(model, data: dict, device: str, n_pairs: int = 5) -> None:
    """A violation game equals its source legal game at every position < t0."""
    viol_idx = (data["src_game"] >= 0).nonzero().flatten()[:n_pairs]
    for i in viol_idx.tolist():
        t0, src = data["t0"][i].item(), data["src_game"][i].item()
        pair = torch.stack([data["tokens"][i], data["tokens"][src]]).long().to(device)
        assert torch.equal(pair[0, :t0], pair[1, :t0])
        _, cache = model.run_with_cache(pair, names_filter=HOOKS[5])
        acts = cache[HOOKS[5]]
        assert torch.equal(acts[0, :t0], acts[1, :t0]), \
            f"violation {i}: activations diverge before t0={t0}"


def main():
    from transformer_lens import HookedTransformer

    device = select_device()
    data = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()

    out = extract(model, data["tokens"], device)
    check_sink(out[HOOKS[5]])
    check_prefix_invariance(model, data, device)

    out_dir = study_dir("straightness") / "traj"
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, acts in out.items():
        name = "token_embed" if h == "token_embed" else f"resid_post_{h.split('.')[1]}"
        torch.save(acts, out_dir / f"{name}.pt")
    print(f"wrote {len(out)} trajectory tensors {tuple(out['token_embed'].shape)} to {out_dir}")


if __name__ == "__main__":
    main()
