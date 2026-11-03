"""Step 2 of the manifold-capture study: layer-5 activations for the corpus games.

One batched run_with_cache pass at blocks.5.hook_resid_post; position 0 is
verified as the attention sink and dropped, leaving rows aligned with
corpus.pt labels ([n_games * 58, 512] fp32).

Usage: python -m spade.manifold_capture.extract_acts
"""

import torch

from spade.common.config import CTX_LEN, D_MODEL, HOOK_POINT, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir

force_ipv4()

BATCH_SIZE = 256


def main():
    from transformer_lens import HookedTransformer

    device = select_device()
    tokens = torch.load(study_dir("manifold") / "corpus" / "corpus.pt",
                        weights_only=True)["tokens"]
    n = tokens.shape[0]
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()

    acts = torch.empty(n, CTX_LEN, D_MODEL)
    for start in range(0, n, BATCH_SIZE):
        batch = tokens[start:start + BATCH_SIZE].long().to(device)
        _, cache = model.run_with_cache(batch, names_filter=HOOK_POINT)
        acts[start:start + BATCH_SIZE] = cache[HOOK_POINT].cpu()
        if (start // BATCH_SIZE) % 4 == 0:
            print(f"extracted {min(start + BATCH_SIZE, n)}/{n}")

    norms = acts.norm(dim=-1)
    sink, rest = norms[:, 0].mean().item(), norms[:, 1:].mean().item()
    assert sink > 3 * rest, f"position-0 sink missing: {sink:.1f} vs {rest:.1f}"

    out_dir = study_dir("manifold") / "acts"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts[:, 1:].reshape(-1, D_MODEL).contiguous(), out_dir / "acts.pt")
    print(f"wrote {n * (CTX_LEN - 1)} rows to {out_dir / 'acts.pt'} "
          f"(pos-0 sink {sink:.1f} vs rest {rest:.1f}, dropped)")


if __name__ == "__main__":
    main()
