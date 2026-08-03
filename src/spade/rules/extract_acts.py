"""Step 2 of the rules study: extract OthelloGPT activations for the pairs.

For every prefix+move sequence from build_pairs.py, saves the layer-5
resid_post activation at position t (the injected move), the activation at
position t-1 (prefix-final, deduplicated to one row per prefix — where the
model computes legality of all candidates), and the model's own probability
of the injected token. Items are bucketed by exact sequence length, so no
padding ever reaches the model.

Usage: python -m spade.rules.extract_acts
"""

import torch

from spade.common.config import CTX_LEN, D_MODEL, HOOK_POINT, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.othello.taxonomy import CATEGORIES, LEGAL

force_ipv4()

BATCH_SIZE = 256


def extract(model, items: dict, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = items["length"].shape[0]
    acts_final = torch.zeros(n, D_MODEL)
    acts_before = torch.zeros(n, D_MODEL)
    p_injected = torch.zeros(n)
    for length in items["length"].unique().tolist():
        idx = (items["length"] == length).nonzero().flatten()
        for start in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            tokens = items["tokens"][batch_idx, :length].long().to(device)
            assert (tokens > 0).all(), "padding/pass token reached the model"
            logits, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
            acts_final[batch_idx] = cache[HOOK_POINT][:, -1, :].cpu()
            acts_before[batch_idx] = cache[HOOK_POINT][:, -2, :].cpu()
            probs = logits[:, -2, :].softmax(-1)
            p_injected[batch_idx] = probs.gather(
                1, items["injected_token"][batch_idx].to(device).unsqueeze(1)
            ).squeeze(1).cpu()
    return acts_final, acts_before, p_injected


def check_prefix_invariance(model, items: dict, device: str, n_pairs: int = 5) -> None:
    """A pair's members must agree exactly at every position before t."""
    legal_primary = (items["category"] == LEGAL) & (items["is_control_legal"] == 0)
    for pid in items["prefix_id"][legal_primary][:n_pairs].tolist():
        pair_idx = (items["prefix_id"] == pid).nonzero().flatten()[:2]
        length = items["length"][pair_idx[0]].item()
        tokens = items["tokens"][pair_idx, :length].long().to(device)
        _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
        acts = cache[HOOK_POINT]
        assert torch.equal(acts[0, :-1], acts[1, :-1]), \
            f"prefix {pid}: activations diverge before the injected move"


def report_p_injected(items: dict, p_injected: torch.Tensor) -> None:
    means = {}
    for c, name in enumerate(CATEGORIES):
        mask = (items["category"] == c) & (items["is_control_legal"] == 0)
        means[name] = p_injected[mask].mean().item()
        print(f"mean p_injected  {name:18} {means[name]:.2e}")
    for name in ["occupied_own", "occupied_opp"]:
        assert means[name] < 1e-2 and means["legal"] > 10 * means[name], \
            f"model assigns implausible probability to {name} moves"


def main():
    from transformer_lens import HookedTransformer

    device = select_device()
    items = torch.load(study_dir("rules") / "pairs" / "items.pt", weights_only=True)
    assert items["length"].max().item() <= CTX_LEN

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    acts_final, acts_before, p_injected = extract(model, items, device)
    check_prefix_invariance(model, items, device)
    report_p_injected(items, p_injected)

    # one prefix-level (t-1) activation per prefix, row k = prefix_id k
    legal_primary = (items["category"] == LEGAL) & (items["is_control_legal"] == 0)
    order = items["prefix_id"][legal_primary].argsort()
    acts_prefix = acts_before[legal_primary.nonzero().flatten()[order]]
    n_prefixes = items["prefix_id"].max().item() + 1
    assert acts_prefix.shape == (n_prefixes, D_MODEL)

    out_dir = study_dir("rules") / "acts"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts_final, out_dir / "acts_final.pt")
    torch.save(acts_prefix, out_dir / "acts_prefix.pt")
    torch.save(p_injected, out_dir / "p_injected.pt")
    print(f"wrote acts_final {tuple(acts_final.shape)}, "
          f"acts_prefix {tuple(acts_prefix.shape)} to {out_dir}")


if __name__ == "__main__":
    main()
