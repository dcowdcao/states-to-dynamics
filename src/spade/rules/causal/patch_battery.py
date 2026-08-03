"""Component-patch battery shared by the pair/forfeit patching experiments.

Clean and corrupted sequences share their prefix and differ only in the final
token, so the final position's full residual carries *all* differing
information — patching it at any layer trivially recovers clean behavior
(recovery == 1 by construction). Localization therefore patches single
components: each layer's attn_out and mlp_out at the final position, clean ->
corrupted, leaving the rest of the stream corrupted. `resid0` (full residual
after block 0) is kept as the tautology sanity check.
"""

import torch

N_LAYERS = 8

PATCH_HOOKS = {"resid0": "blocks.0.hook_resid_post"}
for _L in range(N_LAYERS):
    PATCH_HOOKS[f"attn{_L}"] = f"blocks.{_L}.hook_attn_out"
    PATCH_HOOKS[f"mlp{_L}"] = f"blocks.{_L}.hook_mlp_out"


def battery_logprobs(model, tok_clean: torch.Tensor,
                     tok_corr: torch.Tensor) -> dict[str, torch.Tensor]:
    """Final-position log-probs [B, vocab] for the clean run, corrupted run,
    and every single-component clean->corrupted patch."""
    hook_names = set(PATCH_HOOKS.values())
    clean_logits, cache = model.run_with_cache(
        tok_clean, names_filter=lambda n: n in hook_names)
    runs = {"clean": clean_logits, "corr": model(tok_corr)}
    for key, name in PATCH_HOOKS.items():
        def patch(act, hook, src=cache[name][:, -1, :]):
            act[:, -1, :] = src
            return act
        runs[key] = model.run_with_hooks(tok_corr, fwd_hooks=[(name, patch)])
    return {k: v[:, -1, :].log_softmax(-1).cpu() for k, v in runs.items()}
