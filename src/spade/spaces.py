"""The representation-space API: one way to ask "encode this into space X".

A space is the raw layer-5 residual stream or any trained SAE's latent space, plus the
random-projection and random-TopK controls. Manifold-capture, bracket-flip and the rules
embeddings all need the same four operations, so they live here rather than inside the
rules study where they grew.

Loading a space means loading a checkpoint through spade.evaluation.karvonen_adapter and
wrapping it in StandardizedDictionary — SAEs were trained on scale*(x-mean) activations,
so an unstandardized encode is silently wrong, not merely inaccurate.
"""

import torch

from spade.common.config import D_MODEL, DICT_SIZE
from spade.common.paths import repo_root
from spade.evaluation.karvonen_adapter import (
    StandardizedDictionary,
    load_karvonen_sae,
    load_standardization_params,
)
from spade.saes.registry import entries

MIN_T_BIN = 5
ENCODE_BATCH = 8192

# The distance-SpaDE gamma=1e-2 seed-42 checkpoint. Degenerate (L0 15.4, VE 0.352, where
# seeds 43-46 of the same config give 128 and 0.973) with a Mac smoke config written over
# its real GPU weights. Excluded from every study and from the viewer, decided here once;
# the default SpaDE is gamma=1e-4 (checkpoints/spade_revival/), which takes its slot.
EXCLUDED_SAE = {"SpaDETrainer-othello"}


def load_space(name: str, entry, device: str):
    """encode_fn mapping a raw-activation batch on `device` to features."""
    params = load_standardization_params(
        str(repo_root() / "checkpoints" / "main" / "trainer_0"))
    mean, scale = torch.tensor(params["mean"], device=device), params["scale"]
    if name == "raw":
        return lambda x: scale * (x - mean)
    if name == "random_proj":
        g = torch.Generator().manual_seed(0)
        w = (torch.randn(D_MODEL, DICT_SIZE, generator=g) / D_MODEL ** 0.5).to(device)
        return lambda x: torch.relu((scale * (x - mean)) @ w)
    if name == "random_topk":
        from circuits.dictionary_learning.trainers.top_k import AutoEncoderTopK
        torch.manual_seed(0)
        ae = AutoEncoderTopK(D_MODEL, DICT_SIZE, 80).to(device).eval()
        return StandardizedDictionary(ae, mean, scale).to(device).encode
    ae, _ = load_karvonen_sae(str(entry.ckpt_dir), device=device)
    params = load_standardization_params(str(entry.ckpt_dir), device=device)
    return StandardizedDictionary(
        ae, torch.tensor(params["mean"], device=device), params["scale"]).to(device).encode


def representative_spaces() -> list[tuple]:
    """(space_name, entry_or_None): baselines + one checkpoint per SAE config.

    EXCLUDED_SAE configs are dropped here, which is what makes the exclusion global —
    every study reaches its space list through this function. The source repo filtered
    inside two study modules instead, so three other studies and the viewer kept reading
    the degenerate checkpoint.
    """
    by_config = {}
    for e in entries():
        if e.config in EXCLUDED_SAE:
            continue
        best = by_config.get(e.config)
        if best is None or e.seed == 42 or (best.seed != 42 and e.seed < best.seed):
            by_config[e.config] = e
    return [(n, None) for n in ["raw", "random_proj", "random_topk"]] + \
        [(cfg, e) for cfg, e in sorted(by_config.items())]


def batched_apply(fn, acts: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([fn(acts[i:i + ENCODE_BATCH].to(device)).cpu()
            for i in range(0, len(acts), ENCODE_BATCH)])


def center_by_t(feats: torch.Tensor, t: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Subtract each t bin's legal centroid (visualization-grade: no
    leave-own-prefix-out — fine for embeddings, not for distance metrics)."""
    centered = feats - feats[legal].mean(0)
    for tv in t.unique().tolist():
        bin_legal = legal & (t == tv)
        if int(bin_legal.sum()) >= MIN_T_BIN:
            bin_mask = t == tv
            centered[bin_mask] = feats[bin_mask] - feats[bin_legal].mean(0)
    return centered
