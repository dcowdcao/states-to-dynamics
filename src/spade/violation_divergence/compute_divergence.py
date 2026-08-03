"""Step 3 of the violation-divergence study: how far each branch travels from
its game's reference, per representation space.

Spaces: token_embed + resid_post_0..7 from `$DIR/acts/`, and the 8 seed-42 SAEs
(minus SpaDE γ=1e-2, as in the straightness study) encoding resid_post_5. One
output row per non-reference branch, paired with its game's reference row by
game_id; reference rows are dropped, their divergence being zero by definition.
Resumable: a space whose npz exists is skipped.

`$DIR/divergence/<space>.npz`, float32 unless noted.
Event-locked, [n_branch, 33] at offsets r = s - t0:

    window       int [33]  the offsets themselves, -8..24
    delta        δ̂ = ‖Δ‖ / ‖z_ref‖                                        M1
    cos_t0       cos(Δ(r), Δ(0))                                          M2
    cos_control  cos(Δ_branch, Δ_control), = 1 on the control branch      M2
    topk_share   share of Δ's squared norm in its 10 largest |Δ_j|        M4
    pr           participation ratio of Δ²                                M4
    jaccard      support Jaccard                          SAE spaces only M4
    dl0          L0(branch) − L0(reference)               SAE spaces only M4

Per branch row, [n_branch]:

    peak, peak_r (int), half_life, plateau   divergence.persistence(delta)  M3
    topk_idx     int [n_branch, 10]  the 10 largest |Δ_j| at r = 0, SAE only
    game_id, t0, category, is_control, n_flipped, row

`half_life` is plies elapsed after the peak, +inf where δ̂ never halves inside
the window; add `peak_r` for the absolute offset. `row` indexes
branches.pt, which is where anything else per-row lives (branch_token, n_legal).
M5 (behavioral KL) reads `$DIR/acts/logits.pt` directly and is not a space here.

Spaces are independent and skip-if-exists, so `--shard I --n_shards N` splits
them across N SLURM array tasks with no coordination.

Usage: python -m spade.violation_divergence.compute_divergence [--smoke]
       [--spaces resid_post_5] [--shard 0 --n_shards 6]
"""

import argparse
from collections import defaultdict

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.evaluation.karvonen_adapter import (
    StandardizedDictionary, load_karvonen_sae, load_standardization_params)
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.divergence import (
    WINDOW, align, branch_cosine, direction_cosine, participation_ratio,
    persistence, relative_divergence, support_jaccard, topk_share)

FILE_SPACES = ["token_embed"] + [f"resid_post_{i}" for i in range(8)]
SAE_SOURCE = "resid_post_5"
ROW_BATCH = 64  # 64 seqs x 59 positions x 8192 codes x 4 B = 124 MB per code tensor
TOPK = 10
META_KEYS = ["game_id", "t0", "category", "is_control", "n_flipped"]


def _row_of(game_id: torch.Tensor, mask: torch.Tensor, what: str) -> torch.Tensor:
    """Game id -> the single row of that game satisfying `mask`."""
    assert int(mask.sum()) == len(game_id.unique()), f"not exactly one {what} per game"
    out = torch.full((int(game_id.max()) + 1,), -1, dtype=torch.long)
    out[game_id[mask]] = mask.nonzero().flatten()
    return out


def pairs(meta: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(branch rows, and per branch the reference and control_legal row of its game)."""
    game_id = meta["game_id"]
    is_ref = (meta["category"] == 0) & (meta["is_control"] == 0)
    branch = (~is_ref).nonzero().flatten()
    of_game = game_id[branch]
    return (branch,
            _row_of(game_id, is_ref, "reference")[of_game],
            _row_of(game_id, meta["is_control"] == 1, "control_legal")[of_game])


def sae_encoder(entry, device: str):
    ae, _ = load_karvonen_sae(str(entry.ckpt_dir), device=device)
    params = load_standardization_params(str(entry.ckpt_dir), device=device)
    return StandardizedDictionary(
        ae, torch.tensor(params["mean"], device=device), params["scale"]).to(device).encode


def features(Z: torch.Tensor, rows: torch.Tensor, encode, device: str) -> torch.Tensor:
    """[len(rows), T, width] float32 on `device`; `encode` is None for a raw space."""
    x = Z[rows].float().to(device)
    if encode is None:
        return x
    with torch.no_grad():
        return encode(x.flatten(0, 1)).unflatten(0, x.shape[:2])


def metrics(Fb: torch.Tensor, Fc: torch.Tensor, Fref: torch.Tensor, t0: torch.Tensor,
            sae: bool) -> dict[str, torch.Tensor]:
    """Per-position [n, T] quantities. Support and L0 mean nothing in a dense
    space, where every coordinate is nonzero, so they are SAE-only."""
    m = {"delta": relative_divergence(Fb, Fref),
         "cos_t0": direction_cosine(Fb, Fref, t0),
         "cos_control": branch_cosine(Fb, Fc, Fref),
         "topk_share": topk_share(Fb, Fref, TOPK),
         "pr": participation_ratio(Fb, Fref)}
    if sae:
        m["jaccard"] = support_jaccard(Fb, Fref)
        m["dl0"] = ((Fb != 0).sum(-1) - (Fref != 0).sum(-1)).float()
    return m


def event_locked(Z: torch.Tensor, branch: torch.Tensor, ref: torch.Tensor,
                 control: torch.Tensor, t0: torch.Tensor, encode,
                 device: str) -> dict[str, torch.Tensor]:
    """Every metric aligned to r = WINDOW, plus the top-|Δ| code indices at r = 0.

    Batched over branch rows: a game's reference and control rows are re-encoded
    once per branch of that game, ~3x the minimal encode work, which keeps peak
    memory at three code tensors however many branches a game has. The encode is
    seconds of GPU time either way; the acts tensor load dominates.

    `cos_control` is 1 by construction on the control branch itself, which is
    the cheapest available check that the pairing is right.
    """
    chunks, top = defaultdict(list), []
    for i in range(0, len(branch), ROW_BATCH):
        sl = slice(i, i + ROW_BATCH)
        t = t0[sl].to(device)
        Fb = features(Z, branch[sl], encode, device)
        Fref = features(Z, ref[sl], encode, device)
        Fc = features(Z, control[sl], encode, device)
        for key, value in metrics(Fb, Fc, Fref, t, encode is not None).items():
            chunks[key].append(align(value, t, WINDOW).cpu())
        if encode is not None:
            at_t0 = (Fb - Fref)[torch.arange(len(t), device=device), t]
            top.append(at_t0.square().topk(TOPK, dim=-1).indices.cpu())
    out = {key: torch.cat(v) for key, v in chunks.items()}
    if top:
        out["topk_idx"] = torch.cat(top)
    return out


def checks(space: str, out: dict[str, torch.Tensor]) -> None:
    """δ̂ before t0 is zero by construction; a nonzero one means the alignment,
    not the model, is wrong. NaN δ̂ needs an all-zero reference code, which only
    a sparse space can produce."""
    delta = out["delta"]
    pre = torch.as_tensor(WINDOW) < 0
    assert (delta[:, pre] == 0).all(), f"{space}: δ̂ nonzero before t0"
    nan_frac = delta.isnan().float().mean().item()
    assert space.startswith("sae_") or nan_frac == 0, \
        f"{space}: {nan_frac:.2%} undefined δ̂ in a dense space"
    print(f"{space}: δ̂(0) {delta[:, WINDOW.index(0)].nanmean():.4f}"
          f"  peak {out['peak'].nanmean():.4f} at r {out['peak_r'].float().mean():.1f}"
          f"  plateau {out['plateau'].nanmean():.4f}"
          f"  half-life censored {torch.isinf(out['half_life']).float().mean():.1%}"
          f"  nan {nan_frac:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="the 50-game set under $DIR/smoke/")
    parser.add_argument("--spaces", default="", help="comma-separated filter (smoke tests)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1,
                        help="process spaces[shard::n_shards]; one SLURM array task each")
    args = parser.parse_args()
    assert 0 <= args.shard < args.n_shards, "shard out of range"

    device = select_device()
    base = study_dir("divergence") / "smoke" if args.smoke else study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    branch, ref, control = pairs(meta)
    t0 = meta["t0"][branch]
    print(f"{len(meta['game_id'].unique())} games, {len(branch)} branch rows")

    out_dir = base / "divergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    spaces = [(s, None) for s in FILE_SPACES] + \
        [(f"sae_{e.config}", e) for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    if args.spaces:
        keep = args.spaces.split(",")
        spaces = [s for s in spaces if s[0] in keep]
    # Round-robin, not contiguous blocks: the list is 9 cheap raw spaces then 8
    # SAE ones that carry the encode, so a block split loads every expensive
    # space into one task. The array passes indices and never space names --
    # a space list duplicated in bash is how the cosine sweep silently trained
    # one config instead of nine (PROCESS.md).
    spaces = spaces[args.shard::args.n_shards]

    row_meta = {key: meta[key][branch].numpy() for key in META_KEYS}
    row_meta["row"] = branch.numpy()

    for space, entry in spaces:
        path = out_dir / f"{space}.npz"
        if path.exists():
            print(f"skip {space} (exists)")
            continue
        Z = torch.load(base / "acts" / f"{SAE_SOURCE if entry else space}.pt",
                       weights_only=True)
        positions = t0.unsqueeze(1) + torch.as_tensor(WINDOW)
        assert positions.min() >= 1 and positions.max() < Z.shape[1], \
            f"{space}: window leaves the board or hits the position-0 sink"
        out = event_locked(Z, branch, ref, control, t0,
                           sae_encoder(entry, device) if entry else None, device)
        out |= persistence(out["delta"], WINDOW)
        checks(space, out)
        np.savez(path, window=np.asarray(WINDOW),
                 **{key: v.numpy() for key, v in out.items()}, **row_meta)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
