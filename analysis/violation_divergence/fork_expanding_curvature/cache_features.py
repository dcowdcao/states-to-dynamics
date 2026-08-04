"""One forked game's activations in all nine representation spaces, cached.

The acts file is 1 GB and the eight SAE checkpoints take a couple of minutes to
load; the game itself is 9 branches x 17 positions. Cache it once so the
accounting and the figures are seconds apart.

Usage: python analysis/violation_divergence/fork_expanding_curvature/cache_features.py
"""

from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.paper_style import sae_label
from spade.othello.taxonomy import CATEGORIES
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.compute_divergence import sae_encoder

from window import positions

CACHE = study_dir("divergence") / "fork_expanding" / "game_spaces.npz"
RAW_SPACE = "raw layer 5 (reference space)"


def load() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing; run cache_features.py first")
    z = np.load(CACHE, allow_pickle=False)
    names = [str(s) for s in z["names"]]
    return {"names": names, "F": [z[f"space_{j}"] for j in range(len(names))],
            "pos": z["pos"], "t0": int(z["t0"]), "game": int(z["game"]),
            "category": z["category"], "is_control": z["is_control"]}


def main():
    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)

    n_types = len(CATEGORIES) + 2  # 6 illegal + reference + 2 legal controls
    game = next(g for g in meta["game_id"].unique().tolist()
        if int((meta["game_id"] == g).sum()) == n_types)
    idx = (meta["game_id"] == game).nonzero().flatten()
    t0 = int(meta["t0"][idx[0]])
    pos = positions(t0)

    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True)
    Z = acts[idx][:, pos, :].float().to(device)
    del acts

    spaces = [(RAW_SPACE, None)] + \
        [(sae_label(e.config.replace("Trainer-othello", "")), e)
        for e in seed42_entries() if e.config not in EXCLUDED_SAE]
    out = {}
    for j, (name, entry) in enumerate(spaces):
        if entry is None:
            out[f"space_{j}"] = Z.cpu().numpy()
            continue
        with torch.no_grad():
            f = sae_encoder(entry, device)(Z.reshape(-1, Z.shape[-1]))
        out[f"space_{j}"] = f.reshape(Z.shape[0], Z.shape[1], -1).cpu().numpy()
        print(f"{name}: {out[f'space_{j}'].shape}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, names=np.array([n for n, _ in spaces]), pos=pos, t0=t0, game=game,
        category=meta["category"][idx].numpy(),
        is_control=meta["is_control"][idx].numpy(), **out)
    print(f"wrote {CACHE}  game {game}, t0={t0}, positions {pos[0]}-{pos[-1]}")


if __name__ == "__main__":
    main()
