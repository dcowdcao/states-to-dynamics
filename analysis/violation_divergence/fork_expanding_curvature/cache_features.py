"""One forked game's activations in all nine representation spaces, cached.

The acts file is 1 GB and the eight SAE checkpoints take a couple of minutes to
load; the game itself is 9 branches x 17 positions. Cache it once so the
accounting and the figures are seconds apart.

`--variant rollout` caches the same game from the branch set whose suffixes are
sampled from the model, which stores K rows per branch (36, not 9) and a
`rollout` column naming which sample a row is. The cache is scoped by variant:
an unscoped path would let a rollout run overwrite the spliced features and
silently redraw every spliced figure.

Usage: python analysis/violation_divergence/fork_expanding_curvature/cache_features.py
       [--variant spliced|rollout]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import variant_dir
from spade.figures.paper_style import sae_label
from spade.saes.registry import seed42_entries
from spade.spaces import EXCLUDED_SAE
from spade.violation_divergence.branches import complete_games
from spade.violation_divergence.compute_divergence import sae_encoder

from window import positions

VARIANT_DIRS = {"spliced": "", "rollout": "rollout"}
RAW_SPACE = "raw layer 5 (reference space)"


def cache_path(variant: str) -> Path:
    return variant_dir("divergence", variant=variant) / "fork_expanding" / "game_spaces.npz"


def load(variant: str = "") -> dict:
    cache = cache_path(variant)
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing; run cache_features.py first")
    z = np.load(cache, allow_pickle=False)
    names = [str(s) for s in z["names"]]
    g = {"names": names, "F": [z[f"space_{j}"] for j in range(len(names))],
         "pos": z["pos"], "t0": int(z["t0"]), "game": int(z["game"]),
         "category": z["category"], "is_control": z["is_control"]}
    if "rollout" in z.files:
        g["rollout"] = z["rollout"]
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="spliced", choices=list(VARIANT_DIRS))
    args = parser.parse_args()

    device = select_device()
    variant = VARIANT_DIRS[args.variant]
    base = variant_dir("divergence", variant=variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)

    # First branch-complete game, by branch *type*: the rollout set stores K rows
    # per branch, so a row count finds no complete game there at all.
    game = complete_games(meta)[0]
    idx = (meta["game_id"] == game).nonzero().flatten()
    t0 = int(meta["t0"][idx[0]])
    pos = positions(t0)

    # mmap: the rollout acts file is 4 GB and is indexed a game at a time.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
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

    extra = {"rollout": meta["rollout"][idx].numpy()} if "rollout" in meta else {}
    cache = cache_path(variant)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, names=np.array([n for n, _ in spaces]), pos=pos, t0=t0, game=game,
        category=meta["category"][idx].numpy(),
        is_control=meta["is_control"][idx].numpy(), **extra, **out)
    print(f"wrote {cache}  game {game}, t0={t0}, positions {pos[0]}-{pos[-1]}, "
          f"{len(idx)} rows")


if __name__ == "__main__":
    main()
