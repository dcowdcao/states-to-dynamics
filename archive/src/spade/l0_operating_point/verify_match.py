"""Proof that renders/karvonen_only reproduces Karvonen's published PNGs pixel for pixel."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from spade.common.paths import repo_root

HIS_IMAGES = repo_root() / "external/Karvonen_Research/autoencoders/saved_data/07-31_images"
RENDERS = repo_root() / "figures/l0_operating_point/karvonen_only"
FIGURES = [
    "othello_2var_reconstruct.png",
    "othello_2var_coverage.png",
    "othello_3var_reconstruct.png",
    "othello_3var_coverage.png",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--his-images", type=Path, default=HIS_IMAGES)
    ap.add_argument("--ours", type=Path, default=RENDERS)
    args = ap.parse_args()

    for name in FIGURES:
        his = Image.open(args.his_images / name).convert("RGB")
        ours = Image.open(args.ours / name).convert("RGB")
        if his.size != ours.size:
            print(f"{name}: SIZE MISMATCH his {his.size} ours {ours.size}")
            continue
        diff = np.abs(np.asarray(his, float) - np.asarray(ours, float))
        print(f"{name}: {his.size[0]}x{his.size[1]}  max channel diff {diff.max():.0f}  "
            f"pixels differing {(diff.max(2) > 0).mean():.6f}")


if __name__ == "__main__":
    main()
