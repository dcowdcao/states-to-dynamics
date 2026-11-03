"""One game's rows of the layer-5 branch activations: all make_figures.py reads.

The full tensors are 2.1 GB (spliced) and 8.35 GB (rollout). Game 1235's rows are 0.5 MB and
2.2 MB, so the figures redraw on a CPU from these slices, and the data release ships them.

Writes <divergence dir>/[rollout/]fork_and_board/game_<G>_resid_post_5.pt:
{"rows": the game's row indices into branches.pt, "acts": fp16 [rows, 59, 512]}.

Usage: python analysis/violation_divergence/fork_and_board/game_acts.py
       [--game 1235] [--variant spliced|rollout]
"""

import argparse
from pathlib import Path

import torch

from spade.common.study import variant_dir
from spade.violation_divergence.branches import FIGURE_GAME, Game, load_game

VARIANTS = {"spliced": "", "rollout": "rollout"}


def path(base: Path, game: int) -> Path:
    return base / "fork_and_board" / f"game_{game}_resid_post_5.pt"


def load(base: Path, g: Game) -> torch.Tensor:
    """[len(g.idx), 59, 512] fp16, row for row with g.idx."""
    cut = torch.load(path(base, g.game), weights_only=True)
    assert torch.equal(cut["rows"], g.idx), \
        f"{path(base, g.game)} was cut from another branch tree: re-run game_acts.py"
    return cut["acts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--variant", default="spliced", choices=list(VARIANTS))
    args = parser.parse_args()

    base = variant_dir("divergence", variant=VARIANTS[args.variant])
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    rows = load_game(meta, args.game).idx
    # mmap: only the game's rows are read out of the full tensor.
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    out = path(base, args.game)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "acts": acts[rows].clone()}, out)
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
