"""R5 labels: how much flipping each legal injected move performs.

For every legal item in pairs/items.pt (primary and control), replays the
prefix and records the number of discs the injected move flips and the number
of directions it flips in. Illegal items get -1. Saved aligned with items.

Usage: python -m spade.rules.flip_labels
"""

import torch

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()

from spade.common.study import study_dir
from spade.othello.taxonomy import LEGAL

from circuits.othello_engine_utils import OthelloBoardState, eights, to_string


def flip_stats(state, square: int, color: int) -> tuple[int, int]:
    """(discs flipped, directions flipped in) when `color` plays `square`."""
    r, c = square // 8, square % 8
    n_discs, n_dirs = 0, 0
    for dr, dc in eights:
        buffer = 0
        cr, cc = r + dr, c + dc
        while 0 <= cr <= 7 and 0 <= cc <= 7 and state[cr, cc] != 0:
            if state[cr, cc] == color:
                if buffer:
                    n_discs += buffer
                    n_dirs += 1
                break
            buffer += 1
            cr, cc = cr + dr, cc + dc
    return n_discs, n_dirs


def main():
    items = torch.load(study_dir("rules") / "pairs" / "items.pt", weights_only=True)
    n = len(items["t"])
    n_flipped = torch.full((n,), -1, dtype=torch.long)
    n_directions = torch.full((n,), -1, dtype=torch.long)

    legal_idx = (items["category"] == LEGAL).nonzero().flatten().tolist()
    by_prefix: dict[int, list[int]] = {}
    for i in legal_idx:
        by_prefix.setdefault(int(items["prefix_id"][i]), []).append(i)

    for done, (pid, idxs) in enumerate(by_prefix.items()):
        t = int(items["t"][idxs[0]])
        board = OthelloBoardState()
        board.update(to_string(items["tokens"][idxs[0], :t].long().tolist()))
        for i in idxs:
            square = to_string(int(items["injected_token"][i]))
            discs, dirs = flip_stats(board.state, square, board.next_hand_color)
            assert discs > 0, f"legal item {i} flips nothing"
            n_flipped[i], n_directions[i] = discs, dirs
        if (done + 1) % 500 == 0:
            print(f"{done + 1}/{len(by_prefix)} prefixes")

    out = study_dir("rules") / "pairs" / "flip_labels.pt"
    torch.save({"n_flipped": n_flipped, "n_directions": n_directions}, out)
    legal_mask = n_flipped >= 0
    print(f"wrote {out}: {int(legal_mask.sum())} legal items, "
          f"flips mean {n_flipped[legal_mask].float().mean():.2f} "
          f"max {int(n_flipped.max())}, "
          f"dirs mean {n_directions[legal_mask].float().mean():.2f}")


if __name__ == "__main__":
    main()
