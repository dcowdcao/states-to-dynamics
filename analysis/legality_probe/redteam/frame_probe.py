"""Is the SpaDE collapse about the *turn rule*, or about the mover's colour?

Three arms per space, on the same fit set and the same loop as `rederive.py`:

  board              {empty, theirs, mine} -- with its 3x3 confusion, so a colour failure
                     is separable from an occupancy failure
  occupancy          {empty, occupied} -- the frame-invariant half of the board
  legality_frameless {flips-for-someone, R1, R2} -- the 4-class legality alphabet with
                     its two frame-*sensitive* classes (legal, R3) merged. A space that
                     has lost only the mover's frame scores here exactly as well as a
                     space that has lost nothing.

The 4-class alphabet is built so that R1 and R2 are invariant to the mover's colour
(`frame_check.json`: 0.0 of R1 and 0.0 of R2 relabel under a frame flip) while `legal`
and `R3` are not (0.703 and 1.0). "R1 and R2 survive, R3 dies" is therefore the signature
of *any* colour failure, and needs this control to mean anything about turn-taking.

Usage: python analysis/legality_probe/redteam/frame_probe.py [--spaces a,b]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rederive import (  # noqa: E402
    N_TRAIN, OUT, f1_of, fit_probe, fit_targets, predict, tally)

from spade.common.config import LAYER  # noqa: E402
from spade.common.study import study_dir  # noqa: E402
from spade.spaces import batched_apply, load_space, representative_spaces  # noqa: E402


def confusion(preds, truth, n):
    return np.array([[int(((truth == a) & (preds == b)).sum()) for b in range(n)]
                     for a in range(n)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    legal_labels, board_labels = fit_targets()
    frameless = np.where(np.isin(legal_labels, [0, 3]), 0,
                         np.where(legal_labels == 1, 1, 2)).astype(np.int8)
    occupancy = (board_labels > 0).astype(np.int8)
    raw_acts = torch.load(study_dir("legality") / "acts" / f"resid_post_{LAYER}.pt",
                          map_location="cpu", weights_only=True)

    roster = [(n, e) for n, e in representative_spaces()
              if n == "raw" or (e is not None and e.seed == 42)]
    roster += [("random_proj", None), ("random_topk", None)]
    if args.spaces:
        keep = set(args.spaces.split(",")) | {"raw"}
        roster = [s for s in roster if s[0] in keep]

    rows = []
    for name, entry in roster:
        code = batched_apply(load_space(name, entry, device), raw_acts, device).float()
        out = {"space": name}
        for arm, labels, n_classes, filler in [("board", board_labels, 3, 2),
                                               ("occupancy", occupancy, 2, 2),
                                               ("legality_frameless", frameless, 3, 3),
                                               ("legality", legal_labels, 4, 3)]:
            probe = fit_probe(code[:N_TRAIN].to(device),
                              torch.from_numpy(labels[:N_TRAIN]).long().to(device),
                              n_classes, filler, 0, device)
            preds = predict(probe, code[N_TRAIN:], device)
            truth = labels[N_TRAIN:]
            out[f"{arm}_macro_f1"] = float(f1_of(tally(preds, truth, n_classes).sum(0)).mean())
            if arm == "board":
                matrix = confusion(preds.reshape(-1), truth.reshape(-1), 3)
                rates = matrix / matrix.sum(1, keepdims=True)
                for i, a in enumerate(("empty", "theirs", "mine")):
                    for j, b in enumerate(("empty", "theirs", "mine")):
                        out[f"board_{a}_as_{b}"] = float(rates[i, j])
                both = (preds > 0) & (truth > 0)
                out["colour_accuracy_given_both_occupied"] = float(
                    (preds[both] == truth[both]).mean())
            if arm == "legality":
                out["p_predicts_legal"] = float((preds == 0).mean())
                out["p_predicts_R3"] = float((preds == 3).mean())
        rows.append(out)
        print(f"{name:28} " + "  ".join(
            f"{k} {v:.4f}" for k, v in out.items() if k != "space"), flush=True)
        del code
        pd.DataFrame(rows).to_csv(OUT / "rt_frame.csv", index=False)
    print(pd.DataFrame(rows).set_index("space").round(4).to_string())


if __name__ == "__main__":
    main()
