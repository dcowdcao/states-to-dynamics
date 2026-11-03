"""The one place any experiment gets a probe.

Every consumer reads a probe through here, so "the board probe" names a single
artifact rather than whichever of six recipes a study happened to fit. The
layer sweep is the deliberate exception: it wants one probe per layer, which is
what `board(layer)` gives it.

Shapes match the convention the rest of the repo already loads:
`[512, 8, 8, 3]` over ("empty", "theirs", "mine"), mover-relative.

Usage: from spade.probes.load import board; w = board()          # layer 5
"""

from pathlib import Path

import torch


from spade.common.config import LAYER
from spade.probes.study_dir import probes_dir

CLASSES = ("empty", "theirs", "mine")


def path(layer: int = LAYER, arm: str = "board") -> Path:
    return probes_dir() / "probes" / f"{arm}_layer{layer}.pt"


def board(layer: int = LAYER, flat: bool = False) -> torch.Tensor:
    """[512, 8, 8, 3] board-state probe weights, or [512, 64, 3] when `flat`."""
    w = torch.load(path(layer), weights_only=True)["weight"]
    return w if flat else w.reshape(512, 8, 8, 3)


def card(layer: int = LAYER, arm: str = "board") -> dict:
    """Everything needed to cite the probe: accuracy, corpus, where it stopped."""
    meta = torch.load(path(layer, arm), weights_only=True)
    return {k: v for k, v in meta.items() if k not in ("weight", "probe_state_dict")}


def tpr(layer: int = LAYER):
    """The bilinear read-out at `layer`: logit(s, c) = r_s^T M(h) f_c, over the same
    corpus and protocol as `board`. Call it on [n, 512] for [n, 64, 3] logits."""
    # Local: probes.tpr puts the vendored external/tpr_othello/src on sys.path, and
    # nothing that only wants `board()` should pay that.
    from spade.probes.tpr import from_card
    return from_card(torch.load(path(layer, "tpr"), weights_only=True))
