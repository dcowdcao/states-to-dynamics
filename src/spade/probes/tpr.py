"""The TPR arm of the shared probe protocol: logit(s, c) = r_s^T M(h) f_c.

Architecture and loss are upstream's (`external/tpr_othello`, arXiv:2605.09967),
imported unmodified. What is ours is the recipe, and the recipe is what was
wrong: the bracket-flip fit ran a fixed 120 epochs on 8,000 rows at wd=2.0. It
now runs under `probes.train`'s protocol -- split by game, early stop on
validation accuracy, best-validation weights saved -- over the whole corpus.

The filler alphabet is a parameter: 3 board classes for `probes.train`'s board
target, 4 for `legality_probe`'s legal/R1/R2/R3. Upstream sizes
`filler_embeddings` and scales the loss from a module-level `BOARD_LABEL_OPTIONS`
that it imported *by value*, and there is no per-probe override, so
`filler_classes` rebinds it around the two places that read it.

WEIGHT_DECAY is re-selected on validation by spade.probes.tpr_weight_decay;
2.0 was tuned for 8,000 rows and there are now ~812,000.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F


from spade.common.paths import repo_root

_UPSTREAM = repo_root() / "external" / "tpr_othello" / "src"
sys.path.insert(0, str(_UPSTREAM))
sys.path.insert(0, str(_UPSTREAM / "hook_utils"))

import train_tpr_probe  # noqa: E402  the module, so filler_classes can rebind its global
from intervene_tpr_probe import solve_residual_delta_for_binding_delta  # noqa: E402
from train_tpr_probe import (  # noqa: E402
    TensorProductBoardProbe,
    build_square_selection_mask,
    compute_tpr_probe_loss,
)

ROLE_DIM = 52
FILLER_DIM = 2
N_CLASSES = 3
LR = 1e-2
BETAS = (0.9, 0.99)
WEIGHT_DECAY = 1e-3        # validation argmax over 1e-4..2.0 at layer 5; the old 2.0 costs 0.0009
MAX_EPOCHS = 400           # slack over the ~50 the 812k-row corpus needs; a small fit
                           # takes far more epochs for the same number of steps and passes
                           # its own ceiling (legality_probe.probe.MAX_EPOCHS)


@contextmanager
def filler_classes(n: int) -> Iterator[None]:
    """Rebind upstream's by-value `BOARD_LABEL_OPTIONS`. Without it a 4-class arm
    silently builds 3 fillers, and its loss is scaled by 3 rather than 4."""
    previous = train_tpr_probe.BOARD_LABEL_OPTIONS
    train_tpr_probe.BOARD_LABEL_OPTIONS = n
    try:
        yield
    finally:
        train_tpr_probe.BOARD_LABEL_OPTIONS = previous


class TPR(torch.nn.Module):
    """Bilinear read-out: a residual vector becomes a role x filler binding tensor."""

    def __init__(self, d_model: int, seed: int, weight_decay: float = WEIGHT_DECAY,
                 role_dim: int = ROLE_DIM, n_classes: int = N_CLASSES,
                 filler_dim: int = FILLER_DIM, max_epochs: int = MAX_EPOCHS):
        super().__init__()
        self.max_epochs = max_epochs
        torch.manual_seed(seed)
        with filler_classes(n_classes):
            self.module = TensorProductBoardProbe(
                d_model=d_model, role_dim=role_dim, filler_dim=filler_dim)
        assert self.module.filler_embeddings.shape == (n_classes, filler_dim)
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.register_buffer("square_mask", build_square_selection_mask(
            exclude_center_squares=False, device="cpu"))

    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        """[..., d_model] -> [..., 64, n_classes] logits, squares in row-major order."""
        logits = self.module(acts.float().reshape(-1, 1, acts.shape[-1]))
        return logits.reshape(*acts.shape[:-1], 64, self.n_classes)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(y.long(), self.n_classes).float().reshape(
            -1, 1, 8, 8, self.n_classes)
        with filler_classes(self.n_classes):
            return compute_tpr_probe_loss(
                logits.reshape(-1, 1, 8, 8, self.n_classes), one_hot, self.square_mask)

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=LR, betas=BETAS, weight_decay=self.weight_decay)

    def card(self) -> dict:
        return {"probe_state_dict": {k: v.detach().cpu()
                                     for k, v in self.module.state_dict().items()},
                "role_dim": self.roles.shape[-1], "filler_dim": self.fillers.shape[-1],
                "n_classes": self.n_classes, "weight_decay": self.weight_decay}

    @property
    def roles(self) -> torch.Tensor:
        return self.module.role_embeddings.reshape(64, -1)

    @property
    def fillers(self) -> torch.Tensor:
        return self.module.filler_embeddings

    def edit_direction(self, square: int, from_c: int, to_c: int) -> torch.Tensor:
        """Least-norm residual step realizing dB = r_s (f_to - f_from)^T."""
        delta_binding = torch.outer(self.roles[square],
                                    self.fillers[to_c] - self.fillers[from_c])
        return solve_residual_delta_for_binding_delta(self.module.binding_map, delta_binding)


def from_card(meta: dict) -> TPR:
    """The arm a saved card describes, with its weights loaded and frozen.

    Shape, not metadata, carries the filler alphabet: `filler_embeddings` is
    [n_classes, filler_dim], so cards written before either was a parameter load.
    """
    state = meta["probe_state_dict"]
    n_classes, filler_dim = state["filler_embeddings"].shape
    arm = TPR(state["binding_map"].shape[0], meta["seed"], meta["weight_decay"],
              meta["role_dim"], n_classes, filler_dim)
    arm.module.load_state_dict(state)
    return arm.requires_grad_(False)
