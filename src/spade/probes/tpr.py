"""The TPR arm of the shared probe protocol: logit(s, c) = r_s^T M(h) f_c.

Architecture and loss are upstream's (`external/tpr_othello`, arXiv:2605.09967),
imported unmodified. What is ours is the recipe, and the recipe is what was
wrong: the bracket-flip fit ran a fixed 120 epochs on 8,000 rows at wd=2.0. It
now runs under `probes.train`'s protocol -- split by game, early stop on
validation accuracy, best-validation weights saved -- over the whole corpus.

WEIGHT_DECAY is re-selected on validation by src/probes/tpr_weight_decay.py;
2.0 was tuned for 8,000 rows and there are now ~812,000.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


from spade.common.paths import repo_root

_UPSTREAM = repo_root() / "external" / "tpr_othello" / "src"
sys.path.insert(0, str(_UPSTREAM))
sys.path.insert(0, str(_UPSTREAM / "hook_utils"))

from intervene_tpr_probe import solve_residual_delta_for_binding_delta
from train_tpr_probe import (
    TensorProductBoardProbe,
    build_square_selection_mask,
    compute_tpr_probe_loss,
)

ROLE_DIM = 52
FILLER_DIM = 2
LR = 1e-2
BETAS = (0.9, 0.99)
WEIGHT_DECAY = 1e-3        # validation argmax over 1e-4..2.0 at layer 5; the old 2.0 costs 0.0009
MAX_EPOCHS = 400           # this arm's own ceiling; it plateaus by ~50, so this is slack
CLASSES = 3


class TPR(torch.nn.Module):
    """Bilinear read-out: a residual vector becomes a role x filler binding tensor."""

    max_epochs = MAX_EPOCHS

    def __init__(self, d_model: int, seed: int, weight_decay: float = WEIGHT_DECAY,
                 role_dim: int = ROLE_DIM):
        super().__init__()
        torch.manual_seed(seed)
        self.module = TensorProductBoardProbe(
            d_model=d_model, role_dim=role_dim, filler_dim=FILLER_DIM)
        self.weight_decay = weight_decay
        self.register_buffer("square_mask", build_square_selection_mask(
            exclude_center_squares=False, device="cpu"))

    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        """[..., d_model] -> [..., 64, 3] logits, squares in row-major order."""
        logits = self.module(acts.float().reshape(-1, 1, acts.shape[-1]))
        return logits.reshape(*acts.shape[:-1], 64, CLASSES)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(y.long(), CLASSES).float().reshape(-1, 1, 8, 8, CLASSES)
        return compute_tpr_probe_loss(
            logits.reshape(-1, 1, 8, 8, CLASSES), one_hot, self.square_mask)

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=LR, betas=BETAS, weight_decay=self.weight_decay)

    def card(self) -> dict:
        return {"probe_state_dict": {k: v.detach().cpu()
                                     for k, v in self.module.state_dict().items()},
                "role_dim": self.roles.shape[-1], "filler_dim": FILLER_DIM,
                "weight_decay": self.weight_decay}

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
