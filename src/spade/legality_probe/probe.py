"""4-way bilinear legality probe: role = 64 squares, filler = {legal, R1, R2, R3}.

`logit(s, c) = r_s^T M(h) f_c` through the vendored `TensorProductBoardProbe`, with the
filler alphabet widened from 3 board classes to the 4 legality classes. The input dimension
is read off the activations, so the same `fit` runs on the raw d=512 residual and on a
d=8192 SAE code.

SEED, WEIGHT_DECAY, EPOCHS and BATCH_SIZE are the board probe's retired 8,000-row recipe,
which is what `analysis/legality_probe/` was fit under; they stay here rather than tracking
`spade.probes.tpr`, whose values were re-selected for the 812,000-row canonical corpus.

Upstream sizes `filler_embeddings` and scales the loss from a module-level
`BOARD_LABEL_OPTIONS = 3` that `train_tpr_probe` imported *by value*; there is no
per-probe override. `filler_classes()` rebinds it for the duration of a fit, so a 4-class
legality fit and the committed 3-class board fit can live in one process.

Classes are severely imbalanced — R1 is 54% of squares mid-game, R3 has one taxonomy
member — so `macro_f1` is the metric. Fit on shuffled labels it reads 0.29 against a 0.56
accuracy (PROCESS.md, 2026-08-05); accuracy alone hides that.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from spade.common.config import LAYER
from spade.common.paths import add_tpr_to_path
from spade.common.study import study_dir
from spade.probes.tpr import (
    BETAS,
    LR,
    ROLE_DIM,
    TensorProductBoardProbe,
    build_square_selection_mask,
    compute_tpr_probe_loss,
)

add_tpr_to_path()

import train_tpr_probe  # noqa: E402  the module, so filler_classes can rebind its global
from hook_utils import seed_all  # noqa: E402

CLASSES = ("legal", "R1", "R2", "R3")  # taxonomy.LEGAL, then RULE_MEMBERS in order
FILLER_DIM = 3  # upstream used 2 for 3 classes; see PROCESS.md for the 4-class sweep
EVAL_BATCH = 4096
SEED, WEIGHT_DECAY, EPOCHS, BATCH_SIZE = 1111, 2.0, 120, 1024


@contextmanager
def filler_classes(n: int) -> Iterator[None]:
    """Rebind upstream's by-value `BOARD_LABEL_OPTIONS`, which sizes the filler embeddings
    and scales the loss. Without it a 4-class fit silently builds 3 fillers."""
    previous = train_tpr_probe.BOARD_LABEL_OPTIONS
    train_tpr_probe.BOARD_LABEL_OPTIONS = n
    try:
        yield
    finally:
        train_tpr_probe.BOARD_LABEL_OPTIONS = previous


@dataclass(frozen=True)
class LegalityProbe:
    module: TensorProductBoardProbe
    layer: int
    classes: tuple[str, ...]

    def decode(self, h: Tensor) -> Tensor:
        """[..., d] -> [..., 64, len(classes)] logits, squares row-major (square = row*8 + col)."""
        logits = self.module(h.reshape(-1, 1, h.shape[-1]))
        return logits.reshape(*h.shape[:-1], 64, len(self.classes))


@dataclass(frozen=True)
class Scores:
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_recall: dict[str, float]
    support: dict[str, int]
    n: int  # positions; each contributes 64 labelled squares


def fit(acts: Tensor, labels: Tensor, *, filler_dim: int = FILLER_DIM,
        classes: tuple[str, ...] = CLASSES, seed: int = SEED,
        device: str = "cpu", layer: int = LAYER) -> LegalityProbe:
    """acts [n, d], labels [n, 64] or [n, 8, 8] of class indices. Probe stays on `device`.

    `classes` is the filler alphabet: pass the 3 board classes to refit the board
    probe through this loop.
    """
    n_classes = len(classes)
    with filler_classes(n_classes):
        seed_all(seed)
        acts = acts.to(device)
        module = TensorProductBoardProbe(
            d_model=acts.shape[-1], role_dim=ROLE_DIM, filler_dim=filler_dim).to(device)
        assert module.filler_embeddings.shape == (n_classes, filler_dim), module.filler_embeddings.shape
        assert train_tpr_probe.BOARD_LABEL_OPTIONS == n_classes  # the loss's scale factor
        optimizer = torch.optim.AdamW(module.parameters(), lr=LR,
                                      betas=BETAS, weight_decay=WEIGHT_DECAY)
        targets = one_hot(labels.reshape(-1, 8, 8).to(device), n_classes).float().unsqueeze(1)
        mask = build_square_selection_mask(exclude_center_squares=False, device=device)
        for _epoch in range(EPOCHS):
            for batch in torch.randperm(acts.shape[0], device=device).split(BATCH_SIZE):
                loss = compute_tpr_probe_loss(
                    module(acts[batch].unsqueeze(1)), targets[batch], square_mask=mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    return LegalityProbe(module.requires_grad_(False), layer, tuple(classes))


def macro_f1(probe: LegalityProbe, acts: Tensor, labels: Tensor) -> Scores:
    """Unweighted mean over classes of per-class F1, upstream's `macro_f1_score` definition
    (an absent class scores 0), plus the per-class parts and the accuracy it must not be
    reported without."""
    device = probe.module.binding_map.device
    labels = labels.reshape(-1, 64)
    preds = torch.cat([probe.decode(acts[i:i + EVAL_BATCH].to(device)).argmax(-1).cpu()
        for i in range(0, acts.shape[0], EVAL_BATCH)])

    f1, recall, support = {}, {}, {}
    for c, name in enumerate(probe.classes):
        predicted, actual = preds == c, labels == c
        tp = (predicted & actual).sum().item()
        fp = (predicted & ~actual).sum().item()
        fn = (~predicted & actual).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall[name] = tp / (tp + fn) if tp + fn else 0.0
        denominator = precision + recall[name]
        f1[name] = 2 * precision * recall[name] / denominator if denominator else 0.0
        support[name] = int(actual.sum())

    return Scores(
        macro_f1=sum(f1.values()) / len(f1),
        accuracy=(preds == labels).float().mean().item(),
        per_class_f1=f1, per_class_recall=recall, support=support, n=labels.shape[0])


def _path(space: str, layer: int, filler_dim: int) -> Path:
    """`space` keys the file: without it two spaces' probes at one layer overwrite each
    other and `load` silently returns the wrong one."""
    return (study_dir("legality") / "probes"
            / f"tpr_r{ROLE_DIM}_f{filler_dim}_{space}_layer{layer}.pt")


def save(probe: LegalityProbe, space: str) -> Path:
    path = _path(space, probe.layer, probe.module.filler_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "probe_state_dict": {k: v.cpu() for k, v in probe.module.state_dict().items()},
        "layer": probe.layer,
        "classes": list(probe.classes),
        "space": space,
    }, path)
    return path


def load(space: str, layer: int = LAYER, filler_dim: int = FILLER_DIM) -> LegalityProbe:
    artifact = torch.load(_path(space, layer, filler_dim), map_location="cpu", weights_only=True)
    d_model, role_dim, saved_filler_dim = artifact["probe_state_dict"]["binding_map"].shape
    with filler_classes(len(artifact["classes"])):
        module = TensorProductBoardProbe(
            d_model=d_model, role_dim=role_dim, filler_dim=saved_filler_dim)
    module.load_state_dict(artifact["probe_state_dict"])
    return LegalityProbe(module.requires_grad_(False), artifact["layer"], tuple(artifact["classes"]))
