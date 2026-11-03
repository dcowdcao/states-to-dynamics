"""4-way legality probe: role = 64 squares, filler = {legal, R1, R2, R3}.

`spade.probes.tpr.TPR` with the filler alphabet widened from 3 board classes to the 4
legality classes, fitted by `spade.probes.train.fit`. The input dimension is read off the
activations, so the same `fit` runs on the raw d=512 residual and on a d=8192 SAE code.

The protocol is the project's, not this study's: split by game, early stop when validation
accuracy has plateaued, keep the best-validation weights, and fail rather than stop on a
step budget. `fit_set`'s 40 eval games are the test split and no fit ever sees them --
validation is `VAL_GAMES` of the 160 fit games, so neither early stopping nor the
weight-decay selection can reach the test set.

Classes are severely imbalanced -- R1 is 54% of squares mid-game, R3 has one taxonomy
member -- so `macro_f1` is the metric. Fit on shuffled labels it reads 0.29 against a 0.56
accuracy (PROCESS.md, 2026-08-05); accuracy alone hides that.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from spade.common.config import LAYER
from spade.common.study import study_dir
from spade.legality_probe.fit_set import EVAL_GAMES, TRAIN_GAMES
from spade.probes import train
from spade.probes.tpr import TPR, from_card

CLASSES = ("legal", "R1", "R2", "R3")  # taxonomy.LEGAL, then RULE_MEMBERS in order
FILLER_DIM = 3          # upstream used 2 for 3 classes; see PROCESS.md for the 4-class sweep
VAL_GAMES = 32          # of fit_set's 160 fit games; the 40 eval games are the test split
TRAIN_POOL = TRAIN_GAMES - VAL_GAMES
BATCH = 1024            # 6,400 training rows: the corpus default of 8192 is one step per epoch
MAX_EPOCHS = 3000       # 7 steps an epoch against the shared corpus's ~70, so the shared
                        # ceiling of 400 truncates a fit here rather than bounding it
SEED = 42
EVAL_BATCH = 4096


@dataclass(frozen=True)
class Scores:
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    per_class_recall: dict[str, float]
    support: dict[str, int]
    n: int  # positions; each contributes 64 labelled squares


def split(game: Tensor, n_train_games: int = TRAIN_POOL,
          seed: int = SEED) -> tuple[Tensor, Tensor, Tensor]:
    """Train / validation / test row indices, disjoint by source game.

    `n_train_games` shrinks the training pool for the learning curve. Validation and test
    are the same games at every size, so a smaller fit is a smaller *training* set and
    nothing else.
    """
    order = np.random.default_rng(seed).permutation(TRAIN_GAMES)
    return (train.rows(game, order[VAL_GAMES:VAL_GAMES + n_train_games]),
            train.rows(game, order[:VAL_GAMES]),
            train.rows(game, np.arange(TRAIN_GAMES, TRAIN_GAMES + EVAL_GAMES)))


def fit(code: Tensor, labels: Tensor, tr: Tensor, va: Tensor, weight_decay: float, *,
        classes: tuple[str, ...] = CLASSES, filler_dim: int = FILLER_DIM,
        seed: int = SEED, device: str = "cpu") -> tuple[TPR, train.Fit]:
    """code [n, d], labels [n, 64] class indices, `tr`/`va` row indices from `split`.

    `weight_decay` has no default because there is no one right value: it is selected per
    space by `legality_probe.weight_decay.selected`, and at a shared 2.0 the two SpaDE
    codes read 0.44-0.50 macro-F1 against 0.88-0.93 at their own argmax.

    `classes` is the filler alphabet: pass the 3 board classes to fit the occupancy arm.
    """
    arm = TPR(code.shape[-1], seed, weight_decay, n_classes=len(classes),
              filler_dim=filler_dim, max_epochs=MAX_EPOCHS)
    result = train.fit(arm, code[tr].float(), labels[tr],
                       code[va].float(), labels[va], device, BATCH)
    return arm, result


def macro_f1(arm: TPR, acts: Tensor, labels: Tensor,
             classes: tuple[str, ...] = CLASSES) -> Scores:
    """Unweighted mean over classes of per-class F1, upstream's `macro_f1_score` definition
    (an absent class scores 0), plus the per-class parts and the accuracy it must not be
    reported without."""
    device = arm.module.binding_map.device
    labels = labels.reshape(-1, 64)
    preds = torch.cat([arm(acts[i:i + EVAL_BATCH].to(device)).argmax(-1).cpu()
        for i in range(0, acts.shape[0], EVAL_BATCH)])

    f1, recall, support = {}, {}, {}
    for c, name in enumerate(classes):
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


def path(space: str, target: str, layer: int = LAYER) -> Path:
    """`space` and `target` both key the file: without either, two probes at one layer
    overwrite each other and `load` silently returns the wrong one."""
    return study_dir("legality") / "probes" / f"{target}_{space}_layer{layer}.pt"


def save(arm: TPR, space: str, target: str, card: dict, layer: int = LAYER) -> Path:
    p = path(space, target, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**arm.card(), **card, "space": space, "target": target, "layer": layer}, p)
    return p


def load(space: str, target: str, layer: int = LAYER) -> TPR:
    return from_card(torch.load(path(space, target, layer), map_location="cpu",
                                weights_only=True))


def card(space: str, target: str, layer: int = LAYER) -> dict:
    """Everything needed to cite the probe: where it stopped, what it scored, its corpus."""
    meta = torch.load(path(space, target, layer), map_location="cpu", weights_only=True)
    return {k: v for k, v in meta.items() if k != "probe_state_dict"}
