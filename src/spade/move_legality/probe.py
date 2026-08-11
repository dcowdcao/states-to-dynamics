"""The linear probe itself: L2 multinomial logistic regression on one space's features.

Full-batch L-BFGS rather than SGD, so a fit is the exact optimum of a stated objective
(mean cross-entropy + `l2`*||W||^2 over standardized features) and not a point on an
optimization path -- eleven spaces are compared to each other, and a learning-rate
schedule that suits d=512 and not d=8192 would come out looking like a space difference.

Features are standardized by the *train* split's per-feature mean and SD; a dead feature
(SD 0, common in a sparse dictionary) standardizes to 0 rather than to noise over zero.
The transform is saved inside the probe, because the read-out applies it to activations
the fit never saw and the two have to be the same map.

`balanced` is where the class balance lives. A fit-set item is one (game, position) pair
carrying one row per class, so keeping whole items keeps the classes equal in every split
and identical in their game and move-index marginals. The 2-class form draws ONE illegal
row per item at a stated seed; it never averages the three, which would hand the illegal
class 1/sqrt(3) of its real spread (FIGURE_GUIDELINES 3).
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from spade.common.config import CTX_LEN
from spade.common.study import study_dir
from spade.move_legality.fit_set import CLASSES

BINARY_CLASSES = ("legal", "illegal")
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
MAX_ITER = 300
MIN_SCALE = 1e-6


def classes_for(n_classes: int) -> tuple[str, ...]:
    return BINARY_CLASSES if n_classes == 2 else CLASSES


@dataclass(frozen=True)
class LinearProbe:
    weight: Tensor              # [n_classes, d]
    bias: Tensor                # [n_classes]
    mean: Tensor                # [d] train-split feature mean
    scale: Tensor               # [d] train-split feature SD, dead features at 1
    classes: tuple[str, ...]
    space: str
    l2: float

    def proba(self, feats: Tensor) -> Tensor:
        """[..., d] -> [..., n_classes]. Standardizes with the fit's own statistics."""
        device = feats.device
        z = (feats.float() - self.mean.to(device)) / self.scale.to(device)
        return (z @ self.weight.to(device).T + self.bias.to(device)).softmax(-1)

    def p_illegal(self, feats: Tensor) -> Tensor:
        """One number in [0, 1] per position, whatever the class count: 1 - P(legal)."""
        return 1.0 - self.proba(feats)[..., 0]


@dataclass(frozen=True)
class Scores:
    accuracy: float
    macro_f1: float
    auroc: float                       # illegal (any class > 0) against legal
    per_class_f1: dict[str, float]
    support: dict[str, int]
    n: int


def items(fit: dict) -> Tensor:
    """One id per (game, position) pair, in row order."""
    return fit["game_id"].long() * CTX_LEN + fit["position"].long()


def balanced(fit: dict, n_classes: int, seed: int = 0) -> Tensor:
    """Row indices of a class-balanced set: whole items, one row per class.

    4 classes keeps the items where all three rules have a candidate; 2 keeps every item
    and draws one uniform illegal row from it.
    """
    key, label = items(fit), fit["label"].long()
    if n_classes == len(CLASSES):
        return (torch.bincount(key)[key] == n_classes).nonzero().flatten()

    illegal = (label > 0).nonzero().flatten()
    draw = torch.rand(len(illegal), generator=torch.Generator().manual_seed(seed),
                      dtype=torch.float64)
    order = illegal[(key[illegal].double() + draw).argsort()]   # by item, then uniformly
    first = torch.ones(len(order), dtype=torch.bool)
    first[1:] = key[order][1:] != key[order][:-1]
    legal = (label == 0).nonzero().flatten()
    assert len(order[first]) == len(legal), "an item has a legal row and no illegal row"
    return torch.cat([legal, order[first]]).sort().values


def targets(label: Tensor, n_classes: int) -> Tensor:
    """Class index per row: the label itself at 4 classes, legal-vs-illegal at 2."""
    return label.long() if n_classes > 2 else (label > 0).long()


def standardizer(feats: Tensor) -> tuple[Tensor, Tensor]:
    scale = feats.std(0)
    return feats.mean(0), torch.where(scale > MIN_SCALE, scale, torch.ones_like(scale))


def _optimize(z: Tensor, y: Tensor, n_classes: int, l2: float) -> tuple[Tensor, Tensor]:
    weight = torch.zeros(n_classes, z.shape[1], device=z.device, requires_grad=True)
    bias = torch.zeros(n_classes, device=z.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=MAX_ITER, history_size=10,
                                  tolerance_grad=1e-7, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(z @ weight.T + bias, y) + l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach().cpu(), bias.detach().cpu()


def fit(train: Tensor, train_y: Tensor, val: Tensor, val_y: Tensor, space: str,
        n_classes: int, device: str = "cpu") -> tuple[LinearProbe, list[dict]]:
    """Fit at every l2 in L2_GRID; keep the one with the best val macro-F1."""
    classes = classes_for(n_classes)
    mean, scale = standardizer(train.float())
    z = ((train.float() - mean) / scale).to(device)
    y = train_y.to(device)

    best, sweep = None, []
    for l2 in L2_GRID:
        weight, bias = _optimize(z, y, n_classes, l2)
        probe = LinearProbe(weight, bias, mean, scale, classes, space, l2)
        scored = score(probe, val, val_y)
        sweep.append({"space": space, "n_classes": n_classes, "l2": l2,
                      "val_macro_f1": round(scored.macro_f1, 5),
                      "val_accuracy": round(scored.accuracy, 5),
                      "val_auroc": round(scored.auroc, 5)})
        if best is None or scored.macro_f1 > best[1]:
            best = (probe, scored.macro_f1)
    return best[0], sweep


def score(probe: LinearProbe, feats: Tensor, y: Tensor, batch: int = 8192) -> Scores:
    proba = torch.cat([probe.proba(feats[i:i + batch]).cpu()
                       for i in range(0, len(feats), batch)])
    preds = proba.argmax(-1)
    f1, support = {}, {}
    for c, name in enumerate(probe.classes):
        predicted, actual = preds == c, y == c
        tp = int((predicted & actual).sum())
        precision = tp / max(int(predicted.sum()), 1)
        recall = tp / max(int(actual.sum()), 1)
        f1[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support[name] = int(actual.sum())
    return Scores(accuracy=float((preds == y).float().mean()),
                  macro_f1=sum(f1.values()) / len(f1),
                  auroc=auroc(1.0 - proba[:, 0], (y > 0).float()),
                  per_class_f1=f1, support=support, n=len(y))


def auroc(scores: Tensor, positive: Tensor) -> float:
    """Rank-based with ties averaged. Never flipped: a sub-0.5 value is the finding."""
    order = scores.argsort()
    sorted_scores, sorted_positive = scores[order], positive[order]
    ranks = torch.arange(1, len(order) + 1, dtype=torch.float64)
    _, inverse, counts = sorted_scores.unique(return_inverse=True, return_counts=True)
    ranks = (torch.zeros(len(counts), dtype=torch.float64)
             .index_add_(0, inverse, ranks) / counts)[inverse]
    n_pos = float(sorted_positive.sum())
    n_neg = len(sorted_positive) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[sorted_positive.bool()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def path(space: str, n_classes: int) -> Path:
    return study_dir("move_legality") / "probes" / f"{space}_{n_classes}class.pt"


def save(probe: LinearProbe) -> Path:
    out = path(probe.space, len(probe.classes))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": probe.weight, "bias": probe.bias, "mean": probe.mean,
                "scale": probe.scale, "classes": list(probe.classes),
                "space": probe.space, "l2": probe.l2}, out)
    return out


def load(space: str, n_classes: int) -> LinearProbe:
    a = torch.load(path(space, n_classes), map_location="cpu", weights_only=True)
    return LinearProbe(a["weight"], a["bias"], a["mean"], a["scale"],
                       tuple(a["classes"]), a["space"], a["l2"])
