"""The one training protocol every probe in this project uses.

Two rules the previous fits broke, each of which cost a real result:

1. **Stop on the plateau, never on a step budget.** Training runs until
   validation accuracy has not improved by MIN_DELTA for PATIENCE epochs, and
   the best-validation weights are what gets saved. Hitting `max_epochs` is a
   failure, not a stopping rule, and `fit` raises if it does — the committed
   move-legality probe was capped by its iteration limit at a training objective
   of 1.13 against an achievable 0.54, which cost it 15 points of test accuracy.
2. **Split by game, never by row.** Positions within a game share a board.

`fit` never sees the test split, so nothing can be selected on it. The curve is
returned alongside the weights so a reader can see the plateau rather than take
it on faith.

Two arms share all of that: `board`, the per-square linear read-out, and `tpr`,
the bilinear tensor-product read-out of `probes.tpr`. An arm is an nn.Module
with `forward([n, d_model]) -> [n, 64, 3]` plus `loss`, `optimizer`, `card` and
`max_epochs`; everything else here is arm-agnostic.

Usage: python src/probes/train.py [--arm board|tpr] [--layers 0-7] [--seed 42]
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


from spade.probes.study_dir import probes_dir
from spade.probes.tpr import TPR

MAX_EPOCHS = 400
PATIENCE = 20
MIN_DELTA = 1e-4        # on validation accuracy
BATCH = 8192
SPLIT = (0.7, 0.15, 0.15)   # train / val / test, over GAMES
CLASSES = ("empty", "theirs", "mine")


class Board(torch.nn.Module):
    """Per-square 3-way softmax read-out: logit(s, c) = w[:, s, c] . h."""

    max_epochs = MAX_EPOCHS

    def __init__(self, d_model: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.weight = torch.nn.Parameter(
            torch.randn(d_model, 64, len(CLASSES)) * 0.01)

    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nd,dsc->nsc", acts.float(), self.weight)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, len(CLASSES)), y.reshape(-1).long())

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-2, weight_decay=1e-4)

    def card(self) -> dict:
        return {"weight": self.weight.detach().cpu()}


ARMS = {"board": Board, "tpr": TPR}


@dataclass
class Fit:
    state: dict[str, torch.Tensor]
    curve: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    val_accuracy: float = 0.0


def game_split(n_games: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_games)
    a, b = int(SPLIT[0] * n_games), int((SPLIT[0] + SPLIT[1]) * n_games)
    return order[:a], order[a:b], order[b:]


def rows(games: torch.Tensor, of: np.ndarray) -> torch.Tensor:
    return torch.isin(games, torch.from_numpy(of)).nonzero().flatten()


def relative_board(state: torch.Tensor, mover: torch.Tensor) -> torch.Tensor:
    """[n, p, 64] in {0: empty, 1: theirs, 2: mine}; mine is the player to move next."""
    signed = state * mover[:, :, None, None]
    return torch.where(signed > 0, 2, torch.where(signed < 0, 1, 0)).flatten(2).to(torch.int8)


def corpus() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mover-relative labels [n_rows, 64], the game each row belongs to, and n_games."""
    labels = torch.load(probes_dir() / "corpus" / "labels.pt", weights_only=True)
    y = relative_board(labels["state"], labels["mover"])
    n_games, positions = y.shape[0], y.shape[1]
    return (y.reshape(-1, 64),
            torch.arange(n_games).repeat_interleave(positions), n_games)


def activations(layer: int) -> torch.Tensor:
    return torch.load(probes_dir() / "corpus" / f"acts_layer{layer}.pt",
                      weights_only=True).reshape(-1, 512)


def accuracy(arm: torch.nn.Module, acts: torch.Tensor, y: torch.Tensor) -> float:
    hits = total = 0
    for i in range(0, len(acts), BATCH):
        hits += (arm(acts[i:i + BATCH]).argmax(-1) == y[i:i + BATCH].long()).sum().item()
        total += y[i:i + BATCH].numel()
    return hits / total


def fit(arm: torch.nn.Module, a_tr: torch.Tensor, y_tr: torch.Tensor,
        a_va: torch.Tensor, y_va: torch.Tensor, device: str,
        batch: int = BATCH) -> Fit:
    """Train to a validation plateau, leaving `arm` holding the best-validation state.

    `batch` is the corpus's business, not the arm's: the bracket-flip study fits the
    same arms on 6,800 rows, where the corpus default would be one step per epoch.
    """
    a_tr, y_tr = a_tr.to(device), y_tr.to(device)
    a_va, y_va = a_va.to(device), y_va.to(device)
    arm.to(device)
    opt = arm.optimizer()
    snapshot = lambda: {k: v.detach().clone() for k, v in arm.state_dict().items()}

    out = Fit(state=snapshot())
    best, since = -1.0, 0
    for epoch in range(arm.max_epochs):
        perm = torch.randperm(len(a_tr), device=device)
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = arm.loss(arm(a_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            val = accuracy(arm, a_va, y_va)
        out.curve.append({"epoch": epoch, "train_loss": total / len(a_tr), "val_accuracy": val})
        if val > best + MIN_DELTA:
            best, since = val, 0
            out.state, out.best_epoch, out.val_accuracy = snapshot(), epoch, val
        else:
            since += 1
            if since >= PATIENCE:
                break
        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d} loss {total / len(a_tr):.5f} val {val:.5f}"
                  f" best {best:.5f} @{out.best_epoch}", flush=True)

    assert since >= PATIENCE, (
        f"hit max_epochs={arm.max_epochs} without a validation plateau; raise it rather than "
        f"shipping a probe whose training was truncated")
    arm.load_state_dict(out.state)
    print(f"  stopped at epoch {len(out.curve) - 1}, best {out.best_epoch}:"
          f" val {out.val_accuracy:.5f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="board", choices=sorted(ARMS))
    p.add_argument("--layers", default="0-7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    lo, hi = (int(v) for v in args.layers.split("-")) if "-" in args.layers \
        else (int(args.layers), int(args.layers))

    y, games, n_games = corpus()
    tr, va, te = (rows(games, g) for g in game_split(n_games, args.seed))
    print(f"corpus: {n_games} games, {len(y)} rows"
          f" -> train {len(tr)} val {len(va)} test {len(te)}")

    out = probes_dir() / "probes"
    out.mkdir(parents=True, exist_ok=True)
    sweep, curves = [], []
    for layer in range(lo, hi + 1):
        print(f"layer {layer}", flush=True)
        acts = activations(layer)
        arm = ARMS[args.arm](acts.shape[1], args.seed)
        result = fit(arm, acts[tr], y[tr], acts[va], y[va], args.device)
        with torch.no_grad():
            test = accuracy(arm, acts[te].to(args.device), y[te].to(args.device))
        print(f"  test {test:.5f}", flush=True)
        torch.save({**arm.card(), "layer": layer, "seed": args.seed, "classes": CLASSES,
                    "best_epoch": result.best_epoch, "val_accuracy": result.val_accuracy,
                    "test_accuracy": test, "n_games": n_games,
                    "dataset": "adamkarvonen/othello_45MB_games"},
                   out / f"{args.arm}_layer{layer}.pt")
        sweep.append({"layer": layer, "best_epoch": result.best_epoch,
                      "epochs_run": len(result.curve), "val_accuracy": result.val_accuracy,
                      "test_accuracy": test})
        curves += [dict(layer=layer, **c) for c in result.curve]
        del acts, arm

    import pandas as pd
    pd.DataFrame(sweep).to_csv(out / f"{args.arm}_layer_sweep.csv", index=False)
    pd.DataFrame(curves).to_csv(out / f"{args.arm}_training_curves.csv", index=False)
    print(pd.DataFrame(sweep).to_string(index=False))


if __name__ == "__main__":
    main()
