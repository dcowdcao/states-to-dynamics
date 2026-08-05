"""Bilinear TPR board probe over the committed OthelloGPT activation set.

Wraps the vendored `TensorProductBoardProbe` at d_r=52, d_f=2, so that
`logit(s, c) = r_s^T M(h) f_c`. The module, the loss and the pinv edit direction
are upstream's, imported unmodified; only the fitting loop is ours, because
their trainer reads Othello pickles we deliberately did not vendor.

Usage: python -m spade.bracket_flip.tpr_probe
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from spade.common.paths import add_tpr_to_path, linear_probe_activations
from spade.common.study import study_dir

add_tpr_to_path()

from hook_utils import seed_all
from intervene_tpr_probe import solve_residual_delta_for_binding_delta
from train_tpr_probe import (
    TensorProductBoardProbe,
    build_square_selection_mask,
    compute_tpr_probe_loss,
)

ROLE_DIM = 52
FILLER_DIM = 2
CLASSES = ("empty", "theirs", "mine")
N_TRAIN = 8000  # 8000/2000 split, as analysis/linear_probe/train_linear_probe.py
SEED = 1111  # upstream's default
LR = 1e-2
BETAS = (0.9, 0.99)
# Upstream's 1e-2 was tuned on ~1.1M positions and overfits our 8000; both retuned
# on games 140-159 of the train split, so the eval 2000 stays unseen
# (analysis/bracket_flip/reports/PROBE_REPORT.md).
WEIGHT_DECAY = 2.0
EPOCHS = 120
BATCH_SIZE = 1024

ACTIVATIONS = linear_probe_activations()


@dataclass(frozen=True)
class TPRProbe:
    module: TensorProductBoardProbe
    layer: int

    @property
    def roles(self) -> Tensor:
        return self.module.role_embeddings.reshape(64, ROLE_DIM)

    @property
    def fillers(self) -> Tensor:
        return self.module.filler_embeddings

    @property
    def binding_map(self) -> Tensor:
        """M[d_r, d_f, d_model]. Upstream stores the same tensor as [d_model, d_r, d_f]."""
        return self.module.binding_map.permute(1, 2, 0)

    def decode(self, h: Tensor) -> Tensor:
        """[..., d_model] -> [..., 64, 3] logits over CLASSES, squares in row-major order."""
        logits = self.module(h.reshape(-1, 1, h.shape[-1]))
        return logits.reshape(*h.shape[:-1], 64, len(CLASSES))

    def edit_direction(self, square: int, from_c: int, to_c: int) -> Tensor:
        """Least-norm residual step realizing dB = r_s (f_to - f_from)^T."""
        delta_binding = torch.outer(
            self.roles[square],
            self.fillers[to_c] - self.fillers[from_c],
        )
        return solve_residual_delta_for_binding_delta(self.module.binding_map, delta_binding)


def board_states() -> Tensor:
    """[n, 8, 8] mover-relative labels in {0: empty, 1: theirs, 2: mine}."""
    return torch.load(ACTIVATIONS / "board_states.pt", map_location="cpu").long()


def layer_activations() -> Tensor:
    """[8, n, d_model] resid_post, layer on axis 0."""
    return torch.load(ACTIVATIONS / "all_layer_activations.pt", map_location="cpu")


def accuracy(probe: TPRProbe, acts: Tensor, labels: Tensor) -> float:
    return (probe.decode(acts).argmax(-1) == labels.reshape(-1, 64)).float().mean().item()


def filler_geometry(probe: TPRProbe) -> dict:
    fillers = probe.fillers
    norms = fillers.norm(dim=-1)
    cosines = (fillers @ fillers.T) / torch.outer(norms, norms)
    return {
        "coords": [[round(v, 4) for v in row] for row in fillers.tolist()],
        "norms": dict(zip(CLASSES, [round(v, 4) for v in norms.tolist()])),
        "angles_deg": {
            f"{CLASSES[i]}-{CLASSES[j]}": round(
                math.degrees(math.acos(cosines[i, j].clamp(-1.0, 1.0).item())), 2)
            for i in range(len(CLASSES))
            for j in range(i + 1, len(CLASSES))
        },
    }


def _fit(acts: Tensor, labels: Tensor, seed: int) -> TensorProductBoardProbe:
    seed_all(seed)
    probe = TensorProductBoardProbe(
        d_model=acts.shape[-1],
        role_dim=ROLE_DIM,
        filler_dim=FILLER_DIM,
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    one_hot = torch.nn.functional.one_hot(labels, len(CLASSES)).float().unsqueeze(1)
    square_mask = build_square_selection_mask(
        exclude_center_squares=False, device=acts.device)

    for _epoch in range(EPOCHS):
        for batch in torch.randperm(acts.shape[0]).split(BATCH_SIZE):
            loss = compute_tpr_probe_loss(
                probe(acts[batch].unsqueeze(1)), one_hot[batch], square_mask=square_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return probe.requires_grad_(False)


def path(layer: int, basis: str = "relative") -> Path:
    return (study_dir("bracket") / "tpr_probes"
            / f"tpr_r{ROLE_DIM}_f{FILLER_DIM}_{basis}_layer{layer}.pt")


def _fingerprint(labels: Tensor) -> str:
    """Identifies the training labels, so a probe fit on one set is never
    mistaken for one fit on another at the same layer and basis."""
    return f"{tuple(labels.shape)}:{int(labels.sum())}"


def train(layer: int, seed: int = SEED, labels: Tensor | None = None,
          basis: str = "relative") -> TPRProbe:
    """Fit at `layer` on the first N_TRAIN rows of the committed activation set.

    `labels` is [n, 8, 8] over CLASSES and defaults to the mover-relative board
    states; pass an absolute-color set with basis="absolute" to refit the same
    architecture in that basis. `basis` keys the cache: without it the two fits
    overwrite each other and load() silently returns the wrong probe.
    """
    acts = layer_activations()[layer]
    if labels is None:
        labels = board_states()
    probe = TPRProbe(_fit(acts[:N_TRAIN], labels[:N_TRAIN], seed), layer)

    out = path(layer, basis)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "probe_state_dict": probe.module.state_dict(),
        "layer": layer,
        "seed": seed,
        "role_dim": ROLE_DIM,
        "filler_dim": FILLER_DIM,
        "basis": basis,
        "labels": _fingerprint(labels[:N_TRAIN]),
        "n_train": N_TRAIN,
    }, out)
    return probe


def load(layer: int, basis: str = "relative") -> TPRProbe:
    artifact = torch.load(path(layer, basis), map_location="cpu", weights_only=True)
    module = TensorProductBoardProbe(
        d_model=artifact["probe_state_dict"]["binding_map"].shape[0],
        role_dim=artifact["role_dim"],
        filler_dim=artifact["filler_dim"],
    )
    module.load_state_dict(artifact["probe_state_dict"])
    return TPRProbe(module.requires_grad_(False), artifact["layer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    acts = layer_activations()
    labels = board_states()
    linear = json.loads((ACTIVATIONS / "probe_metadata.json").read_text())["layer_accuracies"]

    report = {"seed": args.seed, "role_dim": ROLE_DIM, "filler_dim": FILLER_DIM, "layers": {}}
    for layer in [int(piece) for piece in args.layers.split(",")]:
        probe = train(layer, args.seed, labels)
        tpr = accuracy(probe, acts[layer][N_TRAIN:], labels[N_TRAIN:])
        report["layers"][layer] = {
            "tpr": round(tpr, 4),
            "linear": round(linear[layer], 4),
            "gap": round(tpr - linear[layer], 4),
            "filler_geometry": filler_geometry(probe),
        }
        print(f"layer {layer}  tpr {tpr:.4f}  linear {linear[layer]:.4f}  "
              f"gap {tpr - linear[layer]:+.4f}", flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
