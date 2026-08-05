"""Validation: the legality labels are in the frame of the player TO MOVE.

`classify` and `get_valid_moves` both read `board.next_hand_color`, so an engine
cross-check is circular and cannot tell the two frames apart. Two things can.

The move the game record actually plays at ply t is legal for the true mover and, for
the other player, is almost always R3 — an oracle from outside the circle, and the
check that settles the question. A probe is the weaker but more diagnostic test: fit
on the wrong frame it still decodes R1 and R2, which barely depend on whose turn it
is, and loses the `legal` class. The probe here is a plain per-square linear
classifier, not the study's TPR probe; it exists to compare two label sets, not to
measure a space.

Usage: python -m spade.legality_probe.frame_check
"""

import copy
import json

import numpy as np
import torch

from spade.common.config import D_MODEL
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe.fit_set import N_TRAIN
from spade.legality_probe.labels import LABELS, board_labels
from spade.othello.taxonomy import classify_all

from circuits.othello_engine_utils import OthelloBoardState, to_string

EPOCHS, BATCH_SIZE, LR, SEED = 100, 1024, 1e-2, 1111


def both_frames(game: list[int], t: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Labels at ply t for the mover and for the player who just moved, and the square
    the record plays next."""
    board = OthelloBoardState()
    board.update(to_string(game[:t]))
    opponent = copy.copy(board)
    opponent.next_hand_color = -board.next_hand_color
    return (board_labels(board, classify_all(board)),
            board_labels(opponent, classify_all(opponent)),
            to_string(game[t]))


def share(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels, minlength=len(LABELS))
    return {name: round(float(counts[i] / counts.sum()), 4) for i, name in enumerate(LABELS)}


def fit(acts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """64 independent 4-way linear classifiers on d_model, as [d_model, 64, 4]."""
    torch.manual_seed(SEED)
    weight = (torch.randn(D_MODEL, 64, len(LABELS)) * 0.01).requires_grad_(True)
    optimizer = torch.optim.Adam([weight], lr=LR)
    for _epoch in range(EPOCHS):
        for batch in torch.randperm(acts.shape[0]).split(BATCH_SIZE):
            logits = torch.einsum("nd,dsc->nsc", acts[batch], weight)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, len(LABELS)), labels[batch].reshape(-1).long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return weight.detach()


def score(acts: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    weight = fit(acts[:N_TRAIN], labels[:N_TRAIN])
    predicted = torch.einsum("nd,dsc->nsc", acts[N_TRAIN:], weight).argmax(-1).numpy().ravel()
    true = labels[N_TRAIN:].numpy().ravel()
    out = {}
    for i, name in enumerate(LABELS):
        tp = int(((predicted == i) & (true == i)).sum())
        out[name] = round(2 * tp / int((predicted == i).sum() + (true == i).sum()), 4)
    out["macro_f1"] = round(float(np.mean([out[name] for name in LABELS])), 4)
    out["accuracy"] = round(float((predicted == true).mean()), 4)
    return out


def main() -> None:
    fit_dir = study_dir("legality") / "fit_set"
    tokens = torch.load(fit_dir / "tokens.pt", weights_only=True)
    saved = torch.load(fit_dir / "labels.pt", weights_only=True)
    position = torch.load(fit_dir / "position.pt", weights_only=True)
    game_id = torch.load(fit_dir / "game_id.pt", weights_only=True)

    mover = np.empty((len(position), 64), dtype=np.int8)
    opponent = np.empty_like(mover)
    played = np.empty(len(position), dtype=np.int64)
    for row in range(len(position)):
        game = tokens[int(game_id[row])].tolist()
        mover[row], opponent[row], played[row] = both_frames(game, int(position[row]) + 1)
    assert np.array_equal(mover, saved.numpy()), "fit_set labels are not the mover-frame labels"

    rows = np.arange(len(played))
    assert (mover[rows, played] == 0).all(), \
        "a move the game record plays is not legal in the mover frame: the frame is wrong"

    acts = torch.load(study_dir("legality") / "acts" / "resid_post_5.pt", weights_only=True)
    report = {
        "label_of_the_played_move": {"mover": share(mover[rows, played]),
                                     "opponent": share(opponent[rows, played])},
        "fraction_relabelled_by_frame": {
            **{name: round(float((mover[mover == i] != opponent[mover == i]).mean()), 4)
               for i, name in enumerate(LABELS)},
            "all": round(float((mover != opponent).mean()), 4),
        },
        "linear_probe": {"mover": score(acts, saved),
                         "opponent": score(acts, torch.from_numpy(opponent))},
    }
    out = repo_root() / "analysis" / "legality_probe" / "results" / "frame_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
