"""Self-tests for the 4-way legality probe: no data, no model, no GPU.

Usage: python -m spade.legality_probe.test_probe
"""

import os
import tempfile

import torch

from spade.legality_probe.probe import (
    CLASSES,
    FILLER_DIM,
    TRAIN_GAMES,
    TRAIN_POOL,
    VAL_GAMES,
    fit,
    load,
    macro_f1,
    save,
    split,
)
from spade.probes import tpr


def test_filler_alphabet_widens_only_inside_the_context():
    outside = tpr.TensorProductBoardProbe(d_model=8, role_dim=4, filler_dim=FILLER_DIM)
    assert outside.filler_embeddings.shape[0] == 3, "upstream's default is no longer 3"
    with tpr.filler_classes(len(CLASSES)):
        inside = tpr.TensorProductBoardProbe(
            d_model=8, role_dim=4, filler_dim=FILLER_DIM, use_bias=True)
        assert inside.filler_embeddings.shape == (4, FILLER_DIM), inside.filler_embeddings.shape
        assert inside.bias.shape == (8, 8, 4), inside.bias.shape
    assert tpr.train_tpr_probe.BOARD_LABEL_OPTIONS == 3


def test_the_arm_scales_its_loss_by_its_own_alphabet():
    """Upstream reads the scale from a module global; an arm that leaves it at 3 while
    training 4 classes reports a loss three quarters the size."""
    arm = tpr.TPR(d_model=8, seed=0, n_classes=4, filler_dim=FILLER_DIM)
    torch.manual_seed(0)
    logits = torch.randn(5, 64, 4)
    y = torch.randint(0, 4, (5, 64))
    correct = (logits.log_softmax(-1) * torch.nn.functional.one_hot(y, 4)).sum(-1).mean(0)
    assert torch.allclose(arm.loss(logits, y), -4 * correct.sum())
    assert tpr.train_tpr_probe.BOARD_LABEL_OPTIONS == 3


def test_split_is_by_game_and_never_reaches_the_eval_games():
    positions = 50
    game = torch.arange(TRAIN_GAMES + 40).repeat_interleave(positions)
    tr, va, te = split(game)
    assert len(tr) == TRAIN_POOL * positions and len(va) == VAL_GAMES * positions
    train_games, val_games, test_games = (set(game[i].tolist()) for i in (tr, va, te))
    assert not (train_games & val_games) and not ((train_games | val_games) & test_games)
    assert test_games == set(range(TRAIN_GAMES, TRAIN_GAMES + 40))
    assert split(game, 32)[1].equal(va), "shrinking the training pool moved validation"
    assert set(game[split(game, 32)[0]].tolist()) < train_games, "the pools are not nested"


def test_planted_signal_is_recovered():
    torch.manual_seed(0)
    n_games, positions, d = 60, 40, 64
    patterns = torch.randint(0, len(CLASSES), (32, 64))  # per-archetype 64-square labels
    codes = torch.randn(32, d)
    which = torch.randint(0, 32, (n_games * positions,))
    acts = codes[which] + 0.1 * torch.randn(n_games * positions, d)
    labels = patterns[which]
    game = torch.arange(n_games).repeat_interleave(positions)
    tr, va, te = (torch.nonzero(torch.isin(game, torch.tensor(g))).flatten()
                  for g in (range(40), range(40, 50), range(50, 60)))

    arm, result = fit(acts, labels, tr, va, 1e-2)
    assert result.best_epoch < len(result.curve) - 1, "stopped on its own best epoch"
    assert arm.fillers.shape == (4, FILLER_DIM)
    assert torch.cdist(arm.fillers, arm.fillers).fill_diagonal_(9.9).min() > 0.1, arm.fillers

    scores = macro_f1(arm, acts[te], labels[te])
    assert scores.macro_f1 > 0.95, scores
    assert set(scores.per_class_f1) == set(CLASSES)
    print(f"   planted: macro-F1 {scores.macro_f1:.4f}  acc {scores.accuracy:.4f}  "
          f"stopped at epoch {len(result.curve) - 1}, best {result.best_epoch}")


def test_save_is_keyed_by_space_and_target():
    arms = [tpr.TPR(16, seed, n_classes=len(CLASSES), filler_dim=FILLER_DIM)
            for seed in range(3)]
    keys = [("raw", "legality"), ("TopKTrainer-othello", "legality"), ("raw", "board")]

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SPADE_LEGALITY_DIR"] = tmp
        for arm, (space, target) in zip(arms, keys):
            save(arm, space, target, {"seed": 0, "classes": list(CLASSES)})
        for arm, (space, target) in zip(arms, keys):
            loaded = load(space, target)
            assert torch.equal(loaded.fillers, arm.fillers), (space, target)
            assert not loaded.module.binding_map.requires_grad
    del os.environ["SPADE_LEGALITY_DIR"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
