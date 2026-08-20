"""Self-tests for the 4-way legality probe: no data, no model, no GPU.

Usage: python -m spade.legality_probe.test_probe
"""

import os
import tempfile

import torch

from spade.legality_probe.probe import (
    CLASSES,
    FILLER_DIM,
    LegalityProbe,
    TensorProductBoardProbe,
    build_square_selection_mask,
    compute_tpr_probe_loss,
    fit,
    filler_classes,
    load,
    macro_f1,
    save,
    train_tpr_probe,
)


def test_filler_alphabet_widens_only_inside_the_context():
    outside = TensorProductBoardProbe(d_model=8, role_dim=4, filler_dim=FILLER_DIM)
    assert outside.filler_embeddings.shape[0] == 3, "upstream's default is no longer 3"
    with filler_classes(len(CLASSES)):
        inside = TensorProductBoardProbe(
            d_model=8, role_dim=4, filler_dim=FILLER_DIM, use_bias=True)
        assert inside.filler_embeddings.shape == (4, FILLER_DIM), inside.filler_embeddings.shape
        assert inside.bias.shape == (8, 8, 4), inside.bias.shape
    assert train_tpr_probe.BOARD_LABEL_OPTIONS == 3


def test_loss_scales_with_the_alphabet():
    torch.manual_seed(0)
    logits = torch.randn(5, 1, 8, 8, 4)
    targets = torch.nn.functional.one_hot(torch.randint(0, 4, (5, 8, 8)), 4).float().unsqueeze(1)
    mask = build_square_selection_mask(exclude_center_squares=False, device="cpu")
    correct = (logits.log_softmax(-1) * targets).sum(-1).mean(0)  # [1, 8, 8]

    with filler_classes(4):
        loss = compute_tpr_probe_loss(logits, targets, square_mask=mask)
    assert torch.allclose(loss, -4 * correct.reshape(1, 64).mean(0).sum()), loss
    with filler_classes(3):
        stale = compute_tpr_probe_loss(logits, targets, square_mask=mask)
    assert torch.allclose(stale, loss * 3 / 4), (stale, loss)


def test_planted_signal_is_recovered():
    torch.manual_seed(0)
    n_train, n_eval, d = 1536, 512, 64
    patterns = torch.randint(0, len(CLASSES), (32, 64))  # per-archetype 64-square labels
    codes = torch.randn(32, d)
    which = torch.randint(0, 32, (n_train + n_eval,))
    acts = codes[which] + 0.1 * torch.randn(n_train + n_eval, d)
    labels = patterns[which]

    probe = fit(acts[:n_train], labels[:n_train])
    assert probe.module.filler_embeddings.shape == (4, FILLER_DIM)
    fillers = probe.module.filler_embeddings
    assert torch.cdist(fillers, fillers).fill_diagonal_(9.9).min() > 0.1, fillers

    scores = macro_f1(probe, acts[n_train:], labels[n_train:])
    assert scores.macro_f1 > 0.95, scores
    assert set(scores.per_class_f1) == set(CLASSES)
    print(f"   planted: macro-F1 {scores.macro_f1:.4f}  acc {scores.accuracy:.4f}  "
          f"per-class F1 {[round(v, 3) for v in scores.per_class_f1.values()]}")


def test_save_is_keyed_by_space():
    with filler_classes(len(CLASSES)):
        modules = [TensorProductBoardProbe(d_model=16, role_dim=4, filler_dim=FILLER_DIM)
            for _ in range(2)]
    spaces = ("raw", "TopKTrainer-othello")

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SPADE_LEGALITY_DIR"] = tmp
        for module, space in zip(modules, spaces):
            save(LegalityProbe(module, 5, CLASSES), space)
        for module, space in zip(modules, spaces):
            loaded = load(space, layer=5)
            assert loaded.classes == CLASSES
            assert torch.equal(loaded.module.filler_embeddings, module.filler_embeddings), space
    del os.environ["SPADE_LEGALITY_DIR"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
