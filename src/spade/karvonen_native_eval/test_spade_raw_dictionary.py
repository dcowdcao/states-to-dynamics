"""Invariants the wrapper must hold for Karvonen's eval stages to be reading what they think.

Run against a real checkpoint:
    python -m spade.karvonen_native_eval.test_spade_raw_dictionary \
        --checkpoint_dir $SPADE_CHECKPOINTS/spade_revival/trainer_0
"""

import argparse

import torch as t

from spade.karvonen_native_eval.spade_raw_dictionary import RawSpaDE, load_spade
from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()
from circuits.dictionary_learning.dictionary import Dictionary


def check(ae: RawSpaDE, meta: dict):
    d, width = ae.activation_dim, ae.dict_size
    # Raw activations with the right first two moments: x = mean + eps/scale.
    x = ae.mean + t.randn(64, d) / ae.scale

    assert isinstance(ae, Dictionary)
    assert callable(ae.scale_biases)

    f = ae.encode(x)
    assert f.shape == (64, width), f.shape
    assert (f >= 0).all(), "sparsemax output must be non-negative"
    assert t.allclose(f.sum(-1), t.ones(64), atol=1e-4), f.sum(-1)[:4]

    x3 = x.view(4, 16, d)
    f3 = ae.encode(x3)
    assert f3.shape == (4, 16, width), f3.shape
    assert t.allclose(f3.reshape(64, width), f, atol=1e-6)
    assert ae.decode(f3).shape == x3.shape
    assert ae(x3).shape == x3.shape
    x_hat, f_out = ae(x3, output_features=True)
    assert x_hat.shape == x3.shape and f_out.shape == f3.shape

    # scale_biases(s) must make the module exactly s-homogeneous, code unchanged.
    x_hat_before = ae(x)
    s = 3.7
    ae.scale_biases(s)
    assert t.allclose(ae(s * x), s * x_hat_before, atol=1e-3), \
        (ae(s * x) - s * x_hat_before).abs().max()
    assert t.allclose(ae.encode(s * x), f, atol=1e-6)
    ae.scale_biases(1 / s)
    assert t.allclose(ae(x), x_hat_before, atol=1e-4)

    # The wrapper must be the standardized model conjugated by the affine map, nothing else.
    # ae.spade.encode() is the training-path sparsemax, so this also pins the replacement
    # projection in spade_raw_dictionary to the one the checkpoint was trained under.
    x_std = ae.scale * (x - ae.mean)
    f_train = ae.spade.encode(x_std)
    assert ((f_train > 0) == (f > 0)).all(), "support differs from the training-path sparsemax"
    assert t.allclose(f_train, f, atol=1e-5), (f_train - f).abs().max()
    assert t.allclose(ae(x), ae.spade(x_std) / ae.scale + ae.mean, atol=1e-4)

    print(f"  arch={meta['arch']} dict_size={width} lambda={meta['lambda']} "
          f"smoke_config={meta['config_is_smoke_test']}")
    print(f"  L0 on moment-matched noise = {(f != 0).float().sum(-1).mean():.1f}")
    print("  OK")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_dir", required=True)
    args = p.parse_args()
    ae, meta = load_spade(args.checkpoint_dir)
    print(args.checkpoint_dir)
    check(ae, meta)


if __name__ == "__main__":
    main()
