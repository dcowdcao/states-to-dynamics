"""SpaDE-family encoders whose simplex projection survives an nnsight trace.

Karvonen's classifier stage calls `ae.encode()` inside `model.trace()`
(circuits/utils.py, `collect_activations_batch`). The `sparsemax` package's projection is a
`torch.autograd.Function` whose forward branches on `input.dim()`; under a trace that is a
proxy and the call dies with `AttributeError: type object '_empty' has no attribute
'__bool__'`. Only SpaDE is affected — his Linear+ReLU encoders trace fine.

The projection below is the same function in plain tensor ops, so the training path keeps
the package and the evaluation path gets a traceable encode. These subclasses add nothing
else: same parameters, same state_dict keys, so `load_sae` instantiates them in place of
the training classes and everything downstream (including the StandardizedSAE wrapper) is
unchanged.
"""

import torch as t

from spade.saes.spade import SpaDEAutoEncoder
from spade.saes.spade_cosine import CosineSpaDEAutoEncoder


def sparsemax(z: t.Tensor, width: int) -> t.Tensor:
    """Euclidean projection of the last dimension onto the simplex.

    `width` is an argument rather than `z.shape[-1]` because a traced tensor's shape is not
    a concrete int.
    """
    z = z - z.max(-1, keepdim=True).values
    zs = z.sort(-1, descending=True).values
    rng = t.arange(1, width + 1).to(zs)
    is_gt = ((1 + rng * zs) > zs.cumsum(-1)).to(zs.dtype)
    k = (is_gt * rng).max(-1, keepdim=True).values
    tau = ((is_gt * zs).sum(-1, keepdim=True) - 1) / k
    return (z - tau).clamp_min(0)


class _TraceSafeEncode:
    def encode(self, x: t.Tensor) -> t.Tensor:
        return sparsemax(self._scores(x).float(), self.dict_size).to(x.dtype)


class TraceSafeSpaDE(_TraceSafeEncode, SpaDEAutoEncoder):
    pass


class TraceSafeCosineSpaDE(_TraceSafeEncode, CosineSpaDEAutoEncoder):
    pass
