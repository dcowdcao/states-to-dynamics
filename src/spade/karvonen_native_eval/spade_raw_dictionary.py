"""Present a standardized-trained SpaDE checkpoint as a raw-activation `Dictionary`.

Karvonen's released OthelloGPT SAEs consume raw layer-5 activations, and his eval stages
feed raw activations. SpaDE/CosineSpaDE were trained on x' = scale*(x - mean) (see
`spade.training.core.StandardizedBuffer`). Folding that affine map into the module makes the
checkpoint a well-defined function of raw activations, so both sides are evaluated as maps
on the same input space and the same tensors can be pushed through both.

`mean` is the only bias this module has: x_hat = f @ Ad.T / scale + mean. That is what
`scale_biases` acts on, and it is why the argument "SpaDE has no b_dec" stops being true
once the wrapper is applied — in raw space it has one, and it is not learned.

Shapes are preserved: [..., d_model] in, [..., dict_size] out. Karvonen's classifier stage
encodes [batch, seq_len, d_model] and indexes the result by (batch, seq_len), so a wrapper
that flattens to 2D silently corrupts that stage.
"""

import json
from pathlib import Path

import torch as t
import torch.nn as nn

from spade.common.paths import add_karvonen_to_path
from spade.evaluation.karvonen_adapter import load_standardization_params
from spade.saes.spade import SpaDEAutoEncoder
from spade.saes.spade_cosine import CosineSpaDEAutoEncoder

add_karvonen_to_path()
from circuits.dictionary_learning.dictionary import Dictionary


def sparsemax(z: t.Tensor, width: int) -> t.Tensor:
    """Euclidean projection onto the simplex, in plain tensor ops.

    Same result as the `sparsemax` package's `Sparsemax(dim=-1)` (equal support, float32
    rounding apart), but it is not a `torch.autograd.Function` and branches on nothing.
    That matters because Karvonen's classifier stage calls `encode` inside an nnsight
    trace, where the package's `if input.dim() <= dim` raises on a proxy tensor.
    """
    z = z - z.max(-1, keepdim=True).values
    zs = z.sort(-1, descending=True).values
    rng = t.arange(1, width + 1).to(zs)
    is_gt = ((1 + rng * zs) > zs.cumsum(-1)).to(zs.dtype)
    k = (is_gt * rng).max(-1, keepdim=True).values
    tau = ((is_gt * zs).sum(-1, keepdim=True) - 1) / k
    return (z - tau).clamp_min(0)


class RawSpaDE(Dictionary, nn.Module):
    """A SpaDE-family autoencoder that consumes and reconstructs raw activations."""

    def __init__(self, spade: nn.Module, mean: t.Tensor, scale: float):
        super().__init__()
        self.spade = spade
        self.activation_dim = spade.activation_dim
        self.dict_size = spade.dict_size
        self.register_buffer("mean", mean)
        self.register_buffer("scale", t.as_tensor(float(scale)))

    def encode(self, x: t.Tensor) -> t.Tensor:
        scores = self.spade._scores(self.scale * (x - self.mean))
        return sparsemax(scores.float(), self.dict_size).to(x.dtype)

    def decode(self, f: t.Tensor) -> t.Tensor:
        return self.spade.decode(f) / self.scale + self.mean

    def forward(self, x: t.Tensor, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        return x_hat

    def scale_biases(self, scale: float):
        """Rescale so the module consumes (and returns) `scale`x-larger activations.

        Karvonen calls this to undo the activation normalization he trains under. Rescaling
        `mean` and compensating in `scale` is exact here: reconstruction scales by `scale`
        and the simplex code is unchanged, whereas his linear SAEs' codes scale by `scale`
        too. Both leave his fraction-of-max thresholds invariant.
        """
        self.mean.mul_(scale)
        self.scale.div_(scale)

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "RawSpaDE":
        ae, _ = load_spade(path, device=device)
        return ae


def load_spade(checkpoint_dir, device=None) -> tuple[RawSpaDE, dict]:
    """Load a SpaDE/CosineSpaDE checkpoint dir as a raw-activation Dictionary.

    Architecture and shapes come from the weights, never from config.json: several
    seed-42 checkpoints carry a Mac smoke-test config written over real GPU weights
    (see PROCESS.md), so the recorded hyperparameters are wrong for those runs.

    Returns (ae, meta). `meta` is the saved trainer config with the weight-derived
    truth (`arch`, `dict_size`, `activation_dim`, `lambda`) overwriting it, plus the
    standardization actually applied and `config_is_smoke_test`.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state = t.load(checkpoint_dir / "ae.pt", map_location="cpu", weights_only=True)
    dict_size, activation_dim = state["Ae"].shape

    is_distance = "lambda_pre" in state
    arch = "SpaDE" if is_distance else "CosineSpaDE"
    spade = (SpaDEAutoEncoder if is_distance else CosineSpaDEAutoEncoder)(activation_dim, dict_size)
    spade.load_state_dict(state)

    std = load_standardization_params(str(checkpoint_dir))
    ae = RawSpaDE(spade, t.tensor(std["mean"]), std["scale"])
    ae.to(device).eval()

    config = json.loads((checkpoint_dir / "config.json").read_text())
    trainer = config["trainer"]
    meta = {
        **trainer,
        "arch": arch,
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "lambda": spade.lambda_val.item() if is_distance else None,
        "standardization_scale": std["scale"],
        "standardization_mean_norm": ae.mean.norm().item(),
        "config_is_smoke_test": trainer["steps"] < 1000,
    }
    return ae, meta
