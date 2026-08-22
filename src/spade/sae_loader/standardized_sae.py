"""A Dictionary on raw activations that wraps one trained on standardized activations.

This study's trainers standardize the activation buffer as `act_std = scale * (act - mean)`,
so the inner dictionary is only meaningful on that scale. The wrapper is itself a Karvonen
`Dictionary`, so Karvonen's evaluation code cannot tell the two provenances apart.

Standardization broadcasts over the trailing dimension: a wrapped dictionary treats 2D and
3D inputs exactly as the unwrapped one does, batch semantics included.
"""

import torch
import torch.nn as nn

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()
from circuits.dictionary_learning.dictionary import Dictionary


class StandardizedSAE(Dictionary, nn.Module):
    def __init__(self, dictionary: Dictionary, mean: torch.Tensor, scale: float):
        super().__init__()
        self.dictionary = dictionary
        self.register_buffer("mean", mean)
        self.scale = scale
        self.activation_dim = dictionary.activation_dim
        self.dict_size = dictionary.dict_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary.encode(self.scale * (x - self.mean))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.dictionary.decode(f) / self.scale + self.mean

    def forward(self, x: torch.Tensor, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        return x_hat

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs):
        raise NotImplementedError(
            "standardization params live beside the checkpoint, not in ae.pt; use load_sae()"
        )
