"""Shared base for the SpaDE-family SAEs (distance and cosine variants).

Both variants encode by projecting a per-atom score through sparsemax, decode with
a no-bias linear map, and train with Adam + gradient clipping + Karvonen's LR
schedule under a feature-weighted sparsity regularizer. They differ only in three
places, which live in the subclasses: the score function (`_scores`), the sparsity
regularizer (`_reg`), and — for the distance variant — a trainable inverse
temperature (lambda).

Parameters Ae and Ad are declared here but, via inheritance, stay top-level
attributes on the concrete autoencoder, so their state_dict keys remain "Ae"/"Ad"
(subclasses adding e.g. "lambda_pre" keep it top-level too). Do not move these into
a submodule — that would prefix the keys and break every saved checkpoint.
"""

from collections import namedtuple
from typing import Optional

import torch as t
import torch.nn as nn
from sparsemax import Sparsemax

from spade.common.config import DEFAULT_LR, DEFAULT_WARMUP_STEPS
from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()
from circuits.dictionary_learning.dictionary import Dictionary
from circuits.dictionary_learning.trainers.trainer import SAETrainer, get_lr_schedule

class SpaDEAutoEncoderBase(Dictionary, nn.Module):
    """Sparsemax-over-scores autoencoder with a no-bias decoder.

    Subclasses implement `_scores(x) -> [batch, dict_size]`, where a higher score
    means the atom is more active for x. Encoding is sparsemax over those scores;
    decoding is `f @ Ad.T`.
    """

    def __init__(self, activation_dim: int, dict_size: int, init_data: Optional[t.Tensor] = None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        # Encoder prototypes (rows are atoms); decoder starts as their transpose.
        self.Ae = nn.Parameter(t.randn(dict_size, activation_dim))
        self.Ad = nn.Parameter(t.randn(activation_dim, dict_size))
        with t.no_grad():
            self.Ad.copy_(self.Ae.T)

        # Optional data-sampled init: encoder rows = sampled standardized activations,
        # not norm-rescaled. Prototypes must sit on the data manifold; randn rows
        # (norm ~sqrt(d)) sit far above the standardized activation bulk.
        if init_data is not None:
            assert init_data.shape == (dict_size, activation_dim), (
                f"init_data must be [{dict_size}, {activation_dim}], got {tuple(init_data.shape)}"
            )
            with t.no_grad():
                self.Ae.copy_(init_data.to(self.Ae.dtype))
                self.Ad.copy_(init_data.t().to(self.Ad.dtype))

        self._sparsemax = Sparsemax(dim=-1)

    def _scores(self, x: t.Tensor) -> t.Tensor:
        raise NotImplementedError

    def encode(self, x: t.Tensor) -> t.Tensor:
        f = self._sparsemax(self._scores(x).float())
        return f.to(x.dtype)

    def encode_with_logits(self, x: t.Tensor):
        """encode(), plus the detached pre-sparsemax score tensor for diagnostics."""
        scores = self._scores(x)
        f = self._sparsemax(scores.float())
        return f.to(x.dtype), scores.detach()

    def decode(self, f: t.Tensor) -> t.Tensor:
        return f @ self.Ad.T

    def forward(self, x: t.Tensor, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        return x_hat

    def scale_biases(self, scale: float):
        """No-op: SpaDE-family autoencoders have no biases in the forward path."""
        pass

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs):
        state_dict = t.load(path, map_location="cpu")
        dict_size, activation_dim = state_dict["Ae"].shape
        ae = cls(activation_dim, dict_size)
        ae.load_state_dict(state_dict)
        if device is not None:
            ae.to(device)
        return ae


class SpaDETrainerBase(SAETrainer):
    """Karvonen-compatible trainer: loss = MSE + gamma_reg * `_reg(x, f)`.

    Subclasses build the autoencoder (`_build_ae`), define the feature-weighted
    regularizer (`_reg`), name its log key and matching instance attribute
    (`_reg_key`), and report any pre-sparsemax score stats (`_score_stats`).
    """

    def __init__(
        self,
        *,
        steps: int,
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        gamma_reg: float,
        lr: float,
        warmup_steps: int,
        decay_start: Optional[int],
        seed: Optional[int],
        device: Optional[str],
        wandb_name: str,
        submodule_name: Optional[str],
        init_data: Optional[t.Tensor],
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.steps = steps
        self.lr = lr
        self.gamma_reg = gamma_reg
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.device = device if device is not None else ("cuda" if t.cuda.is_available() else "cpu")

        self.ae = self._build_ae(activation_dim, dict_size, init_data)
        self.ae.to(self.device)
        self.init_from_data = init_data is not None

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def _build_ae(self, activation_dim, dict_size, init_data) -> SpaDEAutoEncoderBase:
        raise NotImplementedError

    def _reg(self, x: t.Tensor, f: t.Tensor) -> t.Tensor:
        raise NotImplementedError

    def _score_stats(self, scores: t.Tensor) -> dict:
        raise NotImplementedError

    def loss(self, x: t.Tensor, step: int = None, logging: bool = False):
        if not logging:
            x_hat, f = self.ae(x, output_features=True)
            recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
            reg = self._reg(x, f)
            setattr(self, self._reg_key, reg.item())
            return recon_loss + self.gamma_reg * reg

        # Logging path: also capture the pre-sparsemax scores for diagnostic stats.
        f, scores = self.ae.encode_with_logits(x)
        x_hat = self.ae.decode(f)
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        reg = self._reg(x, f)
        setattr(self, self._reg_key, reg.item())
        loss = recon_loss + self.gamma_reg * reg
        losses = {
            "mse_loss": recon_loss.item(),
            self._reg_key: reg.item(),
            "sparsity_loss": reg.item(),
            "loss": loss.item(),
            **self._score_stats(scores),
        }
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(x, x_hat, f, losses)

    def update(self, step, activations):
        activations = activations.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def _base_config(self) -> dict:
        return {
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "gamma_reg": self.gamma_reg,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "steps": self.steps,
            "seed": self.seed,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "init_from_data": self.init_from_data,
        }
