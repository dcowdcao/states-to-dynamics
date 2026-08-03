"""SpaDE (Sparsemax-Distance) SAE in Karvonen's SAETrainer / Dictionary interface.

Encoding: sparsemax(-lambda * ||x - Ae_i||^2)   → sparse weights
Decoding: f @ Ad.T                               → reconstruction (no bias)
Loss:     MSE + gamma_reg * mean_b sum_j ||x - Ad_j||^2 * f_j

Lambda is a trainable inverse temperature via softplus reparameterization. Shared
plumbing lives in spade_base; only the score, regularizer, and lambda are here.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from spade.common.config import DEFAULT_LR, DEFAULT_WARMUP_STEPS
from spade.saes.spade_base import SpaDEAutoEncoderBase, SpaDETrainerBase

# Canonical SpaDE defaults — identical to the production configs in
# spade.training, which reference these constants.
# lambda_init ≈ 1/(2*d_model) = 9.77e-4 for d_model=512 (the paper's theoretical
# inverse temperature), rounded to 1e-3. gamma_reg=1e-4 is the chosen-best point
# (checkpoints/spade_revival), replacing the earlier 1e-2 default.
DEFAULT_LAMBDA_INIT = 1e-3
DEFAULT_SPADE_GAMMA = 1e-4


def softplus_inverse(x, beta=1.0, threshold=20.0):
    """Inverse of torch.nn.functional.softplus."""
    if isinstance(x, float):
        x = t.tensor([x])
    if x * beta < threshold:
        return (1 / beta) * t.log(t.exp(beta * x) - 1.0)
    else:
        return x[0]


class SpaDEAutoEncoder(SpaDEAutoEncoderBase):
    def __init__(self, activation_dim, dict_size, lambda_init=DEFAULT_LAMBDA_INIT, init_data=None):
        super().__init__(activation_dim, dict_size, init_data=init_data)
        self.lambda_pre = nn.Parameter(softplus_inverse(lambda_init))

    @property
    def lambda_val(self):
        return F.softplus(self.lambda_pre)

    def _scores(self, x):
        return -self.lambda_val * t.cdist(x, self.Ae).pow(2)


class SpaDETrainer(SpaDETrainerBase):
    def __init__(
        self,
        steps,
        activation_dim,
        dict_size,
        layer,
        lm_name,
        lambda_init=DEFAULT_LAMBDA_INIT,
        gamma_reg=DEFAULT_SPADE_GAMMA,
        lr=DEFAULT_LR,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        decay_start=None,
        seed=None,
        device=None,
        wandb_name="SpaDETrainer",
        submodule_name=None,
        init_data=None,
    ):
        self.lambda_init = lambda_init
        super().__init__(
            steps=steps, activation_dim=activation_dim, dict_size=dict_size, layer=layer,
            lm_name=lm_name, gamma_reg=gamma_reg, lr=lr, warmup_steps=warmup_steps,
            decay_start=decay_start, seed=seed, device=device, wandb_name=wandb_name,
            submodule_name=submodule_name, init_data=init_data,
        )
        self.logging_parameters = ["lambda_val", "dist_reg_loss"]
        self.dist_reg_loss = -1.0

    _reg_key = "dist_reg_loss"

    def _build_ae(self, activation_dim, dict_size, init_data):
        return SpaDEAutoEncoder(activation_dim, dict_size, lambda_init=self.lambda_init, init_data=init_data)

    def _reg(self, x, f):
        # Distance-weighted L1: penalize feature j by ||x - Ad_j||^2 (Ad is [d, width]).
        dist_penalty = t.cdist(x, self.ae.Ad.T).pow(2)
        return (dist_penalty * f).sum(dim=-1).mean()

    def _score_stats(self, scores):
        # scores = -lambda * ||x - Ae||^2 (input to sparsemax); track magnitude to
        # detect prototype-norm growth that would saturate sparsemax.
        nz = scores.float()
        return {
            "lambda_val": self.ae.lambda_val.item(),
            "neg_lambda_d_sq_mean": nz.mean().item(),
            "neg_lambda_d_sq_max": nz.max().item(),
            "neg_lambda_d_sq_min": nz.min().item(),
            "neg_lambda_d_sq_abs_mean": nz.abs().mean().item(),
        }

    def get_logging_parameters(self):
        return {"lambda_val": self.ae.lambda_val.item(), "dist_reg_loss": self.dist_reg_loss}

    @property
    def config(self):
        return {
            "trainer_class": "SpaDETrainer",
            "dict_class": "SpaDEAutoEncoder",
            **self._base_config(),
            "lambda_init": self.lambda_init,
        }
