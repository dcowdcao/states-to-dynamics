"""CosineSpaDE SAE: SpaDE variant scoring atoms by cosine similarity.

Encoding: sparsemax(cos(x, Ae_i))    → sparse weights
Decoding: f @ Ad.T                   → reconstruction (no bias)
Loss:     MSE + gamma_reg * mean_b sum_j (1 - cos(x, Ad_j)) * f_j

No lambda: cosine is scale-invariant, so gamma is the only sparsity knob and atom
norms don't affect encoding. Shared plumbing lives in spade_base.
"""

import torch as t
import torch.nn.functional as F

from spade.common.config import DEFAULT_LR, DEFAULT_WARMUP_STEPS
from spade.saes.spade_base import SpaDEAutoEncoderBase, SpaDETrainerBase

_COS_EPS = 1e-8
# Kept over the gamma sweep's recommended 20: the sweep selected on reconstruction
# F1 (0.547 vs 0.530), coverage runs the other way (0.537 vs 0.474) and decides it.
DEFAULT_COSINE_GAMMA = 1.0


class CosineSpaDEAutoEncoder(SpaDEAutoEncoderBase):
    def _scores(self, x):
        x_n = F.normalize(x, dim=-1, eps=_COS_EPS)
        ae_n = F.normalize(self.Ae, dim=-1, eps=_COS_EPS)
        return x_n @ ae_n.T


class CosineSpaDETrainer(SpaDETrainerBase):
    def __init__(
        self,
        steps,
        activation_dim,
        dict_size,
        layer,
        lm_name,
        gamma_reg=DEFAULT_COSINE_GAMMA,
        lr=DEFAULT_LR,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        decay_start=None,
        seed=None,
        device=None,
        wandb_name="CosineSpaDETrainer",
        submodule_name=None,
        init_data=None,
    ):
        super().__init__(
            steps=steps, activation_dim=activation_dim, dict_size=dict_size, layer=layer,
            lm_name=lm_name, gamma_reg=gamma_reg, lr=lr, warmup_steps=warmup_steps,
            decay_start=decay_start, seed=seed, device=device, wandb_name=wandb_name,
            submodule_name=submodule_name, init_data=init_data,
        )
        self.logging_parameters = ["cos_reg_loss"]
        self.cos_reg_loss = -1.0

    _reg_key = "cos_reg_loss"

    def _build_ae(self, activation_dim, dict_size, init_data):
        return CosineSpaDEAutoEncoder(activation_dim, dict_size, init_data=init_data)

    def _reg(self, x, f):
        # Cosine misalignment: penalize feature j by (1 - cos(x, d_j)); Ad is [d, width],
        # so normalize its columns (dim=0) to get unit decoder directions.
        x_n = F.normalize(x, dim=-1, eps=_COS_EPS)
        ad_n = F.normalize(self.ae.Ad, dim=0, eps=_COS_EPS)
        mis = 1.0 - (x_n @ ad_n)
        return (mis * f).sum(dim=-1).mean()

    def _score_stats(self, scores):
        cs = scores.float()
        return {
            "cos_score_mean": cs.mean().item(),
            "cos_score_max": cs.max().item(),
            "cos_score_min": cs.min().item(),
            "cos_score_abs_mean": cs.abs().mean().item(),
        }

    def get_logging_parameters(self):
        return {"cos_reg_loss": self.cos_reg_loss}

    @property
    def config(self):
        return {
            "trainer_class": "CosineSpaDETrainer",
            "dict_class": "CosineSpaDEAutoEncoder",
            **self._base_config(),
        }
