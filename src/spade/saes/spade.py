"""SpaDE — Sparsemax-Distance Encoder.

Encoder score = -λ · ‖x − Aeᵢ‖²  (negative squared distance to each prototype), fed to
sparsemax → a sparse point on the simplex. λ is trainable via softplus(lambda_pre).
Regularizer is distance-weighted L1. No decode bias.

The default configuration is λ=1e-3, γ=1e-4 — the chosen-best point, with checkpoints under
checkpoints/spade_revival/. It replaces the earlier γ=1e-2 default, whose seed-42
checkpoint is degenerate and is now excluded everywhere (see spade.spaces).

λ below ~1e-7 is not a smaller λ, it is no λ: it falls under fp32 sparsemax resolution and
those runs train bitwise-identically. Do not add sweep points there.
"""

# DEFAULT_LAMBDA_INIT = 1e-3
# DEFAULT_SPADE_GAMMA = 1e-4

# def softplus_inverse(y):   # init lambda_pre so softplus(lambda_pre) == lambda_init
#     ...

# class SpaDEAutoEncoder(SpaDEAutoEncoderBase):
#     def _scores(self, x):   return -self.lambda_() * cdist(x, self.Ae) ** 2
#
# class SpaDETrainer(SpaDETrainerBase):
#     def _reg(self, x, f):   # distance-weighted L1 over active prototypes
#         ...
