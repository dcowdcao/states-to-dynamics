"""SpaDE — Sparsemax-Distance Encoder.

Encoder score = -λ · ‖x − Aeᵢ‖²  (negative squared distance to each prototype), fed to
sparsemax → a sparse point on the simplex. λ is trainable via softplus(lambda_pre).
Regularizer is distance-weighted L1. No decode bias.
"""

# DEFAULT_LAMBDA_INIT = 1e-3
# DEFAULT_SPADE_GAMMA = 1e-2

# def softplus_inverse(y):   # init lambda_pre so softplus(lambda_pre) == lambda_init
#     ...

# class SpaDEAutoEncoder(SpaDEAutoEncoderBase):
#     def _scores(self, x):   return -self.lambda_() * cdist(x, self.Ae) ** 2
#
# class SpaDETrainer(SpaDETrainerBase):
#     def _reg(self, x, f):   # distance-weighted L1 over active prototypes
#         ...
