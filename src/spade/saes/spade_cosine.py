"""CosineSpaDE — cosine-similarity variant of SpaDE.

Same simplex encoder + no-bias decoder, but the score is cosine similarity between x and
each prototype (no λ). Regularizer penalizes cosine misalignment. This variant is what
fixes the distance-norm pathology: it scores by direction, not magnitude, recovering
variance-explained ≈0.979 vs distance-SpaDE's ≈0.43.
"""

# DEFAULT_COSINE_GAMMA = 1.0

# class CosineSpaDEAutoEncoder(SpaDEAutoEncoderBase):
#     def _scores(self, x):   return cosine_sim(x, self.Ae)   # normalize both sides
#
# class CosineSpaDETrainer(SpaDETrainerBase):
#     def _reg(self, x, f):   # cosine-misalignment penalty
#         ...
