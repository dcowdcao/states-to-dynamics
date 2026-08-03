"""CosineSpaDE — cosine-similarity variant of SpaDE.

Same simplex encoder + no-bias decoder, but the score is cosine similarity between x and
each prototype (no λ). Regularizer penalizes cosine misalignment. This variant is what
fixes the distance-norm pathology: it scores by direction, not magnitude, recovering
variance-explained ≈0.979 vs distance-SpaDE's ≈0.43.
"""

# DEFAULT_COSINE_GAMMA = 1.0
#   Kept over the gamma sweep's recommended 20, which was selected on reconstruction F1
#   (0.547 vs 0.530). Coverage decides it and runs the other way: 0.537 vs 0.474.

# class CosineSpaDEAutoEncoder(SpaDEAutoEncoderBase):
#     def _scores(self, x):   return cosine_sim(x, self.Ae)   # normalize both sides
#
# class CosineSpaDETrainer(SpaDETrainerBase):
#     def _reg(self, x, f):   # cosine-misalignment penalty
#         ...
