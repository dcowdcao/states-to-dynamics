"""Shared base classes for the SpaDE family.

SpaDE and CosineSpaDE differ ONLY in how they score prototypes against an input and how
they regularize; everything else (sparsemax over scores, no-bias decode, Adam + grad-clip +
Karvonen LR schedule, loss = MSE + gamma_reg * reg) is shared here. Subclasses override the
three hooks: `_scores`, `_reg`, `_score_stats`.

Both subclass Karvonen's `Dictionary` and `SAETrainer` so they plug straight into
`trainSAE()`. Contracts that must not drift (loaders + state_dicts depend on them):
  - state_dict keys `Ae` (encode prototypes), `Ad` (decode atoms), `lambda_pre`
  - encode returns a simplex vector via sparsemax
"""

# from circuits.dictionary_learning ... import Dictionary, SAETrainer, get_lr_schedule

# class SpaDEAutoEncoderBase(Dictionary, nn.Module):
#     # Ae, Ad parameters; encode(x) = sparsemax(self._scores(x)); decode(f) = f @ Ad (no bias)
#     def _scores(self, x): raise NotImplementedError
#
# class SpaDETrainerBase(SAETrainer):
#     # optimizer/schedule/grad-clip; loss = mse(x, x_hat) + gamma_reg * self._reg(...)
#     def _reg(self, ...): raise NotImplementedError
#     def _score_stats(self, ...): raise NotImplementedError   # for logging
