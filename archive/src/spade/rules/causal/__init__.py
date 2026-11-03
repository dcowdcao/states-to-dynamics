"""The interventional arm: where in the network are rule consequences computed?

Clean/corrupted activation patching plus a targeted edit, over the dynamics rules.

  patch_battery.py  the hook set (resid0 + attn/mlp per layer) and its log-prob readout
  pair_patch.py     R5: patch between two same-prefix branches that differ in one flip
  unflip_edit.py    R5: edit the flip direction out of the residual stream and measure
                    the log-mass moved onto the counterfactual legal set
  mine_forfeits.py  R4: mine minimal pairs where the turn forfeits
  forfeit_patch.py  R4: behavioural + turn-assignment patching

One trap, written down because it has already cost time: when clean and corrupted share a
prefix, a full-residual patch recovers 1.0 *by construction*. That line is a labeled
ceiling on a plot, never a result.
"""
