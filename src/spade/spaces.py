"""The representation-space API: one way to ask "encode this into space X".

A space is the raw layer-5 residual stream or any trained SAE's latent space, plus the
random-projection and random-TopK controls. Manifold-capture, bracket-flip and the rules
embeddings all need the same four operations, so they live here rather than inside the
rules study where they grew.

Loading a space means loading a checkpoint through spade.evaluation.karvonen_adapter and
wrapping it in StandardizedDictionary — SAEs were trained on scale*(x-mean) activations,
so an unstandardized encode is silently wrong, not merely inaccurate.
"""

# def load_space(name): ...                 # -> encoder callable + metadata
# def representative_spaces(): ...          # the canonical space list used across studies
# def batched_apply(fn, x, batch): ...      # memory-bounded encode over a big tensor
# def center_by_t(x, t): ...                # remove the per-position mean (the t confound)

# EXCLUDED_SAE = {"SpaDETrainer-othello"} — the distance-SpaDE gamma=1e-2 seed-42
# checkpoint. Degenerate (L0 15.4, VE 0.352, where seeds 43-46 of the same config give 128
# and 0.973) with a Mac smoke config written over its real GPU weights. The source repo
# excluded it in the straightness study only, while three other studies and the public
# viewer kept reading it; it is now excluded everywhere, from here, once. The default SpaDE
# is gamma=1e-4 (checkpoints/spade_revival/), which takes its place in the roster.
