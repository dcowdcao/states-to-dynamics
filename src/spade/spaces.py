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

# One provenance decision, currently duplicated in two study modules as
# EXCLUDED_SAE = {"SpaDETrainer-othello"}: the distance-SpaDE gamma=1e-2 checkpoint is
# degenerate (L0 15.4, VE 0.352) and its config.json is a Mac smoke config over real GPU
# weights. Whether it is excluded everywhere or only in some studies is an open question
# in MIGRATION.md; whatever is decided, it is decided here, once.
