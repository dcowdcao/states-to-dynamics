"""The rules-of-Othello program: do the model's representations encode the game's rules?

Rule set in docs/othello_rules.md. The organizing split is constraint rules, whose
violations exist as counterfactual tokens (R1 placement, R2 bracketing, R3 turn — studied
by injecting an illegal move), versus dynamics rules, whose violations cannot appear in a
game record at all (R4 forfeit, R5 flipping — studied by patching and editing).

  flip_labels.py   R5 labels: how many discs and directions each legal move flips
  (build_pairs, extract_acts, embed, causal/: archived under archive/src/spade/rules/)

This package is a study, not a library. The pieces other studies import — game corpora,
board mechanics, the legality taxonomy, the space API, the checkpoint registry — were
lifted out to spade.othello, spade.spaces and spade.saes.registry precisely so that no
study imports from a sibling study.

Working dir: study_dir("rules"). Figures: archive/analysis/rules_manifolds/, archive/analysis/rules_causal/.
"""
