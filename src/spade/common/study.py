"""Netscratch working directories for the per-study pipelines.

Five study packages in the source repo each grew their own `study_dir.py` with the same
body; this is that rule, once. A study's heavy intermediates (extracted activations,
branch tensors, probe fits) live outside the repo under a netscratch root, are resumable
(each step skips what already exists), and are on 90-day retention — the repo alone cannot
regenerate a figure.

`study_dir("bracket")` -> $SPADE_BRACKET_DIR, else /n/netscratch/ba_lab/Lab/$USER/spade_bracket
"""

# _NETSCRATCH = Path("/n/netscratch/ba_lab/Lab")
# def study_dir(name: str) -> Path:   # $SPADE_<NAME>_DIR, else _NETSCRATCH/$USER/spade_<name>
#     # mkdir parents; fail loudly if $USER is unset rather than defaulting to a person
