"""Netscratch working directories for the per-study pipelines.

Five study packages in the source repo each grew their own `study_dir.py` with the same
body; this is that rule, once. A study's heavy intermediates (extracted activations,
branch tensors, probe fits) live outside the repo under a netscratch root, are resumable
(each step skips what already exists), and are on 90-day retention — the repo alone cannot
regenerate a figure.

`study_dir("bracket")` -> $SPADE_BRACKET_DIR, else /n/netscratch/ba_lab/Lab/$USER/spade_bracket
Names in use: rules, bracket, straightness, manifold, divergence, multiseed, legality.

$USER unset is a KeyError, not a fallback to a hardcoded username: the source repo's
multiseed and registry copies defaulted to `dcao`, which silently points another user at
someone else's netscratch tree.
"""

import os
from pathlib import Path

_NETSCRATCH = Path("/n/netscratch/ba_lab/Lab")


def study_dir(name: str) -> Path:
    default = _NETSCRATCH / os.environ["USER"] / f"spade_{name}"
    return Path(os.environ.get(f"SPADE_{name.upper()}_DIR", default))


def variant_dir(name: str, smoke: bool = False, variant: str = "") -> Path:
    """A study tree, its 50-game smoke copy, or a data variant living under either.

    `variant_dir("divergence", smoke=True, variant="rollout")` -> $DIR/smoke/rollout.
    Defaults compose to plain `study_dir(name)`, so an unflagged run reads and
    writes exactly where it always did.
    """
    base = study_dir(name) / "smoke" if smoke else study_dir(name)
    return base / variant if variant else base
