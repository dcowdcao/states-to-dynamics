"""Where the data lives: the per-study working directories and the downloaded weights.

Five study packages in the source repo each grew their own `study_dir.py` with the same
body; this is that rule, once. A study's heavy intermediates (extracted activations,
branch tensors, probe fits) live under one data root, are resumable (each step skips what
already exists), and are too big for git — the repo alone cannot regenerate a figure.

`data_root()` -> $SPADE_DATA_ROOT, else <repo>/data
`study_dir("bracket")` -> $SPADE_BRACKET_DIR, else data_root()/spade_bracket
Names in use: rules, bracket, straightness, manifold, divergence, multiseed, legality,
generation, move_legality, probes, rollouts.
"""

import os
from pathlib import Path

from spade.common.paths import repo_root


def data_root() -> Path:
    return Path(os.environ.get("SPADE_DATA_ROOT", repo_root() / "data"))


def study_dir(name: str) -> Path:
    return Path(os.environ.get(f"SPADE_{name.upper()}_DIR", data_root() / f"spade_{name}"))


def variant_dir(name: str, smoke: bool = False, variant: str = "") -> Path:
    """A study tree, its 50-game smoke copy, or a data variant living under either.

    `variant_dir("divergence", smoke=True, variant="rollout")` -> $DIR/smoke/rollout.
    Defaults compose to plain `study_dir(name)`, so an unflagged run reads and
    writes exactly where it always did.
    """
    base = study_dir(name) / "smoke" if smoke else study_dir(name)
    return base / variant if variant else base
