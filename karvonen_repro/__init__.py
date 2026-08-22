"""Run Karvonen's own board-state evaluation on arbitrary SAEs.

Nothing here reimplements a metric. `evaluate.py` calls
`circuits.full_pipeline.analyze_sae_groups` verbatim; `patches.py` is the entire
surface where our checkpoints meet his code.
"""

import sys
from pathlib import Path


def karvonen_root() -> Path:
    for d in Path(__file__).resolve().parents:
        p = d / "external" / "Karvonen_Research"
        if p.is_dir():
            return p
    raise FileNotFoundError(f"no external/Karvonen_Research above {__file__}")


def add_karvonen_to_path() -> None:
    """Anchored on this file, not on the installed `spade` package.

    The editable `spade` install can point at a different checkout than the one this
    harness lives in, and then `import circuits.*` would silently resolve to that
    checkout's vendored fork.
    """
    k = str(karvonen_root())
    if k not in sys.path:
        sys.path.insert(0, k)
