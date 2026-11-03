"""Repo-root-anchored path resolution + external-fork sys.path injection.

Walk up from this file to the repo root (marker: pyproject.toml) so scripts resolve
paths regardless of where they're invoked. With the package installed (`pip install -e .`)
intra-repo imports no longer need sys.path tricks; the remaining injections put the two
vendored forks on the path so `import circuits.*` and the TPR probe modules resolve.

Editable installs only. A non-editable install puts this file in site-packages, where the
marker walk finds no repo root and fails late — at the first karvonen_root() call, not at
install time. If that ever needs to work, read $SPADE_REPO first and keep the walk as a
fallback; $SPADE_REPO is already the convention in the cluster job scripts.

Study working dirs are NOT here — they live outside the repo, see spade.common.study.
"""

import os
import sys
from pathlib import Path

_MARKER = "pyproject.toml"


def repo_root() -> Path:
    for d in Path(__file__).resolve().parents:
        if (d / _MARKER).exists():
            return d
    raise FileNotFoundError(f"no repo-root marker {_MARKER} above {__file__}")


def karvonen_root() -> Path:
    p = repo_root() / "external" / "Karvonen_Research"
    if not p.is_dir():
        raise FileNotFoundError(f"Karvonen_Research (circuits dependency) not found at {p}")
    return p


def add_karvonen_to_path() -> None:
    """Put Karvonen_Research on sys.path so `import circuits...` resolves."""
    k = str(karvonen_root())
    if k not in sys.path:
        sys.path.insert(0, k)


def tpr_root() -> Path:
    p = repo_root() / "external" / "tpr_othello"
    if not p.is_dir():
        raise FileNotFoundError(f"tpr_othello (TPR probe dependency) not found at {p}")
    return p


def add_tpr_to_path() -> None:
    """Put tpr_othello's src/ and src/hook_utils/ on sys.path, both as roots.

    Exactly the two entries upstream's own entrypoints insert; importing their modules
    without both diverges from the path they were written against.
    """
    src = tpr_root() / "src"
    for p in (str(src), str(src / "hook_utils")):
        if p not in sys.path:
            sys.path.insert(0, p)


def data_dir() -> Path:
    return repo_root() / "data"


def linear_probe_activations() -> Path:
    """The committed board probe and its cached activations.

    Gitignored and 220 MB (MIGRATION.md), so a worktree that does not carry it sets
    $SPADE_LINEAR_PROBE_ACTIVATIONS to a checkout that does.
    """
    return Path(os.environ.get("SPADE_LINEAR_PROBE_ACTIVATIONS",
                               repo_root() / "analysis" / "linear_probe" / "activations"))


def results_dir(experiment: str) -> Path:
    """A paper result: results/figNN_<name>/, or results/shared/ for cross-figure inputs."""
    return repo_root() / "results" / experiment


def build_dir(name: str) -> Path:
    """Gitignored output that is not itself a paper result (scripts/stage_figures.py)."""
    return repo_root() / "build" / name


def checkpoints_dir(experiment: str) -> Path:
    return repo_root() / "checkpoints" / experiment


def figures_dir() -> Path:
    return repo_root() / "figures"
