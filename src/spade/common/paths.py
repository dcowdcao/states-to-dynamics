"""Repo-root-anchored path resolution + Karvonen sys.path injection.

Walk up from this file to the repo root (marker: CLAUDE.md / .git) so scripts resolve
paths regardless of where they're invoked. With the package installed (`pip install -e .`)
intra-repo imports no longer need sys.path tricks; the ONE remaining injection is putting
external/Karvonen_Research on the path so `import circuits.*` resolves.
"""

# def repo_root() -> Path:              # walk up to CLAUDE.md / .git marker
# def karvonen_root() -> Path:          # repo_root() / "external" / "Karvonen_Research"
# def add_karvonen_to_path() -> None:   # sys.path.insert Karvonen root; raise if missing
# def data_dir() -> Path:               # repo_root() / "data" / "activations"
# def results_dir(experiment: str) -> Path:      # results/<experiment>/
# def checkpoints_dir(experiment: str) -> Path:  # checkpoints/<experiment>/
# def figures_dir() -> Path:            # repo_root() / "figures"
