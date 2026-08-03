"""Repo-root-anchored path resolution + external-fork sys.path injection.

Walk up from this file to the repo root (marker: CLAUDE.md / .git) so scripts resolve
paths regardless of where they're invoked. With the package installed (`pip install -e .`)
intra-repo imports no longer need sys.path tricks; the remaining injections put the two
vendored forks on the path so `import circuits.*` and the TPR probe modules resolve.

Editable installs only. A non-editable install puts this file in site-packages, where the
marker walk finds no repo root and fails late — at the first karvonen_root() call, not at
install time. If that ever needs to work, read $SPADE_REPO first and keep the walk as a
fallback; $SPADE_REPO is already the convention in the cluster job scripts.

Study working dirs are NOT here — they live outside the repo, see spade.common.study.
"""

# def repo_root() -> Path:              # walk up to CLAUDE.md / .git marker
# def karvonen_root() -> Path:          # repo_root() / "external" / "Karvonen_Research"
# def add_karvonen_to_path() -> None:   # sys.path.insert Karvonen root; raise if missing
# def tpr_root() -> Path:               # repo_root() / "external" / "tpr_othello"
# def add_tpr_to_path() -> None:        # tpr_root()/src AND tpr_root()/src/hook_utils, both as roots
# def data_dir() -> Path:               # repo_root() / "data" / "activations"
# def results_dir(experiment: str) -> Path:      # results/<experiment>/
# def checkpoints_dir(experiment: str) -> Path:  # checkpoints/<experiment>/
# def figures_dir() -> Path:            # repo_root() / "figures"
