# Finish notes — packaging and the environment

Zone: `pyproject.toml`, `requirements.txt`, `Makefile`, `README.md`. The wave-1 verification
MIGRATION.md specifies — `pip install -e .` in a clean env, `import spade`,
`repo_root()` resolving to this tree — had never been run. `spade` was not installed
anywhere, and `make setup` was a block of comments.

## What changed

- **`pyproject.toml`** declared no dependencies (a commented-out `# dependencies = [...]`),
  so `pip install -e .` installed the package with an empty requirement set. It now carries
  `dynamic = ["dependencies"]` reading `requirements.txt`, which keeps one dependency list
  instead of two. `namespaces = true` and its comment about "seven subpackages with no
  `__init__.py`" described the source repo, not this tree — all eighteen package dirs have
  one and `final_validation` asserts it, so discovery is now `namespaces = false`.
- **`requirements.txt`** asked for `python-chess==1.11.2`. The distribution actually
  installed in `spade_env` is `chess==1.11.2`; `python-chess` is the deprecated alias.
  Dropped `plotly` and `umap-learn`: their only importers are
  `circuits/dictionary_learning/interp.py` and `external/tpr_othello/plot_scripts/`, and
  nothing in this repo imports either — the same "pins with no importer" defect the file's
  own header criticises in the source version. Added `pydantic` (`viewer/server.py` imports
  `BaseModel` directly). The header said the source omitted four hard dependencies; it
  omitted five (`nnsight`, `chess`, `jaxtyping`, `networkx`, `requests`). Inline comments
  moved onto their own lines so setuptools can read the file.
- **`Makefile`** — `setup` and `run` are real. `setup` builds `.venv` from `python3.11`
  (`PYTHON` overrides it), installs editable, and resolves both vendored forks so a missing
  `external/` fails at setup rather than mid-run. `run` prepends `.venv/bin` to `PATH` and
  exports `SPADE_REPO`, which `scripts/run_pipeline.sh` requires (`cd "${SPADE_REPO:?}"`).
  `clean` is gone: `rm -rf results/*` would have deleted the tracked eval CSVs.
- **`README.md`** — the setup line claimed `make setup` vendors the external forks; it does
  not, they are in git. Pipeline stages were listed `extract → train`, the script runs
  `train → extract`. Added the FASRC path that reuses `spade_env`'s stack instead of
  re-downloading ~2 GB of torch.

## Verified

Verification venv, deliberately not `spade_env` — an editable install into `spade_env`
would point at this temporary worktree and dangle when it is removed:

```
/n/home10/dcao/.conda/envs/spade_env/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-deps
```

`--system-site-packages` inherits the working torch 2.5.1+cu124 / transformer-lens 1.17.0
stack; `--no-deps` means only the packaging under test is exercised.

- `import spade` works, from inside the repo and from `/tmp`. `spade.common.paths.repo_root()`
  → `/n/home10/dcao/spade_sae_worktrees/env-install`.
- **102 of 102 modules under `src/spade/` import cleanly**, zero failures. The previous
  session's claim was 101; the tree holds 102 `.py` files and every one of them is an
  importable module name. Nothing was broken — the count was off by one, and the earlier
  check ran under an ad-hoc `PYTHONPATH`, not an install.
- The vendored surface imports too: 14 `circuits.*` modules the package uses, and the three
  `tpr_othello` modules `bracket_flip/tpr_probe.py` imports at module level.
- Every pinned version resolves on PyPI (`pip index versions`, all eight `==` pins).
- Package metadata carries all 21 requirements (`Requires-Dist` in the built dist-info), so
  the `dynamic`-from-file read parses.
- Fresh-clone path: stock `/usr/bin/python3.11` (3.11.11) + stock pip 22.3.1,
  `pip install -e . --no-deps`, `import spade` → same repo root. What was *not* run is the
  full `make setup` dependency resolution — that downloads ~2 GB into a home dir with 22 GB
  free, and every pin was checked against the index instead.
- `make test` → `python -m spade.validation.final_validation`: **23/24 passed, 1 skipped,
  0 failed**, matching the previous session's number. The skip is the checkpoint dispatch
  round-trip.

## Data-blocked

Not fixable here, and not faked:

- **`final_validation`'s one skip** — no `checkpoints/*/trainer_*/ae.pt` in this tree.
  There is no `.pt` file anywhere in the worktree; weights are gitignored and are being
  copied into the main tree separately.
- **`spade.saes.registry.entries()`** raises `AssertionError: missing ae.pt in
  checkpoints/main/trainer_0`. Correct behaviour — it verifies what it lists.
- **`analysis/linear_probe/activations/`** is absent, so `rules/causal/unflip_edit.py` and
  the three `bracket_flip` modules that read `linear_probe.pt` cannot run.
- The wave-3 acceptance number (TopK on `checkpoints/main/trainer_3` → L0 80.00, VE 0.943,
  coverage 0.722) still cannot be checked. Import-clean is not run-clean.
  *Run since, and the number quoted here was wrong: the committed reference is L0 79.93,
  VE 0.9867, coverage 0.7265 — see `docs/finish_notes/acceptance-eval.md`.*

## For David

- **`omegaconf` is a latent runtime gap in the vendored TPR fork.**
  `external/tpr_othello/src/load_model.py` imports it inside `load_model()`, and omegaconf
  is not installed in `spade_env`. `spade.bracket_flip.tpr_probe` imports that module but
  never calls `load_model` — it fits the probe itself, because upstream's trainer reads
  Othello pickles we did not vendor. So nothing we run breaks today, and it is not in
  `requirements.txt`. Anyone running upstream's own `train_tpr_probe.py` entrypoint will hit
  it. Add the pin only if that entrypoint becomes ours.
- **`README.md` still opens with "Status: skeleton"**, which PROCESS.md contradicts — the
  port is done. Left alone: rewriting the front matter is not an environment change.
- MIGRATION.md open question 12 (keep `viewer/server.py` at all?) now also decides whether
  `fastapi`, `uvicorn` and `pydantic` stay in `requirements.txt`. They are the only three
  entries no `spade.*` module imports.
