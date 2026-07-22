# Process log

Running record of processes and critical decisions. Newest first. Keep entries short:
what changed, why, and any decision the reader would otherwise have to reverse-engineer.

---

## 2026-07-22 — New repository architecture (this skeleton)

Rebuilt the project layout from scratch in a sibling worktree (`../spade_sae`, branch
`new-architecture`) because the original repo's top level had grown ~13 parallel
artifact/snapshot directories. The `src/` package was already clean, so the churn is
top-level artifact consolidation + light code renames. Full old→new map in
[ARCHITECTURE.md](ARCHITECTURE.md).

This commit is a **skeleton only**: every `.py` is a header comment + pseudocode. Real
logic is ported from the original repo incrementally, while that repo keeps being edited.

### Critical decisions (asked + answered)

- **Installable package.** `src/spade/` is a real package (`pip install -e .` via
  `pyproject.toml`), replacing the original's `sys.path`-injection hack. `import spade.*`
  resolves everywhere; `make setup` installs it.
- **One training entrypoint per SAE type.** Chosen over the original's single
  parametrized `train_karvonen_matched.py`. Shared loop lives in `training/core.py`; each
  `train_<type>.py` is a thin entrypoint that hands its type's config to `core`. (Note:
  this trades some DRY for per-type clarity — the shared logic must stay in `core.py`, not
  get copied into each entrypoint, or the files will drift.)
- **Analysis code + outputs stay co-located.** Each `analysis/geometry/<study>/` keeps its
  `make_*.py` next to its PNGs + `summary.json`, so a study is self-contained and rerunnable
  in place.
- **Consolidated artifacts.** Four `autoencoders*/` dirs → `checkpoints/<experiment>/`;
  `results/karvonen*/` → `results/<experiment>/`; one deduped `figures/`.

### Deferred

- **Stale-content confirmation.** Which of `8_spade_configs_updated/`,
  `lambda_sweep_results/`, `gamma_sweep_results/`, `W&B Results/`, `wandb/` are truly stale
  vs worth keeping in `archive/` — to be confirmed with David before migrating them.

---

## Carried-over decisions from the original repo (for context)

These predate this restructure but constrain how code must be ported. Sources:
original `docs/Refactor_Cleanup_Progress.md`, `docs/session_logs/`.

- **8 SAE types**: PAnneal, Standard, Gated, TopK (k=80), GatedAnneal, SpaDE, BatchTopK,
  CosineSpaDE. Only the 3 SpaDE-family variants are novel/in-repo; the rest come from the
  vendored `circuits.dictionary_learning` (Karvonen).
- **SpaDE hyperparameters** canonicalized to `DEFAULT_LAMBDA_INIT=1e-3`,
  `DEFAULT_SPADE_GAMMA=1e-2`. Earlier λ=1e-7..1e-8 runs collapse: λ below fp32 sparsemax
  resolution trains bitwise-identical.
- **Load-time dispatch ordering** matters: match `CosineSpaDE` before `SpaDE` and
  `BatchTopK` before `TopK` (substring collisions) when reconstructing a class from its
  `trainer_class` string.
- **Karvonen protocol coupling**: training uses Karvonen's `trainSAE()` loop and
  `NNsightActivationBuffer`; SpaDE subclasses Karvonen's `Dictionary` / `SAETrainer`; eval
  uses the board-reconstruction train/test protocol. `external/Karvonen_Research` is a hard
  runtime dependency.
- **Headline result**: CosineSpaDE reaches variance-explained ≈0.979 vs distance-SpaDE's
  ≈0.43 — the "simplex ceiling" is an artifact of squared-distance scoring under the
  standardized-activation norm pathology, not a limit of the simplex encoder.
