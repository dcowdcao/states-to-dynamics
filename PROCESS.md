# PROCESS.md

Running log of what's being worked on and the non-obvious decisions behind it, so
humans and future agents have context the code alone doesn't show. Newest first.
Keep entries short; the code is the source of truth for *what*, this is the *why*.

---

## New repository architecture skeleton (2026-07-22)

**Goal:** clean-slate layout for the SpaDE / OthelloGPT SAE project, built in this
sibling worktree (`spade_sae`, orphan branch `new-architecture`) because the original
repo's top level had grown ~13 parallel artifact/snapshot dirs. The original `src/`
was already clean, so the churn is top-level artifact consolidation + light code
renames. Full old→new map in [ARCHITECTURE.md](ARCHITECTURE.md).

This is a **skeleton**: every `.py` is a header comment + pseudocode. Real logic ports
over from `../Spade_Research_David_Sumedh_Ba` incrementally, while that repo keeps
being edited. The two worktrees share one `.git`, so content can be diffed and
cherry-picked between them.

**Decisions (user-confirmed):**
- *Installable package.* `src/spade/` is a real package (`pip install -e .` via
  `pyproject.toml`), replacing the original's sys.path-injection hack; `import spade.*`
  resolves everywhere and `make setup` installs it.
- *One training entrypoint per SAE type.* Shared loop lives in `training/core.py`; each
  `train_<type>.py` hands its type's config to `core`. Keep shared logic in `core`, not
  copied into each entrypoint, or the files drift (this trades some DRY for per-type
  clarity — the tradeoff was chosen deliberately).
- *Analysis code + outputs co-located.* Each `analysis/geometry/<study>/` keeps its
  `make_*.py` next to its PNGs + `summary.json`, so a study is self-contained.
- *Consolidated artifacts.* 4 `autoencoders*/` → `checkpoints/<experiment>/`;
  `results/karvonen*/` → `results/<experiment>/`; one deduped `figures/`.

**Pending:** stale-content confirmation with David before migrating —
`8_spade_configs_updated/`, `lambda_sweep_results/`, `gamma_sweep_results/`,
`W&B Results/`, transient `wandb/`, empty `results/runs/Main_Trained_Run_All_SAE/`,
and the stale `autoencoders/trainer_5/config.json`.

**Status:** skeleton committed (5ac74a5). Nothing pushed to a remote.

---

## Carried-over constraints from the original repo (for context)

These predate this restructure but govern how code must port. Sources: original
`docs/Refactor_Cleanup_Progress.md`, `docs/session_logs/`, `PROCESS.md`.

- **8 SAE types**: PAnneal, Standard, Gated, TopK (k=80), GatedAnneal, SpaDE, BatchTopK,
  CosineSpaDE. Only the 3 SpaDE-family variants are novel/in-repo; the rest come from the
  vendored `circuits.dictionary_learning` (Karvonen).
- **SpaDE hyperparameters** canonicalized to `DEFAULT_LAMBDA_INIT=1e-3`,
  `DEFAULT_SPADE_GAMMA=1e-2`. Earlier λ=1e-7..1e-8 runs collapse (λ below fp32 sparsemax
  resolution trains bitwise-identical).
- **Load-time dispatch ordering**: match `CosineSpaDE` before `SpaDE` and `BatchTopK`
  before `TopK` (substring collisions) when reconstructing a class from `trainer_class`.
- **Karvonen coupling**: training uses Karvonen's `trainSAE()` + `NNsightActivationBuffer`;
  SpaDE subclasses Karvonen's `Dictionary` / `SAETrainer`; eval uses the board-
  reconstruction train/test protocol. `external/Karvonen_Research` is a hard runtime dep.
- **Headline result**: CosineSpaDE reaches variance-explained ≈0.979 vs distance-SpaDE's
  ≈0.43 — the "simplex ceiling" is an artifact of squared-distance scoring under the
  standardized-activation norm pathology, not a limit of the simplex encoder.
- **Active in the original repo** (not yet reflected here — port when they land): the
  rules-of-Othello program (`src/legality/` → `src/rules/` umbrella), the interactive
  feature viewer (`feature_viewer/`), and the CosineSpaDE γ-sweep. Watch that
  `sae_registry.py` stays importable — the feature viewer depends on it.
