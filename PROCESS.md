# PROCESS.md

Running log of what's being worked on and the non-obvious decisions behind it, so
humans and future agents have context the code alone doesn't show. Newest first.
Keep entries short; the code is the source of truth for *what*, this is the *why*.

---

## The plan, re-derived against the live repo (2026-08-03)

**Goal:** the 2026-07-22 skeleton was written against the source repo as it stood that
morning, and twelve days of research landed on top of it. Fold all of that drift into the
plan so the port is planned against reality rather than against a two-week-old snapshot.

**Method:** eight parallel read-only surveys of the source repo, one per zone — `src/`,
`analysis/`, top-level artifacts, `shell/` + cluster env, the feature viewer, docs and
constraints, dependencies and packaging, and an audit of the skeleton itself. Findings are
folded into [ARCHITECTURE.md](ARCHITECTURE.md) and [MIGRATION.md](MIGRATION.md).

**What the plan was missing.** Seven code packages (`rules/` and its `causal/` subpackage,
`bracket_flip/`, `violation_divergence/`, `straightness/`, `manifold_capture/`,
`cosine_gamma_range/`), eight analysis studies, a second vendored runtime dependency
(`external/tpr_othello`), the whole figure-style layer (`paper_style.py` +
`FIGURE_GUIDELINES.md`), the feature viewer, 26 of 37 shell scripts, and the netscratch
tier that six studies depend on.

**Three findings that changed the plan rather than extending it:**

- *`rules/` is not a leaf study.* The plan modelled legality as one linear sub-pipeline.
  It is now the layer four other studies import — taxonomy, game streaming, board helpers,
  the checkpoint registry and the representation-space API all live inside it. Left as-is,
  every downstream study would import from a sibling study. The genuinely shared pieces are
  lifted into `spade/othello/`, `spade/saes/registry.py` and `spade/spaces.py`.
- *Three dirs the plan queued for deletion are live inputs.* `8_spade_configs_updated/`,
  `lambda_sweep_results/` and `gamma_sweep_results/` are hard-coded in
  `aggregate_seeds.py` and `figure_all_configs_combined.py`; dropping them removes the
  seed-42 point from every sweep config in the headline figure. The question for them
  changed from *drop?* to *where?*.
- *The per-study convention should be architecture, not repetition.* Five packages
  independently re-derived the same shape: a `study_dir.py` reading `SPADE_<STUDY>_DIR`
  with a netscratch default, numbered resumable steps, and a paired
  `analysis/<study>/make_figures.py` that only draws. That is lifted into
  `spade/common/study.py` and written down as the way a study is built.

**Decisions:**
- *All work happens in this worktree.* The source repo is frozen and read-only, ported
  from and never edited. The cost — no experiment can run here until the pipeline lands in
  wave 3 — is why the port order follows the import graph.
- *Commit at every coherent step, push at every wave boundary.* The skeleton sat
  uncommitted and unpushed for two weeks; that is the failure this rule prevents.
- *`paper_style.py` and `FIGURE_GUIDELINES.md` port in wave 1*, before any figure or
  analysis code, or the port recreates the color-literal drift the guidelines exist to stop.
- *`requirements.txt` gets regenerated, not copied.* The current file cannot build a
  working environment: four pins are wrong, `transformer_lens>=2.0` is incompatible with
  the vendored `circuits`, four hard deps are undeclared, and two pins have no importer.

**Open:** fifteen questions in MIGRATION.md need David's call — the shared-layer split, the
SpaDE γ=1e-2 checkpoint's status across studies, whether `analysis/geometry/` is still
current, netscratch retention, and the tracked-figure policy.

**Status:** plan updated and pushed. No code ported yet.

---

## New repository architecture skeleton (2026-07-22)

**Goal:** clean-slate layout for the SpaDE / OthelloGPT SAE project, built in this
sibling worktree (`spade_sae`, orphan branch `new-architecture`) because the original
repo's top level had grown ~13 parallel artifact/snapshot dirs. The original `src/`
was already clean, so the churn is top-level artifact consolidation + light code
renames.

This is a **skeleton**: every `.py` is a header comment + pseudocode. Real logic ports
over from `../Spade_Research_David_Sumedh_Ba` incrementally. The two worktrees share one
`.git`, so content can be diffed and cherry-picked between them.

**Decisions (user-confirmed):**
- *Installable package.* `src/spade/` is a real package (`pip install -e .` via
  `pyproject.toml`), replacing the original's sys.path-injection hack; `import spade.*`
  resolves everywhere and `make setup` installs it.
- *One training entrypoint per SAE type.* Shared loop lives in `training/core.py`; each
  `train_<type>.py` hands its type's config to `core`. Keep shared logic in `core`, not
  copied into each entrypoint, or the files drift (this trades some DRY for per-type
  clarity — the tradeoff was chosen deliberately).
- *Analysis code + outputs co-located.* Each `analysis/<study>/` keeps its `make_figures.py`
  next to its figures + results, so a study is self-contained.
- *Consolidated artifacts.* 4 `autoencoders*/` → `checkpoints/<experiment>/`;
  `results/karvonen*/` → `results/<experiment>/`; one deduped `figures/`.

**Status:** skeleton committed (5ac74a5). Superseded by the 2026-08-03 entry above, which
keeps all four decisions and corrects the layout they were applied to.

---

## Invariants carried over from the source repo

Moved to [ARCHITECTURE.md](ARCHITECTURE.md) § *Invariants a port must not break* — dispatch
ordering, checkpoint state-dict contracts, hyperparameter canonicalization, the known-stale
configs, the figure-style contract, and the environment pins.
