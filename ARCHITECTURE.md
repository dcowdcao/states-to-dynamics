# Architecture

Clean-slate layout for the SpaDE / OthelloGPT SAE project. This is a **skeleton**: every
`.py` file holds a header comment + pseudocode describing what it will do, not working
code. Real logic gets ported from the source repo (`../Spade_Research_David_Sumedh_Ba`)
one piece at a time.

**All work happens in this worktree.** The source repo is frozen, read-only, and exists
only to be ported from. See [MIGRATION.md](MIGRATION.md) for the rule and the port order.

## Tree

```
spade_sae/
├── pyproject.toml            installable package `spade` (pip install -e .)
├── Makefile                  make setup / make run
├── README.md                 tiered: run it / dev setup / architecture
├── ARCHITECTURE.md           this file — layout + old→new migration map
├── MIGRATION.md              port order, working rules, what must not break
├── FIGURE_GUIDELINES.md      normative figure rules (no color literals outside paper_style)
├── PROCESS.md                running log of processes + critical decisions
├── CLAUDE.md  LICENSE.txt  requirements.txt  .gitignore
│
├── src/spade/                the code package (one import root: `import spade.*`)
│   ├── common/               config, paths, device, net, study dirs  (leaf; everything imports these)
│   ├── othello/              game/board layer shared by every study
│   │   ├── games.py              stream games from HF, legal-prefix collection
│   │   ├── boards.py             replay, flip_set, regular_moves, rays_touch, tokens_of
│   │   └── taxonomy.py           the 7-category legality taxonomy, mapped to rules R1–R3
│   ├── saes/                 ALL SAE definitions + the checkpoint registry
│   │   ├── spade_base.py         shared base: sparsemax encode + no-bias decode + trainer
│   │   ├── spade.py              distance SpaDE  (score = -λ·‖x-Aeᵢ‖²)
│   │   ├── spade_cosine.py       CosineSpaDE     (score = cosine sim, γ-scaled)
│   │   └── registry.py           the verified checkpoint roster (was rules/sae_registry.py)
│   │   # the 5 Karvonen baselines + BatchTopK are NOT redefined here — they come from
│   │   # external/Karvonen_Research (circuits.dictionary_learning).
│   ├── spaces.py             the representation-space API: load_space, representative_spaces,
│   │                         batched_apply, center_by_t  (was rules/embed.py's shared half)
│   ├── extraction/           stream games → cache layer-5 activations + board labels
│   ├── training/             ONE core loop + one thin entrypoint per SAE type + sweep driver
│   │   ├── core.py               shared: standardization, data-init, trainSAE plumbing
│   │   ├── sweep.py              λ/γ grid drivers (grids span multiple checkpoint trees)
│   │   ├── train_standard.py     train_panneal.py  train_gated.py  train_gated_anneal.py
│   │   ├── train_topk.py         train_batch_topk.py
│   │   └── train_spade.py        train_cosine_spade.py
│   ├── evaluation/           karvonen_adapter.py + run_eval.py + merge_revival_results.py
│   ├── multiseed/            aggregate_seeds.py  aggregate_cosine_sweep.py
│   ├── figures/              paper_style.py (the style contract) + shared panels + entrypoints
│   ├── diagnostics/          SpaDE norm / dead-prototype / per-game MSE studies
│   ├── validation/           final_validation.py fast self-tests
│   │
│   │   ── studies: each owns a netscratch dir, numbered resumable steps, and one
│   │      analysis/<study>/make_figures.py that only draws ──
│   ├── rules/                rules-of-Othello program (R1–R5): pairs, embeddings, causal
│   │   └── causal/               patch battery, un-flip edit, pair patching, forfeits
│   ├── bracket_flip/         TPR read-out of a flanked square (+ external/tpr_othello)
│   ├── violation_divergence/ one prefix branched into legal/illegal futures
│   ├── straightness/         ReStraV trajectory geometry
│   └── manifold_capture/     concept-manifold capture vs tiling
│
├── scripts/                  shell orchestration
│   ├── run_pipeline.sh  smoke_test.sh  clean_smoke_test.sh
│   └── slurm/{main,sweeps,multiseed,studies/<study>,diagnostics}/   FASRC jobs
│
├── analysis/                 downstream studies (code + outputs co-located, one per dir)
│   ├── spade_vs_baselines.md
│   ├── linear_probe/             gold-standard board probe (~0.99 acc)
│   ├── geometry/                 13 self-contained sub-studies (older generation)
│   ├── rules_manifolds/  rules_causal/  bracket_flip/  violation_divergence/
│   ├── straightness/  manifold_capture/  cosine_spade_gamma_sweep/
│   └── <each>: README.md + make_figures.py + figures/ + results/
│
├── viewer/                   the interactive SAE feature viewer (a deployable, not a package)
│   ├── web/                      static HF Space: hand-written JS forward pass + SAE encode
│   └── server.py static/         cluster FastAPI variant
├── docs/                     pipeline, rules spec, study designs, cluster runbook, session_logs/
├── external/                 vendored runtime deps: Karvonen_Research/, tpr_othello/
├── reference/                papers, ARENA exercises, PDFs
│
│   ── generated outputs (weights/tensors gitignored; small CSVs + figures tracked) ──
├── checkpoints/<experiment>/  trained SAE weights
├── data/activations/          extracted .pt + metadata.json
├── results/<experiment>/      eval CSVs/tensors
├── figures/                   canonical cross-study figures only
└── archive/                   superseded snapshots + old runs
```

**Outside the repo, and load-bearing.** Six studies keep their heavy artifacts on netscratch
at `/n/netscratch/ba_lab/Lab/$USER/spade_<study>`, resolved by `spade.common.study.study_dir()`
and overridable per study with `SPADE_<STUDY>_DIR`. That tier holds ~92 GB, including the
seeds 43–46 behind every error bar, and it is on **90-day retention** running from
2026-07-22. The repo alone cannot reproduce the figures.

## Old → new migration map

Source repo: `../Spade_Research_David_Sumedh_Ba` (branch `cosine-spade-gamma-sweep`).

### Code — unchanged from the original plan

| Old | New |
|---|---|
| `src/common/*` | `src/spade/common/*` |
| `src/models/spade_base.py` | `src/spade/saes/spade_base.py` |
| `src/models/spade_trainer.py` | `src/spade/saes/spade.py` |
| `src/models/spade_cosine_trainer.py` | `src/spade/saes/spade_cosine.py` |
| `src/extraction/extract_sae_training_data.py` | `src/spade/extraction/extract_activations.py` |
| `src/training/train_karvonen_matched.py` | split → `training/core.py` + `training/sweep.py` + `train_<type>.py` ×8 |
| `src/evaluation/karvonen_adapter.py` | `src/spade/evaluation/karvonen_adapter.py` |
| `src/evaluation/run_karvonen_eval.py` | `src/spade/evaluation/run_eval.py` |
| `src/multiseed/aggregate_seeds.py` | `src/spade/multiseed/aggregate_seeds.py` |
| `src/figures/karvonen_plot.py` + the 2 Karvonen entrypoints | `src/spade/figures/*` |
| `src/diagnostics/*` | `src/spade/diagnostics/*` (shorter names) |
| `src/validation/final_validation.py` | `src/spade/validation/final_validation.py` |
| `shell/*` | `scripts/*` |

### Code — corrections and additions (the 2026-07-22→08-03 drift)

`src/legality/` no longer exists. It was renamed to `src/rules/` and rewritten:
`compute_metrics.py` was deleted and its work split into `embed.py` (descriptive) and a
`causal/` subpackage (interventional). The plan's `legality/*` row is dropped.

| Old | New | Note |
|---|---|---|
| `src/rules/build_pairs.py` | `src/spade/rules/build_pairs.py` + `spade/othello/games.py` | `stream_games` is imported by 4 other studies — it is not rules-specific |
| `src/rules/illegal_taxonomy.py` | `src/spade/othello/taxonomy.py` | same: shared by rules, straightness, violation_divergence |
| `src/rules/causal/boards.py` | `src/spade/othello/boards.py` | engine helpers, also used by bracket_flip |
| `src/rules/sae_registry.py` | `src/spade/saes/registry.py` | shared by the viewer + 5 studies; must stay importable |
| `src/rules/embed.py` | `src/spade/spaces.py` (shared API) + `spade/rules/embed.py` (the rules PCA/Isomap run) | `load_space`/`representative_spaces`/`batched_apply` are the de-facto SAE-space API for manifold_capture and bracket_flip |
| `src/rules/{extract_acts,flip_labels,test_illegal_taxonomy,study_dir}.py` | `src/spade/rules/*` | `study_dir.py` folds into `spade/common/study.py` |
| `src/rules/causal/{patch_battery,mine_forfeits,forfeit_patch,pair_patch,unflip_edit}.py` | `src/spade/rules/causal/*` | |
| `src/bracket_flip/*.py` (10) | `src/spade/bracket_flip/*` | |
| `src/bracket_flip/*.md` (8 reports) | `analysis/bracket_flip/reports/` | results write-ups do not ship inside the package |
| `src/violation_divergence/*.py` (6) | `src/spade/violation_divergence/*` | |
| `src/violation_divergence/*.md` (2) | `analysis/violation_divergence/reports/` | |
| `src/straightness/*.py` (6) | `src/spade/straightness/*` | |
| `src/manifold_capture/*.py` (13) | `src/spade/manifold_capture/*` | |
| `src/cosine_gamma_range/loss_balance.py` | `src/spade/training/sweep_loss_balance.py` | one module; not worth a package |
| `src/multiseed/aggregate_cosine_sweep.py` | `src/spade/multiseed/aggregate_cosine_sweep.py` | |
| `src/evaluation/merge_revival_results.py` | `src/spade/evaluation/merge_revival_results.py` | |
| `src/figures/paper_style.py` | `src/spade/figures/paper_style.py` | **port before any analysis code** — everything downstream imports it |
| `src/figures/{embedding_panels,straightness_panels}.py` | `src/spade/figures/*` | shared panel libraries |
| `src/figures/figure_cosine_gamma_response.py` | `src/spade/figures/*` | |
| six `*/study_dir.py` | `src/spade/common/study.py` → `study_dir(name)` | one netscratch policy, one place |
| `FIGURE_GUIDELINES.md` | `FIGURE_GUIDELINES.md` | repo root; normative, cited by section number from figure code |
| `feature_viewer/` | `viewer/` | a deployable; `export_weights.py` → `src/spade/export/viewer_weights.py` |
| `feature_viewer/{build_space.py,space_src/,hf_space/}` | **not ported** | the dead Docker path; rescue its exact CPU pins into docs first |
| `external/tpr_othello/` | `external/tpr_othello/` | second hard runtime dep; reached by its own path helper |

### Analysis

Every study built after 2026-07-22 already follows the plan's co-location rule — a
top-level `analysis/<study>/` with a README, a `make_figures.py` that only draws, and its
own `figures/`. That shape wins; `analysis/geometry/` keeps its 13 sub-studies as a
self-contained older generation rather than being the organizing unit.

| Old | New |
|---|---|
| `analysis/analysis_spade_vs_baselines.md` | `analysis/spade_vs_baselines.md` |
| `analysis/geometry/` (13 sub-studies, **not 14**) | unchanged |
| `analysis/geometry/legal_vs_illegal/` | deleted upstream; reborn as `analysis/rules_manifolds/` |
| `analysis/{rules_manifolds,rules_causal,bracket_flip,violation_divergence,straightness,manifold_capture,cosine_spade_gamma_sweep}/` | unchanged |
| `analysis/linear_probe/` | unchanged; `activations/` stays gitignored and must be **copied by hand** |
| `analysis/model_data_selection/` | `reference/scoping/` — planning material, not a study |

### Artifacts

| Old | New | Note |
|---|---|---|
| `autoencoders/` | `checkpoints/main/` | |
| `autoencoders_batch_topk/` | `checkpoints/batch_topk/` | |
| `autoencoders_cosine_spade/` | `checkpoints/cosine_spade/` | |
| `autoencoders_spade_sweep/` (trainer_4, 6 — the two with weights) | `checkpoints/spade_sweep/` | |
| `autoencoders_spade_sweep/` (trainer_0–3 — λ=1e-8/1e-7, config-only, collapsed) | `archive/spade_sweep_collapsed/` | VE ≈ −3e-05, L0 = 8192 |
| `autoencoders_spade_lam1e-03_gam1e-04/` | `checkpoints/spade_revival/` | **was missing from the plan**; the default SpaDE config, required by the registry |
| `results/karvonen/` | `results/main/` | |
| `results/karvonen_{batch_topk,cosine_spade,spade_sweep}/` | `results/{batch_topk,cosine_spade,spade_sweep}/` | |
| `results/karvonen_spade_lam1e-03_gam1e-04/` | `results/spade_revival/` | was missing |
| `results/{manifold_capture,straightness,violation_divergence,multiseed}/` | unchanged | were missing |
| `8_spade_configs_updated/karvonen_spade_sweep/` | `results/spade_sweep_8config/` | **live input**, see below |
| `lambda_sweep_results/karvonen_spade_sweep/` | `results/lambda_sweep/` | **live input** |
| `gamma_sweep_results/karvonen_spade_sweep/` | `results/gamma_sweep_lam1e-07/` | **live input**; name disambiguates it from the CosineSpaDE γ-sweep |
| the rest of those three dirs (configs, stray figures, stray SLURM logs) | `archive/<name>/` | |
| `best_spade_lam1e-07_gam5e-03/` | `archive/` | zero readers; the point is already covered via the γ-sweep CSV |
| `results/legality/` | — | never existed; row dropped |
| `figures/` | `figures/`, keeping the `cosine_spade_sweep/` and `spade_revival/` subdirs | there are 8 distinct `Figure_4_*.png`, no true duplicates — flattening destroys 5 |
| `W&B Results/`, `results/Wandb_Export.csv` | `archive/wandb_charts/` | 2026-03-29 exports, no readers |
| `wandb/` (579 MB), `logs/` | not migrated | transient, gitignored |
| `archive/runs/Run_{2,3}_Results/` | see MIGRATION.md open question | 127 MB, 93 MB already in history |

### Three dirs the old plan wanted to drop, that must be kept

The 2026-07-22 plan queued `8_spade_configs_updated/`, `lambda_sweep_results/` and
`gamma_sweep_results/` as "confirm stale before dropping." They are not stale. Each
supplies a hard-coded live input:

- `lambda_sweep_results/.../all_results.csv` and `gamma_sweep_results/.../all_results.csv`
  are entries 5 and 6 of `SEED42_CSVS` in `multiseed/aggregate_seeds.py`. They carry the
  seed-42 point for five of the six sweep configs. Dropping them silently removes seed 42
  from every sweep config in the headline figure.
- `8_spade_configs_updated/.../all_results.csv` is `OLD_SWEEP_CSV` in
  `figures/figure_all_configs_combined.py` — the only surviving record of the chosen-best
  `SpaDE-lam1e-03-gam1e-04` single-seed point.

Still genuinely pending a staleness call: `W&B Results/`, `wandb/`,
`results/runs/Main_Trained_Run_All_SAE/` (an empty stub), and `best_spade_lam1e-07_gam5e-03/`.

### Scripts

The plan's `slurm/{main,sweeps,multiseed,legality}` covers 11 of 37 live scripts and has no
bucket for six study programs. `legality/` is dropped; `studies/<study>/` and `diagnostics/`
are added.

| Old | New |
|---|---|
| `shell/{run_pipeline,local_smoke_test,clean_smoke_test}.sh` | `scripts/{run_pipeline,smoke_test,clean_smoke_test}.sh` |
| `shell/slurm/{run_all,submit_training,submit_extraction,submit_karvonen_eval}.sh` | `scripts/slurm/main/` |
| `shell/slurm/{run_spade_sweep,run_spade_only,submit_*_only_*,submit_spade_sweep_*}.sh` + `shell/cosine_gamma_range/submit_probe.sh` | `scripts/slurm/sweeps/` |
| `shell/multiseed/*` (6) | `scripts/slurm/multiseed/` |
| `shell/slurm/submit_rules.sh`, `submit_rules_causal.sh` | `scripts/slurm/studies/rules/` |
| `shell/slurm/submit_bracket_*.sh` (5) | `scripts/slurm/studies/bracket_flip/` |
| `shell/slurm/submit_violation_divergence*.sh` (2) | `scripts/slurm/studies/violation_divergence/` |
| `shell/slurm/submit_straightness.sh` | `scripts/slurm/studies/straightness/` |
| `shell/slurm/submit_manifold_capture.sh` | `scripts/slurm/studies/manifold_capture/` |
| `shell/slurm/{submit_perf_game_diag,submit_spade_revival}.sh` | `scripts/slurm/diagnostics/` |
| `shell/slurm/submit_legality.sh` | deleted upstream; became the two `rules` scripts |

Every ported script standardizes on `cd "${SPADE_REPO:?}"` and `python -m spade.<module>`.
Eleven live scripts still `cd $HOME/Spade_SAE_Research`, a path that has not existed since
the 2026-07-13 reorg; they abort at the `cd` under `set -e` while their submitters report
success. Do not copy that line forward.

### Docs

| Old | New |
|---|---|
| `docs/Spade_Pipeline_Summary.md` | `docs/pipeline.md` |
| `docs/Spade_vs_Karvonen_Comparison.md` | `docs/spade_vs_karvonen.md` |
| `docs/Karvonen_Pipeline_Summary.md` | `docs/karvonen_reference.md` |
| `docs/fasrc_run_commands.md` | `docs/fasrc_commands.md` (rewritten as a runbook) + `docs/session_logs/` (the historical half) |
| `docs/othello_rules.md` | unchanged — the R1–R8 spec three studies cite |
| `docs/{bracket_flip_design,violation_divergence_design,ReStraV_Summary,activation_cache_plan,othello_experiment}.md` | unchanged |
| `docs/Refactor_Cleanup_Progress.md` | `archive/old_md/` — the 2026-07 refactor is finished history |

## Invariants a port must not break

Scientific and engineering constraints carried over from the source repo. Sources are in
that repo's `PROCESS.md` unless noted.

**Dispatch and checkpoint format**
- `load_karvonen_sae` matches `BatchTopK` before `TopK` and `CosineSpaDE` before `SpaDE`.
  These are substring collisions; reordering silently loads the wrong class, and
  `load_state_dict` may not even error.
- `Ae` / `Ad` / `lambda_pre` stay top-level attributes on the concrete autoencoder. Moving
  them into a submodule changes every saved `state_dict` key.
- Class names `SpaDEAutoEncoder` / `CosineSpaDEAutoEncoder` are load-bearing: the adapter
  dispatches on `trainer_class` strings and the viewer exporter keys on
  `type(ae).__name__`. Module paths may move; class names may not.
- `load_standardization_params` searches the checkpoint dir *and its parent*. The parent
  fallback is required — `standardization_params.json` sits at the save-dir level, not per
  trainer. Any `checkpoints/` reshuffle must move it with its trainers.

**Hyperparameters and provenance**
- The default SpaDE configuration is **λ=1e-3, γ=1e-4** — the chosen-best point, with
  checkpoints under `checkpoints/spade_revival/`. It replaces the source repo's
  `DEFAULT_SPADE_GAMMA=1e-2`, whose seed-42 checkpoint is degenerate and is now excluded
  everywhere (below). λ ≤ 1e-7 collapses — below fp32 sparsemax resolution those runs train
  bitwise-identically, so there is no point sweeping there.
- The **γ=1e-2 seed-42 checkpoint is excluded from every study and from the viewer**, in
  one place (`spade.spaces`). The source repo excluded it in the straightness study only
  while three other studies and the public Space kept reading it. Two consequences: the
  geometry family's "SpaDE is broken for cell-state mapping" conclusion rests on it and
  must be re-run or retired, and the viewer roster keeps eight entries with the λ=1e-3/γ=1e-4
  checkpoint taking its slot.
- CosineSpaDE's operating point is **γ=20**, not the `DEFAULT_COSINE_GAMMA=1.0` default;
  the γ-sweep across 5.3 decades selects it (L0 98, recon-F₁ 0.547).
- **All six** `autoencoders/trainer_*/config.json` are a Mac smoke-test config
  (`device: mps`, `steps: 400`) written over real GPU weights — not just `trainer_5` as the
  old plan said. Do not trust them to label the checkpoints.
- `data/activations/metadata.json` records a 100-game smoke test, not the real extraction.
- Sweep dirs restart `trainer_N` at 0. Three different dirs contain an
  `autoencoders_spade_sweep/trainer_0` with three different λ/γ grids, and four different
  dirs contain a `karvonen_spade_sweep/all_results.csv`. Merging by index overwrites.

**Figures**
- `figures/paper_style.py` is the only place color literals may appear; `FIGURE_GUIDELINES.md`
  is normative and cited by section number from plotting code. Port both before any figure
  or analysis code, or the port recreates exactly the drift the guidelines exist to prevent.
- Every claim-supporting curve carries uncertainty with its semantics and n stated; the
  reference class is styled as a reference; nulls are gray, dashed, and last.

**Environment**
- CPython 3.11 (`spade_env`), torch 2.5.1+cu124, `transformer_lens==1.17.0` — TL 2.x is
  incompatible with `circuits`. `requirements.txt` as it stands cannot build a working env:
  four pins are wrong, four hard deps (`nnsight`, `python-chess`, `jaxtyping`, `pydantic`)
  are undeclared, and two pins have no importer. Regenerate from the working env rather
  than porting the file.
- `repo_root()` walks up for a `CLAUDE.md` / `.git` marker. This survives an **editable**
  install only; a non-editable install resolves into site-packages and fails late, at the
  first `karvonen_root()` call.
- Seven of fifteen live `src/` subdirs have no `__init__.py` and only import because `src/`
  is on `sys.path`. Under `packages.find` they would be silently omitted from the package.
- `external/Karvonen_Research` is unpinned — no commit, no provenance file, a flattened
  submodule and a dangling `.git` pointer. Three files it needs are invisible to git
  (`linear_probes/*.pth` via the root ignore; `circuits/resources/meta.pkl` and one `.sh`
  via the *vendored* `.gitignore`, which git still honours). A fresh clone yields a broken
  `circuits`. Capture the upstream SHA before re-vendoring anything.
- `external/tpr_othello` is pinned at `cc90cea` and has **no upstream license** — flagged
  all-rights-reserved, not redistributable.

**Data**
- `analysis/linear_probe/activations/linear_probe.pt` is gitignored and is a hard
  dependency of `rules/causal/unflip_edit.py` and three `bracket_flip` modules. It must be
  copied by hand; a `git checkout` alone yields a broken tree. Its sibling
  `board_states.pt` has 53 mislabeled rows in 10,000 (parity from `t%2` rather than replay).
- `*.pt` is gitignored, so a `git mv`-based migration leaves ~1.5 GB of `ae.pt` behind while
  moving only the `config.json` files. Weights move by filesystem copy.
- `results/manifold_capture/` is 121 MB of already-committed `.npz`. Re-committing it under
  a new path doubles it in history.

## What is deliberately not decided here

Open questions that need David's call are collected in [MIGRATION.md](MIGRATION.md) rather
than being resolved silently in this document.
