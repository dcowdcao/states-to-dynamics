# SpaDE SAE Research — Refactor & Cleanup Progress

_Handoff snapshot: 2026-07-08; updated 2026-07-12._

> **Read as a record of July 2026.** It surveys a workspace — `Runs_GPU/`,
> `Original_Spade_Repo/`, `Spade_SAE_Research_Codespace/` — that no longer exists, and no
> path or cleanup proposal below applies here. The λ/γ canonicalization it establishes is
> carried live in [`docs/pipeline.md`](../../docs/pipeline.md) §5. It is kept whole for its
> last section, the only record of the deleted `Run_4_9_weight_init` run (reachable at
> commit `7de4353`) and of the unresolved VE 0.995-vs-0.43 discrepancy.

## UPDATE 2026-07-12 — cleanup EXECUTED, code AUDITED, changes COMMITTED

**Disk: repo ~46 GB → 12 GB (~34 GB reclaimed).**
- `Run_1_Results` pruned 29 GB → 2.2 GB — kept every index-referenced deliverable (FIGURES/, results/, _mini_pipeline/, Linear Probe/figures/ incl. probe_accuracy_heatmap.png, Karvonen saved_data/, top-level .md); deleted only bulk (22 GB activations, embedded .git histories, Run-1 checkpoints, ARENA, activation .pt, dup PDF dir).
- Codespace intermediate `checkpoints/` −2.3 GB (all 24 verified to keep a sibling final `ae.pt`). Stale `logs/` + 3 dev changelogs removed. 42 MB duplicate `Spade_Research_Paper.pdf` de-duped.
- `Original_Spade_Repo` re-cloned shallow: 750 MB → 42 MB (`.git` 728 MB → 20 MB; verified `models.py` byte-identical, 0 ahead/behind origin).
- Repo-wide: 13 `__pycache__` + 104 `.DS_Store` + empty `wandb/` stub swept.

**Code audit (4 read-only agents) → code is UP TO DATE and correct.** Refactor behavior-preserving (17/17), no dangling refs, all 8 trainers wired, shell/SLURM paths+flags resolve, extraction/diagnostics/notebook fresh, `train_karvonen_matched.py`'s uncommitted diff is the coherent 6→8 expansion. One real gap FIXED: `figure_karvonen_extension.py` + `figure_all_configs_combined.py` now cover all 8 SAE types (BatchTopK + CosineSpaDE markers/legend); validation guard updated 6→8; sub-README corrected (180 labels, λ production note, 2nd figure script). Re-verified 17/17.

**Committed — both repos branched off main (merge with `git checkout main && git merge <branch>`):**
- Inner `Spade_SAE_Research_Codespace` → branch **`spade-refactor-cleanup`**: (1) refactor + 8-type figures + dead-code removal + doc sync; (2) stale logs/changelogs removal. `train_karvonen_matched.py` WIP intentionally left uncommitted.
- Outer → branch **`repo-cleanup`**: rename + summary archive + PDF dedup + CLAUDE.md + this doc.

**Decisions — both RESOLVED (2026-07-12):**
1. **λ/γ config hygiene — RESOLVED.** Canonicalized to `DEFAULT_LAMBDA_INIT=1e-3`, `DEFAULT_SPADE_GAMMA=1e-2` (were a stale 9.77e-5/0.2 that also disagreed with the docstring's theoretical 9.77e-4), and `get_trainer_configs` now references the constants → one source of truth. Production values unchanged (verified: SpaDE 1e-3/1e-2, CosineSpaDE 1.0); smoke test 17/17. This commit also landed the pending 8-type training expansion (BatchTopK + CosineSpaDE configs), making the branch coherent with its already-committed 8-type figures/docs. On `spade-refactor-cleanup`.
2. **Narrative-doc staleness — RESOLVED (2026-07-12).** All three narrative docs updated to the current 8-type pipeline and committed: `Spade_Pipeline_Summary.md` (added BatchTopK/CosineSpaDE, split Loss-Recovered vs Variance-Explained, train/test split, production hyperparams, CosineSpaDE result); `Spade_vs_Karvonen_Comparison.md` (rewritten — pipeline now Karvonen-aligned, gaps closed, current 8-type results, stale AUROC/overlay claims removed); `analysis_spade_vs_baselines.md` (CosineSpaDE's var-explained 0.979 overturns the "simplex ceiling is fundamental" thesis → the ~0.43 ceiling is specific to distance scoring under the norm pathology, not the simplex).

---
_Below: original 2026-07-08 handoff (pre-execution survey), kept for reference._

## Scope decisions (already made, do not re-ask)

- **Refactor scope**: "Code + structure, keep data." Code changes confined to `Spade_SAE_Research_Codespace`. Run outputs, checkpoints, and result CSVs are untouched.
- **Governance**: `CLAUDE.md` (house style) applies. Ask before deleting fence-sitter files or changing core logic. Prefer new purpose-named folders. Delete dead code.

---

## ✅ DONE & VERIFIED — Code refactor (`Spade_SAE_Research_Codespace`, inner git repo)

Behavior-preserving. **Verified: 17/17** via
`cd Spade_SAE_Research_Codespace/experiments/expt_othello_sae && python scripts/validation/final_validation.py`.

| File | Change |
|---|---|
| `scripts/training/spade_base.py` *(new, 222 ln)* | Shared `SpaDEAutoEncoderBase` (sparsemax-over-scores, no-bias decode, data-init, `from_pretrained`) + `SpaDETrainerBase` (Adam/clip/LR-schedule `update`, single `loss()` scaffold, `_base_config`). Karvonen `sys.path` bootstrap lives here once. |
| `scripts/training/spade_trainer.py` | 292 → 108 ln. Subclass: distance score, distance-weighted-L1 reg, λ, log keys. |
| `scripts/training/spade_cosine_trainer.py` | 265 → 85 ln. Subclass: cosine score + misalignment reg. |
| `scripts/evaluation/karvonen_adapter.py` | Removed 2 dead functions (`compute_coverage`, `compute_board_reconstruction` — zero callers, superseded by `*_from_assignments`) + dead local `max_act_placeholder`. |
| `scripts/evaluation/run_karvonen_eval.py` | Dropped the now-unused imports of those two. |
| `scripts/validation/final_validation.py` | Fixed stale hardcoded `== 6` wandb-name assertion → proper uniqueness check. |

Two concrete trainers: **557 → 193 ln** combined; duplicated plumbing now exists once.

**Contracts preserved (must stay this way):** module filenames + the 4 class names + `softplus_inverse` importable from `spade_trainer`; `trainer_class` strings exactly `"SpaDETrainer"`/`"CosineSpaDETrainer"` (Cosine dispatched first — substring); state_dict keys stay top-level `Ae`/`Ad`/`lambda_pre` (SpaDE) and `Ae`/`Ad` (Cosine). Verified checkpoints load unchanged.

## ✅ DONE — Structure / docs

- `git mv "Model : Data Selection " Model_Data_Selection` (shell-safe; outer repo).
- `git mv` superseded `Spade_Pipeline_Summary_Prechange.md`, `_Post_Change.md`, `.docx` → `Old_md_files/`. Canonical `Spade_Pipeline_Summary.md` stays at root.
- Codespace `README.md` rewritten to describe the real `expt_othello_sae` pipeline (was the upstream synthetic-paper README).

## Deliberately NOT changed (flagged, awaiting decision)

- **λ default** (`9.77e-5`) kept — a scientific hyperparameter, not a refactor call. Named constant + comment added noting the paper value (`9.77e-4`).
- `requirements.txt` kept (it is current).
- Historical prose line `Old_md_files/run_2_arch_anom.md:21` (old folder name) left as archived record.
- **Sub-README** `expt_othello_sae/README.md` says "6 SAE types" but code trains 8 (BatchTopK + CosineSpaDE) — a real staleness, content edit deferred. **[open: fix?]**

## ⚠️ NOT COMMITTED

Changes span **two repos**: outer (doc/structure moves) and inner `Spade_SAE_Research_Codespace` (code). Nothing committed yet — awaiting go-ahead. Other working-tree items (`Main_Trained_Run_All_SAE/`, root PDF, `train_karvonen_matched.py` M, `.DS_Store`s) are **pre-existing**, not from this work.

---

## 🔄 Stale-file cleanup survey (PROPOSE-ONLY — nothing deleted)

All **6 of 6 zones** surveyed and consolidated below.

### 🏆 Biggest win by far — `Runs_GPU/` is **33G**, and `Run_1_Results/` is **29G** of it
`Run_1_Results/` is a full committed **git snapshot of the entire workspace** (not just results):
- `Run_1_Results/Spade_SAE_Research_Codespace/` — **28G** — embedded clone of the live root (incl. **22G `data/` activations + 3.3G checkpoints**, duplicated from root). **ASK-USER** (confirm activations/checkpoints are reproducible/backed-up before pruning).
- Nested `.git` dirs in the archive: `.../Spade_SAE_Research_Codespace/.git` 748M + `Run_1_Results/.git` 271M + `Karvonen_Research/.git` 58M + AIZI 0.8M — **LIKELY** (no purpose in an output archive).
- Embedded copies: `Other_Research_Materials/ARENA/` 43M, `Linear Probe/` 280M, `Karvonen_Research/` 115M, `Model : Data Selection /` 43M — **ASK-USER**.
- **The one decision that reclaims ~28–29G**: prune `Run_1_Results/` down to its actual deliverables (figures/CSVs/`RUN_1_RESULTS_INDEX.md` + `.md` logs). Everything else in the whole repo combined is <2G.

Also in Runs_GPU: `Main_Trained_Run_All_SAE/results/karvonen_spade_sweep_backup_20260404/` **1.0G** dated backup beside the live sweep — **ASK-USER** (likely near-dup). `Run_2_Results/` 30M superseded by Run_3 — **ASK-USER**. 93× `.DS_Store` + 2 `__pycache__` — **SAFE** (~1.1M).

### `Original_Spade_Repo/` — 750M, but **728M is git-history bloat**
Nested public repo (`github.com/Sai-Sumedh/SaeConceptDuality-SpaDE`). Working tree only 22M; the 728M `.git` pack is bloated by removed `wandb` logs + an 11M notebook still reachable in history (`git gc` won't shrink it). Reclaim ~727M only via history rewrite or **re-clone from the public remote** — **ASK-USER** (also decide whether this vendored public repo belongs in the workspace at all). Plus ~50K SAFE cruft, ~11M ASK-USER regenerable figs/pkls.

### `Karvonen_Research/` — 116M (fork w/ submodule)
SAFE: 3× `__pycache__` + 10× `.DS_Store` (~0.58M). ASK-USER (committed upstream baggage): `autoencoders/saved_data/` 22M (paper CSVs/PNGs, unreferenced), `linear_probes/*.pth` 1.6M, identity-AE example folders 88K. **DO NOT DELETE** (tracked + referenced by tests/core): `autoencoders/testing_{chess,othello}/.../ae.pt` (16M each — test fixtures), `circuits/resources/meta.pkl`. Note: `full_pipeline_walkthrough.ipynb` + the submodule show **uncommitted edits** — real work, don't blow away.

### `Linear Probe` + `Model_Data_Selection` + `Other_Research_Materials` — ~412M
SAFE: 16× `.DS_Store` + 1 `__pycache__` (~150K). LIKELY: **byte-identical `Spade_Research_Paper.pdf` (42M) duplicated** across `Other_Research_Materials/` + `Model_Data_Selection/` — keep one; `SAE_Reference_Table.xlsx - SAE Instances.csv` 20K. ASK-USER (regenerable): `Linear Probe/activations/` 181M (`all_layer_activations.pt` 156M), `figures/*.html` 87M, `ARENA/` 41M (only `part53_othellogpt` used), `linear_probe_pipeline_colab.ipynb` 68K.

### Root docs
SAFE: root `.DS_Store` 16K. LIKELY: 3 archived `Old_md_files/` summary dupes ~72K. ASK-USER: `Old_md_files/SAE_Task_Pipeline (1).md` 500K, `run_2_arch_anom.md` 12K. KEEP: `4-4-2026.md`, `5-10-2026.md`, `fasrc_run_commands.md`, `Karvonen_Pipeline_Summary.md`, `Spade_vs_Karvonen_Comparison.md`, root progress PDF (21M, unique).

### `Spade_SAE_Research_Codespace/` internals — 7.3G (checkpoints untracked/gitignored)
SAFE: 5× `__pycache__` + 9 `.pyc` + 32× `.DS_Store` (~0.5M); `EXPT/wandb/` is an empty stub (16K). LIKELY: old Slurm cluster logs `EXPT/logs/` + embedded `*/logs/` (~42M, some duplicated across sweep folders); dev changelogs `session_changelog_2026-03-26.md`, `scripts/training/CHANGES_*.md`. ASK-USER (DATA): **redundant intermediate `checkpoints/` snapshots ~2.3G** (24 trainer dirs each keep `ae_9155/18310/27465.pt` beside the final `ae.pt`); **superseded exploratory sweeps** `lambda_sweep_results/` 1.0G + `gamma_sweep_results/` 529M (~1.5G — distinct λ/γ regions from the current `8_spade_configs_updated/`, but the older runs that led to it); `W&B Results/` 6.9M + `Wandb_Export.csv`; one-off `best_spade_lam1e-07_gam5e-03/figure_karvonen_best_spade.py` (~65% dup of `scripts/figures/figure_karvonen_extension.py`); `analysis_spade_vs_baselines.md`. **KEEP**: `results/` 1.4G (current eval) + `8_spade_configs_updated/` 2.0G (current sweep) + sub-`README.md`. _(EXPT = `experiments/expt_othello_sae/`.)_

### Consolidated tiers
- **SAFE now** (caches + all `.DS_Store` repo-wide + empty wandb stub): a few MB, zero risk (~200+ `.DS_Store` + scattered `__pycache__`).
- **LIKELY**: dedup the 42M PDF; nested `.git` dirs inside `Run_1_Results` (~1.1G); Codespace old logs ~42M; 3 archived summary dupes + dev changelogs.
- **ASK-USER (the real space)**: `Run_1_Results` embedded clone **~28–29G**; `Original_Spade_Repo/.git` 727M (re-clone); Codespace redundant checkpoint snapshots ~2.3G + older sweeps ~1.5G; regenerable `Linear Probe` artifacts ~268M; Karvonen `saved_data` 22M; Main backup 1.0G; Run_2 30M.

### Cleanup next steps
1. All 6 zones surveyed — consolidation above is complete.
2. Get approval per tier. **The one decision that dominates everything: prune `Run_1_Results/` (~29G).** Second tier: Codespace intermediate `checkpoints/` snapshots (~2.3G) + older sweeps (~1.5G) + `Original_Spade_Repo/.git` (727M, re-clone).
3. Execute only approved deletions. **Nothing deleted yet.**

---

## 📋 Separate open items (from earlier PDF-vs-repo audit)

Results in `Spade Research Progress _ David Cao (5).pdf` NOT in the repo:
- **7/07 TPR research direction** (arxiv 2605.09967) — absent everywhere.
- **Written reasoning for sessions 4/7, 4/18, 4/21, 4/29** — repo has the result numbers (in `geometry_analysis/`) but not the prose/decisions/next-steps.
- **"Two-headed SpaDE" idea** (4/7) — never implemented.
- **Discrepancy**: 4/18 note claims SpaDE "Var explained = 0.995"; repo weight-init sweep CSV shows ≈0.43 for the matching config (precision 0.366 matches exactly). Unresolved.
- **`Run_4_9_weight_init` deleted** from working tree (217 files, only in git commit `7de4353`) — the actual "weight-init" run; may be the source of the 0.995 claim.

---

## Quick reference

- Smoke test (code refactor): `cd Spade_SAE_Research_Codespace/experiments/expt_othello_sae && python scripts/validation/final_validation.py` → expect `17/17 PASSED`.
- CLAUDE.md style guide lives at repo root.
