# Port notes: straightness

Source: `../Spade_Research_David_Sumedh_Ba` at `src/straightness/*.py` (6 files) and
`analysis/straightness/**` (3 `make_figures.py`, 3 `README.md`). Logic is unchanged
everywhere; the deltas below are the rewiring the new layout requires, plus the
questionable things carried over as-is.

## Where things landed

| source | here |
|---|---|
| `src/straightness/{build_games,extract_traj,geometry,compute_geometry,classify}.py` | `src/spade/straightness/` |
| `src/straightness/study_dir.py` | deleted — `spade.common.study.study_dir("straightness")` |
| `analysis/straightness/{,sae_profiles/,feature_pca/}{make_figures.py,README.md}` | unchanged paths |

`spade/straightness/__init__.py` was already written in the skeleton and is accurate;
it is left alone.

## Rewiring applied

- `rules.build_pairs.stream_games` → `spade.othello.games.stream_games`;
  `rules.illegal_taxonomy.classify_all` → `spade.othello.taxonomy.classify_all`;
  `rules.sae_registry.seed42_entries` → `spade.saes.registry.seed42_entries`;
  `evaluation.karvonen_adapter` → `spade.evaluation.karvonen_adapter`;
  `figures.{paper_style,straightness_panels,embedding_panels}` → `spade.figures.*`.
- Every intra-repo `sys.path.insert` deleted (4 in the package, 3 in analysis). No
  external-fork injection is added here: `build_games.py` imports
  `circuits.othello_engine_utils` and relies on `spade.othello.games` having called
  `add_karvonen_to_path()` at import, exactly as the source relied on
  `rules.build_pairs` for the same side effect. The import order in `build_games.py`
  is load-bearing for that reason.
- `straightness_dir()` → `study_dir("straightness")` at each call site. Env var
  `$SPADE_STRAIGHTNESS_DIR` and the on-disk `.../spade_straightness` path are
  unchanged, so the existing netscratch tree is read as-is.
- `EXCLUDED_SAE` is no longer defined in `compute_geometry.py`; it is imported from
  `spade.spaces`, per the 2026-08-03 decision that the exclusion lives in one place.
  The rationale comment moved with it. `FILE_SPACES` stays local.
- `Usage:` lines in the five package modules became `python -m spade.straightness.<mod>`.
  The three analysis scripts keep `python analysis/straightness/.../make_figures.py`.

## Cross-package assumptions the parent must reconcile

1. **`collect_legal` and `first_illegal_pos` stayed in `build_games.py`** with their
   exact names, as instructed. `manifold_capture/build_corpus.py` and
   `violation_divergence/branches.py` import both from here. Note that
   straightness's `collect_legal(games, n, skips)` is *not* the same function as
   `rules/build_pairs.py`'s legal-prefix collector, and the `spade/othello/games.py`
   skeleton sketches a third signature, `collect_legal(n, t_min, t_max)`. If the
   othello agent moved a `collect_legal` into `spade.othello.games`, the two are
   different functions sharing a name — do not let a mechanical import rewrite point
   the two study callers at the wrong one.
2. **`check_sink` kept its name** in `extract_traj.py`; `violation_divergence/extract_acts.py`
   imports it.
3. **`results_dir` signature.** The source has `results_dir()` and writes
   `results_dir() / "straightness"`. The skeleton in `spade/common/paths.py`
   specifies `results_dir(experiment: str) -> Path  # results/<experiment>/`, so
   `classify.py` calls `results_dir("straightness")`. Same path either way; if the
   common agent ported the no-arg version, this one call site needs to change back.
4. **`SAEEntry.ckpt_dir`.** `compute_geometry.sae_encoder` reads `entry.ckpt_dir`,
   which is the source registry's field name. The `spade/saes/registry.py` skeleton
   pseudocode names it `checkpoint_dir`. Kept as `ckpt_dir` because the instruction
   is a faithful port and the source is the authority; if the registry agent renamed
   the field, this is the one reader in this study.
5. `spade.spaces.EXCLUDED_SAE` must exist as a set of config names. The skeleton has
   it commented out.

## Carried over as-is (questionable, not changed)

- `build_games.py`'s docstring says "the four game classes" and then lists five;
  `CLASSES` has five. Source wording, left alone.
- `analysis/straightness/make_figures.py` imports `repo_root` and never uses it.
- `classify.py`'s `boot_ci` returns `tuple(np.percentile(...))`, annotated
  `tuple[float, float]` but actually a tuple of `np.float64`.
- The legal-vs-legal null band `[0.42, 0.58]` is only asserted when the test split
  has ≥400 games, so smoke runs silently skip the study's main guardrail.
- `compute_geometry.py`'s `sae_geometry` divides `mean_lat` by `count` with no guard
  for an empty train split; it would raise a `TypeError` on `None`, not a clear error.

## Deliberate small edits, all documented

- `compute_geometry.py`'s `--spaces` usage example named
  `sae_SpaDETrainer-othello`, which is the excluded checkpoint and would now match
  nothing. Changed to `sae_TopKTrainer-othello`. This is the only content edit to a
  docstring in the port.
- READMEs: paths updated (`src/straightness/*` → `src/spade/straightness/*`,
  `src/figures/*` → `src/spade/figures/*`, `shell/slurm/submit_straightness.sh` →
  `scripts/slurm/studies/straightness/submit_straightness.sh`), pipeline commands
  switched to `python -m spade.straightness.*`, and the SpaDE γ=1e-2 exclusion note
  in `sae_profiles/README.md` now says the decision is global and held in
  `spade.spaces.EXCLUDED_SAE` rather than in this study's `compute_geometry.py`.
  No finding, number or table was rewritten.

## Claims hygiene

`docs/claims_status.md` retires two straightness claims — "SpaDE destroys the
curvature signal" and its 79%-cancellation companion — and marks the layer-5
championship effect as not distinguishable from chance. Nothing ported here restates
either: the surviving prose already says every family preserves the separation
(0.01–0.21), the cancellation table already ranges 1–40%, and both READMEs already
report the L5 championship AUROC as 0.509 [0.483, 0.535] and call it a late-layer
effect. No figure title or docstring needed a correction.

## Not ported

`analysis/straightness/**/figures/*.png` (35 files). They are regenerated outputs and
MIGRATION.md open question 10 (tracked-figure policy) is unresolved, so the decision
is not mine to make silently. The three `make_figures.py` each `mkdir` their own
`figures/` on run.

## Verification

No install performed (`spade_env` is shared). Under `conda activate spade_env`
(CPython 3.11.15):

- `python -m py_compile` clean on all 8 ported `.py` files.
- Import-checked with `PYTHONPATH=$WT/src` and stub modules standing in for the
  packages other agents own (`spade.common.*`, `spade.othello.*`, `spade.spaces`,
  `spade.saes.registry`, `spade.evaluation.karvonen_adapter`, `spade.figures.*`,
  `circuits`): all five package modules and all three analysis scripts import, so
  every module-level name resolves. Without the stubs they fail only on names the
  skeleton has not defined yet.
- `geometry.py` checked numerically against the source module in one process:
  `temporal_geometry`, `restrav_features`, `train_mask` and `RESTRAV_FEATURE_NAMES`
  are bit-identical on a random trajectory batch with a forced zero-length step.
- Not runnable end-to-end here: needs a GPU, the netscratch tree, and the packages
  above.
