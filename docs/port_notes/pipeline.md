# Port notes — pipeline zone

Covers `spade/{extraction,evaluation,multiseed,validation}`, ported from the frozen source
repo on branch `port/pipeline`.

| Source | Target |
|---|---|
| `src/extraction/extract_sae_training_data.py` | `spade/extraction/extract_activations.py` |
| `src/evaluation/karvonen_adapter.py` | `spade/evaluation/karvonen_adapter.py` |
| `src/evaluation/run_karvonen_eval.py` | `spade/evaluation/run_eval.py` |
| `src/evaluation/merge_revival_results.py` | `spade/evaluation/merge_revival_results.py` |
| `src/multiseed/aggregate_seeds.py` | `spade/multiseed/aggregate_seeds.py` |
| `src/multiseed/aggregate_cosine_sweep.py` | `spade/multiseed/aggregate_cosine_sweep.py` |
| `src/validation/final_validation.py` | `spade/validation/final_validation.py` (rewritten, see below) |

Algorithms are unchanged. Every `sys.path.insert` that put `src/` on the path is gone; the
only remaining injection is `add_karvonen_to_path()` for the vendored `circuits` fork.

## Path repointing

In-repo result paths follow ARCHITECTURE's artifact map:

| Old | New | Where |
|---|---|---|
| `results/karvonen` | `results/main` | `SEED42_CSVS`, `merge_revival_results.SOURCES`, `run_eval --output_dir` default |
| `results/karvonen_batch_topk` | `results/batch_topk` | `SEED42_CSVS` |
| `results/karvonen_cosine_spade` | `results/cosine_spade` | `SEED42_CSVS` |
| `results/karvonen_spade_sweep` | `results/spade_sweep` | `SEED42_CSVS`, `SOURCES` |
| `lambda_sweep_results/karvonen_spade_sweep` | `results/lambda_sweep` | `SEED42_CSVS` |
| `gamma_sweep_results/karvonen_spade_sweep` | `results/gamma_sweep_lam1e-07` | `SEED42_CSVS` |
| `results/karvonen_spade_lam1e-03_gam1e-04` | `results/spade_revival` | `SOURCES`, merge output |

`merge_revival_results` resolved its root with `Path(__file__).resolve().parents[2]`; it now
uses `spade.common.paths.results_dir`. `SOURCES` is built at import time, so importing the
module calls `repo_root()`.

The last two `SEED42_CSVS` entries look archival and are not: they carry the seed-42 point
for five of the six sweep configs. `final_validation` asserts all six entries survive and
that none still name `karvonen`.

**Netscratch paths are deliberately unchanged.** Seeds 43–46 still read
`$SPADE_MULTISEED_DIR/seed_<n>/results/karvonen/all_results.csv`, and the cosine sweep still
globs `karvonen_cosine_spade_sweep*/all_results.csv`. Those are the directory names on disk
today, and whether the netscratch tree gets renamed is MIGRATION open question 7. The
skeleton's `aggregate_seeds` docstring said `results/main` for seeds 43–46 — following it
would silently drop four of five seeds. Repoint both the disk and this file together, or
neither.

`MULTISEED_DIR` is now `study_dir("multiseed")`, which resolves `$SPADE_MULTISEED_DIR` else
`/n/netscratch/ba_lab/Lab/$USER/spade_multiseed` — same env var, same path. The source's
`os.environ.get("USER", "dcao")` hardcoded-user fallback is gone, since `study.py` is
specified to fail loudly on an unset `$USER`.

## Ported as-is, worth knowing

- `karvonen_adapter.compute_f1_per_feature` imports `tqdm` mid-function, and `run_eval.main`
  imports `fcntl` mid-function. Both kept where they were.
- `run_eval` reaches for `trainer_config.get("trainer_class", trainer_config.get("wandb_name",
  "unknown"))` and swallows an `AttributeError` around the `.logits` unwrap. Soft fallbacks
  the house style would normally reject; changing them changes what lands in the CSV.
- `extract_activations` keeps its logic at module level under `if __name__ == '__main__'`
  rather than in a `main()`. `python -m spade.extraction.extract_activations` works either
  way; wrapping 200 lines was not worth the risk. Its `import numpy as np`, the
  `games_batch_to_state_stack_BLRRC` import and the local `actual_batch` are all dead in the
  original and were carried over unchanged.
- `karvonen_adapter.load_standardization_params`'s docstring said "saved by
  train_karvonen_matched.py". That file is split into eight entrypoints in the new layout, so
  the sentence now says "saved by the training entrypoints". Doc text only.
- The SpaDE branches import from `spade.saes.{spade,spade_cosine}`, written against a
  parallel agent's zone that is still a skeleton on this branch.

## final_validation

Rewritten for the new layout and extended. 24 checks, no GPU, no checkpoints, ~10 s.

Added over the source's 17: every `spade` subpackage imports; every package dir has an
`__init__.py`; the dispatch table resolves all 8 trainer classes to the intended branch and
the three ordering constraints hold (BatchTopK before TopK, CosineSpaDE before SpaDE, Gated
before Standard/PAnneal); the `wandb_name` fallback is still there;
`load_standardization_params` finds the file in the *parent* dir (synthetic tmpdir); the
class names `SpaDEAutoEncoder` / `CosineSpaDEAutoEncoder` are unchanged; model constants are
defined only in `spade.common.config` and `DICT_SIZE == EXPANSION * D_MODEL`;
`BASELINE_CONFIGS` is defined exactly once and names the same 8 trainer classes the dispatch
does; `SEED42_CSVS` is repointed and still six entries long.

The dispatch tests read the ordered branch substrings out of `karvonen_adapter.py`'s source
and simulate first-match, so they need neither `circuits` nor weights. `StandardizedDictionary`
is exercised against a small stand-in `Dictionary` so its math is checked even when the SpaDE
classes are unavailable.

Dropped from the source suite, because their targets no longer exist under these names:
`wandb_name` uniqueness and "spade classes present" (both read
`training/train_karvonen_matched.py`, now split into eight entrypoints), the two shell-script
stage checks (`shell/*.sh` -> `scripts/*.sh`, not ported yet, and both were no-ops when the
file was missing), the `figure_karvonen_extension.py` CLI-arg check, and the
`softplus_inverse` inlining check. Re-add equivalents when the training, scripts and figures
zones land.

Two deliberate style deviations: the test wrapper catches `Exception` (a test harness must
report a broken test, not die on it), and `Skip` is raised for checks that need data —
currently only the checkpoint round-trip, which skips because `*.pt` is gitignored and the
weights have not been copied yet.

`main()` prints a "NOT CHECKED" block: no eval reproduces a known number, no GPU kernel or
nnsight trace runs, the F1 / coverage / board-reconstruction math is not checked numerically,
and no training loop is run.

## Verification

No install; `PYTHONPATH=$WT/src` under `spade_env` (CPython 3.11.15).

- `python -m py_compile` over every `.py` in `src/spade`: clean.
- `python -m spade.validation.final_validation` on this branch as it stands: **11 PASS, 13
  FAIL, 0 SKIP.** All 13 failures are `ImportError`/`AttributeError` from
  `spade.common.{paths,config,device,net,study}`, `spade.saes.{spade,spade_cosine}` and
  `spade.figures.karvonen_plot` still being skeleton stubs — other agents' zones. No failure
  is in the pipeline zone.
- Re-run with those five modules bound to the frozen source repo's real implementations:
  **23 PASS, 1 SKIP, 0 FAIL.** The skip is the checkpoint round-trip (no `ae.pt` in the tree).
- All six ported modules import cleanly under the same binding, including
  `extract_activations` (which imports `circuits.othello_utils` at module level).
- Both aggregators run end-to-end on synthetic CSVs — seed dedup, the `_trainer_N` config
  key, mean/std/`n_seeds`, `FIGURE_CONFIGS` ordering, the two-segment cosine glob and the
  baseline-aggregate concat all behave.

Not verified: nothing ran against real data. `results/*/all_results.csv` and
`checkpoints/*/ae.pt` have not been migrated into this tree yet, so the wave-3 acceptance
number (TopK on `checkpoints/main/trainer_3`: L0 80.00, VE 0.943, coverage 0.722) is still
outstanding.
