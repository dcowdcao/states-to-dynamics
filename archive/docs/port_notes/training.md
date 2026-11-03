# Port notes — `spade.training`

Source: `src/training/train_karvonen_matched.py` (657 lines) and
`src/cosine_gamma_range/loss_balance.py` in the frozen repo.

## Where each piece went

| Source | New |
|---|---|
| `main()`, `StandardizedBuffer`, `sample_init_activations` | `training/core.py` |
| `get_trainer_configs()`, one dict per SAE type | `training/train_<type>.py` ×8, one `config(p)` each |
| `--spade_sweep --cosine_spade_sweep --spade_lambdas --spade_gammas --cosine_spade_gammas --spade_pairs` and the `spade_cfg` closure | `training/sweep.py` |
| `cosine_gamma_range/loss_balance.py` | `training/sweep_loss_balance.py` |

`core.run_training(build_configs, save_dir, ...)` takes a callable, not a config dict: the
configs cannot be built until `run_training` knows `steps` (a function of `--smoke_test`)
and the buffer has been calibrated, so it passes a frozen `TrainParams` down to the
builder and gets the list back. `core.run_cli(config, __doc__)` is a per-type entrypoint's
entire `main()`; `sweep.py` adds its grid flags to `core.arg_parser` and calls
`run_training` directly.

## The config contract, checked

Every `wandb_name` and every trainer class comes out byte-identical. Verified by a
throwaway script (since deleted) that imported the source's `get_trainer_configs`, pulled
the source's `spade_cfg` closure and the CosineSpaDE sweep dict literal out of `main()`
with `ast`, and diffed them against the ported builders at both `steps=36621` and
`steps=100`. All 8 baselines and 14 sweep points matched key-for-key and value-for-value,
classes compared by name. The one difference is deliberate and expected:
`SpaDETrainer-othello`'s `gamma_reg` is now `DEFAULT_SPADE_GAMMA = 1e-4`, not `1e-2`.

The eight names, and the `BASELINES` order in `sweep.py` that fixes `trainer_N`:

    0 PAnnealTrainer-othello   1 StandardTrainer-othello   2 GatedSAETrainer-othello
    3 TopKTrainer-othello      4 GatedAnnealTrainer-othello 5 SpaDETrainer-othello
    6 BatchTopKTrainer-othello 7 CosineSpaDETrainer-othello

That order is why `checkpoints/main/trainer_3` is TopK. Sweep names keep their exact
formats: `SpaDE-lam{lam:.0e}-gam{gam:.0e}` and `CosineSpaDE-gam{gam:g}` — reproducing
`SpaDE-lam8e-08-gam1e-03` and `CosineSpaDE-gam1` / `CosineSpaDE-gam0.05` as the eval CSVs
have them.

## Decisions

**Off-default λ/γ only goes through `sweep.py`.** The skeleton's docstrings sketched
`train_spade.py --lambda_init --gamma`, but the source only ever produced non-default
points through the sweep, under the `SpaDE-lam..-gam..` name. Giving the entrypoint those
flags would let a non-default run be written out as `SpaDETrainer-othello` and collide
with the canonical config under `aggregate_seeds.config_key`. So the two SpaDE entrypoints
train exactly their canonical config, and a one-point grid does the rest — which is
already how `shell/slurm/submit_spade_revival.sh` trained λ=1e-3/γ=1e-4.

**`--trainer_filter` is gone**, superseded by the per-type entrypoints (that is the
PROCESS.md decision). The three `*_only_train.sh` scripts map one-to-one onto
`train_batch_topk` / `train_cosine_spade` / `train_spade`.

**Training all eight into one directory is `python -m spade.training.sweep` with no grid
flag** (optionally with `--spade_pairs`), which is what `submit_training.sh` and
`shell/multiseed/submit_train.sh` did. Running the eight entrypoints separately instead
gives eight directories each holding `trainer_0` — a different on-disk layout from
`checkpoints/main`. Whoever ports `scripts/slurm/` should keep the sweep entrypoint for
those two jobs.

**Refuse-if-exists guard added** (`sweep.refuse_if_populated`): a sweep aborts if
`--save_dir` already holds `trainer_*` dirs, because indices restart at 0 and a rerun with
a different grid overwrites checkpoints while their eval rows survive — the silent
overwrite that already happened three times in the source repo. It is in `sweep.py` only;
the per-type entrypoints keep the source's overwrite-in-place behavior, since each writes
one trainer under its own experiment dir. No `--force` flag: move or delete the directory.
`submit_spade_revival.sh`'s own `if [ ! -f .../ae.pt ]` guard is compatible with it.

**`P_ANNEAL`** (the p-start/p-end/anneal-window/queue-length block) lives in `core.py` and
is spread into the two configs that share it. It is one schedule; the two configs move
together or the comparison is not matched.

## Ported as-is, worth knowing

- `init_data` is sampled on **every** run, including runs with no SpaDE-family trainer in
  them. It costs 8192 activations off the buffer and one `randperm` draw, and moving it
  behind a check would change the RNG stream and the data order for every other SAE.
- `StandardizedBuffer.config` swallows a missing `buffer.config` with `try/except
  AttributeError`. Kept.
- `sweep_loss_balance.main` reads `rows[-1]` and dies with `IndexError` on a sweep dir
  with no `trainer_*` in it; its `SWEEP_DIR` is a SCREAMING_CASE local. Both kept.
- `lr: 3e-4` stays a literal in the config dicts even though `spade.common.config` exports
  `DEFAULT_LR = 3e-4`. The source hardcoded it here and referenced the constant only from
  the model defaults; wiring them together would silently couple all eight configs to one
  knob they were never coupled to.
- `sweep.spade_config` / `sweep.cosine_config` repeat the field list of
  `train_spade.config` / `train_cosine_spade.config`. That duplication is the source's;
  the two pairs must be edited together.
- `train_spade.py`'s docstring says γ=1e-4 where the source's comment said 1e-2, following
  the constant rather than restating a stale value.

## Dependencies on other zones

Written against APIs that are still skeletons here:

- `spade.common.config`: `MODEL_NAME DATASET LAYER D_MODEL CTX_LEN EXPANSION DEFAULT_WARMUP_STEPS`
- `spade.common.net.force_ipv4`, `spade.common.paths.add_karvonen_to_path`
- `spade.common.study.study_dir` — `sweep_loss_balance` uses `study_dir("multiseed")` in
  place of the source's hardcoded `/n/netscratch/ba_lab/Lab/dcao/spade_multiseed`, the one
  place `$USER` was not interpolated. It resolves identically for `dcao`. Note the
  skeleton has `study_dir` create the directory, so importing this module has a mkdir side
  effect.
- `spade.saes.spade`: `SpaDETrainer DEFAULT_LAMBDA_INIT DEFAULT_SPADE_GAMMA` (=1e-4)
- `spade.saes.spade_cosine`: `CosineSpaDETrainer DEFAULT_COSINE_GAMMA` (=1.0)
- `spade.evaluation.karvonen_adapter`: `load_karvonen_sae load_standardization_params`

Entrypoints import `circuits.*` **after** `spade.training.core`, which is what puts the
vendored fork on `sys.path`. Do not let an import sorter reorder those two blocks.

## Verification

No install, no GPU, no training run. `python -m py_compile` on all 11 modules; then, in a
throwaway tree outside the repo holding this package plus stand-ins for the six unported
modules above (taken from the source repo), `import spade.training.core`,
`spade.training.sweep`, `spade.training.sweep_loss_balance` and all eight entrypoints all
succeed against the vendored `circuits`, `--help` renders for the entrypoints and the
sweep, the four real sweep invocations from `shell/` produce the expected config lists
(14 / 8 / 1 / 5 configs, names as above), and the overwrite guard fires.
