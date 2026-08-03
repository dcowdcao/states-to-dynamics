# Port notes — `spade/saes/`

Four files moved from the source repo, logic unchanged:

| Source | New |
|---|---|
| `src/models/spade_base.py` | `src/spade/saes/spade_base.py` |
| `src/models/spade_trainer.py` | `src/spade/saes/spade.py` |
| `src/models/spade_cosine_trainer.py` | `src/spade/saes/spade_cosine.py` |
| `src/rules/sae_registry.py` | `src/spade/saes/registry.py` |

## What changed

Imports are `spade.*`-rooted (`spade.common.config`, `spade.common.paths`,
`spade.saes.spade_base`, `spade.multiseed.aggregate_seeds`). The registry's
`sys.path.insert(0, src/)` is deleted; the only remaining path injection is
`add_karvonen_to_path()` in `spade_base.py`, which puts the vendored fork on the path
before `import circuits...`.

`DEFAULT_SPADE_GAMMA` 1e-2 → **1e-4** (David's call, this session's default SpaDE config
is λ=1e-3 / γ=1e-4, checkpoints under `checkpoints/spade_revival/`). `DEFAULT_LAMBDA_INIT`
stays 1e-3 and `DEFAULT_COSINE_GAMMA` stays **1.0** — deliberately not the γ-sweep's
recommended 20, which was selected on reconstruction F₁ (0.547 vs 0.530) while coverage
runs the other way (0.537 vs 0.474) and decides it. The reason is now a comment beside the
constant so nobody "fixes" it from the sweep README.

`registry._SEED42_DIRS` is rewritten to the new checkpoint layout
(`autoencoders*/` → `checkpoints/{main,batch_topk,cosine_spade,spade_revival}/`; the
`spade_sweep` dir has no seed-42 entry). `MULTISEED_DIR` — an env-var-with-hardcoded-fallback
constant — is gone; seeds 43–46 now resolve through `study_dir("multiseed")`. The netscratch
sub-layout (`seed_<n>/autoencoders/trainer_<i>`) is unchanged; only the in-repo dirs were
renamed. Names `entries`, `seed42_entries`, `SAEEntry`, `_verify`, `_SEED42_DIRS` are kept
exactly: `manifold_capture/spaces.py` imports the two private ones, and the viewer exporter
plus straightness and violation-divergence import `seed42_entries()`.

Everything else is byte-identical to the source apart from three comment edits: the
γ=1.0 rationale above, a pointer from `train_karvonen_matched.get_trainer_configs` (a module
that no longer exists under that name) to `spade.training`, and one docstring line in the
registry recording that the degenerate γ=1e-2 seed-42 entry is excluded **downstream** in
`spade.spaces`, not here. The registry still lists it — it exists on disk, and the exclusion
is one other agent's single decision point.

## Invariants held

Class names `SpaDEAutoEncoder`, `CosineSpaDEAutoEncoder`, `SpaDETrainer`,
`CosineSpaDETrainer` unchanged (the eval adapter dispatches on `trainer_class` strings; the
viewer exporter keys on `type(ae).__name__`). `Ae` / `Ad` / `lambda_pre` are still top-level
attributes — a smoke run confirms `state_dict()` keys are exactly `{Ae, Ad, lambda_pre}` for
SpaDE and `{Ae, Ad}` for CosineSpaDE. `softplus_inverse` is still importable from
`spade.saes.spade` (source `final_validation.py` test 17 asserts on it by name).
`trainer_class` / `dict_class` / default `wandb_name` strings are byte-identical.

## Verification

`python -m py_compile` passes on all four files. The mandated
`PYTHONPATH=$WT/src python -c "import spade.saes.spade"` **fails today**, and not on
anything in this zone: `spade/common/config.py` and `spade/common/paths.py` are still
skeleton stubs (another agent's zone), so `DEFAULT_LR` / `add_karvonen_to_path` do not exist
yet. Re-run it once `spade.common` lands.

To verify anyway, a scratch script stubbed `spade.common.{config,paths,study}` and
`spade.multiseed.aggregate_seeds` with the source repo's values and exercised the real
modules end to end: `circuits` imports resolve against the vendored fork, both autoencoders
encode to rows summing to 1 and decode back to input shape, both trainers build an
optimizer + LR schedule, take an `update()` step, and emit their logging dict
(`lambda_val`, `dist_reg_loss` / `cos_reg_loss`) and `config`.

`registry.seed42_entries()` raises `AssertionError: missing ae.pt in
checkpoints/main/trainer_0` — correct and intended. The ~1.5 GB of `ae.pt` is gitignored and
moves by filesystem copy, which has not happened in this worktree. The check was not
weakened; a registry that silently returns unverifiable entries is worse than one that
crashes.

## Ported as-is, worth knowing

- `spade_base.py` and `spade_cosine.py` both import `DEFAULT_LR` / `DEFAULT_WARMUP_STEPS`
  without using them (only `spade.py` and the cosine trainer signature do — the base and
  `spade_cosine` lines are dead). `spade_cosine.py` also imports `torch as t` unused. Kept:
  this is a move, not a cleanup.
- `_verify` reads `config.get("trainer", config)["wandb_name"]` — a soft `.get` handling two
  config shapes (Karvonen nests under `trainer`, the sweep dirs don't). Kept as written.
- `SpaDETrainerBase.loss` takes `step` and ignores it; `scale_biases` is a required no-op
  from the `Dictionary` interface.
- `softplus_inverse` returns a 0-d/1-element tensor via a float branch and `x[0]`; passing a
  tensor with more than one element would silently take the first. Only ever called with a
  float.
- The registry docstring's "65 entries" count (9 seed-42 + 4 seeds × 14 configs) is carried
  over unverified — the netscratch tier is not reachable from this check.
