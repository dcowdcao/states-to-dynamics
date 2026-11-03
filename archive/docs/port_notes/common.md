# Port notes — `spade.common`

Zone: `src/spade/common/*`, `src/spade/__init__.py`. Source: `src/common/{config,device,net,paths}.py`
plus the six duplicated `study_dir` bodies.

## What landed

- `config.py`, `device.py`, `net.py` — byte-identical bodies to the source, docstrings included.
- `paths.py` — `repo_root`, `karvonen_root`, `add_karvonen_to_path`, `data_dir` ported as-is;
  `results_dir(experiment)` / `checkpoints_dir(experiment)` now take an experiment name;
  `figures_dir()` added; `tpr_root()` / `add_tpr_to_path()` added; `src_dir()` /
  `add_src_to_path()` dropped (the package replaces them).
- `study.py` — new; one `study_dir(name)` for all six studies.

`src/spade/__init__.py` left empty, as the skeleton had it.

## Verified

With `PYTHONPATH=src` under `spade_env` (py3.11.15), no install:

- `from spade.common import config, device, net, paths, study` imports clean.
- `repo_root()` → this worktree; `karvonen_root()` and `tpr_root()` both resolve to the
  vendored forks; `add_karvonen_to_path()` then `import circuits` works.
- `add_tpr_to_path()` makes every import the source's `bracket_flip/tpr_probe.py` performs
  resolve: `from hook_utils import seed_all`, `intervene_tpr_probe.solve_residual_delta_for_binding_delta`,
  `train_tpr_probe.{TensorProductBoardProbe,build_square_selection_mask,compute_tpr_probe_loss}`.
- All six `study_dir` names resolve to existing netscratch trees: rules 69M, bracket 658M,
  straightness 5.3G, manifold 471M, divergence 9.3G, multiseed 68G. `SPADE_BRACKET_DIR`
  override honoured; unset `$USER` raises `KeyError: 'USER'`.

## Decisions

**`$USER` fails loudly.** The five `*/study_dir.py` files used `os.environ['USER']`;
`multiseed/aggregate_seeds.py`, `rules/sae_registry.py` and `cosine_gamma_range/loss_balance.py`
defaulted to the literal `dcao` (the last one hardcodes `/n/netscratch/ba_lab/Lab/dcao/spade_multiseed`
with no `$USER` at all). Ported the loud version: another user running the pipeline would
otherwise silently read and write dcao's netscratch. The three lenient call sites must switch
to `study_dir("multiseed")` when their modules are ported — that is a behaviour change for a
non-dcao user, and the right one.

**No `mkdir` in `study_dir`.** The skeleton pseudocode said "mkdir parents"; the source does
not, and every caller already does `out_dir.mkdir(parents=True, exist_ok=True)` on its own
subdir. Resolving a path should not create 90-day-retention trees as a side effect. Flagging
in case the parent intended otherwise.

**`data_dir()` stays `repo_root()/"data"`.** The skeleton comment said
`repo_root()/"data"/"activations"`. Kept the source semantics — it has zero callers in the
source repo, so nothing pins it either way, but "move, not rewrite" says leave it. If the
intent was for `data_dir()` to point at the activation cache directly, say so and it is a
one-line change; downstream porters should write `data_dir() / "activations"` until then.

**`tpr_root()` raises when missing**, mirroring `karvonen_root()`. The source had no
`tpr_root` at all — `tpr_probe.py` built the path inline off `repo_root()` with no check.

**`add_tpr_to_path()` inserts two entries in the source's order** (`src` then
`src/hook_utils`, each at position 0, so `hook_utils` ends up first). Both are what
upstream's own entrypoints inject; nothing our code imports strictly needs the second, but
diverging from upstream's assumed path is not worth the risk.

## Inconsistencies found in the source (ported as-is, not fixed)

- The `study_dir` default is built eagerly, so `os.environ['USER']` is read **even when
  `SPADE_<X>_DIR` is set** — setting the override does not save you from an unset `$USER`.
  Preserved. Trivially fixable with a lazy default if anyone cares.
- `src/cosine_gamma_range/loss_balance.py:30` hardcodes `/n/netscratch/ba_lab/Lab/dcao/spade_multiseed`
  with no `$USER` interpolation at all — the worst of the three lenient variants. Whoever
  ports `training/sweep_loss_balance.py` should route it through `study_dir("multiseed")`.
- `results_dir()` had eight callers, all of the shape `results_dir() / "<study>"`. Under the
  new signature they become `results_dir("<study>")`; the on-disk path is unchanged.

## Not verified

- `select_device()` returns `cpu` here — no GPU on this node, so the `cuda` branch is
  untested (it is unchanged source).
- `force_ipv4()` was called but no download was attempted.
- Nothing was `pip install -e .`'d (shared env), so `repo_root()`'s behaviour under an
  editable install is still assumed, not demonstrated.
