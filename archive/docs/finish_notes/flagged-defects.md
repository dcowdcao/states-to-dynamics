# Finish notes — the defects the port agents recorded but did not fix

Source of the list: the fifteen files in `docs/port_notes/`, sections "Ported as-is",
"Questionable things ported as-is", "Inconsistencies found in the source". Ground truth
for what the code is supposed to do: the frozen repo at
`../Spade_Research_David_Sumedh_Ba`, read-only.

Everything below was reproduced before it was changed and re-checked after. Nothing here
changes a published number.

## Fixed

### 1. The sweep merge silently dropped CosineSpaDE from the baseline half

`scripts/slurm/sweeps/merge_baseline_results.py` — flagged in `port_notes/scripts.md`
("The filter is wrong and was kept").

The baseline filter was `~sae_type.str.contains("SpaDE", case=False)`. `sae_type` holds
trainer-class strings, so that predicate matches `CosineSpaDETrainer` as well as
`SpaDETrainer` and every `combined_results.csv` lost its CosineSpaDE row.

*Evidence.* The source carries the identical line twice, inline in
`shell/slurm/submit_spade_only_eval.sh:65` and `shell/slurm/submit_spade_sweep_eval.sh:66`.
It was latent there: the baseline CSV was `results/karvonen/all_results.csv`, which holds
six trainer types and no CosineSpaDE row (CosineSpaDE went to
`results/karvonen_cosine_spade/`, BatchTopK to `results/karvonen_batch_topk/`). The port
repoints the baseline at `results/main/all_results.csv`, and `results/main` is where
`spade.training.sweep` now writes all eight baselines — so the bug goes live the next time
either job runs. On an eight-family baseline frame the old predicate keeps six rows, the
new one keeps seven.

*Fix.* Match the swept family exactly: `sae_type != "SpaDETrainer"`. Both callers sweep
distance-SpaDE only, so its baseline row is the one the sweep supersedes; CosineSpaDE is a
separate family and stays a baseline. The docstring paragraph defending the old behaviour
is gone.

*Verified.* New check in `spade.validation.final_validation`, "the sweep merge keeps
CosineSpaDE as a baseline": it runs the real script over a synthetic eight-family baseline
CSV in a tmpdir and asserts CosineSpaDE survives while the superseded SpaDE row does not.
Confirmed the check FAILs against the old predicate and PASSes against the new one.

### 2. `--save-counts` crashed the inactive-prototype diagnostic

`src/spade/diagnostics/inactive_spade_prototypes.py:200`, flagged in `port_notes/figures.md`
("Left as found").

`repo_root / args.save_counts` divides the imported *function*, not its result:
`TypeError: unsupported operand type(s) for /: 'function' and 'str'`. Reachable whenever
the optional flag is passed.

*Evidence.* Same line in the source at
`src/diagnostics/find_inactive_spade_prototypes.py:203`. `main()` already binds
`root = repo_root()` twenty lines earlier and the two sibling paths (`ckpt`, `sp`) resolve
through `root` with the identical `isabs` shape.

*Fix.* `root / args.save_counts`.

*Verified.* An AST pass over `main()` confirms `root` is bound in scope and that no bare
`repo_root` remains on the left of a `/`. Not exercised end-to-end: the script needs a GPU,
the model download and a real checkpoint.

### 3. `branches.py` reached for `classify_prefix` through a sibling study

`src/spade/violation_divergence/branches.py:36`, flagged in `port_notes/divergence.md`
("If it moves to `othello/games.py`, this is a one-line change").

It moved. `spade/othello/games.py` owns `classify_prefix` now, and its own docstring names
violation-divergence as one of the five studies that read the shared corpus layer. The
import still said `from spade.rules.build_pairs import classify_prefix`, which only
resolved because `build_pairs` re-imports the name from `othello.games`.

*Fix.* Import it from `spade.othello.games`, alongside `collect_legal`,
`first_illegal_pos` and `stream_games` which were already coming from there.

*Verified.* `spade.violation_divergence.branches.classify_prefix is
spade.othello.games.classify_prefix` — the same function object before and after, so
behaviour is unchanged. `spade.rules.flip_labels.flip_stats` and
`spade.straightness.extract_traj.check_sink` are left as sibling-study imports: those
symbols have no home in the shared layer and the port brief carried them over on
instruction.

### 4. Two divergence docs contradicted the code they document

- `analysis/violation_divergence/game_trajectory/README.md` advertised
  `[--n_grid 6] [--embedding both]`. `argparse` has no `--n_grid` and `--embedding` is
  `choices=["pca", "isomap"]`; `git log -S n_grid` shows the flag never existed. Usage line
  now matches `--help`.
- `analysis/violation_divergence/make_figures.py`'s docstring listed F1 as "event-locked
  delta-hat and M2's cosines". The two cosine panels were deleted — the comment inside
  `f1_divergence_layer5` says so explicitly, and the panel titles are "(a) divergence from
  the reference game" / "(b) excess over a different legal move". Docstring now says
  "event-locked delta-hat, raw and vs control".

Both stale in the source. Flagged in `port_notes/divergence.md`.

### 5. `build_games.py` said four game classes and listed five

`src/spade/straightness/build_games.py:1`. `CLASSES` has five entries and the docstring
lists five bullets. Flagged in `port_notes/straightness.md`. One word.

## Verified, not changed — the four fixes reported from the previous merge

All four hold.

| Claim | Evidence |
|---|---|
| `EXCLUDED_SAE` is applied, not merely defined | `spade/spaces.py:66` — `representative_spaces()` skips `e.config in EXCLUDED_SAE` inside its `entries()` loop, with the reason in the docstring |
| `LAYER` / `D_MODEL` / `CTX_LEN` are single-sourced | Defined only in `spade/common/config.py`. `final_validation`'s "model constants are defined only in spade.common.config" walks every `.py` under `src/spade` for `^(LAYER\|D_MODEL\|CTX_LEN\|…)\s*=` and passes. Two package modules define *differently named* locals (`probe_space.MAIN_LAYER`, `INPUT_LAYER`) which are not the constant |
| `collect_legal` / `first_illegal_pos` exist once | `def` sites: `spade/othello/games.py:60` and `:86` only, tree-wide. `engine_legal` likewise at `:73`. Straightness, manifold-capture and violation-divergence all import them from there |
| No caller uses the old no-arg `results_dir()` | All 20 call sites pass an experiment name; `paths.results_dir(experiment: str)` has no default, so a no-arg call would be a `TypeError` at import for the module-level ones |

One caveat on the constants row: `analysis/bracket_flip/sae_pca_figure.py:30` and
`analysis/violation_divergence/probe_readout/decode.py:36` set `LAYER = 5` and
`PROBE_LAYER = 5` at module level. Both are byte-identical to the source, both live in
`analysis/` rather than the package, and `final_validation`'s check is scoped to
`src/spade`. Worth knowing that "single-sourced" currently means single-sourced *inside the
package*.

## Also verified while looking

- The `"SpaDE"`-substring mistake appears nowhere else. Every other family test is either
  an exact-match table (`karvonen_plot.TRAINER_CLASS_MAP`,
  `aggregate_cosine_sweep.COSINE_PREFIX = "CosineSpaDE-gam"`,
  `figure_all_configs_combined`'s `startswith(BEST_SPADE_CONFIG)`,
  `paper_style`'s `startswith("SpaDE-")`) or a deliberately Cosine-first ordered chain
  (`karvonen_adapter`'s dispatch, which `final_validation` pins;
  `analysis/manifold_capture/make_figures.py:55`). `manifold_capture/spaces.py`'s
  `POINT_BASED`/`check_codes` and `analysis/straightness/make_figures.py`'s `ours` use
  `"SpaDE" in name` deliberately — both families are prototype/simplex-valued, so matching
  both is the intent.
- All 38 shell scripts anchor with `cd "${SPADE_REPO:?}"`. `port_notes/docs.md` left
  `scripts/slurm/main/run_all.sh` on `cd "$HOME/spade_sae"` as someone else's problem; it
  was fixed on merge. No hardcoded out-of-repo path survives anywhere in `src/`,
  `scripts/`, `analysis/` or `viewer/`.
- `claims_status.md`'s γ=20 sentence, the other item `port_notes/docs.md` left open, was
  rewritten on merge (9db3895) and now records γ=1.0 as the deliberately-kept default.
- `SAEEntry.ckpt_dir` is consistent across all seven readers including the viewer and the
  exporter; the skeleton's `checkpoint_dir` survives nowhere.
- `make_log_dirs.sh` creates both `logs/bracket` and `logs/bracket_flip`, so
  `submit_bracket_sae_pca.sbatch`'s odd log directory does not fail its job at launch. It
  stays odd; it is not a defect.
- `--help` renders for all 44 argparse entrypoints in the package. All 28 analysis scripts
  import. `viewer/server.py` fails only on the absent `ae.pt`, which is the registry
  verifying loudly and correct.

## Regression state after the changes

`final_validation` 24/25 PASS, 1 SKIP (checkpoint round-trip — weights are gitignored and
have not been copied in), 0 FAIL. 7 taxonomy tests, 11 divergence-kernel tests, and all six
manifold-capture selftests pass and reproduce the source numbers.

## Left alone deliberately

**Out of scope by instruction.** The manifold-capture `LOW_VE` / `EXCLUDED_SAE` question
(whether the shattering headline survives excluding the degenerate γ=1e-2 seed-42
checkpoint); the `checkpoints/main/trainer_5/config.json` λ=9.77e-5 / γ=0.2 discrepancy;
all 13 `MIGRATION.md` open questions; anything that moves a published number.

**Entangled with the exclusion question.** `analysis/bracket_flip/sae_pca_figure.py`'s
`FAMILIES` list names `SpaDETrainer-othello` and loads it through `load_space` directly
rather than through `representative_spaces()`, so the global exclusion does not reach it.
Same shape as the manifold-capture case `port_notes/manifold.md` raises. Whether this
figure should drop the excluded checkpoint is the same owner's call, so the code is
untouched.

**A design choice, not a defect.** `spade/figures/figure_karvonen_extension.py` writes its
"No data available" placeholder to `Figure_4_karvonen_extension.png` — the same path the
real figure uses — so a run with a missing CSV overwrites a good figure in place. House
style says crash instead, and I would delete `save_placeholder` and raise. But six shell
callers run under `set -e`, and making this raise turns "pipeline continues with a
placeholder" into "pipeline aborts". That is a pipeline-behaviour decision, so it is
flagged rather than taken. Recommendation: delete the placeholder and raise.

**Figure text.** `analysis/bracket_flip/sae_pca_figure.py`'s suptitle still reads "F8 — …"
where the README calls it Fig 3, left over from the 8→3 prune. Real inconsistency, but
`analysis/bracket_flip/figures/fig3_sae_pca_grid.png` is committed, so changing the code
without regenerating the artifact just moves the inconsistency. Pair it with the figure
re-run.

**Port fidelity, low value.** Dead imports and dead locals carried over verbatim
(`spade_base`/`spade_cosine`'s `DEFAULT_LR`, `spade_cosine`'s `torch as t`, `rules/embed`'s
`CATEGORIES`, `manifold_capture/spaces`'s `batched_apply`, `sae_pca_figure`'s `INK`,
`analysis/straightness/make_figures`'s `repo_root`,
`norm_diagnostics.collect_activation_norms`); `game_trajectory/make_figures.py`'s
`import torch as _t` inside `space_features` while the module already imports torch;
`sweep_loss_balance`'s SCREAMING_CASE local `SWEEP_DIR`; `classify.boot_ci`'s
`tuple[float, float]` annotation over `np.float64`. None of these is wrong, only untidy,
and the brief was defects. They are a cleanup pass, not a bug-fix pass.

**Behaves correctly where it runs.** `study_dir` builds its netscratch default eagerly, so
`os.environ["USER"]` is read even when `SPADE_<X>_DIR` is set and an unset `$USER` raises
despite the override. Trivially fixable with a lazy default. Not fixed: `$USER` is set
everywhere this runs, and the loud-failure policy the constant encodes is deliberate.

**Reports and design docs.** `METRICS_REPORT.md`'s internal contradiction about `half_life`
already carries a correction header pointing at `RED_TEAM.md` F6; those files are dated
records of a review and rewriting the body would destroy the record.
