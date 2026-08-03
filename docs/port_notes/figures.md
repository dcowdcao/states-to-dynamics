# Port notes — `spade/figures` and `spade/diagnostics`

Eleven modules ported from the frozen source repo on branch `port/figures`. Logic is
unchanged; the diffs against the source are imports, artifact paths, and the two
`paper_style` label changes below. Nothing here has been re-run against real data — the
verification was compilation, import, and a synthetic-input render.

| source | here |
|---|---|
| `src/figures/paper_style.py` | `spade/figures/paper_style.py` |
| `src/figures/{karvonen_plot,embedding_panels,straightness_panels}.py` | `spade/figures/` |
| `src/figures/figure_{karvonen_extension,all_configs_combined,cosine_gamma_response}.py` | `spade/figures/` |
| `src/diagnostics/plot_norm_diagnostics.py` | `spade/diagnostics/norm_diagnostics.py` |
| `src/diagnostics/plot_norm_vs_move.py` | `spade/diagnostics/norm_vs_move.py` |
| `src/diagnostics/plot_activation_norms_per_game.py` | `spade/diagnostics/activation_norms_per_game.py` |
| `src/diagnostics/find_inactive_spade_prototypes.py` | `spade/diagnostics/inactive_spade_prototypes.py` |

## What changed in paper_style

This is the one file where the port is not literal, because the two facts it encoded about
SpaDE are no longer true.

**The stale-config comment.** It said `autoencoders/trainer_5/config.json` records a stale
γ=0.2 / λ=9.77e-5, and reasoned from the measured L0=130 that the real setting is γ=1e-2.
The situation is broader: all six `config.json` under `checkpoints/main` are the same Mac
smoke config written over real GPU weights, so none of them labels its own weights. The
comment now says that instead of singling out trainer_5, and drops the L0 inference, which
was only ever load-bearing for the γ=1e-2 label that is also gone.

**The γ=1e-2 label.** `SAE_HPARAMS["SpaDE"]` was `"γ=1e-2"`. That is the seed-42 checkpoint
now excluded from every study and from the viewer, and the default SpaDE configuration is
λ=1e-3 / γ=1e-4. Both SpaDE keys now read `"λ=1e-3, γ=1e-4"`:

```
"SpaDE":                   "γ=1e-2"  ->  "λ=1e-3, γ=1e-4"
"SpaDE-lam1e-03-gam1e-04": "γ=1e-4"  ->  "λ=1e-3, γ=1e-4"
```

The second half of that is needed because `sae_label` strips the `SpaDE-` suffix, so the
two keys render the same checkpoint; leaving them different labels one config two ways in
two figures, which is the drift `paper_style` exists to prevent.

**This needs David's call.** Rows in `results/main/all_results.csv` carry the family string
`SpaDE`, and those rows *are* the excluded γ=1e-2 run. Any figure still drawn from them now
gets the default-config label — wrong for that checkpoint. The alternative is a separate
key for the legacy run, which nothing currently produces, so it would have to be introduced
in the aggregation as well. Deciding to drop those rows entirely would also settle it.

Also repointed one comment: `CATEGORY_STYLE`'s keys follow `othello.taxonomy.CATEGORIES`,
not `rules.illegal_taxonomy.CATEGORIES`.

`CosineSpaDE` stays at `"γ=1.0"`, matching the decision recorded in ARCHITECTURE — the
γ-sweep's recommended γ=20 lost on coverage. `docs/claims_status.md` says the operating
point is γ=20; the two documents disagree and the label follows ARCHITECTURE.

## Path repointing

- `figure_all_configs_combined.OLD_SWEEP_CSV`:
  `8_spade_configs_updated/karvonen_spade_sweep/all_results.csv` →
  `results/spade_sweep_8config/all_results.csv`. A live input, not an archive: it is the
  only surviving record of the chosen-best `SpaDE-lam1e-03-gam1e-04` single-seed point, and
  the figure raises `KeyError`/`IndexError` without it.
- `figure_karvonen_extension` usage line: `results/karvonen/` → `results/main/`. Its actual
  default, `results/multiseed/aggregate_baseline.csv`, was already correct.
- `inactive_spade_prototypes`: `autoencoders/trainer_5/ae.pt` → `checkpoints/main/trainer_5/ae.pt`,
  `autoencoders/standardization_params.json` → `checkpoints/main/standardization_params.json`.
  The params file stays at the save-dir level, one above the trainer dirs.
- `norm_vs_move --output_dir`: `results/runs/Main_Trained_Run_All_SAE/figures/activation_norms`
  → `figures/activation_norms`. That results path is an empty stub in the source repo and is
  pending a staleness call; meanwhile the file this script writes, `plot5_norm_vs_move.png`,
  sits in `figures/activation_norms/` next to the `plot1`–`plot4` that
  `activation_norms_per_game` writes. The numbering is one series and now lands in one place.
- Output subdirectories are unchanged: `figures/` and `figures/cosine_spade_sweep/`. The
  eight distinct `Figure_4_karvonen_extension*.png` in the source come from calling one
  entrypoint with different `--results_csv` / `--output_dir` against a constant `OUT_NAME`,
  so the directory is the only thing distinguishing them. Flattening loses five.

`activation_norms_per_game.DEFAULT_SAE_PATH` is **not** repointed. It is
`archive/runs/Run_3_Results/runpod_autoencoders/spade_w4096_g0.2_s42`, a dict_size-4096
γ=0.2 SpaDE with no equivalent anywhere under `checkpoints/` — swapping in a
`checkpoints/spade_sweep` trainer would change what the diagnostic is diagnosing. It
therefore depends on MIGRATION open question 5 (whether `archive/runs/` comes across at
all). If the answer is no, this default has to be re-chosen, not just re-pathed.

## Ported as-is, questionable

**`karvonen_plot` violates FIGURE_GUIDELINES §1.5 and §2.** It carries its own eight-color
`SAE_CONFIG` palette (not Okabe–Ito; `relu` and `panneal` share `#e41a1c`, `gated` and
`gated_anneal` share `#984ea3`, `sparsemax_dist` and `cosine_spade` share `#ff7f00` — the
exact SAEBench failure §1.5 cites), sets `plt.rcParams` at import time instead of calling
`apply_style`, and calls `matplotlib.use("Agg")` as an import side effect. It also draws at
14×10 in and 12×8.5 in, far above every row of the sizing table. Reconciling it with
`paper_style` would change the colors in every published Figure 4 at once, so it is left
alone; this is the concrete instance of the "nothing is drawn at final printed width" open
defect in `docs/claims_status.md`.

**Zero-length error bars read as measured zeros.** `_error_bars` does
`df[f"{col}_std"].fillna(0)`, so a single-seed config — including the chosen-best SpaDE
point the combined figure exists to show — draws a bar of length zero, visually identical
to a five-seed config with no variance. §1.2 wants the semantics stated; the figure states
"±1 std across seeds" in the suptitle and says nothing about n per point.

**`inactive_spade_prototypes` crashes on `--save-counts`.** Line reads
`repo_root / args.save_counts` where `repo_root` is the imported function, not a call —
`TypeError: unsupported operand type(s) for /`. Only reachable when the optional flag is
passed, and the flag has no caller in the source repo. Left as found.

**Dead and unused code carried over.** `norm_diagnostics.collect_activation_norms` is never
called (`collect_raw_and_standardized` replaced it); in the same file `plot_histogram`
computes `std_val` and drops it, and `main` unpacks `raw_norms`, `ad_norms` and `std_mean`
without using them. `count_firings` unpacks `d` unused.

**Soft fallback on an unknown SAE type.** `get_display_info` falls back to a gray circle
for a `sae_type` missing from `SAE_CONFIG` rather than raising, so a typo in a results CSV
plots silently as an unnamed gray point.

**Both `figure_karvonen_extension` outcomes write the same filename.** When the CSV is
missing it writes a "No data available" placeholder to `Figure_4_karvonen_extension.png` —
the same path the real figure uses, so a failed run overwrites a good figure in place.

## Verification

No `pip install`; `spade_env` is shared. Python 3.11.15, matplotlib 3.10.8, pandas 2.0.3,
Agg backend throughout.

- `python -m py_compile` on all eleven files, clean.
- `paper_style`: imported every name the downstream scripts use, ran `apply_style()` both
  ways, `rule_key`/`role_key`/`save_figure` against a real figure, `sae_label` on a known
  and an unknown family.
- `embedding_panels`, `straightness_panels`, `karvonen_plot`: rendered every public panel
  and `render_grid` (both the color-mapped and the plain branch, with error bars and an
  annotation) on synthetic frames.
- The three figure entrypoints: run end-to-end against a synthetic results tree, including
  the missing-CSV placeholder path. All three produced PNGs at the expected paths.
- The four diagnostics: imported and `--help`-parsed against the real `circuits`,
  `transformer_lens`, `datasets` and matplotlib.

Two dependencies are still skeleton in this worktree, so they were stubbed for the import
checks and are the first thing to re-verify once they land: `spade.common.{paths,config,device}`
(`repo_root`, `add_karvonen_to_path`, the constants, `select_device`) and
`spade.saes.spade.SpaDEAutoEncoder`. Nothing was run on GPU or on real checkpoints.

`spade/validation/final_validation.py` asserts against this zone — that
`figure_karvonen_extension.py` contains `--results_csv` and `--output_dir`, and that
`karvonen_plot.TRAINER_CLASS_MAP` covers all eight trainer classes. Both still hold.

## Not adopted

`spade.common.paths` declares `results_dir(experiment)`, `checkpoints_dir(experiment)` and
`figures_dir()`. These modules resolve `repo_root() / "results/..."` instead, the shape the
source uses, so the port diff stays readable as pure path repointing. Switching them over
is a follow-up once `common` is real, and it is the natural moment to also give the
diagnostics' `--output_dir` defaults a single owner.
