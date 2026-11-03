# karvonen_repro

Karvonen's own board-state evaluation, run on our checkpoints. Not a reimplementation:
every number comes out of `external/Karvonen_Research/circuits/`.

## Run it

```bash
python -m karvonen_repro.evaluate --output-dir OUT CKPT_DIR [CKPT_DIR ...]
```

`OUT/f1_results.csv` is the result. Defaults are his published settings
(`--n-inputs 1000`, `--batch-size 25`); `--device` defaults to cuda if present. On the
cluster: `karvonen_repro/submit_eval.sbatch`, whose header documents its env vars.

Needs `adamkarvonen/othello_45MB_games` and `Baidicoot/Othello-GPT-Transformer-Lens`
in the HF cache — his code fetches both by name and a compute node may not have the
network to.

## Two modes

`--mode repro` (default) is his protocol exactly: `get_evals` averages **one** batch
of 25 activation vectors, so `l0`, `l1_loss`, `l2_loss`, `frac_variance_explained`,
`cossim`, `l2_ratio` and `frac_alive` are n=25 estimates. This is the only mode whose
output may be compared column-by-column against his published CSV — see PROTOCOL.md
section 2 for the evidence and section 3 for the resulting per-column tolerances.

`--mode measurement` averages 100 batches (2500 activations; `--eval-batches` to
change it). Use it for any number we report ourselves. At n=25 the
`frac_variance_explained` estimator has sd 0.14 and a 0.13 downward bias at L0 10;
publishing a SpaDE-vs-Karvonen comparison on that would be indefensible.

The board-state stage is identical in both modes — always 1000 games x 59 positions.
Every row carries `eval_protocol_mode`, `eval_n_batches` and `eval_n_activations` so
it cannot be misread later.

To reproduce his table (the validation tooling is archived, see below):

```bash
python -m karvonen_repro.validation_checkpoints --released-root DIR   # the 15 checkpoints
python -m karvonen_repro.evaluate --mode repro --output-dir OUT $(...)
python archive/karvonen_repro/compare_to_published.py --harness OUT/f1_results.csv
```

## Why this exists

`archive/src/spade/evaluation/run_eval.py` reimplements the protocol and drifts from it —
no >=0.95-precision / >=10-firings feature gate, one label per feature instead of
many, a per-feature threshold instead of one global one, macro-averaging over cells
instead of micro-pooling. Worth about 0.2 absolute F1. This harness does not
reimplement anything, so it cannot drift.

## What runs

`evaluate.py` calls `full_pipeline.analyze_sae_groups` once per checkpoint. That is
his whole pipeline, unedited:

| stage | function | produces |
| --- | --- | --- |
| SAE metrics | `get_eval_results.get_evals` → `evaluation.evaluate` | l0, l1/l2 loss, frac_alive, FVE, cossim, l2_ratio, loss_{original,reconstructed,zero}, frac_recovered |
| aggregation | `eval_sae_as_classifier.aggregate_statistics` | per-feature board-state counts at 11 thresholds |
| labelling | `analysis.analyze_results_dict` | the >=0.95-precision, >=10-firings gate; many labels per feature |
| reconstruction | `eval_board_reconstruction.test_board_reconstructions` → `calculate_F1_scores` | micro-pooled F1 per threshold on held-out games |
| coverage | `f1_analysis.add_coverage_to_df` → `best_f1_average` | `*_best_average_f1` and the `*_f1_threshold_*` family |

Board functions and every constant (thresholds `arange(0, 1.1, 0.1)`, high_threshold
0.95, significance_threshold 10, n_inputs, batch size) come from his
`pipeline_config.Config`. For Othello that is
`games_batch_to_state_stack_mine_yours_BLRRC`,
`games_batch_to_state_stack_mine_yours_blank_mask_BLRRC` (the paper's headline
metric) and `games_batch_to_valid_moves_BLRRC`.

Column names and meanings are his: `full_pipeline.initialize_dataframe` and
`append_results` build every row. The output is a superset of the columns in his
published `06-10-othello-trained-results_with_07-10_topk.csv`, minus the two unnamed
index columns pandas wrote into it.

## Layout

- `evaluate.py` — the CLI. Chooses checkpoints, output paths and what is already done.
- `patches.py` — the four module-level rebinds that let his pipeline see our
  checkpoints. Read this file; it is the entire divergence surface.
- loader: `spade.sae_loader.load_sae`, shared with the rest of the reproduction work.
- `submit_eval.sbatch` — SLURM.
- `PROTOCOL.md` — the pass/fail spec.
- Archived under `archive/karvonen_repro/` (the paper runs the harness on a checkpoint
  glob only): `validation_checkpoints.py` + `validation_set.csv` (the 15 released
  checkpoints), `spade_checkpoints.py` + `spade_set.csv` (the study's 44 SpaDE and
  CosineSpaDE checkpoints), and the spec's evidence `compare_to_published.py`,
  `probe_eval_batch.py`.

`OUT/` holds `checkpoints/<sweep parent>/<group>/<trainer>/` (symlinks to `ae.pt`,
`config.json` and the sweep's `standardization_params.json`, plus every pkl his
pipeline writes), the same nesting again under `per_checkpoint/`, the shared
`othello_{train,test}_dataset.pkl`, and the concatenated `f1_results.csv`.
The `<group>/<trainer>` tail is load-bearing: `compare_to_published.py` identifies a
row by the last two components of `autoencoder_path`. The level above it is what makes
the mirror injective — every seed of the multi-seed study names its sweep directory
`autoencoders`.

Checkpoints are mirrored rather than used in place because his pipeline writes its
pkl artifacts next to `ae.pt`, and the released SAEs already carry his — symlinking
those would write through and destroy them.

## Two things to know

**Raw activations.** `load_sae` must return a module that consumes raw model
activations. His SAEs were trained on raw residual stream, ours on standardized;
folding that transform into the returned module is the loader's job. The harness
never applies one, and nothing downstream would notice a loader that forgot.

**`l1_loss` means nothing for SpaDE.** `evaluation.py:183` is
`f.norm(p=1, dim=-1).mean()`, and a SpaDE code is a point on the simplex, so the column
is 1.000000 for every SpaDE checkpoint at every sparsity — measured, not argued. Drop it
from any SpaDE table rather than reporting a 60x "improvement" over his SAEs.

**One output dir per (n_inputs, mode, checkpoint set), and one process at a time.**
n_inputs is baked into the artifact filenames his code writes, and a second n_inputs
in the same directory makes `get_all_results_file_names` see two candidates and
refuse. `analyze_sae_groups` also writes its per-group `results.csv` to a fixed path
and reads it back, so two concurrent runs sharing OUT will swap rows. Re-running the
same command against the same OUT is the resume path: finished checkpoints are
skipped.
