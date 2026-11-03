# 7-14-2026 — Multi-seed training for error bars: context and findings

Goal: for every SAE hyperparameter config in the figures, train with multiple
random weight-init seeds, then add seed-based error bars to every plot.

## Repo / environment state (verified this session)

- Fresh clone at `~/Spade_Research_David_Sumedh_Ba` (old broken repo preserved at
  `~/Spade_SAE_Research_BACKUP`, 51 GB, git history was empty).
- `external/Karvonen_Research` and `external/Original_Spade_Repo` were empty
  broken gitlinks in the clone (no `.gitmodules`); restored by copying from the
  backup. `src/validation/final_validation.py` passes 17/17 after restore.
- Conda env `spade_env` works (torch 2.5.1+cu124, nnsight imports OK).
- SLURM available; partition `kempner_h100` up (85 mixed nodes), account
  `kempner_ba_lab`, fairshare 0.99 for dcao.
- Home quota nearly full (7.4 GB free of 95 GB). Lab scratch is writable at
  `/n/netscratch/ba_lab/Lab/dcao/` — big artifacts go there
  (`spade_multiseed/`). NOTE: netscratch has a 90-day retention policy.
- The 28 GB extracted eval data (train/test activations + board states) exists in
  the backup at
  `Spade_SAE_Research_BACKUP/Spade_SAE_Research_Codespace/experiments/expt_othello_sae/data/activations/`
  — being copied to `/n/netscratch/ba_lab/Lab/dcao/spade_multiseed/data/activations/`
  (no re-extraction needed). Seed-42 checkpoints (`ae.pt`) for all configs also
  live in that backup tree (`autoencoders*/`).
- All `shell/slurm/*.sh` scripts and `docs/fasrc_run_commands.md` still hardcode
  the old repo path `$HOME/Spade_SAE_Research` — stale for the new clone name.

## Seed flow through training (traced to file:line)

`train_karvonen_matched.py:436` hardcodes `seed = 42`; `get_trainer_configs`
puts `"seed"` into every trainer config; `trainSAE` (training.py:157) passes it
to each trainer `__init__`, and each trainer's config.json records it.
`trainSAE` itself never seeds anything globally.

Per-trainer, does the config `seed` change weight init?

| Trainer | Seed controls init? | Where |
|---|---|---|
| StandardTrainer | yes | standard.py:41-43 seeds before `AutoEncoder` (`t.randn` dictionary.py:57) |
| PAnnealTrainer | yes | p_anneal.py:48-50 |
| GatedSAETrainer | yes | gdm.py:41-43 (`t.randn_like` dictionary.py:209) |
| TopKTrainer | yes | top_k.py:184-186 (nn.Linear kaiming init) |
| GatedAnnealTrainer | yes | gated_anneal.py:47-49 |
| BatchTopKTrainer | yes | batch_top_k.py:126-128 |
| SpaDETrainer | **NO** | spade_base.py:133-135 seeds, but Ae/Ad are overwritten by `init_data` (spade_base.py:52-58) |
| CosineSpaDETrainer | **NO** | same base-class path |

Randomness NOT controlled by the per-trainer seed:
1. **`init_data` sampling** (`train_karvonen_matched.py:160`, global `t.randperm`
   with no prior `t.manual_seed`) — this IS the entire SpaDE/CosineSpaDE weight
   init. Must seed the global RNG (per run seed) at the top of `main()` so
   different seeds give different SpaDE prototypes.
2. Which buffer batches land in the init pool (same reason).
3. Training-data order for all 8 trainers: `NNsightActivationBuffer` uses global
   `t.randperm` (buffer.py:74/225/370), unseeded. A top-of-main global seed also
   makes data order vary per seed (and be reproducible).

Conclusion: adding `--seed` needs (a) CLI arg replacing the hardcoded 42,
(b) a global `t.manual_seed(args.seed)` (+cuda) at the top of `main()` before
buffer calibration / init sampling. With that, Karvonen trainers vary by weight
init, SpaDE variants vary by prototype sampling, and data order varies too.

## Plot inventory (what needs error bars)

Seed-dependent AND reproducible in this repo — the Karvonen Figure-4 2x2 grid
family (all rendered by `src/figures/karvonen_plot.py::render_grid`):
- `src/figures/figure_karvonen_extension.py` → `figures/Figure_4_karvonen_extension.png`
  (reads `results/karvonen/all_results.csv`; also produced the `_spade_sweep` /
  `_spade_v2` / lambda / gamma PNG variants — same script, different CSV, renamed on scp).
- `src/figures/figure_all_configs_combined.py` → `figures/Figure_4_all_configs_combined.png`
  (baseline CSV + lambda_sweep_results + gamma_sweep_results CSVs, deduped).
- `best_spade_lam1e-07_gam5e-03/figure_karvonen_best_spade.py` (standalone copy).

Seed-dependent but NOT reproducible here (inputs missing from this tree; source
only): all 11 SAE-based `analysis/geometry/*` scripts (need
`Runs_GPU/Main_Trained_Run_All_SAE/...`, per-SAE `f1_TFRRC.pt` sets),
`plot_activation_norms_per_game.py` plots 3–4, archived Run_2/Run_3 PNGs
(no generator scripts exist for most), W&B UI chart exports.

Seed-independent (no SAE involved; no error bars needed):
`plot_norm_diagnostics.py`, `plot_norm_vs_move.py`, activation-norm plots 1–2,
`analysis/geometry/{activations_pca_probe_overlay,linear_probe_directions_pca}`,
`analysis/linear_probe` notebook figures.

## Hyperparameter configs appearing in current figures

Metrics CSV schema (`all_results.csv`): sae_name, sae_type, wandb_name, ...,
frac_recovered, coverage, reconstruction_f1, l0, variance_explained, seed
(already has a `seed` column, all rows currently 42).

- Baseline 8 (one trainSAE job trains them together, sharing the LM stream):
  PAnneal, Standard, Gated, TopK(k=80), GatedAnneal,
  SpaDE(λ=1e-3, γ=1e-2 defaults), BatchTopK(k=80), CosineSpaDE(γ=1.0).
  (results/karvonen has the first 6; BatchTopK and CosineSpaDE were trained in
  separate follow-up jobs → results/karvonen_cosine_spade etc.)
- SpaDE sweep points in figures (combined figure after dedup):
  λ=8e-8/γ=1e-3, λ=1e-7 × γ∈{1e-3, 5e-3, 8e-3, 1e-2, 3e-2}.
- Additional sweep points that exist in CSVs but are dropped/degenerate:
  λ=5e-8 & λ=1e-8 (collapsed, frac_recovered<0, L0=8192), λ=9e-8 (dup of 8e-8).
  Old 8_spade_configs_updated grid (λ∈{1e-3,1e-4} × γ∈{1e-4..1e-1}) is superseded.

## Cost anchors (from run_all.sh comments / past runs)

- Baseline 6-SAE training job: ~8–16 h on one H100 (36,621 steps, 300M tokens).
- SpaDE-sweep training job (8 configs): ~2–4 h.
- Eval of 6 SAEs: ~4–8 h (≈40–60 min per SAE, dominated by encoding 5.3M train
  samples + F1 + loss_recovered).
- Checkpoints: ~130 MB per SAE (ae.pt) + 25/50/75% checkpoints.

## Decisions (user-approved)

- 5 seeds total: 42 (reused from existing runs) + 43, 44, 45, 46.
- Scope: the 14 configs appearing in figures — 8 baseline SAEs + 6 SpaDE sweep
  points (lam8e-8/gam1e-3, lam1e-7 x gam{1e-3,5e-3,8e-3,1e-2,3e-2}). Collapsed
  sweep points (lam1e-8/5e-8, 9e-8 dup) excluded.
- Multi-GPU: device-parallel trainers (bit-identical per-SAE math), NOT DDP
  batch-sharding (would change BatchTopK/p-anneal semantics and break
  comparability with seed 42 / Karvonen). Eval parallelized as SLURM arrays.
- Precision stays fp32 (no bf16/TF32) so new seeds are numerically comparable
  to the reused seed-42 replicate.
- W&B stays on for real runs (same project; runs distinguished by config.seed,
  run names unchanged since wandb_name is the aggregation grouping key).

## What was implemented

- `train_karvonen_matched.py`: `--seed` arg; global `t.manual_seed` at top of
  main() (fixes SpaDE init_data + data order); `--spade_pairs "lam:gam,..."`
  appends SpaDE configs so one job trains all 14; `spade_cfg()` helper
  (dedupes sweep branch); device round-robin across visible GPUs when >1
  (buffer/LM on cuda:0; trainers self-move batches — verified all 8 trainers
  call `.to(self.device)` in update()).
- `external/.../dictionary_learning/training.py` log_stats: per-trainer copy
  moved to `trainer.device` (only cross-device gap; update() self-moves).
- `shell/multiseed/`: submit_train.sh (array 43-46 == seed, gpu:4, 14 configs
  per job), submit_eval.sh (array 0-13 == trainer idx, SEED via env, gpu:1,
  --n_samples 59000 matching original protocol), run_multiseed.sh (chains
  eval arrays afterok per-seed training task).
- `src/multiseed/aggregate_seeds.py`: merges seed-42 repo CSVs + per-seed
  netscratch CSVs → results/multiseed/{all_seeds,aggregate,aggregate_baseline}.csv
  (mean + <metric>_std + n_seeds per config; key = wandb_name minus _trainer_N).
- `src/figures/karvonen_plot.py`: `_error_bars()` draws ±1-std x/y bars when
  <col>_std columns present. figure_karvonen_extension default CSV →
  aggregate_baseline.csv; figure_all_configs_combined rewritten to read
  aggregate.csv; best_spade figure rewritten as thin shared-renderer client.
- Restored from backup: BatchTopK seed-42 metrics → results/karvonen_batch_topk/;
  external/Karvonen_Research + Original_Spade_Repo (broken gitlinks in clone);
  28 GB eval activations → netscratch spade_multiseed/data/activations/.
- Error-bar rendering verified with synthetic 8% std (all 4 panels OK).

## Outcome (completed 7-14-2026)

All 5 seeds x 14 configs trained + evaluated; aggregate CSVs and the three
error-bar figures regenerated and committed. Training took ~7h35m/seed
(4 seeds in parallel, 4 GPUs each); evals ~5 min each on netscratch data.
Two transient NFS FileLock failures in seed-46 evals (HF datasets cache);
fixed via HF_USE_SOFTFILELOCK=1 in both submit scripts + rerun.
Findings: TopK/BatchTopK L0 pinned (std<0.07); classic baselines stable
(std<1.7); default SpaDE wildly seed-sensitive (L0 105±50, frac_recovered
dips to 0.74 at one seed); sweep SpaDEs moderate (L0 std 4-25);
lam8e-8 vs lam1e-7 at gam1e-3 train to bitwise-identical models (lambda
below fp32 resolution in sparsemax scores — same as old 8e-8/9e-8 dup).

## Geometry-plot regeneration (7-15-2026)

All 13 analysis/geometry scripts hardcoded /Users/dcao/Downloads/... paths
(never ported in the reorg); no rendered PNGs existed anywhere. Ported all 13
to repo-relative paths (REPO = parents[3] of each script; outputs to each
script's own dir; sys.path → external/Karvonen_Research + src/*). Staged
seed-42 f1 tensors + checkpoints from the backup into gitignored repo paths.
Substitution (user-approved): the old 8-config sweep's λ=1e-3 row is lost
everywhere (hollow dirs), so SpaDE γ=1e-4/γ=1e-2 inputs use the surviving
λ=1e-4 siblings (results/karvonen_spade_sweep/trainer_4 and trainer_6),
commented at each site. linear_probe.pt was also lost; extracted
analysis/linear_probe/train_linear_probe.py from the notebook and retrained —
layer 5 primary, eval acc 0.9896 (matches the original notebook exactly).
Regenerated all 210 PNGs (5 CPU scripts locally; probe + 8 remaining scripts
in one 11-min GPU job). PNGs stay gitignored per repo convention; the
regenerated JSON/CSV intermediates are committed. plotly + umap-learn
installed into spade_env. Original_Spade_Repo still unrecoverable.

## History (submitted 7-14-2026)

- Smoke test PASSED on attempt 3 (job 30789876) after two fixes it caught:
  (1) sparsity_warmup_steps default 2000 >= smoke steps → now scaled via
  min(2000, steps//2), real runs unchanged; (2) TopK/BatchTopK step-0
  geometric-median assigned b_dec.data before the x device move → b_dec
  stranded on cuda:0 while module on cuda:3 (single-GPU-invisible bug),
  fixed by moving x to self.device first. Cross-seed weight check passed
  for Standard + all SpaDE variants.
- Real chain submitted: train array 30794093 (tasks 43-46 == seeds, gpu:4
  each), eval arrays 30794094-97 (14 tasks each, chained afterok per seed).
- Committed + pushed as 905050a. external/Karvonen_Research now tracked as
  regular files (gitlinks removed). Original_Spade_Repo unrecoverable —
  every local copy is a hollow .git with no objects; need its GitHub URL
  to restore.
- After all evals: `python src/multiseed/aggregate_seeds.py` then regenerate
  the three figures (see run_multiseed.sh tail for commands); commit again.
- Possibly stale now: results/all_configs_combined.csv (old combined CSV,
  superseded by results/multiseed/) — ask user before deleting.
