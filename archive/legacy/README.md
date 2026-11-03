# archive/

Superseded material, kept for provenance. Nothing here is read by the pipeline and nothing
here is a current result. Entries are named after the live thing they were split from, so
what did not go to `results/lambda_sweep/` is in `archive/lambda_sweep/`.

| what | why it is here | use instead |
|---|---|---|
| `spade_sweep_collapsed/` | four λ=1e-8/1e-7 SpaDE configs whose runs collapsed — VE ≈ −3e-05, L0 = 8192, no weights ever saved | `checkpoints/spade_sweep/` (λ=1e-4, trainer_4 and 6) |
| `lambda_sweep/` | training configs for λ ∈ {5e-8, 8e-8, 9e-8, 1e-7}; that region either collapses or trains bitwise-identically, so it was not swept further | `results/lambda_sweep/all_results.csv` — still a live seed-42 input |
| `spade_sweep_8config/` | training configs for the λ ∈ {1e-3, 1e-4} × γ ∈ {1e-4 … 1e-1} grid, superseded by retraining its chosen best on its own | `checkpoints/spade_revival/` (λ=1e-3, γ=1e-4), the current SpaDE default |
| `main_mac_smoke_eval/` | the Mac/`mps` smoke evaluation of `checkpoints/main/trainer_0–5`, N=520 — the run whose `config.json`s are MIGRATION.md question 3 | `results/main/all_results.csv` (N=59000) |
| `Wandb_Export.csv` | 2026-03-29 W&B export: parameters and final state for every training run. No reader in `src/`; it is the wandb record MIGRATION.md question 15 cites | each checkpoint's own `config.json` |
| `docs/Refactor_Cleanup_Progress.md` | the July 2026 refactor and disk cleanup, over a workspace layout that no longer exists | `docs/pipeline.md` §5 for the λ/γ fact it establishes |

Three sweep generations reused the same `autoencoders_spade_sweep/` save-dir name, so the
configs archived here are the only record of which hyperparameters produced which
`trainer_N` row in the matching `results/` CSV. Read them together.

## Deliberately not here

The frozen source repo `../Spade_Research_David_Sumedh_Ba` still holds all of it.

- **Images.** Every superseded figure — three sweep `Figure_4_*.png`,
  `best_spade_lam1e-07_gam5e-03/`, the six `W&B Results/` chart exports (~8 MB) — waits on
  MIGRATION.md question 10, the tracked-figure policy.
- **The source repo's own `archive/`**: `runs/Run_{2,3}_Results/`, `old_md/`,
  `gpu_runs_plans/`, `run_notes/` — 127 MB, 93 MB of it already in history. MIGRATION.md
  question 5. If it is carried in, its markdown joins `docs/` here.
- **SLURM stdout/stderr from the sweep evals** — 1.2 MB of progress-bar spam over numbers
  the eval CSVs already carry.
