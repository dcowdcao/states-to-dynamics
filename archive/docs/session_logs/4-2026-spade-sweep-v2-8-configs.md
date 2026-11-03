# April 2026 — SpaDE sweep v2: 8 configs, dead-neuron tracking, norm diagnostics

Historical record, reconstructed from the append-only cluster command log. The
exact submission date is not recorded; this run executes the grid and the
tooling decided in the [4-4-2026](4-4-2026.md) PI session, so it followed it
closely. Its outputs are what is now `results/spade_sweep_8config/`.

**Paths here are as they were** — `$HOME/Spade_SAE_Research`, `shell/slurm/`, a
Mac working copy under `/Users/dcao/Downloads`, scripts run as paths. All of that
is gone; the current workflow is [`../fasrc_commands.md`](../fasrc_commands.md).

## Why

The previous λ values were about four orders of magnitude too small. The
theoretical default from the SpaDE paper is λ ≈ 1/(2·512) = 9.77e-4, so the
sweep moved to λ ∈ {1e-3, 1e-4} crossed with γ ∈ {1e-4, 1e-3, 1e-2, 1e-1} — eight
runs on one shared activation stream.

| Run | lambda_init | gamma_reg |
|---|---|---|
| 1 | 1e-3 | 1e-4 |
| 2 | 1e-3 | 1e-3 |
| 3 | 1e-3 | 1e-2 |
| 4 | 1e-3 | 1e-1 |
| 5 | 1e-4 | 1e-4 |
| 6 | 1e-4 | 1e-3 |
| 7 | 1e-4 | 1e-2 |
| 8 | 1e-4 | 1e-1 |

## What shipped with it

- Sweep defaults moved to the grid above.
- `sparsity_loss` added to the SpaDE W&B losslog (same value as
  `dist_reg_loss`).
- `frac_dead_neurons` tracking added to
  `external/Karvonen_Research/circuits/dictionary_learning/training.py` for
  **all** SAE types, not just SpaDE. Because that edit lives in the vendored
  fork, `external/` had to be uploaded alongside `src/` — a step easy to miss.
- A new 2×2 norm-diagnostic figure: activation norms and weight norms, pre and
  post standardization. Now `spade.diagnostics.norm_diagnostics`.

W&B was watched for `frac_dead_neurons` (should fall over training for every
type), `sparsity_loss`, and `frac_variance_explained` with a target of ≥ 0.9 for
at least some configs.

Training 2–4 h for the eight configs, eval plus figures 30–60 min.

## What became of it

The λ=1e-3 rows are the ones that mattered. `SpaDE-lam1e-03-gam1e-04` is the
**current default SpaDE configuration**, and its combined CSV is the only
surviving record of that single-seed point — `spade.figures.figure_all_configs_combined`
reads it directly, which is why `results/spade_sweep_8config/` is a live input
rather than an archived sweep.

Two later consequences worth knowing when reading this log:

- The λ=1e-3 checkpoints themselves were lost in a later reorg; the surviving
  λ=1e-4 siblings were substituted in the geometry figures, commented at each
  site. The λ=1e-3 γ=1e-4 point was eventually retrained, and that retrain is
  **not a reproduction** of this run: same config, same seed, VE 0.995 against
  0.434 and L0 1808 against 961. The trainer moved underneath it.
- γ=1e-2 became the shipped default for a while and has since been replaced by
  γ=1e-4. Its seed-42 checkpoint is degenerate and is now excluded from every
  study and from the viewer. See [`../claims_status.md`](../claims_status.md).
