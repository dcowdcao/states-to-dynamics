# SpaDE vs Karvonen: remaining differences and current results

This project runs on **Karvonen's exact training and evaluation infrastructure**.
Training streams OthelloGPT activations through Karvonen's buffer and trains every
SAE through Karvonen's `trainSAE()` loop; evaluation runs Karvonen's board-state
and loss-recovered metrics. Most of the gaps this document once catalogued are
therefore **closed**.

It records (1) the gaps that were closed, (2) the few genuine differences that
remain, and (3) the current results.

---

## 1. Gaps that were closed

The prior pipeline (historical Run 3) differed from Karvonen on several axes. The
current one matches on all of them:

| Axis | Old Spade (historical Run 3) | Current = Karvonen |
|---|---|---|
| **Activation collection** | Offline pre-extraction of all games to `.pt`, loaded via `DataLoader` | Streams layer-5 `resid_post` on the fly via Karvonen's `NNsightActivationBuffer` (`n_ctxs=1000`, `ctx_len=59`), wrapped in `StandardizedBuffer` (Anthropic Apr-2024 center+scale). No offline extraction for training. |
| **Training loop** | One SAE at a time, hyperparameters read from a CSV row | All 8 SAEs trained **simultaneously on one shared stream** via `trainSAE()`; configs built in code. No CSV. |
| **Dictionary size** | 4096 (8×) only | **8192 (16×)** for all types |
| **Training duration** | ~2,540 steps, 2 epochs | **36,621 steps** (300M tokens / batch 8192), warmup 1000, seed 42, lr 3e-4 (TopK auto-lr ≈ 2.83e-4) |
| **Turn range** | turns 4–56 | **turns 0–59** (`context_length=59`) — all turns, matching Karvonen |
| **Baseline trainers** | p-annealing reimplemented locally | Karvonen's **own trainer classes** used directly (`PAnnealTrainer`, `StandardTrainer`, `GatedSAETrainer`, `GatedAnnealTrainer`, `TopKTrainer`, `BatchTopKTrainer` from `circuits.dictionary_learning`) |
| **Fidelity metric** | Variance explained only | Both **`frac_recovered`** (Loss Recovered — loads OthelloGPT, runs it 3× original/reconstructed/zero-ablated) and variance explained, plus L0, coverage, reconstruction F1 |

Everything above is now shared infrastructure and is **not** a source of
difference.

---

## 2. Genuine remaining differences

### 2a. The novel SpaDE and CosineSpaDE architectures

The research contribution is two SAE architectures absent from Karvonen. Code
lives in `src/spade/saes/`: `spade_base.py` (the shared `SpaDEAutoEncoderBase` +
`SpaDETrainerBase`), `spade.py` (SpaDE), `spade_cosine.py` (CosineSpaDE).

**SpaDE (Sparsemax-Distance)** — default configuration `λ_init=1e-3`, `γ=1e-4`:

```
lambda = softplus(lambda_pre)          # trainable inverse temperature
scores = -lambda * ||x - Ae_i||^2      # squared L2 distance to each encoder prototype
hidden = Sparsemax(scores)             # sparse soft assignment, sums to ~1
x_hat  = Ad @ hidden                   # decode (no bias)
```

Regularizer: distance-weighted L1, `gamma * mean(sum(dist_i^2 * hidden_i))`.

**CosineSpaDE** — default `γ=1.0`, no λ: replaces the squared-distance score with
cosine similarity (scale-invariant), so γ is the only sparsity hyperparameter.
Because the cosine misalignment term is O(1) per atom against SpaDE's O(d_model)
distance, its natural γ scale is ~500× larger than distance-SpaDE's.

Distinguishing properties of both against every Karvonen architecture:

- **Distance/similarity-based encoding** to learned prototypes, not a linear
  projection plus ReLU/TopK/gate.
- **Sparsemax** output — a proper distribution with exact zeros, not thresholded
  activations.
- **No decoder bias and no decoder unit-norm constraint**, by design, since
  sparsemax outputs already sum to ~1.
- SpaDE has a **trainable inverse temperature** λ.

### 2b. Config selection: fixed Pareto points vs a broad sweep

Karvonen sweeps ~240 configs per run (20 sparsity penalties × 2 expansions per
type, plus a wide TopK k-sweep). This pipeline instead trains **8 fixed configs** —
one Pareto-selected operating point per type, chosen from Karvonen's own study —
plus optional SpaDE λ/γ and CosineSpaDE γ sweeps through `spade.training.sweep`.

Consequence: these baselines are **single Karvonen-style operating points, not a
wide sweep**. Read them as "a good Karvonen config for this type at 16×," not as
the best point on the full Pareto frontier.

The 8 types: **PAnneal, Standard, Gated, TopK (k=80), GatedAnneal, SpaDE,
BatchTopK (k=80), CosineSpaDE**. No JumpReLU — it is not trained here.

### 2c. Variance-explained formula nuance

The R² formulas still differ slightly across codebases:

- **This repo** (`spade.evaluation.run_eval`): total sum-of-squares R²,
  `1 − Σ(x − x̂)² / Σ(x − mean)²`.
- **Karvonen:** sums per-dimension variances,
  `1 − Σ_d var(residual_d) / Σ_d var(x_d)`.

For centered data these converge; with a mean offset the total-SS form is the
standard R². The numerical gap is typically small but real.

Board-state metrics use a **train→test assignment split**
(`learn_feature_assignments` on train, then `compute_coverage_from_assignments` /
`compute_board_reconstruction_from_assignments` on test), matching Karvonen. The
older single-dataset functions were removed. Board-state encoding
(`mine/yours/blank`, 4 center squares masked, shared
`games_batch_to_state_stack_mine_yours_BLRRC`) is **consistent** between pipelines
and is not a confound.

---

## 3. Current results

59,000 test samples, seed 42. `frac` = Loss Recovered (fidelity); `var` = variance
explained (R²).

| SAE | L0 | frac_recovered | var_explained | coverage | recon_F1 |
|---|---|---|---|---|---|
| **PAnneal** | 339.5 | 1.000 | 0.999 | 0.664 | 0.541 |
| **Standard** | 1094.9 | 1.000 | 0.998 | 0.587 | 0.568 |
| **Gated** | 49.7 | 0.999 | 0.982 | 0.516 | 0.567 |
| **TopK** (k=80) | 79.9 | 1.000 | 0.987 | 0.726 | 0.684 |
| **GatedAnneal** | 32.3 | 0.998 | 0.971 | 0.548 | 0.500 |
| **BatchTopK** (k=80) | 80.2 | 1.000 | 0.987 | 0.689 | 0.630 |
| **SpaDE** default (λ=1e-3, γ=1e-4) | 1808.2 | 1.000 | 0.995 | 0.570 | 0.515 |
| **SpaDE** best swept (λ=1e-7, γ=0.005) | 316.2 | 0.959 | 0.433 | 0.314 | 0.421 |
| **CosineSpaDE** (γ=1.0) | 173.6 | 0.992 | 0.979 | 0.535 | 0.530 |

CosineSpaDE also reports reconstruction precision 0.489 / recall 0.727; the SpaDE
default reports 0.366 / 0.956; BatchTopK 0.656 / 0.666.

Coverage and recon_F1 for the five `results/main/` baselines (PAnneal, Standard,
Gated, TopK, GatedAnneal) reproduce only to ±0.014: the activations behind those
rows no longer exist (MIGRATION.md open question 14). L0 and var_explained are
stable, and the BatchTopK, CosineSpaDE and SpaDE-default rows reproduce exactly
(`docs/finish_notes/acceptance-eval.md`).

### Reading the results

- **CosineSpaDE** reaches near-baseline fidelity (frac 0.992, var 0.979) at
  L0 173.6 — dramatically better reconstruction than distance-SpaDE at the same
  simplex encoding, with competitive coverage and recon-F1.
- **Distance-SpaDE** at its best *swept* point stays in the characteristic
  low-fidelity soft-assignment regime (var 0.433); its interpretation is "which
  prototype is this closest to?" rather than "which directions are present?".
  That 0.433 is single-seed — the multiseed aggregate for the family is
  0.849 ± 0.278, so quote the cosine-vs-distance contrast per seed or
  qualitatively, never as a clean mean.
- **The SpaDE default row is dense.** λ=1e-3/γ=1e-4 reconstructs almost perfectly
  (var 0.995) at L0 1808, which is denser than Standard ReLU. It is the
  reconstruction-quality point of the family, not a sparsity result. It is also
  not a reproduction of the lost original at the same config and seed (var 0.995
  vs 0.434, L0 1808 vs 961) — the trainer moved underneath it.
- **TopK** is the strongest baseline on board-state metrics (coverage 0.726,
  recon-F1 0.684), with BatchTopK close behind at the same L0.
- Because only one operating point per baseline is trained (§2b), these baselines
  are single Karvonen-style configs, not the argmax over a full sweep.

The main run's original SpaDE entry (L0 15.4, var 0.352) does not appear above: it
is the γ=1e-2 seed-42 checkpoint, whose evaluation is degenerate where seeds 43–46
of the same config give L0 128 and var 0.973, and it is excluded from every study
and from the viewer. See `docs/claims_status.md`. Re-running that row from
`checkpoints/main/trainer_5` will not return it — the `ae.pt` there gives L0 127.76
/ var 0.973 and is not the run its config and metrics describe (MIGRATION.md open
question 15).

### CosineSpaDE γ sweep (19 γ from 0.05 to 1e4, × 5 seeds, mean ± std)

γ is CosineSpaDE's only sparsity knob, and it works — but only above the point
where the sparsity regularizer can outvote the MSE term. The loss is `MSE + γ·reg`
with `reg = mean_b Σ_j (1−cos(x,d_j))·f_j`; sparsemax puts f on the simplex
(`Σ_j f_j = 1`), so reg is a convex combination of misalignments, bounded in [0,2]
regardless of how many atoms are active. It is not an L0 penalty, and raising γ
buys loss *share*, never a bigger penalty. Measured on trained checkpoints:
MSE ≈ 10.7, reg ≈ 0.78, so parity needs **γ ≈ 16**
(`spade.training.sweep_loss_balance`).

L0 is therefore a sigmoid in log γ whose inflection sits at that measured parity
point, running 179 → 20.5 and flooring there: past γ ≈ 500 nothing moves.

| γ | L0 | frac_recovered | coverage | recon_F1 | var_explained |
|---|---|---|---|---|---|
| 0.05 | 179.1 ± 3.0 | 0.991 | **0.544** | 0.529 | 0.979 |
| **1** (default) | 172.7 ± 2.9 | 0.991 | 0.537 | 0.530 | 0.979 |
| 10 | 128.7 ± 3.4 | 0.988 | 0.499 | 0.536 | 0.975 |
| 20 | 98.1 ± 1.9 | 0.984 | 0.474 | **0.547** | 0.972 |
| 50 | 58.9 ± 0.5 | 0.972 | 0.341 | 0.516 | 0.965 |
| 100 | 41.6 ± 0.3 | 0.955 | 0.214 | 0.436 | 0.956 |
| 200 | 29.3 ± 0.2 | 0.903 | 0.110 | 0.341 | 0.938 |
| 500 | 21.5 ± 0.2 | 0.587 | 0.056 | 0.220 | 0.895 |
| 1e4 | 20.6 ± 0.5 | 0.102 | 0.050 | 0.148 | 0.854 |

What the wide range establishes that the original [0.05, 5] grid could not:

- **γ is a real sparsity lever**, contrary to the first reading of the narrow
  grid, which sat entirely in the flat left tail of the sigmoid.
- **CosineSpaDE is strictly dominated at every matched sparsity.** At L0 ≈ 30
  GatedAnneal gets coverage 0.537 against 0.110; at L0 ≈ 50–59 Gated gets 0.523
  against 0.341; at L0 ≈ 80–98 TopK gets 0.736 against 0.474. Sweeping γ moves
  CosineSpaDE *along* the L0 axis but never toward the baseline frontier.
- **Fidelity and variance explained dissociate at the sparse end.** At γ=1e4 the
  SAE still explains 0.854 of the variance while recovering only 0.102 of the
  loss — it reconstructs the activation vector in L2 while destroying the model's
  behaviour. Variance explained alone is not a safe health check for this family.

**The operating point remains γ = 1.0.** The sweep's own write-up recommends γ=20
on reconstruction F₁ (0.547 at L0 98 vs 0.530 at L0 173), and that recommendation
was considered and not adopted: coverage decides here and runs the other way
(0.537 vs 0.474), the rest of the trade is L0 173 → 98 and loss recovered
0.991 → 0.984, and the F₁ peak is thin — 0.011 over γ=10, with seed dispersion
reported for L0 only. Do not restore γ=20 from the sweep README.

Full aggregate: `results/multiseed/aggregate_cosine_sweep.csv`. Figures:
`figures/cosine_spade_sweep/Figure_4_karvonen_extension.png` (the Karvonen 2×2)
and `Figure_cosine_gamma_response.png` (metrics vs γ, with the parity line).
