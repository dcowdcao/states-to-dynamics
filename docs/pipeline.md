# Pipeline

End-to-end reference for the 8-type SAE pipeline: model and data, standardization,
all 8 SAE architectures with equations and hyperparameters, every eval metric with
its formula, the results tables, and the Pareto analysis.

Stages (each maps to a `spade.*` subpackage):

1. **extract** — `spade.extraction` → `data/activations/` (acts + board labels)
2. **train** — `spade.training.train_<type>` → `checkpoints/<experiment>/`
3. **eval** — `spade.evaluation.run_eval` → `results/<experiment>/all_results.csv`
4. **aggregate** (multi-seed) — `spade.multiseed.aggregate_seeds` → `results/multiseed/`
5. **figures** — `spade.figures.*` → `figures/`

Plus `spade.diagnostics` for the SpaDE norm studies.

Standardization is part of the contract, not a preprocessing detail: SAEs are
trained on `scale*(x - mean)`, so anything that encodes without applying it is
silently wrong rather than approximately right. The parameters are written beside
the checkpoints at save-dir level and re-applied at load, at export, and again in
the browser viewer.

Where a number here conflicts with `docs/claims_status.md`, that file wins. Read
it before quoting anything from this document in a paper, caption or abstract.

---

## 1. The model: OthelloGPT

- **Model:** `Baidicoot/Othello-GPT-Transformer-Lens` (pre-trained, from HuggingFace)
- **Type:** decoder-only GPT-style transformer (Li et al., arXiv:2210.13382)
- **Layers:** 8 transformer blocks
- **Hidden dimension (d_model):** 512
- **Attention heads:** 8 (head dim 64)
- **MLP width:** 2048 (4× d_model)
- **Activation:** GELU
- **Context length:** 59 tokens (one per move in a game)
- **Vocabulary:** 61 tokens (60 playable squares + 1 special token)

OthelloGPT is trained to **predict the next legal move**. It receives **only a
sequence of move tokens** — never a board image or an explicit board
representation. To predict legal moves it must internally represent the board:
which squares hold which pieces, whose turn it is, which moves are legal. The
point of this project is to use SAEs to extract and interpret that internal
representation.

## 2. Data

**Source:** `taufeeque/othellogpt` from HuggingFace — 100,000 Othello games.

**Activation extraction:**

- **Layer:** 5 (0-indexed), hook point `blocks.5.hook_resid_post` — the residual
  stream after layer 5.
- **Context length:** 59 tokens per game.
- **Buffer:** `NNsightActivationBuffer` with `n_ctxs=1000`,
  `refresh_batch_size=64`, `out_batch_size=8192`.

For evaluation, ground-truth labels come from replaying the game:

- **Board state:** 8×8×3 one-hot tensor (opponent piece / empty / my piece), in
  the "mine/yours" perspective, which flips each turn so the current player is
  always "mine".
- **Valid moves:** 8×8 binary mask of the legal moves at that position.

Two data caveats live in `docs/claims_status.md` and matter to anyone reusing
these artifacts: `data/activations/metadata.json` records a 100-game smoke test
rather than the real extraction, and the committed linear-probe board labels have
53 mislabeled rows in 10,000 because parity was taken from move index rather than
engine replay. Any new probe derives parity from replay.

## 3. Activation standardization

Raw activations are **standardized** before any SAE sees them, using the Anthropic
April-2024 convention. Implemented by `StandardizedBuffer` in
`spade.training.core`.

```
act_standardized = scale * (act - mean)
```

- **mean** = E[act], the empirical mean over the first 100 calibration batches.
- **scale** = sqrt(d_model / E[‖act − mean‖²]), so E[‖act_standardized‖²] = d_model.

This gives **unit variance per dimension on average** (E[‖x'‖²] = 512), **not**
unit norm. The expected norm of a standardized activation is √512 ≈ 22.6 — worth
holding onto, because position 0 is an attention sink whose norm is ~300, and
every trajectory study excludes it for that reason.

### Computed values

| Parameter | Baselines run | SpaDE sweep runs |
|---|---|---|
| scale | 0.5064 | 0.5058 |
| d_model | 512 | 512 |

The ~0.1% difference comes from different calibration batches off the streaming
buffer; the procedure is identical. Calibration is also mildly seed-dependent
(0.505683 at seed 42, 0.505401 at seed 46).

### How it is applied

- **Training:** `StandardizedBuffer` wraps the raw `NNsightActivationBuffer`. It
  calibrates on the first 100 batches, caches them, then yields every subsequent
  batch with `scale * (act − mean)` applied. `trainSAE()` is called with
  `normalize_activations=False`, since the buffer already did it.
- **Evaluation:** the `StandardizedDictionary` wrapper in
  `spade.evaluation.karvonen_adapter` loads `standardization_params.json` and
  applies the same transform before encoding, then inverts it after decoding.
- **Saved metadata:** the 512-dim mean vector and the scale factor go to
  `standardization_params.json` beside each trained SAE. It sits at the
  **save-dir** level, not per trainer, and the loader's parent-directory fallback
  depends on that — any reshuffle of `checkpoints/` must move it with its trainers.

This matches the Karvonen codebase's `standardise_data` exactly:

```python
train_mean = torch.mean(training_data.data, dim=0, keepdims=True)
training_data.data = training_data.data - train_mean
expected_sq_norms = torch.mean(torch.sum(torch.square(training_data.data), dim=-1), dim=0)
scaling = torch.sqrt(data_dim * 1 / expected_sq_norms)
training_data.data = scaling * training_data.data
```

## 4. SAE architectures

Eight SAE types are trained and compared, all through a single `trainSAE()` loop
over one shared standardized activation stream: five Karvonen baselines (§4a–4e),
BatchTopK (§4g), and the two novel prototype-distance variants SpaDE (§4f) and
CosineSpaDE (§4h). Every type uses **width 8192 (16× expansion)** on
**d_model = 512**.

### 4a. p-Anneal (PAnnealTrainer)

Standard ReLU autoencoder with **p-annealing** on the sparsity penalty: the
sparsity function is Lp^p, starting at p=1.0 (plain L1) and annealing to p=0.2,
which reduces the penalty on small-but-nonzero activations and so encourages
sparser solutions. Anneal starts at step 10,000; `sparsity_queue_length=10`,
`n_sparsity_updates=10`. **initial_sparsity_penalty = 0.042105**.

### 4b. Standard ReLU (StandardTrainer)

```
x_centered = x - b_dec
hidden     = ReLU(W_enc @ x_centered + b_enc)
x_hat      = W_dec @ hidden + b_dec
```

Regularizer: L1 on the hidden activations. **l1_penalty = 0.058684**. The decoder
uses `ConstrainedAdam`, which projects gradients perpendicular to the column
directions and renormalizes them to unit norm.

### 4c. Gated SAE (GatedSAETrainer)

```
x_centered = x - b_dec
x_enc      = W_enc @ x_centered
gate       = ReLU(x_enc + b_gate)
magnitude  = ReLU(exp(r_mag) * x_enc + b_mag)
hidden     = (gate > 0).float() * magnitude
x_hat      = W_dec @ hidden + b_dec
```

Separate gating and magnitude paths, plus an auxiliary reconstruction loss with a
detached decoder. **l1_penalty = 1.368421**. Optimizer betas (0.0, 0.999).

### 4d. TopK SAE (TopKTrainer)

```
x_centered = x - b_dec
post_relu  = ReLU(W_enc @ x_centered + b_enc)
hidden     = TopK(post_relu, k)      # zero out all but the top k
x_hat      = W_dec @ hidden + b_dec
```

**k = 80** — exactly 80 features fire per input. An auxiliary loss revives dead
features (`auxk_alpha = 1/32`). Learning rate is auto-computed:
2e-4 / sqrt(8192/16384) ≈ 2.83e-4.

### 4e. Gated + p-Anneal (GatedAnnealTrainer)

The gated architecture with the same p-annealing schedule.
**initial_sparsity_penalty = 0.955263**.

### 4f. SpaDE (Sparsemax-Distance) — the novel contribution

```
lambda = softplus(lambda_pre)          # trainable inverse temperature, forced positive
scores = -lambda * ||x - Ae_i||^2      # negative squared L2 distance to each encoder prototype
hidden = Sparsemax(scores)             # sparse probability distribution
x_hat  = Ad @ hidden                   # decode (NO decoder bias)
```

Instead of a linear encoder plus a nonlinearity, SpaDE computes the **distance
from the input to each encoder prototype** and pushes that through sparsemax,
which yields a sparse probability vector summing to ~1.

- **λ** (inverse temperature) is a single trainable scalar controlling how sharp
  the sparsemax distribution is — higher λ is sparser.
- **Regularizer:** distance-weighted L1, `gamma * mean(sum(dist_i² * h_i))`.
  Features are penalized more when the input is far from their prototype, which
  pushes each feature to fire only for nearby inputs.
- **No decoder normalization**, unlike every other type here.

### 4g. BatchTopK (BatchTopKTrainer)

The batch-wise variant of TopK from Karvonen's `dictionary_learning`. Rather than
keeping the top-k activations per sample, it keeps the top-(k × batch_size)
across the whole batch during training, then applies a single EMA-calibrated
global threshold at inference. **k = 80** per-sample average,
`auxk_alpha = 1/32`.

### 4h. CosineSpaDE (CosineSpaDETrainer)

A prototype SAE like SpaDE, but scoring atoms by **cosine similarity** instead of
negative squared Euclidean distance:

```
scores = cos(x, Ae_i)                  # cosine similarity to each encoder prototype
hidden = Sparsemax(scores)             # sparse simplex weights
x_hat  = Ad @ hidden                   # decode (no bias)
```

- **No λ:** cosine is scale-invariant, so γ is the only sparsity knob.
- **Regularizer:** cosine-misalignment-weighted L1,
  `gamma * mean(sum((1 - cos(x, Ad_j)) * h_j))`.
- Designed to sidestep SpaDE's distance-norm pathology, where large-norm
  activations inflate squared distances. Because the misalignment term is O(1) per
  atom against SpaDE's O(d_model) distance, its natural γ scale is ~500× larger.

SpaDE and CosineSpaDE share their sparsemax and no-bias-decoder plumbing through
`spade.saes.spade_base`. The class names `SpaDEAutoEncoder` and
`CosineSpaDEAutoEncoder` are load-bearing — the evaluation adapter dispatches on
`trainer_class` strings and the viewer exporter keys on `type(ae).__name__` —
and `CosineSpaDE` must be matched before `SpaDE` (as `BatchTopK` must before
`TopK`), because those are substring collisions and a reordering silently loads
the wrong class.

## 5. Hyperparameters

### Shared across all types

- **Learning rate:** 3e-4 (TopK: 2.83e-4)
- **Dictionary size:** 8192 (16× expansion of d_model=512)
- **Training steps:** 36,621
- **Warmup steps:** 1000
- **Seed:** 42 for the headline run; 43–46 add the error bars (§7d)
- **Standardization:** applied (§3)
- **Device:** CUDA (A100 / H100)

### Per-type hyperparameters

Selected from the Pareto-optimal points of Karvonen's own sweep
([karvonen_reference.md](karvonen_reference.md) §10.4):

| SAE type | Key hyperparameter | Value | Karvonen sweep origin |
|---|---|---|---|
| p-Anneal | initial_sparsity_penalty | 0.042105 | trainer33, 16× |
| Standard ReLU | l1_penalty | 0.058684 | trainer30-v2, 16× |
| Gated | l1_penalty | 1.368421 | trainer27-v2, 16× |
| TopK | k | 80 | trainer_47, 16× |
| Gated+p-Anneal | initial_sparsity_penalty | 0.955263 | trainer38, 16× |
| BatchTopK | k | 80 | Karvonen `dictionary_learning` (auxk_alpha = 1/32) |

### SpaDE-family defaults

**The default SpaDE configuration is λ_init = 1e-3, γ = 1e-4**, with weights under
`checkpoints/spade_revival/`. It replaces the earlier γ=1e-2 default, whose seed-42
checkpoint is degenerate and is now excluded from every study and from the viewer,
in one place (`spade.spaces`).

**CosineSpaDE's default is γ = 1.0** (no λ), kept deliberately over the γ-sweep's
own recommendation of γ=20: the sweep selected on reconstruction F₁ (0.547 vs
0.530), but coverage runs the other way (0.537 vs 0.474) and is the deciding
metric here. Do not change it back from the sweep write-up.

Three provenance traps around these numbers:

- **Checkpoint `config.json` files record the wrong run length and device.** All six
  `checkpoints/main/trainer_*/config.json` carry `steps: 400` and `device: mps` from
  a Mac smoke config; the runs are 36,621 steps on `cuda:0`. **λ and γ in those
  files are correct.** In particular the SpaDE one's `lambda_init 9.77e-05,
  gamma_reg 0.2` is a real configuration that really ran —
  `SpaDETrainer-othello_trainer_5`, finished 2026-03-26, degenerate — not a label
  read off a stale file. `docs/claims_status.md` attributes the same evaluation
  (L0 15.4, VE 0.352) to the γ=1e-2 checkpoint, and it wins.
- **`checkpoints/main/trainer_5/ae.pt` is not the run its config, its wandb record
  and its eval row describe.** Those three agree on L0 15.43 / VE ≈ 0.36; the file
  on disk reruns at L0 127.76 / VE 0.973. trainer_0–4 all match their records.
  What the weights are is MIGRATION.md open question 15
  ([acceptance-eval.md](finish_notes/acceptance-eval.md)), so any result that
  loaded them measured an unidentified model.
- **The revived `lam1e-03 gam1e-04` checkpoint is not a reproduction** of the lost
  original at the same config and seed: VE 0.995 vs 0.434, L0 1808 vs 961. The
  trainer moved underneath it. Label it as that config *under the current
  trainer*, never as a replication.

### The distance-SpaDE λ/γ exploration (history)

13 unique distance-SpaDE configs were trained across three phases, all at
dict_size 8192, lr 3e-4, 36,621 steps, seed 42. This exploration produced the
"best swept" point quoted in §7b. It is not the current default.

**Phase 1 — initial configs**

| # | λ_init | γ | L0 | Status |
|---|---|---|---|---|
| 1 | 9.77e-5 | 0.2 | 15.4 | finished — heavily over-regularized; the weights now in `checkpoints/main/trainer_5` are not this run (§5) |
| 2 | 5e-6 | 0.1 | 31.5 | finished — never Karvonen-evaluated |

**Phase 2 — λ sweep at fixed γ=0.001**

| # | λ_init | γ | L0 | Status |
|---|---|---|---|---|
| 3 | 1e-8 | 0.001 | 8192 | collapsed (every feature fires, negative fidelity) |
| 4 | 1e-8 | 0.01 | 8192 | collapsed |
| 5 | 5e-8 | 0.001 | 8192 | collapsed |
| 6 | 8e-8 | 0.001 | 899.5 | finished |
| 7 | 9e-8 | 0.001 | 899.5 | finished — identical to #6 |

**Phase 3 — γ sweep at fixed λ=1e-7**

| # | λ_init | γ (actual) | L0 | Status |
|---|---|---|---|---|
| 8 | 1e-7 | 0.001 | 898.4 | finished |
| 9 | 1e-7 | 0.005 | 316.8 | **finished — best of this exploration** |
| 10 | 1e-7 | 0.008 | 176.4 | finished |
| 11 | 1e-7 | 0.01 | 164.4 | finished |
| 12 | 1e-7 | 0.015 | 141.1 | finished |
| 13 | 1e-7 | 0.025 | 100.9 | finished |

**Do not sweep λ ≤ 1e-7 again.** That whole region sits below fp32 sparsemax
resolution: configs there either collapse outright (λ ≤ 5e-8 — L0 8192, negative
fidelity, near-zero F1) or train **bitwise-identically** to each other. 8e-8 and
9e-8 are one model, and so are 8e-8 and 1e-7 at γ=1e-3 when re-run under the
multiseed harness. The λ values that matter are 1e-3 and 1e-4.

**Naming discrepancies in the γ sweep.** Two runs carry misleading W&B names: the
run labeled `gam1e-02` (trainer_2) actually has γ=0.015, and `gam3e-02`
(trainer_3) actually has γ=0.025. The γ values in the W&B config fields are
authoritative. This is one instance of a general hazard — sweep dirs restart
`trainer_N` at 0, three different directories hold a `trainer_0` with three
different λ/γ grids, and four hold an `all_results.csv`. Merging by index
overwrites.

**Config #2** (λ=5e-6, γ=0.1, L0 31.5) finished training but was never evaluated.
Its L0 is comparable to Gated+p-Anneal's 32.3, which makes it a potentially
interesting comparison point that remains unmeasured.

## 6. Evaluation metrics — exactly how each is computed

### 6a. L0 (sparsity)

Mean number of active (non-zero) features per input.

```python
L0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean()
```

Count the features above 1e-8 per sample, average over samples. Lower is better —
fewer features per input is a more interpretable SAE.

### 6b. Loss recovered and variance explained — two distinct metrics

The pipeline computes **two** reconstruction-quality metrics. Do not conflate them.

**Loss recovered (`frac_recovered`)** — the fraction of OthelloGPT's cross-entropy
loss preserved when the SAE reconstruction is spliced back in. Computed by running
the *live model* three times (original, SAE-reconstructed, zero-ablated):

```python
frac_recovered = (loss_recon - loss_zero) / (loss_orig - loss_zero)
```

**Variance explained (R²)** — reconstruction quality in activation space only, no
model pass:

```python
variance_explained = 1 - sum((x - x_hat)**2) / sum((x - mean(x))**2)
```

Both: 1.0 is perfect, higher is better. **They diverge sharply.** Distance-SpaDE
reaches `frac_recovered ≈ 0.96` at `variance_explained ≈ 0.43` — the
reconstruction preserves the model's *behaviour* far better than the raw
activation *variance*. The dissociation runs the other way too: at the extreme
sparse end of the CosineSpaDE γ sweep, variance explained holds 0.854 while loss
recovered falls to 0.102. Neither metric is a health check on its own.

Our R² is the total sum-of-squares form, `1 − Σ(x − x̂)² / Σ(x − mean)²`
(`spade.evaluation.run_eval`); Karvonen sums per-dimension variances instead. For
centered data the two converge; with a mean offset the total-SS form is the
standard R². The gap is small but real.

### 6c. Coverage

Average best-achievable F1 across all board-state labels — does each board concept
have at least one SAE feature that can detect it?

Computed in `spade.evaluation.karvonen_adapter`:

1. Start from `f1_TFRRC`, shape [T=10, F=8192, 8, 8, C=3]: F1 for every
   (threshold, feature, square, class).
2. Max over thresholds → `f1_FRRC`, [F, 8, 8, C].
3. Max over features → `best_f1_RRC`, [8, 8, C].
4. Mask the 4 center squares (d4, d5, e4, e5 — they never change, so predicting
   them is trivial).
5. Average over the remaining 60 squares × 3 classes = **180 labels**.

Higher is better.

### 6d. Per-feature F1 (the foundational computation)

Treat each SAE feature as a binary classifier for each board-state label. For each
feature `f` and each threshold fraction `t` from
[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]:

1. **Binarize:** prediction = 1 if `feature_acts[f] > t × max(feature_acts[f])`.
2. Per square (r,c) and class: TP = fires and label true; FP = fires and label
   false; FN = does not fire and label true.
3. `F1 = 2·Precision·Recall / (Precision + Recall)`.

This produces the 5-D tensor `f1_TFRRC` of shape [10, 8192, 8, 8, 3].

### 6e. Reconstruction F1

Can the features *collectively* reconstruct the full board? This tests
composition, not individual feature quality.

1. **Assign each feature its single best label:** for each of the 8192 features,
   take the (threshold, row, col, class) with the highest F1 in `f1_TFRRC`. That
   is the feature's role.
2. **Determine active features per test sample:** active if the activation exceeds
   its assigned threshold × its max activation.
3. **Vote:** each active feature votes for its assigned (square, class) cell,
   accumulated with `scatter_add`.
4. **Binarize:** a cell is predicted True if it received ≥ 1 vote.
5. **Score:** TP/FP/FN per cell, then precision, recall and F1 per cell,
   **macro-averaged** over the 60 valid squares × 3 classes.

Higher is better.

**Train/test split.** Each feature's assignment is learned on a *training* split
via `learn_feature_assignments`; coverage and reconstruction F1 are then evaluated
on a held-out *test* split via `compute_coverage_from_assignments` and
`compute_board_reconstruction_from_assignments`. This keeps the feature-labeling
step from overfitting the evaluation set. The older single-dataset
`compute_coverage` / `compute_board_reconstruction` functions were deleted.

## 7. Results

59,000 test samples (1,000 games × 59 turns), 10 threshold levels.

### 7a. The full 8-type comparison, seed 42

Ordered by reconstruction F1.

| SAE type | L0 | Loss recovered | Var explained | Coverage | Recon F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| **TopK (k=80)** | 79.9 | 0.9996 | 0.9867 | **0.7265** | **0.6841** | 0.7174 | 0.7047 |
| BatchTopK (k=80) | 80.2 | 0.9999 | 0.9871 | 0.6889 | 0.6300 | 0.6563 | 0.6660 |
| Standard ReLU | 1094.9 | 0.9999 | 0.9978 | 0.5872 | 0.5679 | 0.4243 | 0.9514 |
| Gated | 49.7 | 0.9989 | 0.9824 | 0.5160 | 0.5673 | 0.6790 | 0.5395 |
| p-Anneal | 339.5 | 1.0000 | 0.9986 | 0.6641 | 0.5410 | 0.5608 | 0.6794 |
| CosineSpaDE (γ=1.0) | 173.6 | 0.9920 | 0.9789 | 0.5346 | 0.5299 | 0.4893 | 0.7271 |
| SpaDE (λ=1e-3, γ=1e-4) | 1808.2 | 0.9999 | 0.9952 | 0.5697 | 0.5145 | 0.3661 | 0.9559 |
| Gated+p-Anneal | 32.3 | 0.9981 | 0.9708 | 0.5479 | 0.5004 | 0.7470 | 0.4268 |

The SpaDE row is the current default configuration
(`checkpoints/spade_revival/`, `results/spade_revival/`). Its L0 of 1808 is denser
than Standard ReLU, so it is a reconstruction-quality point, not a sparsity result.

**Coverage and reconstruction F1 in the five `results/main/` rows** — p-Anneal,
Standard ReLU, Gated, TopK, Gated+p-Anneal — reproduce only to ±0.014, because the
activation set they were evaluated on no longer exists (MIGRATION.md open question
14). L0 and variance explained are stable across the two sets. The BatchTopK,
CosineSpaDE and SpaDE rows come from evals whose activations survive and reproduce
to 0.000000 ([acceptance-eval.md](finish_notes/acceptance-eval.md)).

The main run's original SpaDE entry (L0 15.4, loss recovered 0.597, coverage
0.061, VE 0.352) is **excluded everywhere**: that seed-42 evaluation is degenerate
where seeds 43–46 of the same config give L0 128 and VE 0.973. Any conclusion that
rested on it — in particular the geometry family's "SpaDE is broken for cell-state
mapping" — has to be re-run on the current set or retired before it is quoted.
Note what the exclusion does *not* settle: the `ae.pt` now in
`checkpoints/main/trainer_5` does not reproduce that degenerate row (§5, open
question 15).

### 7b. Distance-SpaDE at its best swept point (λ=1e-7, γ=0.005)

| Metric | Value |
|---|---|
| L0 | 316.2 |
| Loss recovered | 0.959 |
| Variance explained | 0.4325 |
| Coverage | 0.314 |
| Reconstruction F1 | 0.421 |
| Recon precision | 0.343 |
| Recon recall | 0.747 |

**VE ≈ 0.43 is single-seed.** The multiseed aggregate for the distance-SpaDE
family is 0.849 ± 0.278 (§7d). Quote the cosine-vs-distance contrast per seed or
qualitatively, never as a clean mean.

### 7c. Selecting the best distance-SpaDE config

Of the 13 configs, 3 collapsed (L0 = 8192), 1 was never evaluated (#2), and 1 was
a duplicate (#7 = #6). The 7 remaining functional configs form a monotonic
trade-off controlled by γ at fixed λ=1e-7:

| γ | L0 | Loss rec | Coverage | F1 |
|---|---|---|---|---|
| 0.001 | 898 | 0.966 | 0.422 | 0.480 |
| **0.005** | **317** | **0.959** | **0.314** | **0.421** |
| 0.008 | 176 | 0.952 | 0.247 | 0.447 |
| 0.01 | 164 | 0.950 | 0.224 | 0.449 |
| 0.015 | 141 | 0.943 | 0.203 | 0.453 |
| 0.025 | 101 | 0.921 | 0.168 | 0.438 |

γ=0.001 has the highest raw metrics but at an L0 near Standard ReLU's 1095, which
demonstrates nothing about sparsity. γ=0.005 was selected because its L0 of 317
sits squarely in the baseline range (closest to p-Anneal's 340 — neither
suspiciously sparse nor suspiciously dense), it has the highest loss recovered
among configs with a reasonable L0, and the highest coverage among them. Its
reconstruction F1 is only marginally below the sparser configs, and the gains in
loss recovered (+0.7–1.6pp) and coverage (+6.7–9.0pp) compensate.

### 7d. Multiseed aggregate (seeds 42–46, mean ± std, n = 5)

Every error bar in the headline figures comes from here
(`results/multiseed/aggregate_baseline.csv`).

| SAE type | L0 | Loss recovered | Coverage | Recon F1 | Var explained |
|---|---|---|---|---|---|
| PAnneal | 338.6 ± 0.7 | 1.0000 | 0.6696 ± 0.0095 | 0.5439 | 0.9986 ± 0.0000 |
| Standard | 1095.2 ± 1.7 | 0.9999 | 0.5838 ± 0.0033 | 0.5677 | 0.9978 ± 0.0000 |
| Gated | 50.3 ± 0.6 | 0.9991 | 0.5228 ± 0.0101 | 0.5710 | 0.9823 ± 0.0001 |
| TopK | 79.9 ± 0.0 | 0.9997 | 0.7363 ± 0.0117 | 0.6874 | 0.9869 ± 0.0001 |
| GatedAnneal | 31.0 ± 1.0 | 0.9972 | 0.5374 ± 0.0079 | 0.4929 | 0.9695 ± 0.0011 |
| BatchTopK | 80.2 ± 0.1 | 0.9998 | 0.6854 ± 0.0136 | 0.6348 | 0.9870 ± 0.0001 |
| CosineSpaDE | 173.3 ± 2.6 | 0.9913 | 0.5365 ± 0.0040 | 0.5290 | 0.9787 ± 0.0002 |
| SpaDE (γ=1e-2) | 105.5 ± 50.3 | 0.9158 | 0.1793 ± 0.0660 | 0.4261 | 0.8489 ± 0.2775 |

TopK and BatchTopK have L0 pinned by construction (std < 0.1); the classic
baselines are stable (std < 1.7). The SpaDE γ=1e-2 row is the seed-sensitive one —
L0 105 ± 50, loss recovered dipping to 0.74 at one seed, VE 0.849 ± 0.278 — and
its seed-42 member is the excluded checkpoint. That dispersion is why the default
moved off γ=1e-2.

### 7e. CosineSpaDE γ sweep (19 γ from 0.05 to 1e4, × 5 seeds)

γ is CosineSpaDE's only sparsity knob, and it works — but only above the point
where the sparsity regularizer can outvote the MSE term. The loss is `MSE + γ·reg`
with `reg = mean_b Σ_j (1 − cos(x, d_j))·f_j`; sparsemax puts f on the simplex
(`Σ_j f_j = 1`), so reg is a convex combination of misalignments, bounded in
[0, 2] no matter how many atoms are active. It is not an L0 penalty, and raising γ
buys loss *share*, never a bigger penalty. Measured on trained checkpoints
(`spade.training.sweep_loss_balance`): MSE ≈ 10.7, reg ≈ 0.78, so parity needs
**γ ≈ 16**. L0 is a sigmoid in log γ inflecting exactly there, running 179 → 20.5
and then flooring: past γ ≈ 500 nothing moves.

| γ | L0 | Loss recovered | Coverage | Recon F1 | Var explained |
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

Two things the wide range establishes that the original [0.05, 5] grid could not:

- **CosineSpaDE is strictly dominated at every matched sparsity.** At L0 ≈ 30
  GatedAnneal reaches coverage 0.537 against 0.110; at L0 ≈ 50–59 Gated reaches
  0.523 against 0.341; at L0 ≈ 80–98 TopK reaches 0.736 against 0.474. Sweeping γ
  moves CosineSpaDE *along* the L0 axis, never toward the frontier.
- **Fidelity and variance explained dissociate at the sparse end**, as above.

The operating point stays **γ = 1.0**. The sweep's own write-up recommends γ=20 on
reconstruction F₁ (0.547 vs 0.530), the one metric with an interior maximum; that
recommendation was not adopted, because coverage decides and runs the other way
(0.537 vs 0.474), and the F₁ peak is thin — 0.011 over γ=10, with seed dispersion
reported for L0 only.

Aggregate: `results/multiseed/aggregate_cosine_sweep.csv`. Figures:
`figures/cosine_spade_sweep/Figure_4_karvonen_extension.png` and
`Figure_cosine_gamma_response.png`.

### 7f. Distance-SpaDE vs p-Anneal at comparable L0

The most informative comparison is at similar sparsity: distance-SpaDE
(λ=1e-7, γ=0.005) at L0 ≈ 317 against p-Anneal at L0 ≈ 340.

| Metric | SpaDE (L0=317) | p-Anneal (L0=340) | Delta |
|---|---|---|---|
| Loss recovered | 0.959 | 1.000 | **−4.1pp** |
| Coverage | 0.314 | 0.664 | **−35.0pp** |
| Reconstruction F1 | 0.421 | 0.541 | **−12.0pp** |
| Recon precision | 0.343 | 0.561 | −21.8pp |
| Recon recall | 0.747 | 0.679 | **+6.8pp** |

Distance-SpaDE underperforms on loss recovered, coverage and F1 but **beats
p-Anneal on recall**: when a SpaDE feature fires it captures more of the true
positives. The gap is largest on coverage, meaning many board concepts have no
dedicated feature even though the features that exist fire reliably.

### 7g. Weighted Pareto analysis

To compare SAE quality beyond eyeballing the 4-plot figure, a weighted score
combines the four Karvonen metrics, with weights reflecting their importance for
interpretability: loss recovered 25% (necessary but not sufficient — nearly
everything scores well), coverage 30% (the most discriminating metric), recon F1
30% (does the dictionary compose?), and an L0 penalty of 15% normalized against
the baseline range.

| SAE type | L0 | Loss rec (25%) | Coverage (30%) | Recon F1 (30%) | L0 penalty (15%) | **Score** |
|---|---|---|---|---|---|---|
| **TopK** | 79.9 | 0.250 | 0.218 | 0.205 | 0.136 | **0.809** |
| p-Anneal | 339.5 | 0.250 | 0.199 | 0.162 | 0.107 | **0.718** |
| Gated | 49.7 | 0.250 | 0.155 | 0.170 | 0.140 | **0.714** |
| Gated+p-Anneal | 32.3 | 0.249 | 0.164 | 0.150 | 0.142 | **0.706** |
| Standard ReLU | 1094.9 | 0.250 | 0.176 | 0.170 | 0.000 | **0.596** |
| SpaDE (best swept) | 316.2 | 0.240 | 0.094 | 0.126 | 0.110 | **0.570** |

**TopK is the clear winner** — best coverage (0.7265), best reconstruction F1
(0.6841), near-perfect loss recovery (0.9996) at a moderate L0 of 80, well inside
the interpretable range. The best distance-SpaDE config scores lowest, driven by
the coverage gap. This table is single-seed and predates the BatchTopK and
CosineSpaDE evaluations; it ranks the six SAEs in it and nothing more.

### 7h. Observations

1. **TopK dominates the Pareto frontier.** At L0=80 it has the highest coverage
   and reconstruction F1 of any SAE tested, and it is the only one that excels on
   all four metrics at once. Its fixed-k architecture suits OthelloGPT's structure.
2. **P-annealing substantially improves both the ReLU and Gated architectures.**
   Gated (F1 0.567) against Gated+p-Anneal (0.500): p-annealing trades some F1 for
   a much lower L0 (50 → 32). For ReLU, p-Anneal at L0 340 is vastly sparser than
   raw Standard ReLU at 1095 with comparable coverage.
3. **Distance-SpaDE's coverage deficit is the critical weakness.** At L0 ≈ 317 it
   covers 31.4% of board-state labels against p-Anneal's 66.4% at similar
   sparsity — roughly two-thirds of (square, class) pairs have no high-quality
   dedicated feature.
4. **Distance-SpaDE has high recall and low precision.** Recall 0.747 exceeds
   every baseline while precision 0.343 is the lowest: features fire broadly,
   catching true positives along with many false ones. Sparse in count, not
   selective in what they respond to.
5. **The γ–L0 trade-off is smooth and monotonic.** Across the λ=1e-7 γ sweep,
   increasing γ consistently lowers L0 (898 → 101), loss recovered (0.966 →
   0.921) and coverage (0.422 → 0.168), with no sudden transitions. γ is a simple
   dial on the sparsity–fidelity trade.
6. **The λ collapse threshold is sharp**, and the whole region below it is not
   worth sweeping (§5).
7. **CosineSpaDE rescues the prototype approach on reconstruction.** Swapping
   squared-distance scoring for cosine similarity lifts variance explained from
   0.43 to 0.979 and coverage from 0.314 to 0.535 while keeping the sparsemax
   simplex and the no-bias decoder unchanged. That localizes the failure to the
   distance-norm interaction with standardized activations, not to the
   distance/sparsemax idea itself. It does **not** put CosineSpaDE on the
   frontier — at matched sparsity it is still strictly dominated (§7e).
8. **Reconstruction quality is not concept-manifold preservation.** CosineSpaDE at
   VE 0.98 destroys the rule-category manifold curve that the dense revived SpaDE
   preserves. A high VE is not evidence that a dictionary carries the structure a
   downstream study needs.

## 8. Infrastructure

- **GPU:** Harvard FAS RC Kempner cluster (H100; earlier runs on RunPod and A100).
- **Pipeline:** `scripts/run_pipeline.sh` locally, `scripts/slurm/` on the
  cluster — see [fasrc_commands.md](fasrc_commands.md).
- **Logging:** W&B during training, project `spade-vs-karvonen-othello`.
- **Reproducibility:** each run writes `config.json`, the shared
  `standardization_params.json`, and intermediate checkpoints — with the
  config-label caveat in §5.
- **Evaluation:** the Karvonen pipeline on 59,000 test samples at 10 threshold
  levels.

## 9. The studies built on this pipeline

The studies built on this pipeline — rules, bracket flip, violation divergence,
straightness, manifold capture — have their own stage sequences, each documented
in its package docstring and driven by `scripts/slurm/studies/<study>/`. They
share a shape: numbered steps that skip what already exists, writing to a
netscratch working directory resolved by `spade.common.study.study_dir()`, with
all plotting deferred to `analysis/<study>/make_figures.py`.

They also share a checkpoint roster. `spade.saes.registry` holds the verified set
and `spade.spaces` is the representation-space API they load through, which is
where the γ=1e-2 seed-42 exclusion is made — once, rather than per study.
