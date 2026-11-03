# Karvonen reference

Background reference for the upstream study this project extends: Adam Karvonen,
"Evaluating Sparse Autoencoders with Board Games." It covers the model, data, SAE
architectures, training hyperparameters, evaluation metrics, results, and the
final selected configs our baselines are drawn from.

This describes **Karvonen's** pipeline, not this repo's run scripts — for those
see [pipeline.md](pipeline.md) and [othello_experiment.md](othello_experiment.md).

Source code: `external/Karvonen_Research/circuits/`.
Saved results: `external/Karvonen_Research/autoencoders/saved_data/`.

## 1. Model

**Model:** `Baidicoot/Othello-GPT-Transformer-Lens` (HuggingFace)

| Parameter | Value |
|---|---|
| Architecture | 8-layer GPT-2-style Transformer |
| Residual stream dimension (d_model) | 512 |
| MLP hidden dimension | 2048 |
| Context length | 59 tokens (max Othello game length) |
| Library | TransformerLens + NNsight |

- SAEs were trained on **layer-5 residual stream** activations
  (`model.blocks[layer].hook_resid_post`).
- A random-weight baseline (`adamkarvonen/RandomWeights8LayerOthelloGPT2`) with
  identical architecture was also evaluated.

**Code reference:** `circuits/utils.py` — `get_model()` and `get_submodule()`.

## 2. Data

**Dataset:** `taufeeque/othellogpt` (HuggingFace) — sequences of Othello moves
encoded as token integers, each up to 59 tokens long.

### SAE training data

| Parameter | Value |
|---|---|
| Total tokens | 300,000,000 |
| Batch size (SAE training) | 8,192 tokens |
| Total training steps | 36,621 (300M / 8192) |
| Buffer contexts | 1,000 sequences loaded at a time |
| Refresh batch size | 64 (model inference batch) |
| Context length | 59 |

### Evaluation data

| Dataset | Size | Split | Purpose |
|---|---|---|---|
| SAE metrics (L0, FVE, loss recovered) | 1,000 sequences | train | `get_eval_results` |
| Feature-to-board-state statistics | 1,000 sequences | train | `aggregate_statistics` |
| Board reconstruction F1 | 1,000 sequences | test | `test_board_reconstructions` |

### Board-state functions evaluated

| Function | Output shape | Description |
|---|---|---|
| `mine_yours_BLRRC` | 8×8×3 | Three classes: opponent's pieces, empty, my pieces (relative to the current player) |
| `mine_yours_blank_mask_BLRRC` | 8×8×3 (blank channel zeroed) | Same, but blank squares masked out — mine vs yours only |
| `valid_moves_BLRRC` | 8×8×1 | Binary: which squares are legal moves for the current player |

**Why blank_mask matters:** blank squares are the "easy" class, since most squares
are empty most of the time. Masking them forces the metric onto the harder
question of whose pieces are where.

**Code reference:** `circuits/othello_utils.py`, `circuits/pipeline_config.py`.

## 3. SAE architectures

### 3.1 AutoEncoder (standard)

Used by PAnnealTrainer, StandardTrainer.

```
Encoder: f = ReLU(x @ W_enc + b_enc)    → sparse feature vector of size dict_size
Decoder: x_hat = f @ W_dec + b_dec      → reconstructed activation
```

- Decoder columns constrained to **unit norm** via `ConstrainedAdam`.
- Weight init: random normal, column-normalized to magnitude 0.1.
- Encoder and decoder share structure at init (decoder = encoder.T).

### 3.2 GatedAutoEncoder

Used by GatedSAETrainer, GatedAnnealTrainer. Based on
[Rajamanoharan et al. 2024](https://arxiv.org/abs/2404.16014): a separate gate
decides which features fire, decoupling "should this feature fire?" from "at what
magnitude?".

### 3.3 AutoEncoderTopK

Used by TrainerTopK. Based on [Gao et al. 2024](https://arxiv.org/abs/2406.04093):
instead of an L1 penalty, directly select the top-k features by activation. L0
equals k exactly. Uses the geometric median for decoder-bias initialization.

### Expansion factors

Every trainer type was tested at two expansions: **8×** (dict_size 4,096) and
**16×** (dict_size 8,192).

**Code references:** `circuits/dictionary_learning/dictionary.py`,
`circuits/dictionary_learning/trainers/`.

## 4. Training hyperparameters

### 4.1 PAnnealTrainer (40 SAEs)

P-annealing gradually transitions the sparsity penalty from L1-like (p=1) toward
L0-like (p→0.2), producing cleaner sparse features.

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 (fixed) |
| Warmup steps | 1,000 |
| Expansion factors | 8×, 16× |
| Sparsity function | Lp^p |
| **Sparsity penalty (alpha)** | **20 values in linspace(0.025, 0.05)** |
| p_start → p_end | 1.0 → 0.2 |
| Anneal start | step 10,000 |
| n_sparsity_updates | 10 |
| Sparsity queue length | 10 |
| Seed | 42 |
| **Total SAEs** | **2 × 20 = 40** |

### 4.2 StandardTrainer (80 SAEs)

Standard L1-penalized SAE following Anthropic's "Towards Monosemanticity."

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 (fixed) |
| Warmup steps | 1,000 |
| Expansion factors | 8×, 16× |
| **L1 penalty** | **20 values in linspace(0.035, 0.08)** |
| Seed | 42 |
| **Total SAEs** | **2 × 20 = 40** (plus duplicates with a `-v2` suffix = 80 in the data) |

### 4.3 GatedSAETrainer (80 SAEs)

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 (fixed) |
| Warmup steps | 1,000 |
| Expansion factors | 8×, 16× |
| **L1 penalty** | **20 values in linspace(1.0, 2.0)** |
| Seed | 42 |
| **Total SAEs** | **2 × 20 = 40** (plus `-v2` = 80 in the data) |

### 4.4 GatedAnnealTrainer (40 SAEs)

Gated architecture with the p-annealing schedule.

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 (fixed) |
| Warmup steps | 1,000 |
| Expansion factors | 8×, 16× |
| Sparsity function | Lp^p |
| **Sparsity penalty (alpha)** | **20 values in linspace(0.15, 1.0)** |
| p_start → p_end | 1.0 → 0.2 |
| Anneal start | step 10,000 |
| **Total SAEs** | **2 × 20 = 40** |

### 4.5 TrainerTopK

| Parameter | Value |
|---|---|
| Expansion factors | 8×, 16× |
| **k (top-k features)** | **40 values in linspace(10, 400)** |
| Seed | 42 |
| **Total SAEs** | **2 × 40 = 80** (40 in the final Othello data) |

### Common parameters

| Parameter | Value |
|---|---|
| Activation dimension | 512 |
| Total training tokens | 300,000,000 |
| Training batch size | 8,192 |
| Total steps | 36,621 |
| Seed | 42 |
| Target layer | 5 (residual stream) |
| Submodule | `model.blocks[5].hook_resid_post` |

**Code reference:** `circuits/sae_training/train_saes_parallel.py`.

## 5. Key concepts

### 5.1 Batch size

The number of activation vectors processed together before one weight update;
Karvonen uses 8,192. Larger batches (4096–32768) give smoother gradients and allow
higher learning rates at the cost of memory; smaller ones (256–1024) act as
regularization. For SAEs specifically, larger batches supply more diverse
activations per step, which reduces dead features — a rare feature may simply
never appear in a small batch, and the sparsity signal is noisier there too. Rule
of thumb: doubling the batch size affords roughly a √2 increase in learning rate.

### 5.2 L0 (sparsity)

```python
l0 = (f != 0).float().sum(dim=-1).mean()
```

Average number of non-zero features per input. Lower is sparser and more
interpretable. L0 is not differentiable, so an L1 penalty is used during training
as a proxy. For TopK SAEs, L0 = k exactly. A typical good range is 10–100
depending on model and dictionary size.

### 5.3 L0 vs L1

| | L0 | L1 |
|---|---|---|
| What it measures | Count of non-zero activations | Sum of activation magnitudes |
| Role | Evaluation metric | Training loss penalty |
| Differentiable? | No | Yes |

### 5.4 Expansion factor

Dictionary size ÷ activation dimension. 8× at d_model=512 gives 4,096 features.
Higher expansion buys capacity at the cost of dead features and compute.

## 6. Evaluation metrics

### 6.1 L0 (average active features)

```python
l0 = (f != 0).float().sum(dim=-1).mean()
```

Range [0, dict_size].

### 6.2 Fraction variance explained (FVE)

```python
total_variance = torch.var(x, dim=0).sum()
residual_variance = torch.var(x - x_hat, dim=0).sum()
FVE = 1 - residual_variance / total_variance
```

1.0 is perfect; 0.0 is no better than predicting the mean; negative means the
reconstruction is worse than the mean, i.e. something is broken. Unlike MSE it is
normalized, so it is comparable across layers, models and runs. The core trade-off
is sparsity (lower L0) against reconstruction (higher FVE); the goal is the Pareto
frontier.

**Code reference:** `circuits/evaluation.py:201-203`.

### 6.3 Fraction loss recovered (frac_recovered)

```python
loss_original      = CrossEntropyLoss(model(x))
loss_zero          = CrossEntropyLoss(model(x_zeroed))
loss_reconstructed = CrossEntropyLoss(model(x_hat))
frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
```

How much downstream performance survives replacing the layer's activations with
the SAE reconstruction. A task-level metric: imperfect FVE may not matter if the
lost information is not used.

**Code reference:** `circuits/evaluation.py:13-154`.

### 6.4 Reconstruction F1 (board reconstruction F1)

The core game-specific metric:

1. **Feature labeling** (`analysis.py`): for each feature at each activation
   threshold T, compute the board state present when that feature fires above T.
   A board state present ≥ 95% of the time when the feature fires labels that
   feature as a classifier for it.
2. **Board reconstruction** (`eval_board_reconstruction.py`): on a held-out test
   set, use the labels of the firing features to predict the board, and compare
   to the true board.
3. **F1:** TP = the label says "piece X on square Y" and the board has it; FP =
   the label says so and it is absent; FN = the board has it and no feature
   predicted it. `F1 = 2·P·R / (P+R)`.

Two variants: **F1 per class** (`f1_score_per_class`, best taken across T) and
**F1 per square** (`f1_score_per_square`, where a feature predicting both the
correct and an incorrect class on a square is penalized as all-false-positive
there).

Thresholds: `T = torch.arange(0.0, 1.1, 0.1)`. The last index uses a special
"max-active" masking rule.

**Code reference:** `circuits/eval_board_reconstruction.py`, `circuits/analysis.py`.

### 6.5 Coverage

How many individual board square/class combinations have at least one SAE feature
that classifies them well.

```python
f1_TRRC, _ = torch.max(f1_TFRRC, dim=1)   # max over the feature dimension
coverage = (f1_TRRC > 0.8).sum()          # count cells above an F1 threshold
```

`best_average_f1` takes the best F1 per square across all features, then averages
over squares — the overall "quality of coverage."

**Code reference:** `circuits/f1_analysis.py`.

### 6.6 Other metrics

| Metric | Formula | Meaning |
|---|---|---|
| **L1 loss** | `f.norm(p=1, dim=-1).mean()` | Sum of absolute feature values |
| **L2 loss** | `torch.linalg.norm(x - x_hat, dim=-1).mean()` | Raw reconstruction error |
| **Cosine similarity** | `(x_normed * x_hat_normed).sum(-1).mean()` | Directional alignment |
| **L2 ratio** | `norm(x_hat) / norm(x)` | Under-scaled (<1) or over-scaled (>1) reconstruction |
| **Frac alive** | `(active_features != 0).sum() / dict_size` | Fraction of the dictionary ever used |

### 6.7 Feature-labeling thresholds (`analysis.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `high_threshold` | 0.95 | Board state present ≥ 95% of the time the feature fires (precision) |
| `low_threshold` | 0.1 | State appears ≤ 10% of the time it does not fire (recall) |
| `significance_threshold` | 10 | Feature must fire ≥ 10 times to be considered |

### 6.8 Threshold mechanism: static vs dynamic

Neither threshold set is learned. Both are hardcoded — but a data-dependent
scaling step makes the activation thresholds adaptive per feature.

**Layer 1 — activation thresholds (the T dimension): "when does a feature count as
firing?"** Set in `pipeline_config.py` as
`f1_analysis_thresholds = torch.arange(0.0, 1.1, 0.1)`. These are not raw cutoffs.
In `aggregate_statistics()` (`eval_sae_as_classifier.py:465-467`) each fraction is
multiplied by that feature's empirical maximum activation:

```python
thresholds_TF11 = einops.repeat(thresholds_T, "T -> T F 1 1", F=num_features)
max_activations_1F11 = einops.repeat(max_activations_f, "F -> 1 F 1 1")
thresholds_TF11 = thresholds_TF11 * max_activations_1F11
```

So threshold 0.5 for a feature whose empirical max is 12.0 becomes a cutoff of
6.0 — per-feature normalization by dynamic range. The maxima come from
`get_firing_features()` (`utils.py:275-308`), which passes ~500 sequences through
the SAE and also identifies the alive features.

**Layer 2 — feature-labeling thresholds: "which features are good classifiers?"**
`analysis_high_threshold = 0.95`, `analysis_low_threshold = 0.1`,
`analysis_significance_threshold = 10`. All three are author-chosen constants:
0.95 for precision, 0.1 for recall (the feature is not merely coincidentally
correlated), 10 to prevent labeling on noise.

**How the "best" threshold is chosen for reporting.** Argmax of F1 across the T
dimension, excluding the last index, which uses the special max-active masking:

```python
best_idx = board_reconstruction_results[function_name]["f1_score_per_square"][:-1].argmax()
```

Data-driven selection among a fixed grid, not an optimized value.

| Component | Static or dynamic? | How determined |
|---|---|---|
| Threshold fractions (0.0 … 1.0) | Static | Hardcoded grid in `pipeline_config.py` |
| Per-feature threshold scaling | **Data-driven** | Multiplied by that feature's empirical max activation |
| Feature-labeling cutoffs (0.95, 0.1, 10) | Static | Author-chosen constants |
| Which threshold is "best" for reporting | **Data-driven** | Argmax of F1 across thresholds on test data |

**Code references:** `circuits/eval_sae_as_classifier.py:465-467`,
`circuits/utils.py:275-308`, `circuits/pipeline_config.py`,
`circuits/full_pipeline.py:121`.

## 7. Evaluation pipeline

`circuits/full_pipeline.py` runs these steps per SAE:

1. **`get_eval_results`** — 1,000 train sequences → L0, L1, L2, FVE, cosine
   similarity, L2 ratio, loss recovered. Output `eval_results.pkl`.
2. **`aggregate_statistics`** — 1,000 train sequences; per feature × threshold,
   counts how often each board state appears when the feature fires and when it
   does not. Output `results.pkl` (on/off trackers, shape T × F × 8 × 8 × C).
3. **`analyze_results_dict`** — labels features as classifiers under the
   95% / 10% / min-10 rules. Output `feature_labels.pkl`.
4. **`test_board_reconstructions`** — 1,000 **test** sequences; reconstructs
   boards from the labels and computes TP/FP/FN/TN, precision, recall and F1 per
   threshold. Output `reconstruction.pkl`.
5. **`add_coverage_to_df`** — per-square coverage at various F1 thresholds,
   merged into the final CSV.

Outputs: `results.csv` (per-SAE metrics) and `f1_results.csv` (the same augmented
with coverage).

**Code reference:** `circuits/full_pipeline.py`, `circuits/pipeline_config.py`.

## 8. The 4-plot figure

L0 on the x-axis in every panel:

1. **Coverage vs L0** — fraction of board square/class combinations whose best
   feature clears an F1 threshold.
2. **Reconstruction F1 vs L0** — `best_average_f1`.
3. **Fraction variance explained vs L0** — activation-space fidelity.
4. **Fraction loss recovered vs L0** — downstream behavioral fidelity.

**Each point is one trained SAE** from the sweep; different sparsity penalties or
k values put it at a different L0. Points are colored by trainer type and/or
expansion factor. The plots reveal the **Pareto frontier**: you want the upper
left, high metric at low L0. As L0 increases all four metrics increase with
diminishing returns; the interesting region is where the curves inflect.

This repo reproduces the same 2×2 layout via
`python -m spade.figures.figure_karvonen_extension`, adding the SpaDE family as
extra points.

## 9. Results by trainer type

### 9.1 PAnnealTrainer — best overall

| Metric | Min | Median | Max |
|---|---|---|---|
| L0 | 37.6 | 55.2 | 360.4 |
| mine_yours F1 | 0.687 | 0.935 | **0.948** |
| blank_mask F1 | 0.539 | 0.941 | **0.953** |
| valid_moves F1 | 0.707 | 0.877 | **0.918** |
| FVE | 0.932 | 0.951 | 0.997 |
| Frac recovered | 0.999 | 1.000 | 1.001 |

Highest F1 at the lowest L0, and the most reliable: its worst case (0.687) beats
every other trainer's worst case. The 8× SAEs (L0 ~38–80) all clear F1 0.92.

### 9.2 StandardTrainer — highest raw F1, unreliable

| Metric | Min | Median | Max |
|---|---|---|---|
| L0 | 0.0 | 77.0 | 460.0 |
| mine_yours F1 | 0.000 | 0.862 | **0.952** |
| blank_mask F1 | 0.000 | 0.832 | 0.954 |
| valid_moves F1 | 0.000 | 0.811 | **0.939** |
| FVE | −0.000 | 0.892 | 0.977 |
| Frac recovered | −0.811 | 0.998 | 1.000 |

26 of 80 SAEs failed completely (L0 = 0, all features dead). The peak F1 needs
L0 = 457. Extremely hyperparameter-sensitive.

### 9.3 GatedSAETrainer — stable, lower ceiling

| Metric | Min | Median | Max |
|---|---|---|---|
| L0 | 60.7 | 138.6 | 570.8 |
| mine_yours F1 | 0.395 | 0.865 | 0.906 |
| blank_mask F1 | 0.171 | 0.850 | 0.913 |
| valid_moves F1 | 0.198 | 0.737 | 0.793 |
| FVE | 0.929 | 0.964 | 0.999 |
| Frac recovered | 0.999 | 1.000 | 1.000 |

No failures, but F1 plateaus around 0.906.

### 9.4 GatedAnnealTrainer — lowest L0 at decent F1

| Metric | Min | Median | Max |
|---|---|---|---|
| L0 | 45.9 | 98.7 | 455.8 |
| mine_yours F1 | 0.351 | 0.780 | 0.875 |
| blank_mask F1 | 0.075 | 0.746 | 0.883 |
| valid_moves F1 | 0.099 | 0.716 | 0.787 |
| FVE | 0.930 | 0.956 | 0.998 |
| Frac recovered | 1.000 | 1.000 | 1.000 |

The lowest L0 of any trainer at usable F1, but F1 tops out at 0.875 — gating plus
p-annealing over-constrains.

### 9.5 TrainerTopK — direct L0 control

| Metric | Min | Median | Max |
|---|---|---|---|
| L0 | 10.0 | 205.0 | 390.0 |
| mine_yours F1 | 0.440 | 0.793 | 0.906 |
| blank_mask F1 | 0.183 | 0.646 | 0.870 |
| valid_moves F1 | 0.369 | 0.662 | 0.763 |
| FVE | 0.795 | 0.988 | 0.998 |
| Frac recovered | 0.849 | 1.000 | 1.000 |

L0 equals k exactly; the sweet spot is k ≈ 80–130. Frac alive is extremely low
(0.01%) — nearly the whole dictionary is dead.

## 10. Pareto analysis and final selected SAEs

### 10.1 Pareto frontier method

An SAE is **Pareto-dominated** (and excluded) if another SAE of the same trainer
type has equal-or-lower L0 **and** equal-or-better values on **all five**
performance metrics simultaneously (mine_yours F1, blank_mask F1, valid_moves F1,
FVE, frac_recovered), with at least one strict improvement. Most SAEs survive,
because the five metrics trade off against each other.

### 10.2 Composite score for final selection

```
Score = 0.30 × mine_yours_F1
      + 0.15 × blank_mask_F1
      + 0.15 × valid_moves_F1
      + 0.15 × FVE
      + 0.10 × frac_recovered
      - 0.15 × (L0 / L0_max)
```

mine_yours F1 carries the heaviest weight as the study's central question;
frac_recovered carries the least because nearly every non-dead SAE exceeds 0.999
and so it does not discriminate; the L0 term (normalized by L0_max = 570.8)
rewards getting good metrics cheaply.

### 10.3 Sensitivity analysis

Four weight schemes were tested; the selections are remarkably stable.

| Weight scheme | PAnneal | Standard | Gated | GatedAnneal | TopK |
|---|---|---|---|---|---|
| F1-heavy (0.50 MY) | trainer27 (L0=63) | trainer26-v2 (L0=199) | trainer27-v2 (L0=94) | trainer38 (L0=47) | trainer_47 (L0=80) |
| Balanced (default) | **trainer33 (L0=48)** | trainer30-v2 (L0=132) | trainer27-v2 (L0=94) | trainer38 (L0=47) | trainer_47 (L0=80) |
| Sparsity-heavy (−0.30 L0) | trainer33 (L0=48) | trainer34-v2 (L0=99) | trainer39-v2 (L0=63) | trainer38 (L0=47) | trainer_43 (L0=40) |
| Equal-all (1/6 each) | trainer33 (L0=48) | trainer30-v2 (L0=132) | trainer33-v2 (L0=75) | trainer38 (L0=47) | trainer_47 (L0=80) |

PAnneal's trainer33 wins under 3 of 4 schemes; the GatedAnneal and TopK
selections are identical across all four.

### 10.4 Final selected SAEs

These are the operating points our baselines are trained at.

| Trainer | Config | L0 | MY F1 | BM F1 | VM F1 | FVE | Frac rec | Composite |
|---|---|---|---|---|---|---|---|---|
| **PAnnealTrainer** | **trainer33** | **48.0** | **0.9389** | **0.9424** | **0.9183** | **0.9474** | **1.0000** | **0.7903** |
| StandardTrainer | trainer30-v2 | 131.7 | 0.9211 | 0.9125 | 0.8842 | 0.9365 | 1.0000 | 0.7517 |
| GatedSAETrainer | trainer27-v2 | 94.4 | 0.9054 | 0.9091 | 0.7653 | 0.9522 | 0.9997 | 0.7408 |
| TrainerTopK | trainer_47 | 80.0 | 0.9019 | 0.8700 | 0.7483 | 0.9620 | 0.9996 | 0.7365 |
| GatedAnnealTrainer | trainer38 | 47.4 | 0.8746 | 0.8834 | 0.7373 | 0.9370 | 0.9998 | 0.7336 |

### 10.5 Pareto comparison at fixed L0 budgets

| L0 budget | Best trainer | Best MY F1 | Runner-up |
|---|---|---|---|
| L0 < 50 | **PAnneal** (L0=48) | 0.939 | GatedAnneal (L0=47, F1=0.875) |
| L0 < 100 | **PAnneal** (L0=74) | 0.948 | GatedSAE (L0=94, F1=0.905) |
| L0 < 200 | **PAnneal** (L0=74) | 0.948 | Standard (L0=199, F1=0.936) |
| L0 < 400 | Standard (L0=390) | 0.950 | PAnneal (L0=74, F1=0.948) |

**PAnneal dominates the interpretable regime (L0 < 200).** Standard catches up
only at very high L0, which defeats the purpose.

### 10.6 Why PAnneal trainer33 is the recommended comparison target

- Best composite score (0.7903), 5 points ahead of Standard (0.7517).
- Near-best F1 on all three board functions at the lowest L0 of any selected SAE.
- valid_moves F1 0.918 against Standard's 0.884, the next best.
- 2.7× L0 efficiency over Standard at comparable mine_yours F1.
- Hyperparameters: 8× expansion, initial sparsity penalty ≈ 0.032 (the geometric
  middle of the 0.025–0.05 sweep), p-annealing 1.0 → 0.2 from step 10,000.

### 10.7 Coverage caveat

Per-feature coverage — individual squares with a single-feature F1 above 0.8 —
favors GatedAnneal and Gated over PAnneal, whose max coverage at that threshold is
only 19.3%. PAnneal reaches high board reconstruction F1 by **combining several
features per square**, while Gated SAEs have individual features more cleanly
aligned to single squares but fewer that contribute to a correct reconstruction.

## 11. Clarifications

### 11.1 What is ground truth?

The actual board state at each position, computed by replaying the game with an
Othello engine:

```python
board = OthelloBoardState()
for move in game:
    board.umpire(move)      # apply the move, flip pieces
    state = board.state     # 8x8: -1 white, 0 empty, +1 black
```

OthelloGPT only ever sees move tokens. Replaying gives, for each position, an
8×8×3 one-hot tensor (opponent piece / empty / my piece), and every F1-based
metric scores against it.

### 11.2 What is blank-mask F1?

Board-state F1 with the blank class zeroed out, so the metric evaluates mine vs
yours only. Blanks are the easy class — a feature that merely detects "this square
is empty" would score well for free.

### 11.3 Coverage vs reconstruction F1

Both use F1 against board-state labels, but they ask different questions.

**Coverage: "does each board concept have a good feature?"** For each
(square, class), take the single best feature across the dictionary and all
thresholds, then average over the 180 cells. Features are tested **individually**,
and one feature may be claimed by many cells. Answers: *is the information present
somewhere in the dictionary?*

**Reconstruction F1: "can the features work together to predict the whole board?"**
Each feature is assigned to exactly one (square, class) — its single best — and
active features vote for their assigned label. Features are tested
**collectively**; if two cells need the same feature, only one gets it. Answers:
*is there enough coverage to reconstruct everything at once?*

They diverge when the dictionary is redundant (many features for one concept) or
polysemantic (one feature partially matching many concepts without committing to
one). Coverage is the lenient test, reconstruction F1 the strict one, and
Coverage ≥ Reconstruction F1 always holds; the gap measures redundancy and
polysemanticity. Reconstruction F1 is computed per square per class and
macro-averaged, so one wrong square lowers the average slightly rather than
invalidating the rest.

### 11.4 Fraction variance explained vs fraction loss recovered

**FVE (R²)** asks whether the reconstructed activation vectors look like the
originals numerically: `1 − SS_res / SS_tot`, a pure tensor comparison, fast.

**Fraction loss recovered** asks whether the model *behaves* the same. It requires
three model runs with intervention, and is slow.

They disagree in both directions: an SAE can reconstruct 99.9% of the activation
while corrupting a tiny decision-critical component (high FVE, low loss
recovered), or spread moderate error across unimportant dimensions (moderate FVE,
high loss recovered). FVE is whether the reconstruction *looks* right; loss
recovered is whether it *works* right. Loss recovered is the stronger metric; FVE
is the cheaper proxy. This repo's CosineSpaDE γ-sweep is a live example of the
dissociation — at γ=1e4, FVE holds 0.854 while loss recovered falls to 0.102.

## Appendix: file paths

All paths are relative to `external/Karvonen_Research/`.

| Purpose | Path |
|---|---|
| SAE training script | `circuits/sae_training/train_saes_parallel.py` |
| Evaluation (L0, FVE, loss) | `circuits/evaluation.py` |
| Board state functions | `circuits/othello_utils.py` |
| Feature-to-board statistics | `circuits/eval_sae_as_classifier.py` |
| Feature labeling | `circuits/analysis.py` |
| Board reconstruction F1 | `circuits/eval_board_reconstruction.py` |
| Coverage computation | `circuits/f1_analysis.py` |
| Full evaluation pipeline | `circuits/full_pipeline.py` |
| Pipeline config | `circuits/pipeline_config.py` |
| Othello results CSV | `autoencoders/saved_data/06-10-othello-trained-results_with_07-10_topk.csv` |
| Othello F1 results CSV | `autoencoders/saved_data/othello-6-10-trained-f1_results.csv` |
| SAE architectures | `circuits/dictionary_learning/dictionary.py` |
| PAnneal trainer | `circuits/dictionary_learning/trainers/p_anneal.py` |
| Standard trainer | `circuits/dictionary_learning/trainers/standard.py` |
| Gated SAE trainer | `circuits/dictionary_learning/trainers/gdm.py` |
| Gated Anneal trainer | `circuits/dictionary_learning/trainers/gated_anneal.py` |
| TopK trainer | `circuits/dictionary_learning/trainers/top_k.py` |

Three files `circuits` needs at runtime are invisible to git — the *vendored*
`.gitignore`, which git still honours, hides `circuits/resources/meta.pkl` and
`circuits/dictionary_learning/pretrained_dictionary_downloader.sh`, and the root
ignore hides `linear_probes/*.pth`. A fresh clone yields a broken `circuits`; those
three must be copied by hand.
