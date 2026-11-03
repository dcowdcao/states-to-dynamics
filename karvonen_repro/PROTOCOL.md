# Reproducing Karvonen's published Othello SAE table

```
karvonen_repro/
  PROTOCOL.md            this file: data protocol, pass criteria, failure modes
archive/karvonen_repro/
  validation_set.csv     the 15 released checkpoints to re-evaluate
  compare_to_published.py  python archive/karvonen_repro/compare_to_published.py --harness <run>/results.csv
  probe_eval_batch.py    evidence for section 2; python archive/karvonen_repro/probe_eval_batch.py --trainers trainer_0 trainer_5
```

The target is `external/Karvonen_Research/autoencoders/saved_data/06-10-othello-trained-results_with_07-10_topk.csv`
(280 rows, 433 columns). This document is the spec the harness has to match, the
per-column pass criteria, and the list of things that break a reproduction
without anyone having written a bug.

Everything below was read out of the vendored code and, where a number is quoted,
confirmed numerically. `probe_eval_batch.py` reproduces the measurements.

## 1. Evaluation data protocol

### Datasets

| stage | HF dataset | split | games | drawn how |
|---|---|---|---|---|
| `eval_results` (SAE metrics) | `adamkarvonen/othello_45MB_games` | `train` | `n_inputs * 10` = 10 000 requested | first 10 000 rows, in order |
| `aggregate_statistics` (feature labelling) | same | `train` | 1 000 | first 1 000 rows, in order |
| `test_board_reconstructions` | same | `test` | 1 000 | first 1 000 rows, in order |

`construct_othello_dataset` takes `dataset[split]` and breaks at `i >= n_inputs`
(`eval_sae_as_classifier.py:104`). There is no shuffle and no seed: the split is
the HF split, and the sample is a prefix of it. The train split has 792 498 rows,
the test split 198 125.

Feature labelling and board reconstruction therefore use **different splits**
(train vs test), and the eval-results dataset is a **superset of the labelling
set** (first 10 000 train games vs first 1 000).

### Shapes and knobs (all from `pipeline_config.Config`, all defaults)

```
eval_sae_n_inputs              = 1000     # aggregation / feature labelling
eval_results_n_inputs          = 1000     # SAE metrics ("evaluate")
board_reconstruction_n_inputs  = 1000
batch_size                     = 25       # also the buffer refresh and out batch size
context_length                 = 59       # config.json buffer.ctx_len; games are 60 tokens, truncated to 59
f1_analysis_thresholds         = arange(0.0, 1.1, 0.1)   # 11 values
analysis_high_threshold        = 0.95
analysis_significance_threshold= 10
indexing_function              = None     # Othello uses every token position
```

`get_recommended_indexing_functions(othello=True)` returns `[None]`, so **all 59
token positions are used**. (Chess subsamples to the ~20 "dots" positions; Othello
does not.)

### The activation buffer, and the thing that matters most

`prep_data_ae_buffer_and_model` builds an `NNsightActivationBuffer` with
`n_ctxs = min(512, n_inputs) = 512`, `ctx_len = 59`, `refresh_batch_size = 25`,
`out_batch_size = 25`. Its first refresh loops `while len(activations) < 512*59 =
30208`, appending `25*59 = 1475` activations per forward pass, so it stops at
**30 975 activations drawn from the first 525 games** and then hands out
25-activation batches by seeded `randperm`.

`get_eval_results.get_evals` calls `circuits.evaluation.evaluate` **without
`n_batches`**, and the default is `n_batches=1`. So every reconstruction column in
the published CSV — `l0`, `l1_loss`, `l2_loss`, `frac_variance_explained`,
`cossim`, `l2_ratio`, `frac_alive` — is computed on **one batch of 25 activation
vectors**, not on 1000 games.

Three independent confirmations:

* Every `l0` in all 280 rows is an exact multiple of `1/25 = 0.04`
  (max deviation 9.8e-4, i.e. float32 rounding). `l0` is a mean over exactly 25 tokens.
* In the sibling `matryoshka_othello/results.csv` run, `frac_alive * dict_size` is
  an exact integer (196/4096, 379/4096, …) — the count of distinct features that
  fired among 25 tokens.
* Rebuilding the buffer on CPU and re-drawing 25 activations 200 times reproduces
  the published values (§2).

`loss_original`, `loss_reconstructed`, `loss_zero`, `frac_recovered` are computed
separately, by `loss_recovered` on `buffer.text_batch(25)`, which pulls the **next
25 games in order** — games 525–549 of the train split. No RNG at all. This is
why `loss_original = 2.054068` and `loss_zero = 4.913147` are identical in all 280
rows and in the matryoshka run.

### Feature labelling and board reconstruction

`aggregate_statistics` seeds `torch.manual_seed(0)` and calls `get_firing_features`
for `min(int(1000*0.5), 1000) * 59 = 29 500` activations in batches of 25 (1 180
draws). The buffer starts at 30 975 slots, refreshes once (at draw 636, growing to
31 325), and consumes 800 of the 1 000 train games. A feature is "alive" if it
fired strictly more than 0 times among the 29 500 drawn activations —
94 % of the buffer, so the alive set is nearly but not exactly permutation-invariant.
`max_activations` (per-feature max over the drawn set) inherits the same near-invariance.
Thresholds are `f1_analysis_thresholds ⊗ max_activations`, so both the labelling
and the reconstruction stage depend on that draw.

`analysis.analyze_results_dict` then labels a (feature, square, class) cell as a
classifier when the class is present on ≥95 % of the firings and the cell has ≥10
firings. Deterministic given the aggregation results.

`test_board_reconstructions` replays the labels on the first 1 000 **test** games,
all 59 positions, batch 25, over the same 11 thresholds. Two gotchas:

* Threshold index 10 is **not** threshold 1.0. `aggregate_feature_labels`
  overwrites `active_boards_sum_TBLRRC[-1]` with the "only the highest threshold
  each feature is still active at" reconstruction. So `last_f1_score_per_class` is
  that variant, and `best_idx = f1_score_per_square[:-1].argmax()` ranges over 0–9
  (thresholds 0.0–0.9 × max activation).
* `best_idx` is chosen by argmax **on the test set itself**, and the reported
  `best_f1_score_per_class` is read at that index. It is a selected maximum, not a
  held-out score.

`f1_analysis.add_coverage_to_df` adds the `*_f1_threshold_*` columns afterwards
from the saved `results.pkl`; they are deterministic given that pickle.

### Model

`Baidicoot/Othello-GPT-Transformer-Lens` loaded through `HookedTransformer` and
wrapped in NNsight; SAE input is `blocks[5].hook_resid_post` when
`activation_dim == 512`.

## 2. What is reproducible, and the settled `frac_variance_explained` question

`probe_eval_batch.py` rebuilds the 30 975-activation buffer on CPU, evaluates a
released SAE on the whole buffer and on 200 independent 25-activation draws, and
compares to the published row. Four matryoshka checkpoints spanning L0 10–334:

```
matryoshka trainer_0  (k=10, d=4096)      published   whole buffer   25-draw mean   25-draw sd        25-draw range
  l0                                       9.920000      10.016013      10.003600     0.867125     [ 8.0800, 13.7200]
  l1_loss                                 54.032108      66.687065      67.290344    22.206812     [45.2566,187.2460]
  l2_loss                                 12.195415      12.116427      12.052073     0.691412     [10.1511, 13.8312]
  frac_variance_explained (centered)       0.692539       0.921253       0.787476     0.138939     [ 0.6107,  0.9900]
  frac_variance_explained (uncentered)          ---       0.923757       0.803225     0.127491     [ 0.6376,  0.9913]
  cossim                                   0.840330       0.837765       0.838945     0.014611     [ 0.7962,  0.8747]
  l2_ratio                                 0.864878       0.851618       0.851910     0.017220     [ 0.7961,  0.8978]
  frac_alive                               0.047852       0.612793       0.049116     0.004199     [ 0.0388,  0.0593]

matryoshka trainer_5  (k=112, d=4096)
  l0                                     119.879997     112.597641     111.986992     7.949931
  frac_variance_explained (centered)       0.955996       0.989020       0.970345     0.019294     [ 0.9477,  0.9985]
  frac_variance_explained (uncentered)          ---       0.989358       0.972527     0.017701
  cossim                                   0.978547       0.979195       0.979295     0.001566

matryoshka trainer_15 (k=317, d=4096)
  l0                                     333.919983     317.402222     315.893402    18.403629
  frac_variance_explained (centered)       0.987962       0.996889       0.991522     0.005521     [ 0.9847,  0.9996]
  frac_variance_explained (uncentered)          ---       0.996889       0.991919     0.005190
  cossim                                   0.994075       0.994022       0.994009     0.000409

matryoshka trainer_30 (k=215, d=8192)
  l0                                     225.639999     215.029022     213.863617    15.544454
  frac_variance_explained (centered)       0.984472       0.996039       0.989263     0.006988     [ 0.9813,  0.9994]
  frac_variance_explained (uncentered)          ---       0.996154       0.990035     0.006411
  cossim                                   0.992551       0.992566       0.992577     0.000514
```

**The 0.9347-vs-0.7954 gap is sampling, not formula and not a loading error.**

* Formula. His per-dimension centered `1 - Σ_d Var(x−x̂)_d / Σ_d Var(x)_d` and our
  total sum-of-squares version differ by 0.0025 at L0 10 and by ≤0.0012 at higher
  L0. Two orders of magnitude too small to explain 0.14.
* Sampling. The n=25 estimator is both noisy (sd 0.139 at L0 10) and *biased low*
  (25-draw mean 0.787 vs population 0.921, because both variance sums are estimated
  on 24 degrees of freedom in a 512-dim space). The published 0.6925 sits 0.7 sd
  below that already-biased mean and inside the observed range [0.611, 0.990].
  A whole-set measurement of 0.9347 against a published 0.7954 at L0 = 10 is the
  same picture with a different checkpoint.
* Loading error is ruled out by the columns that *do* agree: `cossim` matches the
  whole buffer to ≤0.003 at every L0, and `l2_ratio` to ≤0.013. A mis-loaded SAE
  would move those.

The bias shrinks fast with L0 (0.13 → 0.019 → 0.007 → 0.006). So
`frac_variance_explained` is usable at L0 ≳ 100 and meaningless below ~30.

### `frac_alive` is not reproducible with the vendored code

For all 240 non-top_k rows, `l0 / frac_alive` is a power of two to a relative
error of 7e-13 — i.e. published `frac_alive == l0 / dict_size` exactly. That is
`(f != 0).float().mean()`, not the `active_features` accumulation in the vendored
`evaluation.py`. For all 40 top_k rows `frac_alive` is exactly `1/4096` or
`1/8192`, one "alive" feature regardless of L0 — a different bug again (the CSV is
a concatenation of a 2024-06-10 run and a 2024-07-10 run).

The current code computes something else entirely (whole-buffer 0.613 vs published
0.048 in the probe above; the matryoshka run, which used the current code, matches
the 25-draw distribution). **`frac_alive` must be excluded from the comparison, not
tolerated.** Its one use is free information: it recovers `dict_size` for every
published row, giving expansion factor 8 (d=4096) or 16 (d=8192), twenty of each
per group and per sweep.

## 3. Pass criteria

Tolerance form: pass iff `|reproduced − published| ≤ max(abs_tol, rel_tol × |published|)`.

### Tier G — gates. Any failure means the protocol is wrong; stop and fix before reading anything else.

| column | tol | why |
|---|---|---|
| `loss_original` | abs 1e-4 | Games 525–549 of the train split, no RNG, SAE-independent. Published 2.054068 in every row. A miss means wrong dataset revision, wrong context length, wrong batch size, or a buffer that consumed a different number of games. |
| `loss_zero` | abs 1e-4 | Same batch, ablated submodule. Published 4.913147 in every row. |
| `<func>_num_squares` | exact | Count of 1s in the ground-truth test tensor. Depends only on the first 1000 test games: 3 776 000 / 2 006 000 / 506 839 for mine_yours / blank_mask / valid_moves. Detects HF dataset drift on its own. |
| `<func>_board_reconstruction_board_count` | exact | 1000 × 59 = 59 000. |
| `eval_sae_n_inputs`, `eval_results_n_inputs`, `board_reconstruction_n_inputs` | exact | Config echo; catches a mis-set `Config`. |
| `trainer_class`, `sae_class` | exact string | Catches loading the wrong checkpoint or the wrong dictionary class. |

### Tier D — deterministic given the SAE. Only float-reduction order should move these.

| column | tol | why |
|---|---|---|
| `loss_reconstructed` | abs 1e-3 | Same 25 games; the only nondeterminism is GPU reduction order in a 25×58×61 cross-entropy. |
| `frac_recovered` | abs 5e-4 | `(recon − zero)/(orig − zero)` with denominator −2.859, so 1e-3 on the loss maps to 3.5e-4 here. |

### Tier R — board reconstruction. Deterministic except for the alive-feature draw (94 % buffer coverage).

| column | tol | why |
|---|---|---|
| `<func>_best_f1_score_per_class` | abs 0.01 | The F1 curve moves only through which features are alive and their max activation. 94 % coverage moves per-feature thresholds by well under 1 %; 0.01 absolute is ~3× that headroom and still an order of magnitude tighter than the between-checkpoint spread (0.35–0.95). |
| `<func>_zero_f1_score_per_class`, `<func>_last_f1_score_per_class` | abs 0.01 | Same. |
| `<func>_best_idx` | abs 1 (integer) | Argmax over 10 thresholds on a curve that is flat near its peak; the published distribution is concentrated on {2,3} (231/280 rows). Requiring exact equality would fail on ties. A jump of ≥2 is real disagreement. |
| `<func>_zero_L0`, `<func>_best_L0` | rel 5 % | Mean alive features firing per test token; moves with the alive set. |
| `<func>_best_num_true_positive_squares` and the false-positive / false-negative / multiple-classes counterparts | rel 5 % | Integer counts over 59 000 boards; same dependence. |
| `num_alive_features` | rel 5 % | "Fired at least once in 29 500 of 31 325 buffer slots". Features that fire once or twice can fall either side; a 5 % move is expected, more is not. |
| `<func>_high_precision_counts_per_T`, `<func>_high_precision_and_recall_counts_per_T` | rel 10 %, elementwise | Serialised 11-vectors; counts of labelled cells, one further threshold step removed from the alive draw. |

### Tier S — sampling-dominated (the n=25 batch). Loose per checkpoint; the real test is set-level.

| column | tol | why |
|---|---|---|
| `l0` | TopK/BatchTopK SAEs: rel 1e-3. Others: rel 25 % | For `AutoEncoderTopK`, L0 is exactly `k` at every token, so the sample size is irrelevant and the published values are exact integers (10, 20, …, 390). Everywhere else the measured 25-draw relative sd is 5.8–8.7 %, so 25 % is ~3 sd. |
| `cossim` | abs 0.01 | Best-behaved of the batch metrics: published tracks the whole buffer to ≤0.003 at every L0 measured, 25-draw sd 0.0004–0.015. |
| `l2_ratio` | abs 0.02 | 25-draw sd 0.001–0.017. |
| `l2_loss` | rel 15 % | 25-draw relative sd 2–6 %. |
| `l1_loss` | rel 40 % | Heavy-tailed: at L0 10 the 25-draw range is [45, 187] around a population 66.7. Nothing tighter is defensible. |
| `frac_variance_explained` | abs 0.02 for published ≥ 0.95; **advisory only** below | The n=25 estimator's own spread is ±0.14 at L0 10 and ±0.006 at L0 300. Below 0.95 there is no tolerance that both admits an honest reproduction and rejects a broken one. Report the deviation, do not gate on it. |

### Excluded

`frac_alive` — the published values come from code that is not in the tree (§2).
The comparison script reports it as EXCLUDED with the reason and never as a pass.

### Set-level criteria, over the whole validation set

Per-checkpoint tolerances on tier S are too loose to prove anything on their own.
The paper's claims are about the shape of the L0-versus-F1 frontier, so the
reproduction has to hold at that level too:

* Spearman ρ ≥ 0.98 between published and reproduced `l0`, and between published
  and reproduced `<func>_best_f1_score_per_class`, across the validation set.
* Median signed relative deviation of `l0` within ±5 % — a one-sided bias means a
  systematic protocol difference (wrong batch size, wrong split) hiding under
  per-checkpoint noise.

### Verdict

REPRODUCED requires: every tier-G and tier-D check passes on every checkpoint; at
least 90 % of tier-R checks pass; both set-level criteria hold. Tier-S failures are
reported and counted but do not by themselves fail the run. Anything else is
FAILED, and `compare_to_published.py` names the offending checkpoint and column.

## 4. Validation set

`validation_set.csv` — 15 of the 280 released checkpoints. Coverage:

* all five trainer classes (`StandardTrainer`, `GatedSAETrainer`,
  `GatedAnnealTrainer`, `PAnnealTrainer`, `TrainerTopK`);
* both expansion factors (`dict_size` 4096 and 8192);
* both the original and the `-v2` sweeps, for the two groups that have a `-v2`
  (gated, standard);
* L0 from 10 to 571, including both sides of the F1 peak — the gated and
  gated_anneal high-L0 entries have `best_f1` of 0.45 and 0.35, so a harness that
  silently reconstructs boards from a broken feature-label set cannot pass by
  regressing everything toward "good";
* the top_k pair `trainer_38` / `trainer_78`, identical L0 (390) at the two
  expansion factors, which isolates dict-size handling.

Deliberately excluded: the 24 `standard` original-sweep checkpoints with `l0 = 0`
(dead SAEs, `best_f1 ≈ 0.004`). They match trivially and would inflate the pass rate.

Cost note: the released SAEs are only downloadable as whole-group zips
(`othello-trained_model-layer_5-2024-05-23.zip`, 9.3 GB, and
`othello-trained_model-layer_5-2024-07-10_topk.zip`, 3.0 GB). Restricting the
validation set saves evaluation time, not download.

## 5. Ways a reproduction can miss that are not harness bugs

**The n=25 eval batch.** The dominant one. `l0`, `l1_loss`, `l2_loss`,
`frac_variance_explained`, `cossim`, `l2_ratio` are single-batch estimates. Even a
byte-identical rerun on a different GPU draws a different `randperm` and lands
somewhere else in the distributions in §2.

**Seeded `randperm` is device- and version-dependent.** `torch.manual_seed(0)` is
set in `full_pipeline.py` before `get_evals` and inside `aggregate_statistics`, but
the permutation is generated on `device`. CPU and CUDA generators differ, CUDA
generators differ across CUDA/PyTorch versions, and multi-GPU runs
(`Config.N_GPUS > 1` with `joblib` sharedmem) interleave draws from a shared global
generator, so his own run is not reproducible from the seed alone.

**HuggingFace dataset drift.** Nothing pins a revision of
`adamkarvonen/othello_45MB_games`. The protocol is "first N rows of the split", so
a re-upload, a re-shard, or a change in `datasets`' row ordering silently changes
the whole evaluation. `<func>_num_squares` and `loss_original` are the tripwires.

**`best_idx` is selected on the evaluation set.** The reported F1 is a maximum over
10 thresholds picked on the same 1 000 test games it is reported on. Near a tie the
argmax jumps and `best_f1_score_per_class`, `best_L0` and all the `best_num_*`
counts jump with it, discontinuously. Two runs can agree everywhere and still show a
step in these columns.

**The alive-feature draw.** `get_firing_features` sees 94 % of the buffer, and
"alive" is `firing rate > 0`. Features that fire once or twice flip in and out,
which changes `num_alive_features`, the per-feature max activations, every
threshold derived from them, and hence the labels and the reconstruction.

**The CSV is two runs, months apart.** The 240 non-top_k rows are 2024-06-10; the
40 top_k rows are 2024-07-10 with a different `evaluation.py` (§2) and a different
threshold-name formatting in the coverage columns
(`f1_threshold_0.1_*` for 240 rows, `f1_threshold_0.10000000149011612_*` for 40).
Column-by-column comparison across the whole CSV will show spurious NaNs unless
threshold names are normalised.

**The vendored `dictionary_learning` has moved on.** `evaluation.py` in the tree
computes `frac_alive` differently from both published runs, and the dictionary
classes have gained `threshold` buffers and `scale_biases`. Loading a 2024
checkpoint with 2025 classes can silently change inference (e.g.
`AutoEncoderTopK.encode(use_threshold=...)`).

**Float nondeterminism.** Different GPU architecture, TF32 on or off, cuDNN
algorithm selection, and batch-order reductions move the losses in the last few
digits. This is why the tier-G tolerance is 1e-4 rather than exact equality.

**Dead and degenerate checkpoints.** 24 of the 40 original `standard` checkpoints
have `l0 = 0`. Their `frac_variance_explained` is 0 or −2e-6 and `best_f1 ≈ 0.004`.
Ratio-based tolerances divide by ~0 on these; they are excluded from the validation
set for that reason.
