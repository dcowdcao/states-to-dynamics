# A1 — TPR probe adapter

`tpr_probe.py` fits the bilinear TPR of Lee, Viégas & Wattenberg on our cached
OthelloGPT activations. Acceptance met: **layer-5 eval accuracy 0.9827 vs the
linear probe's 0.9896, a gap of 0.69 points** (budget: 1 point).

Reproduce: `python src/bracket_flip/tpr_probe.py` (~2 min, CPU, all 8 layers).
Probes land in `$SPADE_BRACKET_DIR/tpr_probes/tpr_r52_f2_layer{L}.pt`.

## 1. What upstream does

Paths below are relative to `external/tpr_othello/src/`.

**Parameterization** (`train_tpr_probe.py:39-90`). Three parameter tensors, no bias
by default:

| name | shape | init (`:64-72`) |
|---|---|---|
| `binding_map` | `[d_model, d_r, d_f]` | `randn / sqrt(d_model)` |
| `role_embeddings` | `[8, 8, d_r]` | `randn / sqrt(d_r)` |
| `filler_embeddings` | `[3, d_f]` | `randn / sqrt(d_f)` |

The forward pass (`:80-90`) is two einsums: contract the residual against the
binding map to get a per-token binding tensor `T = einsum("bpd,drf->bprf", h, U)`,
then unbind with roles and fillers, `einsum("bprf,xyr,cf->bpxyc", T, R, F)`. That
is exactly `logit(s, c) = r_sᵀ M(h) f_c`. Note the index order: upstream stores the
map as `[d_model, d_r, d_f]`, the design doc writes it `[d_r, d_f, d_model]`. Same
tensor, and `TPRProbe.binding_map` exposes the doc's order.

At `d_r=52, d_f=2` the probe has 512·52·2 + 64·52 + 3·2 = **56,582 parameters**
against the linear probe's 512·64·3 = 98,304, i.e. **57.6%** — reproducing the
paper's "57.5% of the linear probe's parameters" and confirming that r=52/f=2 is
the config the design doc names. Upstream's own defaults are r=16/f=8
(`:162-163`), and their intervention hardcodes r16_f8 checkpoints
(`intervene_tpr_probe.py:79`).

**Loss** (`train_tpr_probe.py:219-229`). Softmax cross-entropy over the 3 classes,
but assembled by hand: take the correct-class log-prob, average it over the batch,
multiply by `BOARD_LABEL_OPTIONS = 3`, average over positions, then **sum over the
64 squares**. Net effect is `192 × mean CE`. With AdamW the scale is nearly
irrelevant to the update direction but does change how the decoupled weight decay
trades against the gradient, so I kept it verbatim rather than substituting
`F.cross_entropy`.

**Optimizer** (`:589-594`). AdamW, `lr=1e-2`, `betas=(0.9, 0.99)`,
`weight_decay=1e-2`. Batch 128 *games* — with 59 positions each that is ~7.5k
board-states per step. One epoch (`:155`), validating every 100 batches (`:643`),
keeping the best-by-validation-loss checkpoint and early stopping at patience 10
(`:669-699`). Their training set is ~19k games ≈ **1.1M positions**.

**Data split** (`train_probe.py:258-303`). Load game pickles, keep only games of
exactly `block_size + 1` moves, shuffle with a seeded generator, then slice from
the end: last `test_size=1000` games are test, the 512 before that are validation,
the rest train. Labels come from replaying each game with their own engine
(`train_probe.py:74-158`) and are **mover-relative** (`:167-184`): class 0 empty;
at even move indices class 1 is black and class 2 white, at odd indices they swap.
Class 1 is therefore the player who just moved ("theirs") and class 2 the player to
move ("mine") — identical to our `{0: empty, 1: theirs, 2: mine}`, so no relabeling
is needed. They derive the swap from move-index parity, which is the exact failure
mode the design doc warns about; it is safe for them because their pickles contain
no forfeits, and irrelevant for me because I consume labels our pipeline already
built by engine replay.

**Accuracy** (`:315-322`). `argmax` over the class axis against the one-hot target,
counted per (game, position, square) and averaged — all 64 squares, centre squares
included unless `--exclude-center-squares` (`:206-216`, default off). Macro-F1 is
also reported (`train_probe.py:191-207`); the design doc's target is accuracy, so
that is what I match.

## 2. What I implemented

Upstream imports cleanly, so this **wraps** rather than reimplements. Taken
unmodified: `TensorProductBoardProbe`, `compute_tpr_probe_loss`,
`build_square_selection_mask`, `seed_all`, and
`solve_residual_delta_for_binding_delta`. Ours is only the fitting loop and the
`TPRProbe` facade.

The loop had to be ours because their `train()` (`train_tpr_probe.py:497`) reaches
`load_probe_dataset` → `data_utils.load_pickle_sequences` (`data_utils.py:32-45`),
which needs the game pickles that PROVENANCE.md records as deliberately not
vendored. It also runs the model live to fill a cache, whereas our activations are
already on disk.

Deliberate divergences, all hyperparameter-level — the parameterization, the loss
and the pinv edit are upstream's:

1. **Batching in board-states, not games.** Our set is a flat `[10000, 512]`; I add
   a length-1 position axis so their loss body is used verbatim. Because every game
   contributes exactly 50 positions, their mean-over-games-then-mean-over-positions
   equals a mean over our flat axis, so the loss is numerically identical.
2. **`weight_decay = 2.0`, not `1e-2`, and 120 epochs at batch 1024.** Their value
   was tuned on 1.1M positions and overfits our 8000 badly (details in §6).
3. **Fixed epoch budget, no early stopping and no best-checkpointing.** Our
   committed set has only the 8000/2000 convention; early stopping on the 2000
   would select on the very number acceptance is judged by. This matches
   `archive/analysis/linear_probe/train_linear_probe.py`, which also trains a fixed 200/500
   epochs, so the two probes are compared under the same protocol.
4. **`fit_binding_to_residual_linear_map` not ported** (`train_tpr_probe.py:352-447`,
   called at `:712-730`). It ridge-fits a map from flattened binding tensors back to
   the residual and stores it in their artifact, but their intervention never uses
   it — `FIXED_RESIDUAL_PROJECTION = "pinv"` (`intervene_tpr_probe.py:77`) routes
   through `solve_residual_delta_for_binding_delta` instead. Dead weight for us.

Data checks I ran rather than assumed:

- `all_layer_activations.pt` is `[8, 10000, 512]`, **layer on axis 0**: `all[5]` is
  bit-identical to `primary_layer_activations.pt`, and mean row norm rises
  monotonically 6.85 → 33.95 across axis 0. Confirmed, not inferred from the name.
- `board_states.pt` is `[10000, 8, 8]` int64 over `{0, 1, 2}`; class frequencies
  0.461 / 0.290 / 0.249.
- Rows are `game * 50 + turn` with turns 5..54 (`move_indices.pt`), so **position 0
  is excluded by construction** and the acceptance criterion about the attention
  sink is satisfied for free.
- Re-running the linear probe's exact recipe on this data reproduces **0.9896** at
  epochs 100 and 200, which confirms both the target and my split handling.

`decode` was checked against an independent einsum written from the doc's equation
using the `[d_r, d_f, d_model]` map: max absolute difference **0.0**.

## 3. Accuracy, all 8 layers

Eval = rows 8000:10000 (the last 40 games), never trained on, never used to select
anything. Linear column is `probe_metadata.json: layer_accuracies`.

| layer | TPR (r52 f2) | linear | gap |
|---|---|---|---|
| 0 | 0.8844 | 0.8903 | −0.0059 |
| 1 | 0.9325 | 0.9360 | −0.0034 |
| 2 | 0.9624 | 0.9652 | −0.0029 |
| 3 | 0.9775 | 0.9789 | −0.0014 |
| 4 | 0.9831 | 0.9867 | −0.0036 |
| **5** | **0.9827** | **0.9896** | **−0.0069** |
| 6 | 0.9718 | 0.9870 | −0.0153 |
| 7 | 0.9673 | 0.9860 | −0.0187 |

Layer 5 passes with 0.31 points of margin. The gap widens at layers 6–7, where the
linear probe stays flat and the TPR falls ~1.5–1.9 points: the low-rank
factorization loses more where the residual is largest (mean norm 30.4 and 34.0).
Worth knowing for M5 — a TPR depth curve peaking at layer 4/5 partly reflects the
probe, not only the model.

Layer 5 detail:

- per-class eval accuracy: empty 0.9966 (n=59,000), theirs 0.9725 (n=37,210),
  mine 0.9687 (n=31,790)
- train accuracy 0.9976, so a 1.5-point train/eval gap remains after retuning
- seed variance (1111 / 7 / 2026): 0.9827 / 0.9830 / 0.9833 — spread 0.0006

## 4. Filler geometry (layer 5, seed 1111)

| filler | coordinates | norm |
|---|---|---|
| empty | (+0.5603, −0.4560) | 0.7224 |
| theirs | (+0.3685, +0.5573) | 0.6681 |
| mine | (−0.4789, −0.0661) | 0.4834 |

| pair | angle | cos | separation |
|---|---|---|---|
| empty–theirs | 95.67° | −0.0988 | 1.0313 |
| empty–mine | 133.00° | −0.6820 | 1.1099 |
| theirs–mine | 131.33° | −0.6604 | 1.0520 |

**{mine, theirs} are not near-antipodal: 131.3°, i.e. 48.7° short of 180°.** The
three angles sum to 360.00°, which means the origin lies strictly inside the
triangle the fillers form (barycentric signs +0.2554, +0.4803, +0.2425, all
positive). The picture is a three-spoke star, not an axis with a third point off it.

**Caveat that matters for F1.** These coordinates are gauge-dependent. The bilinear
form is invariant under `f → A f`, `M → M A⁻ᵀ` for any `A ∈ GL(2)`, so angles and
norms in filler space are properties of the fit, not of the model. Measured
directly: across seeds 1111 / 7 / 2026 the theirs–mine angle is 131.3° / 113.3° /
104.5° while eval accuracy moves by 0.0006. Anything drawn in this plane must fix a
gauge (e.g. an affine map sending the three fillers to a canonical triangle) or it
will not be comparable across probes, seeds, or label conventions.

What survives the gauge freedom: the triangle is non-degenerate and the origin is
strictly interior in all three seeds (angle sums 360.0 / 360.1 / 359.9). Antipodality
means two fillers are collinear through the origin, which would put the origin on an
edge — so **no two fillers are antipodal in any basis**, and the mine↔theirs
displacement is a genuine chord of the triangle rather than a reflection.

**The plane is readable, which was the question.** The decision boundaries are three
lines through the origin, cutting it into sectors of **121.3° (empty), 115.6°
(theirs), 123.1° (mine)** — near-perfect thirds. Real readouts
`u_s = r_sᵀ M(h)` sit well inside their sector: median `|u_s|` is 9.08 / 9.00 / 7.59
by true class, and the class-mean readout aligns with its own filler at cos 0.949
(empty), 0.888 (theirs), 0.981 (mine). With `d_f = 2` there is no dimensionality
reduction, so F1 can be drawn directly. The one design consequence is that "traverses
the mine↔theirs axis" should be phrased as a chord between two 120° sectors, not as a
sign flip along an axis.

## 5. `edit_direction` sanity check

`edit_direction(s, from, to)` builds `ΔB = r_s (f_to − f_from)ᵀ`
(`intervene_tpr_probe.py:496-500`) and solves it back to the residual with
`pinv` (`:519-525`), which is upstream's own function. The one divergence: upstream
normalizes the result to unit norm and sweeps a scale (`:560`, `:81-92`). I return
the **unnormalized** least-norm solution, because M3 wants it as a decomposition
component where the magnitude is the whole point. Callers wanting upstream's
protocol normalize and sweep themselves.

Checks on layer-5 eval activations (n = 200 per transition, one square per
activation so events stay independent):

- **The solve is exact.** `‖Wᵀd − ΔB‖∞ = 3.0e-7`. `W` is `[512, 104]` and
  surjective onto binding space, so the edit realizes the intended `ΔB` exactly,
  not approximately.
- **The direction is always right: 200/200 for all six ordered transitions.**
  Analytically the target square's logit moves by `‖r_s‖²(Δf · f_c)`, so the
  from→to gap closes by `‖r_s‖²‖Δf‖² > 0` per unit. With `‖r_s‖` = 1.67 mean
  (1.35–1.92) and `‖Δf‖²` ≈ 1.06–1.23, that is ≈3.1 logits per unit.
- **Unit scale is not enough.** `‖d‖` is 2.8–3.9% of `‖h‖` (0.656 vs mean 23.23 at
  square 27), and the median scale that closes the gap is `λ*` ≈ 1.8–2.8.

| transition | median λ* | flip @1 | @2 | @4 | @8 |
|---|---|---|---|---|---|
| empty→theirs | 2.82 | 0.050 | 0.260 | 0.735 | 1.000 |
| empty→mine | 2.17 | 0.150 | 0.435 | 0.875 | 1.000 |
| theirs→empty | 2.05 | 0.150 | 0.445 | 0.920 | 1.000 |
| theirs→mine | 1.84 | 0.235 | 0.535 | 0.960 | 1.000 |
| mine→empty | 1.78 | 0.210 | 0.555 | 0.960 | 1.000 |
| mine→theirs | 1.98 | 0.150 | 0.455 | 0.960 | 1.000 |

**Collateral damage**, counted as other squares (of 63) whose decoded class changes:

| | mean squares changed | fraction with zero collateral | worst case |
|---|---|---|---|
| at each event's first flipping scale | 0.28 – 0.53 | 0.65 – 0.79 | — |
| at fixed scale 8 (flips 100%) | 0.87 – 1.75 | 0.18 – 0.49 | 9 |

The mechanism is exact and predictable: square `s' ≠ s` moves by
`(r_{s'} · r_s)(Δf · f_c)`. 64 roles in 52 dimensions cannot be mutually
orthogonal — off-diagonal `|cos|` of the role Gram is 0.0545 mean but 0.3773 max,
so the worst-aligned square receives 38% of the intended push. Collateral is
therefore structural, not noise, and is concentrated on a small set of squares
determined by the role Gram alone. If M6 is revived, use the smallest scale that
flips (≈4) rather than a fixed large one: it roughly triples the clean rate.

## 6. What did not work

1. **Upstream's `weight_decay = 1e-2` fails acceptance.** At layer 5 it peaks at
   eval 0.9743 (epoch 15, batch 1024) and decays to 0.9685 by epoch 60, with train
   accuracy 0.9995 — plain overfitting of 56,582 parameters on 8000 board-states,
   against the 1.1M they fit on. Full-batch was no better (best 0.9734 at epoch
   100). Three seeds agreed (0.9733 / 0.9734 / 0.9755), so it is not an unlucky init.
2. **How `wd = 2.0` was chosen, and why it is not leakage.** I split the *training*
   8000 into games 0–139 (fit) and games 140–159 (select), leaving the eval 2000
   untouched, and swept wd × epochs. Best inner-validation accuracy over the epoch
   grid {10, 20, 30, 40, 60, 80, 120}: 0.9663 (wd 1e-2), 0.9680 (1e-1), 0.9705
   (3e-1), 0.9752 (1.0), **0.9784 (2.0)**, 0.9769 (3.0), 0.8239 (10.0). The surface
   is flat over wd ∈ [1.5, 2.5] and 80–260 epochs
   (0.9762–0.9784), so the choice is not a knife-edge. wd=2.0 / 120 epochs is the
   argmax and sits in the interior of the flat region.
3. **Upstream's default r=16 / f=8 underfits badly at our budget**: eval 0.8276
   after 40 epochs, train 0.8564. Not a problem for us (the design doc specifies
   r=52 / f=2) but a warning that their config does not transfer to a small set.
4. **Filler angles are not reproducible across seeds** (131.3° / 113.3° / 104.5°
   for theirs–mine) even though accuracy is stable to 0.0006. §4 explains why this
   is expected rather than a bug, but it is a real constraint on F1.
5. **Unit-magnitude edits mostly do not flip the class** (5–24%). Not a defect —
   the edit realizes `ΔB` exactly; the decoded margin at a confident square is
   simply larger than one unit of `ΔB` buys. It does mean any downstream use must
   sweep scale, exactly as upstream does.
6. **Layers 6 and 7 miss the linear probe by 1.5–1.9 points.** Only layer 5 was
   required, and layers 0–5 are all within 0.7 points, but M5's depth curve should
   not read a TPR-specific late-layer droop as a fact about the model.
7. **The 0.9896 target is the layer sweep's number, not the saved probe's.**
   `probe_metadata.json: layer_accuracies[5] = 0.98958` comes from the 200-epoch
   quick probe; loading the committed `linear_probe.pt` (500 epochs) and scoring it
   on the same eval rows gives **0.98838**. I compared against 0.9896, the number
   the design doc names and the stricter of the two. Against 0.9884 the gap is 0.57
   points.
8. **Worktree provenance.** This worktree was branched from `main`, which predates
   the bracket-flip prep commits, so `docs/bracket_flip_design.md`,
   `external/tpr_othello/` and `src/bracket_flip/study_dir.py` were absent. I
   restored them into the working tree from commit `26309a0` (`git restore
   --source`), unmodified. The integrator should merge onto
   `cosine-spade-gamma-sweep`, which already carries them.
9. **`analysis/linear_probe/activations/` is gitignored**, so no worktree carries
   it. `tpr_probe.py` resolves it from the repo root and honours
   `SPADE_LINEAR_PROBE_ACTIVATIONS` as an override; I ran with the override
   pointing at the shared checkout. On the shared checkout the default just works.

## 7. Label convention (for A2 and integration)

Every number above is **mover-relative**: `{0: empty, 1: theirs, 2: mine}` with
"mine" the player to move next, taken from the committed `board_states.pt`, which
`train_linear_probe.py:76-81` derives by engine replay. Absolute colour is out of
scope here.

`train(layer, seed=1111, labels=None)` takes labels as an argument; `None` loads
the committed mover-relative set. To fit the absolute-colour TPR for the M1 basis
test, pass A2's `[n, 8, 8]` label tensor — nothing else changes, since the
convention appears nowhere else in the module. Verified by refitting layer 5 on
labels with classes 1 and 2 transposed: accuracy 0.9828 against 0.9827, as it must
be if the convention is data and not code. If A2's activations live elsewhere,
point `SPADE_LINEAR_PROBE_ACTIVATIONS` at that cache; the file layout it expects is
`all_layer_activations.pt` `[8, n, 512]` plus `board_states.pt` `[n, 8, 8]`.

Two things to keep in mind when the absolute-colour probe is compared against this
one. Filler geometry is not comparable across the two fits without fixing a gauge
(§4). And accuracy should be compared at matched wd — wd is doing real work here,
so a difference in accuracy between conventions could otherwise be a difference in
regularization.
