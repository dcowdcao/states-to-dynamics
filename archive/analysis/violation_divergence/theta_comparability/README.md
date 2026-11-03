# Is the straightness study's "mean turning angle" the fork study's?

**Verdict: NOT COMPARABLE. Separate panels, separate axes.** No rescale, no
resample and no reindex reconciles them. Both pipelines call the same function on
layer-5 resid_post and both report degrees, but they hand it different sequences:
the straightness study passes **every ply**, the violation-divergence fork studies
pass **every second ply**. OthelloGPT's layer-5 board code is mover-relative, so
the frame inverts each ply and the per-ply path is a near-reversal sawtooth.
Sampling every second ply stays inside one mover's frame and removes it. The two
numbers are 34° apart, they are **anticorrelated across games** (r = −0.51), and
every class contrast **flips sign** between them.

```
python analysis/violation_divergence/theta_comparability/measure.py
python analysis/violation_divergence/theta_comparability/make_figures.py
```

## Where the two rules come from

Both import `spade.straightness.geometry.temporal_geometry`, so the estimator is
shared and so is the unit. θ is `rad2deg(acos(cos(Δz_i, Δz_{i+1})))`, clamped, in
[0, 180]; NaN at a zero-length step; nan-aware moments. **Degrees in both, and
that is the only thing that transfers.**

| | straightness | violation-divergence fork |
|---|---|---|
| layer-5 tensor | `$SPADE_STRAIGHTNESS_DIR/traj/resid_post_5.pt` | `$SPADE_DIVERGENCE_DIR/acts/resid_post_5.pt` |
| | \[10000, 59, 512] fp16, `blocks.5.hook_resid_post` | \[15531, 59, 512] fp16, same hook, same model |
| positions | 1…58, every ply (`compute_geometry`, position 0 sink dropped) | `fork_plane.positions()`: 1…58, every `STRIDE=2` plies at *t0*'s parity |
| what one θ spans | 2 plies (`z_{t+1}−z_t` against `z_t−z_{t−1}`) | 4 plies (`z_{t+2}−z_t` against `z_t−z_{t−2}`) |
| what one plotted point is | one angle | `mean θ` of ReStraV's R²¹ over a W=8 window = mean of **6** angles spanning **14 plies** |
| plotted at | the angle's vertex ply (`profile_panel`: index *k* → position *k*+2) | the window's **right edge** — its 6 vertices lie at *t*−12…*t*−2 |
| centering | ships `raw` **and** `centered` (minus the mean legal train trajectory); the class-baseline panels draw `raw` | never centres the activations |
| standardizing | none | none on the `fixed` panels (real degrees); **z-scored** on every PCA plane and every Euclidean separation statistic (`fit_plane`, `measure.scaled`) |

Two of these differences are cosmetic on their own. The stride is not.

## 1. The level

Layer 5, raw, per-game mean turning angle, mean ± SD over n = 2000 games per
class. Odd-parity stride-2 is shown because `positions()` samples at *t0*'s
parity: parity changes nothing.

| class | per-ply (straightness) | stride-2 (fork) | stride-2, odd parity |
|---|---|---|---|
| legal | 142.02 ± 0.94 | 107.71 ± 1.12 | 107.90 ± 1.14 |
| rollout | 141.98 ± 0.95 | 107.77 ± 1.13 | 107.95 ± 1.14 |
| violation | 141.68 ± 1.02 | 107.86 ± 1.10 | 108.03 ± 1.17 |
| random_tokens | 138.11 ± 1.51 | 109.52 ± 1.65 | 109.98 ± 1.65 |
| championship | 142.03 ± 1.05 | 108.96 ± 1.11 | 108.81 ± 1.12 |

`142.02` reproduces the straightness README's committed median (142.0°), so the
per-ply column is the published quantity and not a re-derivation of it.

Step length moves with it: legal 29.15 per ply against 18.88 at stride 2. A
stride-2 step is *shorter* than two per-ply steps precisely because the per-ply
ones point back at each other.

## 2. The contrast, and the sign flip

Δ = class mean − legal mean, 95% percentile bootstrap over games (1000
resamples). AUROC is the straightness study's own convention and split — score =
mean θ of the OOD class, odd-index test half, >0.5 means curvier than legal.

| class | per-ply Δ (deg) | per-ply AUROC | stride-2 Δ (deg) | stride-2 AUROC |
|---|---|---|---|---|
| rollout | −0.04 [−0.09, +0.02] | 0.496 [0.470, 0.521] | +0.06 [−0.01, +0.13] | 0.517 [0.493, 0.543] |
| violation | **−0.34 [−0.40, −0.28]** | **0.397 [0.372, 0.421]** | **+0.15 [+0.08, +0.22]** | **0.533 [0.508, 0.560]** |
| random_tokens | −3.91 [−3.98, −3.83] | 0.014 [0.010, 0.019] | +1.81 [+1.71, +1.90] | 0.828 [0.809, 0.845] |
| championship | +0.01 [−0.05, +0.08] | 0.509 [0.483, 0.535] | **+1.25 [+1.17, +1.32]** | **0.782 [0.762, 0.802]** |

Read the violation row. The straightness study's layer-5 headline is that a game
carrying an illegal move is **straighter** than legal (AUROC 0.40, the committed
number). Under the fork's sampling the same games are **curvier** (0.533), and the
two intervals do not overlap or touch 0.5 from the same side. FIGURE_GUIDELINES
§4.5 exists because the direction is the finding; here the direction is a property
of the sampling rule.

Championship is the other way round and worse: per-ply it is indistinguishable
from legal (0.509, the straightness README says so explicitly), while stride-2
reports 0.782. The stride manufactures an effect where the published measurement
finds none.

`corr(per-ply, stride-2) = −0.508` over all 10,000 games. **Anticorrelated**, so
there is no monotone reparametrization between them — not an offset, not a scale
factor, not a resample of one onto the other's grid. Per-game difference
−32.80 ± 2.91°, ratio 0.7678 ± 0.0181; the spread on both rules out a constant.

## 3. The two studies are measuring the same games

They are not two populations that happen to disagree. **1998 of the 2000
straightness legal games are also a fork reference branch, token for token**, and
their layer-5 activations agree *exactly* in 1991 of those 1998 rows (max |Δ|
2.0e−3 against a mean ‖z‖ of 23.4 on the other 7 — the fp16 batch-composition
noise `extract_acts.spans` documents). Reading the same games out of either
study's tensor gives the same number to three decimals:

| read from | per-ply mean θ | stride-2 mean θ |
|---|---|---|
| `straightness/traj/resid_post_5.pt` | 142.023 | 107.712 |
| `divergence/acts/resid_post_5.pt` | 142.023 | 107.712 |

So the 34.3° is the sampling rule and nothing else. Not the game set, not the
extraction pass, not the dtype, not the layer.

## 4. The same activation rows, through both pipelines

Game 1235, *t0* = 20, layer 5 raw, the fork study's own `FIGURE_GAME` and its nine
branches — read once by `curvature_plane.figure_game` and once by
`temporal_geometry` on the same `acts/resid_post_5.pt` rows.

| quantity | reference branch | all 9 branches |
|---|---|---|
| straightness: per-ply mean θ | **143.72** | 142.86 |
| fork, `whole` variant: R²¹ `mean θ` | **107.12** | 108.56 |
| fork, `windowed` variant (w=8): R²¹ `mean θ` | 105.61 | 106.83 |
| straightness: per-ply mean *d* | 29.25 | 28.85 |
| fork, `whole` variant: R²¹ `mean d` | 18.65 | 19.15 |

36.6° apart on identical rows. For scale: the nine branches span **2.65°** of
`mean θ` between them, which is already 8× the entire violation-vs-legal
separation the straightness study reports (0.34°). Overlaid, the fork's branches
would sit as a tight cluster 35° below all the class baselines and a reader would
conclude the branches are dramatically straighter than any game class. They are
not; they are measured differently.

## 5. The indexing, which bites separately

At w = 8 the fork's descriptor at plotted position *t* covers 8 sampled positions
= 14 plies, and its 6 angles are vertexed at *t*−12 … *t*−2:

```
plotted at t = 20 (r =  +0): sampled positions 6..20, angles vertexed at 8..18
plotted at t = 18 (r =  -2): sampled positions 4..18, angles vertexed at 6..16
plotted at t = 16 (r =  -4): sampled positions 2..16, angles vertexed at 4..14
```

Straightness plots one angle at the ply it is vertexed on. So even after the
stride, the fork's point at *t* is a **trailing 14-ply average whose vertex
centroid is at *t* − 7**. For an event-locked overlay this is fatal on its own:
the fork's *r* = 0 point contains the branch in 1 of its 7 steps and does not fill
the window until *r* = +12 (`curvature_plane`'s docstring says so), while
straightness's *r* = 0 point is the single turn vertexed at *t0*. Two curves with
a peak at "0" would be reporting two different events.

## 6. Figure

`figures/theta_sampling_mismatch.png` — the overlay, drawn so it does not have to
be drawn again. (a) the legal baseline's θ profile over the game under each rule,
the mean over 2000 games, with the fork study's own R²¹ windowed curve for
game 1235's nine branches landing on the stride-2 line and nowhere near the
per-ply one. (a) draws no interval: the ±1.96 SEM band it had is ≤ 0.34° wide at
every *t* against the 34.4° gap the panel is about, so it was two invisible
ribbons, and `profile sem` is still in `measure.py`'s output for anyone who wants
the number. (b) Δ vs legal per drawn class under each rule, 95% percentile
bootstrap over games, on an axis symmetric about zero with both directions named.
`random_tokens` is in the tables but not the figure: it has no `CLASS_STYLE` entry
and is neither drawn nor scored anywhere in the repo.

The two sampling rules have **no entry in `paper_style.py`** — this is a one-off
diagnostic, so the incumbent per-ply rule takes `REFERENCE` and the fork's takes
`OKABE_ITO[1]` with its own marker, both imported rather than written as literals.
If an overlay figure ever ships, that pair earns a `SAMPLING_STYLE` dict there.

## What a figure author should do

- **Two panels, two y-axes.** Straightness class baselines in one, fork branch
  curves in the other. Label each axis with its sampling rule, not just "mean
  turning angle (deg)", since that name is currently ambiguous across the repo.
- **A shared axis is available only by recomputing one side.** Running the fork's
  branches through the per-ply rule (positions 1…58, one angle per point) puts
  them on the straightness axis honestly — that is what row 1 of §3 is. Going the
  other way, re-deriving the straightness class baselines at stride 2, is also
  legitimate but it is a **new measurement whose published AUROCs do not apply**:
  §2's right-hand columns are what that costs.
- Whichever side is recomputed, say so in the caption and do not reuse the other
  study's committed numbers next to it.
- If the recomputed side is the fork's, drop the W=8 window too, or state the
  trailing-window lag in the caption. A 14-ply trailing average against a
  per-position curve is still not the same curve.

## Caveats

- Layer 5 raw only. The same two rules apply to every layer and to the SAE
  latent spaces, where the conclusion can only be stronger: SAE codes are sparse,
  so zero-length steps and undefined angles are common and the two samplings see
  different NaN sets.
- Straightness's `centered` variant is not measured here. The class-baseline
  panels draw `raw`, which is what the fork uses, so raw-to-raw is the right
  comparison — but a figure must not mix the two variants either.
- Precision differs cosmetically — `curvature_plane` works in float32,
  `fork_expanding_curvature.window.r21` casts to double, straightness upcasts
  fp16 to float32. Nothing at this magnitude turns on it.
