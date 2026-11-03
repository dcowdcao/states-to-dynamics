# A4 — activation-space geometry (M3, M4)

Layer-5 `resid_post` only. The measurement never needs the TPR: the predicted flip
direction is computed twice, once from the committed linear probe and once from the
TPR, and the two agree. Contract values below are the TPR ones so they line up with
the design doc's wording; the linear-probe column is the independent replication.

Reproduce (~4 min, CPU):

```
SPADE_LINEAR_PROBE_ACTIVATIONS=<checkout>/analysis/linear_probe/activations \
  python analysis/bracket_flip/geometry_figures.py [--camera-ready]
```

The override is needed in a worktree because `analysis/linear_probe/activations/` is
gitignored; on the shared checkout the default resolves.

Outputs: `analysis/bracket_flip/figures/{f3_activation_geometry,f4_specificity}` and
`analysis/bracket_flip/results/{summary,variance_shares,flip_cos_per_event,sanity}.csv`.
Every number below is a row of one of those CSVs.

## The quantity being measured

A flip is the **absence** of a perspective swap, so each square is scored against its
**retain direction**: the residual step that keeps its mover-relative class where the
swap would have exchanged it.

| square | class before *m* | retain direction | happens? |
|---|---|---|---|
| flanked *s* | theirs | `theirs - mine` at *s* | yes — *s* stays *theirs* |
| unflipped opponent disc | theirs | `theirs - mine` at that square | no — it swaps |
| unflipped mover's disc | mine | `mine - theirs` at that square | no — it swaps |

Both probes are linear in the filler contrast, so one 64x512 matrix covers both signs.
TPR: `pinv(M)[r_q (f_theirs - f_mine)^T]` via `tpr_probe.edit_direction(q, mine, theirs)`.
Linear probe: `W[:, q, theirs] - W[:, q, mine]`, read-only from `linear_probe.pt`.
Median cos between the two probes' directions for the same square: **0.822** (n=64
squares), so the replication is a real second measurement, not a restatement.

Two contrasts:

- **Δ = after − before** — what M3/M4 name. Carries the whole perspective swap.
- **Δ' = after − control**, strict controls only (**n = 1596** of 2000) — cancels the
  prefix, the move number and the parity by construction, so only the flip and the
  m-vs-m' placement survive.

## Cross-validation contract

| key | value | n | note |
|---|---|---|---|
| `n_events` | **2000** | 2000 | all headline single-flip events; none dropped |
| `n_events_strict` | 1596 | 2000 | `control_kind == strict` |
| `cos_flip_median` | **+0.1748** | 1596 | Δ', TPR direction |
| `cos_null_median` | **−0.0019** | 46,930 squares in 1596 events | Δ', TPR direction |
| `var_swap` | **0.6894** | 2000 | Δ, matched-control direction |
| `var_flip` | **0.0067** | 2000 | Δ, TPR direction, swap removed |
| `var_resid` | **0.3039** | 2000 | Δ, = 1 − var_swap − var_flip |

**Read the definition before comparing.** `cos_flip_median` is quoted on Δ' because
that is the flip-attributable contrast the design doc specifies. If A5 computed it on
the raw Δ instead, the matching pair is **−0.0200 / −0.0945**, also below. Both rows
are in `summary.csv`, keyed by `contrast` and `direction_source`, so a comparison is a
lookup rather than an argument.

### All four (contrast x probe) combinations

| contrast | probe | `cos_flip_median` | `cos_null_median` | P(cos at *s* > 0) |
|---|---|---|---|---|
| Δ' = after − control (strict, n=1596) | TPR | **+0.1748** | −0.0019 | **1.000** (1596/1596) |
| Δ' = after − control (strict, n=1596) | linear | **+0.1628** | −0.0017 | **1.000** |
| Δ = after − before (n=2000) | TPR | −0.0200 | −0.0945 | 0.288 |
| Δ = after − before (n=2000) | linear | −0.0052 | −0.0840 | 0.423 |

On Δ the flanked square is not near zero in absolute terms — it sits at −0.020 — but
it is **+0.075 above the null** and the null is where every other disc sits. That
offset is the swap of the other ~28 discs (median 28, range 8–60 per event) leaking
onto *s*'s direction through the role overlap PROBE_REPORT.md §5 predicts. Measured on
the residual-space directions rather than the roles themselves, off-diagonal |cos|
between the retain directions of different squares is 0.049 median, **0.341 max**
(TPR) — close to the role Gram's own 0.055 / 0.377, as it should be, since the pinv
solve is linear in `r_q`. A flanked square therefore cannot read exactly zero on Δ even
when its own class is unchanged: the floor is structural. The Δ' contrast
removes that leakage — the null lands at −0.002 — which is what the matched control was
built for.

Restricting Δ' to the 404 **loose** controls instead gives +0.1651 (TPR) / +0.1529
(linear), P(>0) = 0.998 / 1.000: strictness moves the median by 0.010, so the result is
not an artifact of the strict subset. (Recompute from `flip_cos_per_event.csv`, column
`cos_flip_after_control_*` grouped by `control_kind`.)

### Variance shares of Δ, with their null

95% percentile bootstrap over events, 1000 resamples.

| component | Δ = after − before (n=2000) | Δ' = after − control, strict (n=1596) |
|---|---|---|
| `var_swap` (matched control) | 0.6894 [0.6864, 0.6925] | 0.0847 [0.0828, 0.0864] |
| `var_resid` | 0.3039 [0.3008, 0.3069] | 0.8948 [0.8931, 0.8966] |
| `var_flip` (TPR) | 0.00669 [0.00654, 0.00685] | 0.02054 [0.02000, 0.02108] |
| `var_flip` (linear) | 0.00621 [0.00606, 0.00635] | 0.01913 [0.01865, 0.01963] |
| `var_flip_null` (TPR, unflipped disc) | 0.00152 [0.00138, 0.00167] | 0.00422 [0.00380, 0.00469] |
| `var_turn` (global to-move axis) | 0.05487 [0.05431, 0.05541] | 0.000263 [0.000245, 0.000283] |
| isotropic chance `1/(d−1)` | 0.00196 | 0.00196 |

`var_swap + var_flip + var_resid = 1` exactly (the flip direction is orthogonalized
against the swap direction before projecting). `var_flip_null` and `var_turn` are
scored against the same total but are outside that split.

`var_flip` is small in absolute terms because one square's flip is one direction in a
512-d residual that is simultaneously rewriting a whole board. The number that carries
the claim is the ratio to the empirical null: **4.4x** on Δ and **4.9x** on Δ'
(3.4x and 10.5x against isotropic chance). The confidence intervals do not overlap.

## What I found that contradicts the brief

**The global perspective swap is not the dominant axis, and it is not near-identical
across events.** The premise is half right and the half that is wrong changes how M3
has to be built.

- There *is* a single global axis and it *is* the perspective swap: PC1 of the
  per-*t*-centered pooled activations separates "who is to move" at the position each
  row was read at, and the leading direction of the Δ ensemble matches the to-move axis
  at **|cos| = 0.995**.
- That axis carries only **5.5%** of Δ's energy (`var_turn`). Rank 1 is 5.5%, rank 10
  is 27%, rank 20 is 41%.
- Across events the swap is **near-orthogonal**: median cos between the matched-control
  displacements of two different events is **0.00024** (n = 4.0M pairs). The mean
  displacement is 1.27 activation units against an rms length of 29.4 — 0.19% of the
  energy.
- The swap that matters is board-specific, because which discs exist differs per board.
  The right estimator is the one the study already built: `control − before` explains
  **68.9%** of Δ, at median cos 0.831 (IQR [0.800, 0.856]). Matching that with a shared
  subspace takes **rank 73**.

So M3's decomposition uses the matched control as the per-event swap direction, not a
global axis, and reports the global axis separately as `var_turn`. A rank-1 global
"swap" component would have been 5.5%, and the 63% difference would have landed in the
residual and been read as noise.

Two consequences worth carrying forward:

- On Δ' the to-move axis is **0.000263** of the energy — the control cancels the turn
  signal to 0.5% of what it is on Δ. That is independent evidence the control matching
  is doing what the design claims.
- The to-move signal is **twice as large at position t−1 as at position t** (class means
  ±4.56 vs ±2.29 on the axis, within-class sd 0.40–0.48). `before` is read at t−1, so
  its cloud is wider in F3(d). This is a property of the model, not a bug in the
  extraction, but it means "before vs after" comparisons in this space are not
  position-matched — another reason Δ' is the contrast to trust.

## Sanity checks (both pass)

| check | n | value | threshold |
|---|---|---|---|
| `before` identical across events sharing a prefix | 2 pairs (4 rows) | **max abs diff 0.0** over all 8 layers | < 1e-4 |
| strict-control contrast shorter than the raw before/after difference | 1596 | **1.0000** of events | == 1.0 |

Only 2 of the 2000 prefixes collide (one event per game, and prefixes at t ≥ 5 are
almost always unique), so the first check is thin but exact — the two colliding pairs
are bit-identical, not merely close. `extract_acts.check_prefix_invariance` covers the
same invariant against a separate forward pass on 8 events (6.44e-06, DATA_REPORT.md).

Median displacement lengths at layer 5, strict events: ‖after − before‖ 30.15,
‖control − before‖ 30.02, ‖after − control‖ **16.87**.

## Figures

**F3 `f3_activation_geometry`** — the perspective swap is board-specific; the flip is
what is left. Layer 5, transformer blocks indexed from 0.

- **(a)** Displacement field in a per-event frame: *y* is the TPR-predicted flip
  direction for *s*, *x* is what remains of the matched control's displacement after
  *y* is removed, both in activation units. `before` is the origin by construction;
  neither axis is defined by the `after` condition, so the vertical gap is a
  measurement. One arrow per event, matched non-flipping move → flanking move, 120 of
  n=1596 strict events drawn (stated in-panel); ellipses are 50% probability contours
  of a Gaussian fit over all 1596. Median *y*: control **−3.50**, after **−0.59**.
- **(b)** The marginal of (a): KDE of *y* per condition. Median −3.50, IQR
  [−4.25, −2.75] for the control; −0.59, IQR [−1.24, +0.08] after the flanking move.
  Band = 95% percentile bootstrap over events, 1000 resamples, n=1596; Gaussian kernel,
  Scott's rule at the full sample, held fixed across resamples.
- **(c)** Variance shares, point + 95% bootstrap CI over events, log axis because the
  values span three decades (bars would have to start at zero and would hide the flip
  term). Isotropic chance drawn as a labeled gray dashed line; the unflipped-disc null
  in NULL styling, listed last.
- **(d)** Per-*t*-centered PCA of the pooled conditions, cloud drawn once in shuffled
  order and rasterized, centroid + 50% Gaussian contour per condition, per-component
  variance ratios on the axes (2.4% / 1.7%). The condition centroids coincide because
  every condition is half black-to-move; PC1's two lobes are the to-move axis.
- **(e)** The same embedding colored by move number *t* — the §4.6 nuisance panel. It is
  uninformative, which is the point: the per-*t* centering removed it.
- **(f)** Cumulative variance of the best shared rank-*k* subspace of Δ, with the
  per-event control share as a reference line. Band = 95% bootstrap over events, 200
  resamples (each replicate is a 512x512 eigendecomposition).

**F4 `f4_specificity`** — only the flanked square resists the swap, on either probe.
2x2: rows are the direction source (TPR / committed linear probe), columns are the
contrast (Δ / Δ'). Per panel: KDE for the flanked square (hero), unflipped mover's
discs and unflipped opponent's discs, plus the pooled unflipped-disc null as a gray
dashed labeled line, last in the legend. The two unflipped classes are a light-to-dark
ramp of one hue with a shared marker (§4.6), so they also take different dashes to stay
separable in grayscale. Bands are 95% percentile bootstrap over **events** (1000
resamples) — the resampling unit is the event even for the null, whose values are
per-square, so the band reflects the right source of variation. Bandwidths (Scott at
the full sample, fixed across resamples) are printed per panel, 0.0035–0.0103 in cosine
units. n is on every legend entry.

The small secondary bump near cos ≈ −0.17 in the unflipped-opponent curve on Δ' is the
role-overlap tail: squares whose retain direction is the worst-aligned with *s*'s
(|cos| up to 0.341) pick up a share of the flip. Structural, quantified above, not
noise.

## Choices a reader might argue with

1. **The swap direction is the matched control, not a fitted global axis.** Justified
   above; the global alternative is reported as `var_turn` so both are visible.
2. **`var_flip` is a rank-1 projection share**, so its scale is set by the dimension.
   It is always quoted next to `var_flip_null` and `1/(d−1)`; the ratio is the claim.
3. **The empirical null draws one unflipped disc per event** for `var_flip_null`
   (seeded, `SEED = 42`) so the null carries one direction per event like the flip term
   does. The cosine null in F4 uses **all** unflipped discs — 58,434 squares over 2000
   events — and bootstraps over events.
4. **`cos_flip_median` is quoted on Δ'.** The raw-Δ pair is in the same table.
5. **Loose controls are excluded from Δ'** per the design doc; their separate medians
   are quoted above and differ by 0.010.
