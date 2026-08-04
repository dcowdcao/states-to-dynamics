# Can the nine fork panels share one basis in R^21?

**Yes — and the plane that wins needs no fitting at all.**

A shared fitted plane costs nothing against the per-space refit (mean separation
1.21 vs 1.12 pooled SDs; it wins in 6 of 9 spaces). Fixed named axes — Δ mean *d*
vs Δ mean θ — beat both in **every** space, 1.32–2.00 SDs against 0.70–1.80 for
the refit, while capturing only 0.11–0.20 of the descriptor variance the fitted
planes are built to maximise. Variance is not the quantity that matters here, and
the figure is better off not spending an 8-trajectory sample fitting for it.

The caption disclaimer changes, but only halfway: a shared basis over
*per-space-z-scored* descriptors makes **arrangement and direction** comparable
cell to cell, not absolute magnitude. That is the honest wording now on the
shared and named grids.

Everything below is game 0, t0 = 10, anchors t ≥ t0 (10 windows of W = 8 sampled
positions each), the same game and window as the calibration number.

## The numbers

Illegal-vs-control centroid distance in pooled within-group SDs (per coordinate,
so 2-D and 21-D are on one scale), min–max over the nine spaces:

| plane | separation | its permutation-null median | separation ÷ null | p (exact, n = 28) |
|---|---|---|---|---|
| activation space, refit per space (status quo) | 0.13–0.39 | 0.21–0.37 | 0.45–1.05 | 0.50–0.89 |
| R^21, refit per space | 0.70–1.80 | 0.39–0.81 | 1.15–2.77 | 0.04–0.39 |
| **R^21, one shared fitted plane** | **0.70–1.64** | 0.40–0.81 | 1.07–2.16 | 0.04–0.46 |
| **R^21, fixed named axes (mean d, mean θ)** | **1.32–2.00** | 0.56–0.70 | **2.32–3.10** | **0.04–0.18** |
| R^21, one shared Isomap | 0.35–0.88 | 0.21–0.41 | 1.61–2.68 | 0.11–0.29 |
| full 21-D (no projection; the ceiling) | 1.69–3.07 | — | — | — |

Calibration check: scoring the status-quo activation plane over *all* positions
≥ t0, as the calibration number does, gives **0.08–0.45** against the quoted
0.06–0.33 — the same measurement, to within whatever differs in the exact pooling
convention. The metric implementation is comparable.

Variance kept by each plane, per space:

| plane | fraction of that space's descriptor variance |
|---|---|
| refit per space (by construction the maximum) | 0.35–0.63 |
| shared fitted plane | 0.27–0.55 |
| named axes | 0.11–0.20 |

The shared plane's own top-2 hold 25% + 18% = 44% of the *pooled* variance, and
it leads on **mean θ** (PC1) and **max θ** (PC2): fitting on the pooled
descriptors lands on an angle plane, which is most of why it costs so little.

## What each option costs

**Shared fitted plane.** Loses 0.03–0.13 of per-space variance and, in three
spaces, a little separation (TopK 1.80 → 1.64; GatedAnneal 0.90 → 0.87;
CosineSpaDE unchanged). In the other six it *gains* — BatchTopK 0.95 → 1.48 — for
the ordinary reason that an unsupervised per-space fit is free to spend its two
directions on that space's idiosyncratic noise. So: no cost worth naming.

**Named axes.** Loses two thirds of the variance the fitted planes keep and wins
on every space anyway. Cost: the reader must accept that the two axes were chosen
by the parent straightness study rather than by this game, and that a third
component might have carried more.

**Per-space refit.** Costs the entire comparability claim, which was the point of
the exercise, and buys nothing.

**Isomap.** Raw separation drops to 0.35–0.88 and no space reaches the p floor.
Its null is also lowest, so after normalisation it is not embarrassing
(1.61–2.68), but as an ordering statistic it is the weakest of the four. At 90
points per space (810 pooled) with 12 neighbours the fit is defensible; the
result is simply that the nonlinear embedding buys nothing here. It ships because
§4.8 requires a nonlinear companion to a separation claim, and it is evidence
*for* the linear planes, not against them.

## The red flag, checked

R^21 raises the raw separation ~5x over the activation plane. That is exactly the
"large separation from n = 1" the brief warned about, so it is scored against an
exact permutation null — all 28 ways of relabelling the 8 non-reference branches
6/2 — rather than asserted.

Part of the jump is null inflation: the null median roughly doubles (0.2–0.4 →
0.4–0.8) when the same points are pushed through the descriptor, because with
only 8 effective trajectories any 6/2 split of eight heterogeneous branches opens
a centroid gap. But it does not explain the jump. After dividing each plane by
its own null median, the status-quo plane sits at **0.45–1.05** — at or below a
random relabelling, i.e. it does not separate the classes at all — while the
named plane sits at **2.32–3.10**. The net gain is ~3x, not ~5x, and it is real
on this game.

The direction is consistent and interpretable, and it is the one claim here that
is comparable across spaces in absolute units, since degrees are degrees
everywhere: over the post-fork window an illegal branch turns **+1.3° to +2.7°**
more than the reference in all nine spaces, a legal control **−0.3° to +1.0°**.

## Sample size: is the fit supported?

No, and that is the finding. 90 rows per space sounds adequate for a 21-D fit
until you notice the 10 windows per branch overlap in 7 of their 8 samples; the
effective unit is the **branch**, of which there are 8 (plus the reference, which
is exactly zero by construction). A plane fit in 21 dimensions from 8 effective
points is not supported, and the permutation null bears that out — the fitted
planes' nulls are the highest of the four. This is the strongest argument for the
named axes: they are the only plane whose orientation does not consume the
sample, and they are also the only one that reaches p = 0.04 in six of nine
spaces.

## Costs that are not about the basis

Routing through R^21 costs the figure its zero-by-construction prefix. A full
21-D descriptor needs 8 sampled positions = 15 plies, and game 0 forks at t0 = 10
with the analysis window already clipped at the start of the board, so *every*
plotted window straddles t0 and the first point sits at t = 16. The grids keep
the muted prefix line from the first sampled position to the first anchor and
label it as prefix + warm-up; a game with t0 ≥ 15 would restore a genuine zero
segment. W was fixed at the minimum and not tuned — another sub-study owns that
question.

## What should change in the parent figure

If `space_fork_grid` is redrawn through R^21, use the **named** plane, keep the
shared limits and the ticks, and replace "planes refit per space (arb. units, not
comparable across cells)" with "same 21 named coordinates in every cell:
arrangement and direction are comparable, magnitudes are in each space's own
descriptor SD". Keep the refit grid as the control it now is.

Figures and how to rerun: `analysis/r21_shared_basis/`.
