# The fork in a supervised curvature plane

`analysis/violation_divergence/game_trajectory/space_fork_grid_pca` draws one
game's nine futures in the top-2 uncentred PCs of the activation displacement
`D(t) = z_branch(t) − z_ref(t)`. PCA is unsupervised, so those are the directions
of most variance, not the directions that separate the branches — the point
`analysis/straightness/feature_pca` makes in its own figure title.

This sub-study replaces the plane with a **supervised** one, living in the 21-D
ReStraV curvature descriptor (R²¹) rather than in activation space:

- **axis 1** — the LR-21 probe's discriminant direction, applied to the
  descriptor *difference* `φ_branch(t) − φ_ref(t)`. The projection is linear, so
  the coordinate is exactly the branch's probe log-odds minus the reference's:
  *how much more violation-like than the game as played*. The reference is 0 by
  construction, and so is every branch before *t0*.
- **axis 2** — the leading uncentred PC of the same differences with axis 1
  projected out. Unsupervised, refit per space, arb. units.
- **axis 3** — move index, as in the parent figure.

## Pipeline

```
python analysis/violation_divergence/curvature_discriminant/branch_geometry.py   # d/theta per branch, 9 spaces
python analysis/violation_divergence/curvature_discriminant/evaluate.py          # the honesty numbers
python analysis/violation_divergence/curvature_discriminant/make_figures.py      # -> figures/
```

Intermediates go to `$SPADE_DIVERGENCE_DIR/curvature_discriminant/`.

## The three design choices, and why

**Window.** The descriptor is computed on a **trailing 8-ply window ending at
t**. Trailing, not centred, because a centred window would put the injection's
effect at `t < t0` and imply the branches separate before they do. 8 plies
because `restrav_features` slices `d[:, :7]` and `theta[:, :6]`: a window of *W*
sampled positions yields *W−1* steps and *W−2* angles, so 21 dimensions need
*W ≥ 8*, and below that the function silently returns a shorter vector instead of
raising (`window_features` asserts the width). 8 is also the *best* window —
held-out AUROC falls monotonically with length in every space (raw L5:
8 → 0.72, 10 → 0.64, 12 → 0.58), because a violation perturbs a fixed handful of
`d`/`theta` entries and a longer window only dilutes them. At exactly 8 the four
moments are deterministic functions of the raw block, so the descriptor spans 13
dimensions rather than 21.

The window slides **every ply**, not every second ply as in the parent figure: at
stride 2 an 8-position window would be 16 plies out of a 17-position figure,
which is not a sliding window at all. The parent's stride-2 sampling exists
because the mover-relative board code inverts each ply and the raw path is a
sawtooth; the hope here was that differencing against the reference at the same
*t* would cancel it, since the frame flip is common to both. `evaluate.py`
measures that instead of assuming it (`parity_ratio`) — and it only *halves* the
alternating component, which is still enough to hide the fork. So the **plotted
points** are subsampled back to every 2nd ply at *t0*'s parity, exactly as the
parent figure does; only the window is stride 1.

**Differenced, not raw.** Two objects were available: the branch's own trajectory
descriptor `φ_b(t)`, and the descriptor of the displacement `D(t)`. Neither
works. `φ_b(t)` carries the strong *t*-dependent curvature clock every class
shares (`analysis/straightness/curvature_vs_t.png`), so all nine branches move
together and the figure collapses to a ribbon — exactly what FIGURE_GUIDELINES
4.8 says to take out of the plane first. `D(t)` is degenerate for the reference
(zero displacement ⇒ `d = 0`, `theta` undefined) and its geometry is a different
object from the trajectory geometry the probe was trained on. The paired
difference `φ_b(t) − φ_ref(t)` keeps both terms in the probe's own domain, and
the clock, common to both, cancels.

**Scaling.** The descriptor mixes degrees with activation-scale lengths whose
variances differ ~450× across dictionaries, so the difference is divided by the
training split's per-column SD before projection, following
`spade.straightness.classify.lr_auc`. The z-scoring *mean* cancels in the
difference; only the scale survives.

## The honesty guard

A direction fit on legal-vs-violation separates legal from violation by
construction. Four things keep the figure from being circular, and all four are
reported by `evaluate.py`:

1. **Disjoint games.** Both studies draw legal games from the same seeded stream,
   so divergence `game_id` *k* IS straightness legal game *k* — until divergence
   skips a prefix at *k* = 696 and the indices shift. The mapping is therefore
   looked up by tokens, not assumed. The probe is fit only on straightness pairs
   whose source game index is even **and not in the evaluation set**; the second
   condition is what actually enforces the guard, because the plotted game is
   chosen for the figure and its index can land on either side (game 1235 is
   straightness game 1236, which is even).
2. **The two legal controls are the real test.** They are legal moves at *t0*
   that the probe has never seen in any form. If the axis read legality they land
   at 0 with the reference; if it read "a different move was played" they land
   with the illegal branches.
3. **R1 is doubly held out.** The straightness study injects only empty-square
   violations (categories 3–6), so no occupied-square move is anywhere in the
   probe's training set.
4. **Every separation is divided by its own permutation null.** A gap in SDs is
   inflated by its own denominator, so the reported number is the ratio to the
   same statistic on the same plane with the branch labels shuffled — exactly,
   over all C(8,2) = 28 assignments of 6 illegal and 2 control branches, giving a
   floor on the exact *p* of 1/28 = 0.036. The aggregate AUROC gets the matching
   null: labels permuted **within** each game, so t0, board and game length are
   all preserved.

Nothing is refit under the permutation, and here that is legitimate rather than
lazy: axis 1 comes from other games' labels and axis 2 from the whole evaluation
population, so this game's labels cannot move the plane. A plane fit on the
plotted game has no such licence, and this is where **effective sample size**
bites. The 17 plotted positions of a branch are windows overlapping in 7 of 8
samples, so one game is 8 branches in 21 dimensions, not 136 points. Any plane
fit on that game — supervised or not — spends the whole sample, and will separate
whatever it was pointed at. That is a stronger objection to a fitted per-game
plane than circularity is, and the design's answer is to fit neither axis here.

And one thing the figure cannot escape: **the axis is calibrated at one move
index**. It is fit on the window ending at *t0*, the only window whose paired
contrast is a single token, matching the divergence branch design. Panel (j)
measures the transport directly and it is gone by *r* = 2, so the 3-D panels give
a full marker only to the branch position and draw the rest of the path faint.

## Figure

`figures/space_fork_grid_discriminant.png` — the nine spaces, plus a bottom row
carrying the numbers a 3-D projection cannot: (j) does the axis survive the move
index, (k) legality or novelty, (l) where each rule group lands.

## Results (575 held-out games, 5175 branches; probe trained on 1000 pairs)

At *r* = 0, illegal vs the two **held-out legal controls**, against a within-game
label-permutation null of 0.500 [0.483, 0.517]:

| space | probe AUROC | illegal-vs-control | random dir. (95th) | ‖Δφ‖ alone | flip ρ |
|---|---|---|---|---|---|
| raw layer 5 | 0.723 | 0.634 [0.619, 0.650] | 0.624 | 0.587 | **+0.42** |
| GatedAnneal | 0.927 | **0.859** [0.849, 0.869] | 0.809 | 0.692 | −0.12 |
| GatedSAE | 0.912 | 0.824 [0.813, 0.834] | 0.795 | 0.686 | −0.10 |
| TopK | 0.879 | 0.812 [0.800, 0.825] | 0.779 | 0.642 | −0.06 |
| BatchTopK | 0.864 | 0.803 [0.791, 0.816] | 0.761 | 0.623 | −0.05 |
| CosineSpaDE | 0.683 | 0.635 [0.618, 0.650] | 0.670 | 0.666 | +0.21 |
| PAnneal | 0.684 | 0.631 [0.617, 0.644] | 0.609 | 0.570 | +0.20 |
| SpaDE | 0.719 | 0.609 [0.595, 0.623] | 0.624 | 0.612 | +0.27 |
| Standard | 0.642 | 0.553 [0.538, 0.569] | 0.570 | 0.589 | +0.29 |

**What holds.** The axis reads legality, not novelty. The legal controls land at
+0.00 to +0.06 log-odds — on the reference — while R2 lands at +0.38 to +4.46 and
R3 at +0.30 to +3.37. R1, which no straightness violation contains in any form,
still lands at +1.26 to +2.16 in the four sparse dictionaries: real
out-of-distribution generalization. The group ordering R2 > R3 > R1 ≫ controls is
the taxonomy's own "how much board computation to detect" order.

**What does not.** Three things, and each is bigger than the effect above.

1. *Most of the separation is magnitude, not direction.* A **random** line
   through the same R²¹ reaches 0.57–0.81, and the plain norm ‖Δφ‖ reaches
   0.57–0.69 — beating the fitted axis outright in CosineSpaDE, SpaDE and
   Standard. The descriptor difference is simply larger for a branch that
   perturbed the trajectory more. Supervision clears the random-direction 95th
   percentile in 6 of 9 spaces, by 0.02–0.05 in the weak ones.
2. *The axis is calibrated at one move index.* By *r* = 2 the illegal-vs-control
   AUROC is 0.32–0.65 — at chance, or **inverted** in the four sparse
   dictionaries. A fork drawn over move index in this plane is a fork drawn in a
   coordinate that means something at exactly one height.
3. *On one game it separates nothing.* Game 1235 scores **0.09–1.05×** its own
   exact permutation null across the nine spaces, exact *p* 0.36–1.00. The fixed
   named axes (mean *d*, mean θ) do no better on this game, 0.24–1.10×. Neither
   reproduces the 1.07–2.77× and 2.32–3.10× measured on game 0 — the n = 1
   statistic is not stable across games, which is what 8 branches in 21
   dimensions predicts.

**The flip confound**, where it bites. Within the legal controls alone — legality
constant — the axis rises monotonically with discs flipped in raw layer 5:
1 flip −0.24, 2 → +0.07, 3 → +0.38, 4 → +0.51, 5 → +0.88 (Spearman ρ = +0.42).
A 5-flip legal move therefore scores as violation-like as the average R2
violation. This is the *weak* regime: the four sparse dictionaries have ρ = −0.05
to −0.12 and their control bins stay flat, so there the axis is not a
board-change read-out. Weak axis ⇒ board-change read-out; strong axis ⇒ legality.

**Parity.** Differencing against the reference removes only 3–46% of the
mover-frame alternating component (`parity_ratio` 0.54–0.97), so the plotted
points stay at stride 2.
