# Recovering the spliced move without the counterfactual

Every other read-out of this branch tree scores `phi_branch(t) - phi_ref(t)`: it is handed
the game as it was actually played. That is an oracle a detector does not have. This
sub-study asks what is left when it is taken away -- one trajectory, no reference branch,
no knowledge that anything was spliced -- and reports it across the nine spaces.

## What it cannot claim

**This is not a competitive detector and is not offered as one.** The supervised probe in
`analysis/move_legality` localizes the same event at 0.874 and separates it at AUROC 0.960,
and the model's own `1 - P(played)` does it at 1.000 with no probe at all. The number here
is what survives with no labels and no counterfactual, which is a different question.

**The norm-only baseline has not been run on this tree, and it may own the result.** On the
straightness corpus, `-z` of the activation norm alone -- no step, no angle, no trajectory --
reaches 0.808 AUROC in GatedAnneal, `d_t` regresses on `(||z_t||, ||z_{t+1}||)` at R^2 0.87-0.88,
and partialling the norm out drops GatedAnneal from 0.587 to 0.173 while raw layer 5 *improves*.
`curvature_discriminant`'s npz carries `d` and `theta` but not `norm`, so the check needs one
GPU encode pass. Until it is run, treat this as a magnitude claim, not a geometry claim.

**Seed 42 only, and 575 of the 3,185 branch-complete games.** The population is whatever
`curvature_discriminant/meta.npz` holds; it was selected for that study's probe-holdout guard
and predates `extend_branches`. Rebuilding `d`/`theta` over the full tree is the obvious
extension and needs no new design.

## The statistic

Per move index `t`, fit the mean and SD of step length over legal games, and score
`g_t = -(d_t - mu(t)) / sigma(t)`: how much shorter than usual the model's state update is.
Then contrast `g` across each candidate move `c` -- the mean over the `m` steps after it minus
the mean over the `m` before -- and scan the whole game. Peak is the detection score, argmax
the estimated change point.

Three choices, each of which was the difference between working and not:

**Per-move-index, not pooled.** Move index explains 81.3% of the variance of `d`. Pooled into
one Gaussian the score reports how late in the game you are: 0.034 hit / 0.461 AUROC, i.e.
chance, against 0.283 / 0.673 for the profile baseline. This is the thing the paired forms get
for free by subtraction, and it is why `curvature_discriminant` says `phi_b(t)` alone
"collapses to a ribbon". It also absorbs the mover-relative parity sawtooth, since parity is a
function of the move index -- which is what lets the scan run at every ply rather than stride 2.

**One-sided, not a distance.** A Mahalanobis over the full 21-D ReStraV descriptor, or any norm
of a difference, throws the sign away, and then it fires *harder* on real human championship
games -- legal in every position -- than on the violations: 0.694-0.706 against 0.616. The
one-sided version puts championship at 0.376, the opposite tail. Two-sidedness detects unusual,
not illegal.

**Paired across the candidate, not against the population alone.** Writing
`phi_t = mu_t + b_game + eps_t`, the unpaired form scores `b + eps` and competes against
`Sigma_b + Sigma_eps`; the contrast scores `eps_R - eps_L` and the game-level term cancels. Its
equal-width windows also leave `argmax` unbiased in position, which the `sqrt(c(n-c)/n)` factor
in CUSUM/GLR/E-divisive does not. It is the MOSUM statistic (Eichinger & Kirch, *Bernoulli*
24(1), 2018), equivalently the window-sliding discrepancy of Truong, Oudre & Vayatis (2020,
§5.2.1) with a Mahalanobis cost.

### How the window was chosen, and what that does and does not buy

`MOSUM, m = 4` is picked on the straightness violation corpus by `AUROC + hit rate`, the
criterion the study is read by. Three caveats a reader is owed:

- **The two corpora are not independent.** Their splices differ -- straightness resamples the
  suffix and injects only empty-square violations, this tree keeps the suffix and carries R1
  and a matched legal control -- but **501 of the 573 scored games share their source legal
  game** with a selection-corpus violation, and the source game is exactly what carries the
  game-level idiosyncrasy the MOSUM cancels. This is weaker than disjointness.
- **The selection task is not the reported task.** The straightness corpus has no legal-splice
  arm, so `sweep()` scores violations against their *unspliced* source game -- a
  `vs_reference`-shaped contrast, while the headline is `vs_control`. On this tree those differ
  a lot (GatedAnneal 0.811 vs 0.616).
- **The criterion sums two incommensurable scales and the margin is 0.0074.** On AUROC alone
  the *unpaired* forward window wins (0.641 vs 0.618); on hit rate alone MOSUM wins (0.198 vs
  0.168). The theory above is what breaks the tie, not the sweep. Under the unpaired window
  Standard reaches 0.532-0.537 rather than 0.511, so "four at chance" is window-dependent.

What bounds the damage: re-running the grid on this corpus afterwards -- reported, never used
to re-select -- puts the same setting on top for GatedAnneal and within 0.0001 of the top for
raw layer 5.

## Results

`figures/changepoint_by_space.png`, `results/changepoint_localization.csv`. Two panels over
one sort order, so a space is the same column in both: (a) does the score know the game
contains an illegal move, (b) does its argmax say which ply. They separate the spaces very
differently -- (a) spans 0.51-0.62, (b) spans 0.04-0.34 -- which is the point of drawing both.

**Caption.** The figure carries no note under its axes, so this paragraph is its caption and
it is not standalone without it. Both panels: nine representation spaces, one illegal branch
per source game drawn at seed 42, n = 575 branch-complete games, whiskers 95% percentile
bootstrap CIs over games (1,000 resamples, shared draws across spaces). Score is the
per-move-index standardized step-length drop contrasted across each candidate move (MOSUM,
m = 4 plies each side) and scanned blind over the whole game -- 34 candidates, no knowledge
that a move was spliced, no counterfactual branch; the window was chosen on the straightness
violation corpus. (a) AUROC of the illegal branch's peak against the same game's legal-splice
branch, never flipped, so a bar at 0.5 is the finding. (b) exact-match rate of the argmax
against t0. Gray bands are empirical nulls: in (a) each game's legal control scored against
the second legal control it carries, in (b) those two arms and the untouched reference game,
none of which contain an illegal move. Hue is the architecture family, hatch its second
dictionary, and raw layer 5 keeps reference styling wherever it sorts.

| space | AUROC vs legal splice | vs raw | vs clean game | hit@t0 | within ±1 |
|---|---|---|---|---|---|
| GatedAnneal | **0.616** [0.591, 0.640] | **+0.055** [+0.030, +0.080] | 0.811 | **0.343** | 0.485 |
| GatedSAE | 0.604 [0.578, 0.627] | +0.043 [+0.019, +0.065] | 0.745 | 0.270 | 0.431 |
| BatchTopK | 0.593 [0.570, 0.616] | +0.032 [+0.010, +0.056] | 0.729 | 0.242 | 0.362 |
| TopK | 0.590 [0.567, 0.611] | +0.029 [+0.009, +0.050] | 0.712 | 0.205 | 0.348 |
| raw layer 5 | 0.561 [0.538, 0.583] | — | 0.660 | 0.101 | 0.301 |
| CosineSpaDE | 0.553 [0.533, 0.574] | -0.008 [-0.036, +0.023] | 0.619 | 0.057 | 0.143 |
| SpaDE | 0.540 [0.519, 0.561] | -0.021 [-0.042, +0.001] | 0.593 | 0.040 | 0.179 |
| PAnneal | 0.533 [0.508, 0.555] | **-0.029** [-0.047, -0.009] | 0.598 | 0.085 | 0.210 |
| Standard | 0.511 [0.493, 0.529] | **-0.050** [-0.074, -0.026] | 0.518 | 0.054 | 0.148 |

**Four dictionaries clear raw layer 5, two sit below it, and two are indistinguishable.** The
marginal CIs overlap and cannot settle that -- the `vs raw` column is the paired difference on
the same bootstrap resamples, which can. Uncorrected for eight comparisons. The ordering
reproduces `curvature_discriminant`'s *supervised* ordering from a completely unrelated
statistic, which is the strongest internal check available.

**Localization is the stronger half.** GatedAnneal recovers the exact ply 0.343 of the time
against a uniform chance of 1/34 = 0.029 (11.6x) and against its own legal-splice control arm
at 0.038 (8.9x). All three false-alarm arms -- the legal control, the second legal control, and
the untouched reference game -- sit at 0.021-0.045 in every space, i.e. at uniform chance, so
the detector does not fire at `t_0` when the move played there is legal.

**`hit@t0` is exact-match and understates raw layer 5 about twofold.** The mean MOSUM profile
has a right shoulder: raw's argmax lands on `t0` and on `t0+1` equally often (0.101 each),
because mean `g` at the first perturbed step is +0.43 against +1.36 at the second. Offset 0 is
the mode in every SAE space. The `within ±1` column is the same statistic at one ply tolerance.

**Per rule, R2 is easiest and R1 hardest** (GatedAnneal 0.117 / 0.473 / 0.304 for R1/R2/R3),
the same ordering the supervised probe finds.

**Curvature proper carries almost none of this.** The `theta` block scores 0.53-0.58 and only
in the *straighter* direction. That agrees with RECURVE (Shin et al., NeurIPS 2024), which
scores time-series boundaries as `1 - NORM(kappa)` and proves curvature is *lower* at a change
point, and with this study's own finding that violations are straighter at L5-L7. The naive
reading -- sharp turn implies anomaly -- is the wrong sign.

## The same score read as a detector, not as a forced choice

`located()` names exactly one move per sequence. A legal game therefore cannot come back
clean, `hit_control` is a pick-a-position rate rather than a false-alarm rate, a game with
two illegal moves could score at most 1/2, and the 33 candidates that are not the argmax are
never scored at all. `results/threshold_localization.csv` and
`results/threshold_localization_regenerated.csv` read the same `s` through a **cut** instead:
`s >= tau` at every candidate of every arm that ground truth covers, so a sequence gets zero,
one, or many flags. The unit of a row there is a *position*, not a game.

`tau` is the cut maximizing balanced accuracy `(TPR + TNR)/2` -- Youden's J on the same ROC
the AUROC summarizes. It is found exactly: accuracy is piecewise constant in `tau` with
breakpoints only at observed scores and is not unimodal, so a grid can straddle the optimum
and a bisection can settle on a local one. `changepoint.threshold` sweeps every breakpoint in
the one sort the ROC needs. It is fitted **5-fold by source game** and reported out of fold.

### What has ground truth, and what does not

| | spliced tree | regenerated suffix |
|---|---|---|
| positions | 26,252 over 4 disjoint arms | 63,580 over 870 violations + 1,000 legal games |
| positive | 575 (2.19%) | 870 (1.37%) |
| scored band | `t` in [14, *t0*] | all 34 candidates |

The spliced tree's cut at *t0* is the same one the argmax columns take, for the same reason:
past *t0* a branch replays the source game's suffix onto a board that no longer exists. It is
applied to the legal arms too, so the two classes keep identical move-index marginals -- an
unmatched cut would let a threshold separate *position* rather than legality. The regenerated
corpus needs no cut: its continuation is drawn from the engine's legal moves. That legality
holds on a board built by `build_games.place_no_flip` (disc placed, nothing flipped, turn
passed), which is a **stated convention and not a fact** -- no legal play reaches that state.

The four disjoint arms are `illegal`, `reference`, `control`, `control2`. R1/R2/R3 are
re-draws from the same illegal branches as `illegal` and would count those rows twice.

### What the cut finds

Raw layer 5 and the best space, both corpora:

| corpus | space | `hit@t0` | per-position AUROC | `tau` | balanced accuracy (OOF) | flags per legal game |
|---|---|---|---|---|---|---|
| spliced | raw L5 | 0.101 | 0.744 [0.723, 0.765] | 0.869 | 0.680 [0.660, 0.700] | 2.63 |
| spliced | GatedAnneal | 0.343 | **0.855** [0.837, 0.872] | 1.156 | **0.781** [0.763, 0.798] | 1.63 |
| regenerated | raw L5 | 0.155 | 0.761 [0.745, 0.777] | 0.782 | 0.704 [0.688, 0.720] | 8.17 |
| regenerated | GatedAnneal | 0.434 | **0.894** [0.883, 0.905] | 0.960 | **0.819** [0.806, 0.831] | 6.15 |

**The space ordering is the argmax ordering.** Spearman 0.93 against `hit_illegal` on the
spliced tree; the top six are identical and raw layer 5 is fifth in both. Only the three
spaces at chance reshuffle. The cut is a more sensitive read of the same signal, not a
different one -- which is the check that it measures what the study already measures.

**The cut is not overfit, but for the weak spaces it is not fixed either.** In-sample minus
out-of-fold is <= 0.02 everywhere. `fold_thresholds` carries the five per-fold cuts, and they
agree to 0.001 for GatedSAE while Standard swings 0.459 to -0.267 and SpaDE 0.010 to 0.310. A
threshold is only a threshold where the score separates; the accuracy CI holds the cut fixed
and so does **not** cover that variance.

**Abstention is what the cut buys, and it is not enough to make this an alarm.** On the
spliced tree 0.61-0.93 of all-legal games draw no flag at all, against **1.000 by
construction** under the argmax. But at 1.4-2.2% prevalence the balanced operating point
still puts 1.6-14 flags on a legal game. The two corpora are not comparable on this column:
a spliced legal arm is scored over `t <= t0`, about 12 positions, while a regenerated legal
game is scored over all 34, so the regenerated 0.98-1.00 is mostly band length. Per position
this is a graded score, not a usable detector, and the forced choice was hiding that.

**Plain accuracy is a degenerate objective here.** `threshold_plain_accuracy` sits near 5 in
every space: at 1-2% prevalence the accuracy-maximizing cut flags almost nothing and scores
~0.98 by calling every move legal. It is in the table to show that, not to be used.

### Figure notes

`threshold_detection.png` (spliced) and `threshold_detection_regenerated.png` (regenerated),
one figure per corpus, two panels sharing one sort order so a space is the same column in
both: (a) per-position AUROC, tie-averaged, illegal positions against legal ones; (b)
out-of-fold balanced accuracy at the fitted cut. Every interval is a 95% percentile bootstrap
over **source games as clusters**, 1,000 resamples, seed 42 -- games and never positions,
because one game's candidates overlap in `m - 1` steps. n is 575 games / 26,252 positions
(spliced) and 1,000 games / 63,580 positions (regenerated). The dashed gray line is 0.5.
Panel (a)'s axis label differs between the two figures on purpose: the regenerated corpus has
no matched legal splice, so its negatives are moves in unspliced legal games.

**The model's own illegal moves are harder than spliced ones.** The regenerated table also
scores the 2,000 free-running rollouts, whose legality is adjudicable up to `first_illegal`
and no further; 96 of those first illegal moves fall in the band. At the headline cut
`rollout_tpr` is 0.30-0.56, below each space's own pooled TPR. Read it only beside
`rollout_flagged_per_legal_game`: SpaDE has the highest rollout TPR (0.563) because it flags
the most (14.3 per legal rollout), not because it detects best.

## Pipeline

```
python analysis/violation_divergence/changepoint_localization/test_threshold.py   # no data, no GPU
python analysis/violation_divergence/changepoint_localization/measure.py      # -> results/
python analysis/violation_divergence/changepoint_localization/make_figures.py [--camera-ready]
PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/measure_regenerated.py
PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/make_figures_regenerated.py
```

Each `measure` script writes two tables off one scan: the argmax one it always wrote, and the
thresholded one. Adding the second moved no number in the first -- both `changepoint_*.csv`
reproduce byte-identically.

`measure.py` is CPU-only and takes under a minute: it reads the cached `d` that
`curvature_discriminant/branch_geometry.py` already wrote for all nine spaces, and the
straightness study's own geometry for the legal profile. No GPU pass, no re-encoding.
Arms come from `spade.move_legality.localize.arms`, so these rows are column-comparable with
that study's `localization_by_space.csv`.
