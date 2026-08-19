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

## Pipeline

```
python analysis/violation_divergence/changepoint_localization/measure.py      # -> results/
python analysis/violation_divergence/changepoint_localization/make_figures.py [--camera-ready]
```

`measure.py` is CPU-only and takes under a minute: it reads the cached `d` that
`curvature_discriminant/branch_geometry.py` already wrote for all nine spaces, and the
straightness study's own geometry for the legal profile. No GPU pass, no re-encoding.
Arms come from `spade.move_legality.localize.arms`, so these rows are column-comparable with
that study's `localization_by_space.csv`.
