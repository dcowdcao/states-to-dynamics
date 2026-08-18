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

The window (`MOSUM`, `m = 4`) is picked on the straightness violation corpus, which shares no
game with this one, so the reported numbers are not selection-inflated. Re-running the grid on
this corpus afterwards -- reported, never used to re-select -- puts the same setting at the top
for GatedAnneal (0.936 on `AUROC + hit`, against 0.905 for the runner-up) and within 0.0001 of
the top for raw layer 5, so nothing was lost by choosing it elsewhere.

## Results

`figures/changepoint_auroc_by_space.png`, `results/changepoint_localization.csv`.

| space | AUROC vs legal splice | vs clean game | hit@t0 | control hit |
|---|---|---|---|---|
| GatedAnneal | **0.604** [0.578, 0.630] | 0.808 | **0.332** | 0.038 |
| BatchTopK | 0.590 | 0.733 | 0.217 | 0.028 |
| GatedSAE | 0.587 | 0.731 | 0.259 | 0.045 |
| TopK | 0.584 | 0.713 | 0.203 | 0.024 |
| raw layer 5 | 0.557 | 0.656 | 0.117 | 0.026 |
| CosineSpaDE | 0.538 | 0.603 | 0.037 | 0.024 |
| PAnneal | 0.526 | 0.591 | 0.083 | 0.021 |
| SpaDE | 0.523 | 0.577 | 0.037 | 0.026 |
| Standard | 0.506 | 0.515 | 0.050 | 0.035 |

Chance for the hit rate is **0.020-0.029** -- the circular-shift surrogate, not the flat
`1/34`. The argmax of an autocorrelated score is not uniform and the candidates of one game
overlap in `m-1` steps, so rolling each curve (which preserves its shape and destroys only the
alignment to `t0`) is the honest null. The legal-vs-legal null is 0.492-0.512 in every space,
which is what makes the 0.5 line empirical.

**Four dictionaries clear raw layer 5 and four are at chance.** The ordering reproduces
`curvature_discriminant`'s *supervised* ordering (GatedAnneal > GatedSAE > TopK > BatchTopK,
then raw, PAnneal, SpaDE, Standard) from a completely unrelated statistic, which is the
strongest internal check available. Standard's CI includes 0.5.

**Localization is the stronger half.** GatedAnneal recovers the exact ply 17x more often than
the shift-preserving chance level, and its legal-splice control sits at chance -- so the
detector does not fire at `t_0` when the move played there is legal. Detection is the weaker
half because the peak of a whole-game scan is a noisier statistic than the argmax.

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
