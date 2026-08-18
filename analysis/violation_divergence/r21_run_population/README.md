# Run from the reference, over the population

`../r21_shared_basis/run_from_reference` asks the question the nine fork grids raise and
none of them can answer — how far an illegal branch actually runs from the reference, and
how much of that run each candidate plane keeps — off **one game**, where the whole sample
is 8 non-reference branch trajectories and no interval is available. This is the same two
panels over **n = 100 branch-complete source games**, with the mean and its 95% CI.

Nothing about the quantity changes: same 21-D ReStraV descriptor, same W = 8 window, same
four candidate planes, same anchors (t − t₀ = 6…24), same ratio to that game's own legal
control. What changes is that the plane fits, the z-scoring and the control divisor are all
**per game**, so a game is one independent item and the mean over games has an interval that
means something.

```
python analysis/violation_divergence/r21_run_population/measure.py [--n_games N]
python analysis/violation_divergence/r21_run_population/make_figures.py [--camera-ready]
```

`measure.py` writes `$SPADE_DIVERGENCE_DIR/r21_run_population/runs.npz` at ~1.6 s/game,
dominated by the eight SAE encodes; `--n_games` defaults to the 100 this figure draws.
`make_figures.py` reads it and writes the figure plus
`results/violation_divergence/r21_run_population.csv`, one row per (space, series) with the
mean, its 95% CI and the median across games.

## The figure

`figures/run_from_reference_population.png`

**Over 100 games an illegal branch does not run further from the reference than a second
legal move does — and every plane on offer moves it further out than the space it is a
plane of.** One column per representation space, raw layer 5 first. Mean over games, bars
= 95% CI of that mean; the dashed line at 1 is the legal control the ratio divides by, and
the gray triangles are the empirical null (a *third* legal move against that same control,
full R²¹).

| space | activation refit | refit | shared | named | full | control 2 |
|---|---|---|---|---|---|---|
| raw layer 5 | 1.04 ± 0.04 | 1.04 ± 0.08 | 0.98 ± 0.08 | 1.02 ± 0.09 | 0.86 ± 0.05 | 1.00 ± 0.06 |
| PAnneal | 1.04 ± 0.04 | 1.03 ± 0.08 | 0.99 ± 0.08 | 1.04 ± 0.10 | 0.85 ± 0.05 | 1.00 ± 0.07 |
| Standard | 1.08 ± 0.04 | 1.06 ± 0.09 | 1.03 ± 0.09 | 1.04 ± 0.11 | 0.86 ± 0.05 | 1.01 ± 0.07 |
| GatedSAE | 1.02 ± 0.04 | 1.00 ± 0.08 | 0.94 ± 0.07 | 1.00 ± 0.08 | 0.85 ± 0.05 | 0.99 ± 0.06 |
| TopK | 1.01 ± 0.03 | 0.95 ± 0.07 | 0.94 ± 0.07 | 0.99 ± 0.09 | 0.83 ± 0.05 | 1.00 ± 0.06 |
| GatedAnneal | 1.02 ± 0.04 | 1.04 ± 0.10 | 0.97 ± 0.07 | 1.02 ± 0.08 | 0.86 ± 0.05 | 0.99 ± 0.06 |
| BatchTopK | 1.02 ± 0.04 | 0.95 ± 0.07 | 0.91 ± 0.07 | 0.93 ± 0.07 | 0.83 ± 0.05 | 0.97 ± 0.06 |
| CosineSpaDE | 1.07 ± 0.06 | 1.10 ± 0.09 | 1.05 ± 0.09 | 1.09 ± 0.12 | 0.89 ± 0.05 | 0.99 ± 0.07 |
| SpaDE | 1.09 ± 0.05 | 1.06 ± 0.10 | 1.02 ± 0.10 | 0.99 ± 0.11 | 0.87 ± 0.05 | 1.02 ± 0.08 |

**The null lands on 1.** A third legal move scores 0.97–1.02 in all nine cells, which is
what the ratio is supposed to give when nothing distinguishes the numerator from the
denominator. That is the check the single game could not run: there it scattered over
0.88–1.53.

**The per-space structure of the n = 1 figure does not survive.** On one game the nine
cells spread over 0.48–4.03 and invited a reading about which dictionary separates the
violation; over 100 games every cell lies in 0.83–1.10 and the spread across cells is
comparable to a single cell's CI. The differences between spaces in that figure were
game-level noise. Game 0, the game it draws, is one of these 100 items.

**Every candidate plane inflates the run, and the status-quo one inflates it most.** Paired
per game against full R²¹, the activation plane adds +0.19 (per-space range +0.15…+0.22),
`refit` +0.17, `named` +0.16, `shared` +0.13. The single-game study's headline — *the
excursion the fork panels show is largely the plane* — holds on the population, with an
interval, and the plane that the nine grids are actually drawn in is the worst offender.
This is the plane keeping the coordinates where the displacement has its variance and
dropping the 19 (or 8190) that dilute it, so a 2-D panel overstates the excursion by
~15–25% of a control's run.

**A mean over the six illegal categories is not the mean of their runs.** `full R²¹` is
0.83–0.89 while every individual category sits at 0.93–1.17. The mean is taken on the
displacement *vectors*, exactly as the fork panels take it, so the incoherent part of the
six cancels before the run is measured, and the group mean is shorter than any member.
The gap between the dark line and the faint ones in panel (b) is that incoherence, not a
property of any branch. Read against the null this cuts both ways: the null is a single
branch and carries no such cancellation, so `full R²¹ < control 2` (paired per game,
−0.10…−0.17, CI ±0.05–0.06, below in all nine spaces) is a statement about the *centroid*
of the six illegal categories, while the honest per-category statement is that an illegal
move's curvature displacement runs about as far from the reference as a second legal
move's — 0.93–1.17, straddling 1.

**The two halves of R²¹ agree here.** On one game the 11 step-length and 10 turning-angle
components disagreed sharply (0.34 vs 0.94 in TopK). Paired per game over 100, θ − d is
−0.04…+0.11 with CIs of ±0.06, and only GatedSAE and TopK reach their own interval. The
disagreement was a property of that game, so the panel is retitled: it shows what the
single R²¹ number averages, not a conflict inside it.

## Reading it honestly

- **Everything is refit per game.** The z-scoring is per space and per game; `refit` is a
  top-2 of that game-and-space's own descriptors; `shared` is a top-2 of that game's nine
  spaces pooled; the activation plane is a top-2 of that game's raw displacement. A plane
  fit across games would turn the population into one estimate and the interval into a
  fiction. It also means `shared` is "one plane for the nine cells of this game", which is
  the claim the single-game study makes, held 100 times — not one plane for the corpus.
- **The illegal series is a mean of displacement vectors, the null is one branch.** The two
  estimators are not matched (FIGURE_GUIDELINES 3), which is why the per-category members
  are drawn and quoted above; the mean-of-six is kept because it is the quantity the
  single-game figure and the fork panels draw.
- **A ratio of runs, so the mean is a mean of ratios, and it is right-skewed.** The CSV
  carries the median beside every mean for that reason. Mean − median is ≤0.05 for full
  R²¹ and the empirical null, but 0.06–0.12 for `refit` and up to 0.15 for `named` in
  CosineSpaDE: the fitted and named planes are the ones a game with a barely-moving control
  can pull upward. The plane-inflation numbers above are means of the paired difference;
  on medians they are +0.19 (activation, unchanged), +0.10 `refit`, +0.08 `named`, +0.07
  `shared` — the ordering holds and the status-quo plane is the only one the skew does not
  touch.
- **The anchors are all at r ≥ +6**, which is where a W = 8 trailing descriptor window
  first clears the branch point inside the study's r ∈ [−8, 24] window. This figure is
  about the settled excursion, not about what happens at the branch move.
- **The population is the first 100 branch-complete games** (ids 0–140), which all fall in
  the round-robin block: t₀ spreads over [10, 34], 1–6 games each. Every quantity is
  event-locked to each game's own t₀. The tree holds 3185 branch-complete games in all;
  the sweep runs ~1.6 s/game, so `--n_games` is what decides the interval's width.
- **Isomap is absent**, as in the single-game figure: it is not a linear image of R²¹ and
  its reference is not at the origin, so a distance in it is not a share of anything.
- **Spliced only.** The rollout twin resamples every branch's suffix, which moves where the
  mean over K rollouts is taken and puts every anchor drawn here deep in the regime the
  rollout study found saturated.
