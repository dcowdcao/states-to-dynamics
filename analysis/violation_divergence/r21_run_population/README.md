# Run from the reference, over the population

`../r21_shared_basis/run_from_reference` asks the question the nine fork grids raise and
none of them can answer — how far an illegal branch actually runs from the reference, and
how much of that run each candidate plane keeps — off **one game**, where the whole sample
is 8 non-reference branch trajectories and no interval is available. This is the same two
panels over the study's shared **n = 100** population — the first 100 branch-complete
games forking at t₀ = 20 — with the mean and its 95% CI.

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
| raw layer 5 | 1.06 ± 0.05 | 0.99 ± 0.08 | 0.96 ± 0.07 | 0.93 ± 0.08 | 0.85 ± 0.05 | 1.06 ± 0.07 |
| PAnneal | 1.05 ± 0.05 | 1.02 ± 0.09 | 0.97 ± 0.08 | 0.96 ± 0.09 | 0.84 ± 0.05 | 1.06 ± 0.07 |
| Standard | 1.12 ± 0.08 | 1.05 ± 0.09 | 0.98 ± 0.08 | 0.96 ± 0.10 | 0.85 ± 0.05 | 1.10 ± 0.08 |
| GatedSAE | 1.05 ± 0.06 | 1.01 ± 0.07 | 0.98 ± 0.07 | 0.98 ± 0.08 | 0.87 ± 0.04 | 1.07 ± 0.06 |
| TopK | 1.06 ± 0.09 | 0.96 ± 0.07 | 0.94 ± 0.07 | 0.93 ± 0.10 | 0.84 ± 0.04 | 1.06 ± 0.06 |
| GatedAnneal | 1.06 ± 0.09 | 1.00 ± 0.07 | 0.95 ± 0.07 | 0.96 ± 0.08 | 0.85 ± 0.05 | 1.04 ± 0.06 |
| BatchTopK | 1.07 ± 0.09 | 0.98 ± 0.07 | 0.96 ± 0.08 | 0.94 ± 0.08 | 0.84 ± 0.05 | 1.05 ± 0.07 |
| CosineSpaDE | 1.14 ± 0.11 | 1.21 ± 0.11 | 1.08 ± 0.11 | 1.15 ± 0.12 | 0.93 ± 0.06 | 1.06 ± 0.07 |
| SpaDE | 1.09 ± 0.06 | 1.00 ± 0.09 | 0.96 ± 0.08 | 0.92 ± 0.09 | 0.85 ± 0.06 | 1.08 ± 0.08 |

**The null is near 1, and slightly above it.** A third legal move scores 1.04–1.10 with CI
half-widths of 0.06–0.08, so its interval covers 1 in six of the nine cells and clears it
in three. On the single game it scattered over 0.88–1.53, so the check could not be run
there at all. The residual few percent is a property of the two controls, not of the
spaces: `control_legal_2` runs a little further from the reference than `control_legal`
does in every cell, which is why it is the null the illegal branches are read against
rather than the divisor.

**The per-space structure of the n = 1 figure does not survive.** On one game the nine
cells spread over 0.48–4.03 and invited a reading about which dictionary separates the
violation; over 100 games every cell lies in 0.84–1.21 and the spread across cells is
comparable to a single cell's CI. CosineSpaDE is the one cell that stands out (1.08–1.21
in the three R²¹ planes), and it is also the cell with the widest interval.

**Every candidate plane inflates the run, and the status-quo one inflates it most.** Paired
per game against full R²¹, the activation plane adds +0.22 (per-space range +0.19…+0.26),
`refit` +0.17, `shared` +0.12, `named` +0.11. The single-game study's headline — *the
excursion the fork panels show is largely the plane* — holds on the population, with an
interval, and the plane the nine grids are actually drawn in is the worst offender. This
is the plane keeping the coordinates where the displacement has its variance and dropping
the 19 (or 8190) that dilute it, so a 2-D panel overstates the excursion by ~10–25% of a
control's run.

**A mean over the six illegal categories is not the mean of their runs.** `full R²¹` is
0.84–0.93 while every individual category sits at 0.92–1.33. The mean is taken on the
displacement *vectors*, exactly as the fork panels take it, so the incoherent part of the
six cancels before the run is measured, and the group mean is shorter than any member.
The gap between the dark line and the faint ones in panel (b) is that incoherence, not a
property of any branch. Read against the null this cuts both ways: the null is a single
branch and carries no such cancellation, so `full R²¹ < control 2` (paired per game,
−0.12…−0.24, CI ±0.05–0.07, below in all nine spaces) is a statement about the *centroid*
of the six illegal categories, while the honest per-category statement is that an illegal
move's curvature displacement runs about as far from the reference as a second legal
move's, or slightly further.

**The two halves of R²¹ agree here.** On one game the 11 step-length and 10 turning-angle
components disagreed sharply (0.34 vs 0.94 in TopK). Paired per game over 100, θ − d is
−0.02…+0.06 against CIs of ±0.06–0.07 — not one space clears its own interval. The
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
  can pull upward. The plane-inflation numbers above are means of the paired difference; on
  medians they are +0.18 (activation), +0.11 `refit`, +0.07 `shared`, +0.04 `named` — the
  ordering holds and the status-quo plane stays the largest.
- **The anchors are all at r ≥ +6**, which is where a W = 8 trailing descriptor window
  first clears the branch point inside the study's r ∈ [−8, 24] window. This figure is
  about the settled excursion, not about what happens at the branch move.
- **The population is the study's shared one**: the first 100 branch-complete games
  forking at t₀ = 20 (`branches.games_at_t0`, ids 10…2044, 1696 available), the same games
  `../divergence_by_space`'s n = 100 twin and the other n-game figures draw. One fixed t₀
  is what lets those figures be read against each other. The sweep is ~1.6 s/game, so
  `--n_games` is what decides the interval's width.
- **Isomap is absent**, as in the single-game figure: it is not a linear image of R²¹ and
  its reference is not at the origin, so a distance in it is not a share of anything.
- **Spliced only.** The rollout twin resamples every branch's suffix, which moves where the
  mean over K rollouts is taken and puts every anchor drawn here deep in the regime the
  rollout study found saturated.
