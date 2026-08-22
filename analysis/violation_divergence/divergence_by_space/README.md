# Divergence by space: how far the fork panels' branches actually run

The fork grids in `../fork_and_board` and `../game_trajectory` draw the branches
separating in nine representation spaces, and a 3-D projection can only ever show the
*shape* of that separation. The question it poses and cannot answer is how much: at
r = +6, is the illegal branch twice as far from the reference as a second legal move, or
a tenth as far? This is that number, on the study's event-locked axis, over the whole
population.

It is the companion to those figures, not a standalone result. Same nine cells in the
same order with raw layer 5 first, same rule grouping, same `branch_style` colours and
markers, same control-as-null. `../make_figures.py`'s F1 is the thing it replaces: F1
draws δ̂ = ‖Δ‖ / ‖z_ref‖ at layer 5 and nowhere else, on a raw activation axis that no
fork panel is drawn in.

```
python analysis/violation_divergence/divergence_by_space/measure.py       [--plane r21] [--variant rollout] [--n_games 100]
python analysis/violation_divergence/divergence_by_space/make_figures.py  [--plane r21] [--variant rollout] [--n_games 100] [--raw] [--camera-ready]
```

`measure.py` writes `$SPADE_DIVERGENCE_DIR[/rollout]/divergence_by_space/curves.npz`
(~35 min spliced, ~45 min rollout on one core, I/O-bound); `make_figures.py` reads it and
writes the figure plus `results/violation_divergence/divergence_by_space[_rollout].csv`,
one row per (space, series, measured_in, r) with the mean and its 95% CI.

`--n_games N` draws the study's **shared N-game population** instead of the whole tree:
the first N branch-complete games forking at t₀ = 20 (`branches.games_at_t0`, 1696 exist),
which is the same set every other N-game sweep here runs over. It is a separate artifact
at every step — `curves_nN.npz`, `divergence_by_space_nN.png`, `..._nN.csv` — so it never
overwrites the whole-tree figure. n = 100 takes ~2 min.

## The figures

`figures/divergence_by_space.{png,pdf}` · `figures/divergence_by_space_rollout.{png,pdf}`
· `figures/divergence_by_space_n100.png` ·
`figures/divergence_by_space_r21_n100.{png,pdf}` ·
`figures/divergence_by_space_rollout_n100.{png,pdf}` ·
`figures/divergence_by_space_r21_rollout_n100.{png,pdf}`

**In the plane the fork panels draw, R2 and R3 run 6–24% further from the reference than
a legal alternative does — and roughly half of that is the plane, not the branch.** One
panel per representation space, raw layer 5 first so the eye calibrates on the reference
space (§4.6). x is r = s − t₀, event-locked to the branch move: the dashed vertical rule
is the branch position, the shaded band left of it is the shared prefix, where every
branch is bitwise identical to the reference and the distance is 0 by construction. y is
each branch's distance from the reference branch, **in the same 2-D PCA plane its fork
panel is drawn in** (solid, filled markers), with the same distance in that cell's full
space underneath (dotted, open markers); both divided by the same cell's own legal
control, so the null sits at 1 on average over r ≥ 0. One curve per constraint rule, the
mean over its member categories (n = 2, 3, 1), with the count in the legend entry;
`control_legal` is the null, in gray. Bands are 95% CIs of the mean over **n = 1553
branch-complete source games** — one game is one item, so no clustering correction — and
are about ±0.015 wide, hairline at this n. t₀ round-robins over [10, 34] and the window
is r ∈ [−8, +24] sampled every 2 plies at t₀'s parity, so every game contributes to every
offset and n is constant across the x-axis. Panels whose plane curve leaves the y-cap
carry the value they reach.

`../fork_and_board`'s panel (c) is **this same quantity collapsed over r ≥ 0** — the bar
is the in-plane curve's mean, the open marker is the full-space curve's mean, and the
divisor is identical. Read the two together: (c) gives one number per rule per space for
one illustrative game, this gives the whole trajectory of that number over 1553.

## The n = 100 twin

`figures/divergence_by_space_n100.png` is the same figure over the shared 100-game
population, so it can be read beside the other n = 100 figures in this study — same games,
same t₀ = 20, same event-locked axis. Mean over r ≥ 0 in the plane, against the
whole-tree figure above:

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw layer 5 | 0.989 (0.993) | 1.157 (1.142) | 1.130 (1.122) |
| CosineSpaDE | 1.009 (1.017) | 1.387 (1.236) | 1.305 (1.206) |
| SpaDE | 1.008 (1.019) | 1.224 (1.213) | 1.174 (1.176) |
| TopK | 0.983 (0.973) | 1.086 (1.064) | 1.097 (1.069) |

Whole-tree value in brackets. Every cell reproduces to within 0.02 except CosineSpaDE,
whose R2/R3 run 0.10–0.15 higher — that cell's r = 0 spike is the one place the population
matters: it reaches 4.4× at n = 100 against 3.1× on the whole tree, and its CI half-width
there is 0.59 against 0.11. Away from r = 0 the CI is 0.05–0.09, so the plateau where the
claim lives carries the same reading at n = 100 as at n = 1553.

The rule means hide the same amount of within-rule structure in both: the within- ÷
between-rule spread of the displacements is a median 0.924 [IQR 0.842–0.983] over these
100 games, above 1 in 21% of them, against 0.945 [0.872–1.020] over the whole tree. The twin's sweep computes that
statistic per game rather than quoting `rule_groups`' whole-tree number, and
`make_figures.py` prints it for the population drawn.

## The R^21 twin

`figures/divergence_by_space_r21_n100.{png,pdf}` is the same figure with the reference
subtracted one step later: each branch's trajectory goes through ReStraV's 21 curvature
components first, and the difference is taken there —
Δφ(t) = φ(z_b(t)) − φ(z_ref(t)), φ the W = 8 trailing window of
`curvature_plane.windowed`. Nine cells, same colours and markers, same
ratio-to-the-legal-control, same 95% CI over the same 100 games. The plane is
`curvature_plane.fit_plane`'s uncentred top-2 of Δφ, refit per game, and both curves
are measured in the column-scaled field, because R²¹ mixes degrees with
activation-scale lengths.

```
python analysis/violation_divergence/divergence_by_space/measure.py      --plane r21 --n_games 100  # ~20 min
python analysis/violation_divergence/divergence_by_space/make_figures.py --plane r21 --n_games 100 [--camera-ready]
```

**The x axis is shorter: r = −4 … +24, 15 offsets.** φ's window ends at t and spans 14
plies, so it needs seven sampled positions of context and `positions` can only supply
five at t₀ = 20 — the first two offsets have no full-width window. Two branch-free
offsets survive, and `measure.py` asserts Δφ is exactly zero on both, which is what
pins where the branch is once R²¹ smears it.

**The separation is the same event, read through a sliding window.** In the
displacement figure the rule-minus-control gap is a spike at r = 0 (+0.33 in raw layer
5) decaying to +0.03 by r = +24. In R²¹ it starts at +0.11, climbs to **+0.34 at
r = +12**, and collapses to +0.06 at r = +14. Every space peaks at r = +12 (GatedSAE's
R3 at +10), and r = +12 is exactly where the branch move fills all seven steps of the
W = 8 window, and r = +14 is the first offset whose window no longer contains the
step that crosses t₀. So R²¹ does not
find separation the displacement figure missed — it spreads one ply's worth over seven
and reads it out as a ramp. **Nothing about the peak's position is a claim about the
model.**

Integrated over the window that does buy separation. Mean over r ≥ 0 in the plane,
displacement → R²¹, raw layer 5: R1 0.989 → 1.040, R2 1.157 → 1.228, R3 1.130 → 1.204;
in the full space R1 0.909 → **0.977**, R2 1.022 → 1.047, R3 1.017 → 1.038. The one
qualitative change is R1: illegal *placement* sits 8–10% **below** the control
everywhere in the displacement figure and is within 3% of it in R²¹, in every space.
The plane inflates in 27 of 27 cells here too, median 1.134 against 1.108 (range
1.051–1.309).

The price is precision and one unreadable panel. The CI half-width over r ≥ 0 roughly
doubles — median 0.148 against 0.094 — because φ of a short window is a noisier
statistic than a displacement. And CosineSpaDE's R2/R3 leave the y-cap for seven
consecutive offsets rather than the displacement figure's one at r = 0 (peak 2.98× at
r = +12 against 4.44× at r = 0), so that cell shows its annotated peak and little else.
Reading it needs the CSV.

The rule grouping is no better in R²¹: the within- ÷ between-rule spread over these 100
games is a median 0.886 [IQR 0.752–1.022] against the displacement field's 0.924
[0.842–0.983] — the same "barely coherent" verdict, with a wider spread.

## Why the plane, and what the plane costs

The y quantity is `branch_panels.plane_displacement`'s rule, resolved in r instead of
averaged over it: the in-plane distance from the reference branch, where the plane is the
uncentred top-2 PCA of the displacement D(t) = z_b(t) − z_ref(t), refit per game and per
space exactly as `fork_plane.pca2` fits it. The reference's D is exactly zero, so it
projects to the origin and distance-from-the-reference and distance-from-the-origin are
the same number.

A plane holds only part of a displacement, and a different part of each branch, so a
branch that leaves the plane reads as a branch that stayed near the reference. That is
why the full-space distance ‖D‖ rides underneath every in-plane curve. Mean over r ≥ 0,
spliced tree, as a multiple of the cell's legal control:

| space | R1 plane \| full | R2 plane \| full | R3 plane \| full |
|---|---|---|---|
| raw layer 5 | 0.993 \| 0.905 | 1.142 \| 1.009 | 1.122 \| 1.009 |
| PAnneal | 0.992 \| 0.905 | 1.140 \| 1.009 | 1.123 \| 1.010 |
| Standard | 1.007 \| 0.909 | 1.197 \| 1.026 | 1.163 \| 1.018 |
| GatedSAE | 0.983 \| 0.900 | 1.064 \| 0.972 | 1.070 \| 0.987 |
| TopK | 0.973 \| 0.895 | 1.064 \| 0.978 | 1.069 \| 0.993 |
| GatedAnneal | 0.984 \| 0.897 | 1.071 \| 0.971 | 1.072 \| 0.983 |
| BatchTopK | 0.988 \| 0.899 | 1.086 \| 0.979 | 1.085 \| 0.993 |
| CosineSpaDE | 1.017 \| 0.918 | 1.236 \| 1.016 | 1.206 \| 1.013 |
| SpaDE | 1.019 \| 0.920 | 1.213 \| 1.019 | 1.176 \| 1.013 |

**The plane inflates the rule/control ratio in 27 of 27 cells, median 1.108×, range
1.077–1.217.** The two SpaDE codes inflate most (1.16–1.22 on R2/R3), TopK least (1.08).
The single-game figures report a much larger inflation — `../fork_and_board`'s game 1235
reaches 4.08 in the plane against 1.53 in the full space, 2.7× — because that game is an
argmax over branch separation, not a draw. Over the population the inflation is about
11%, and the honest reading of a fork panel is "the spread you see is roughly a tenth
larger than the branch actually is", not "several times".

The sharpest place to see the mechanism is r = 0 in raw layer 5. In the full space the
legal control is displaced 1.384× the window mean and R2 1.528×, a 10% gap. In the plane
the control is 1.133× and R2 1.711×, a 51% gap. The plane is fit where the *illegal*
branches travel, so it keeps 51% more of R2's displacement than of the control's, and
the ratio read off it is that difference in shares, not a difference in distance.

## What the full space says

**No rule ends up further from the reference than a legal alternative does.** Over r ≥ 0
the full-space ratios are 0.895–0.920 for R1, 0.971–1.026 for R2 and 0.983–1.013 for R3;
nothing exceeds the control by more than 2.6% and R1 sits 8–10% below it in every space.
This reproduces the parent study's `ratio_illegal_control_L5` = 0.979 in a different
estimator (in-plane-fit spaces, per-game control normalisation, stride-2 window) and
extends it from layer 5 to all eight dictionaries. **Eight SAE dictionaries do not make
an illegal move look further away than a legal one.** The one thing that changes across
the grid is how much the *plane* exaggerates.

## Cross-panel comparability

The plane is refit per game and per space, and its units are the space's own. Measured
here, the legal control's mean full-space displacement runs from **13.19 in raw layer 5 to
0.0225 in SpaDE — 585×** for the same branches in the same games. Nine raw curves side by
side would be nine unit systems, and nine panels side by side get compared whatever a
disclaimer says. Every curve is therefore divided by **that cell's own `control_legal`,
averaged over r ≥ 0**: dimensionless, a within-cell contrast by construction, the null at
1 by construction, and the same form `../fork_and_board`'s panel (c) and the parent
study's `ratio_illegal_control_L5` already use. The divisor is in the y-axis label rather
than only here, because a reader who misses it misreads every panel.

A scalar divisor, not the control's curve: dividing pointwise would pin the control to a
flat 1 and throw away its shape, and 0/0 in the shared prefix. With a scalar the control
keeps its own trajectory and averages to 1.

## The raw pair

`figures/divergence_by_space_n100_raw.{png,pdf}` ·
`figures/divergence_by_space_r21_n100_raw.{png,pdf}` ·
`figures/divergence_by_space_rollout_n100_raw.{png,pdf}` ·
`figures/divergence_by_space_r21_rollout_n100_raw.{png,pdf}`

```
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100 --variant rollout
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100 --plane r21
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100 --plane r21 --variant rollout
```

`--raw` is a flag on the figure, not a sweep. `measure.py` already writes both distances
raw — `plane` and `full` in each npz, in each cell's own units — and the division is
`normalised()`'s, so the raw pair reads the same four n = 100 npz the ratio pair does and
needs no re-measure.

**Everything but the y-axis is the ratio figure's: same nine cells in the same order, same
`branch_style` colours and markers, same 95% CI over the same 100 games, same shaded
prefix and branch rule.** What goes is the ratio's own furniture — the dashed null at
y = 1, the "above 1" note, the 2.0 cap and the callouts that read against it — because
none of it means anything on a distance axis.

**Read one figure for the size of a distance and the other for the comparison between
panels. A number does not carry across.** The raw figure is the only place a cell's
magnitude appears: the ratio pins every panel's control at 1, so nine panels that differ
700-fold look identical. The ratio figure is the only form in which the nine may be set
side by side at all, and it is the only one whose rule-vs-control read is paired per game.
Neither substitutes for the other, and the styling being identical is exactly why this
paragraph is here.

The legal control's distance from the reference, mean over r ≥ 0 in each cell's own units,
n = 100 — the divisor the ratio figure removes:

| space | spliced plane \| full | rollout plane \| full |
|---|---|---|
| raw layer 5 | 8.486 \| 13.25 | 7.512 \| 14.53 |
| PAnneal | 3.964 \| 6.382 | 3.467 \| 6.763 |
| Standard | 2.057 \| 3.532 | 1.715 \| 3.631 |
| GatedSAE | 3.814 \| 6.229 | 3.349 \| 6.480 |
| TopK | 4.317 \| 6.616 | 3.988 \| 7.320 |
| GatedAnneal | 4.583 \| 7.191 | 4.049 \| 7.752 |
| BatchTopK | 4.458 \| 6.829 | 4.050 \| 7.478 |
| CosineSpaDE | 0.0370 \| 0.0820 | 0.0313 \| 0.0815 |
| SpaDE | 0.0121 \| 0.0226 | 0.00961 \| 0.0219 |

**700× between the top and the bottom of that column** (raw layer 5 against SpaDE, in
the plane, spliced; 586× in the full space, which is the whole tree's 585× above, and
782×/664× on the rollout tree). That is the number "Cross-panel comparability" asserts,
now on an axis.

The same table for the R²¹ pair, where the span is 1.4–1.5×:

| space | spliced plane \| full | rollout plane \| full |
|---|---|---|
| raw layer 5 | 3.123 \| 5.405 | 2.849 \| 4.987 |
| PAnneal | 3.104 \| 5.384 | 2.894 \| 5.018 |
| Standard | 3.000 \| 5.302 | 2.841 \| 4.991 |
| GatedSAE | 3.354 \| 5.495 | 2.880 \| 5.055 |
| TopK | 3.388 \| 5.663 | 2.928 \| 5.186 |
| GatedAnneal | 3.350 \| 5.553 | 2.833 \| 5.057 |
| BatchTopK | 3.313 \| 5.595 | 2.953 \| 5.170 |
| CosineSpaDE | 2.444 \| 4.453 | 1.976 \| 3.843 |
| SpaDE | 3.022 \| 5.192 | 2.791 \| 4.824 |

**These are not physical units and the R²¹ pair is not a raw figure in the sense the other
pair is.** `curvature_plane.fit_plane` divides each of the 21 descriptor columns by its SD
over that game's Δφ rows, `Plane.project` re-applies the same scale, and `measure.py` takes
the full-space norm of the scaled field too — so both R²¹ curves are lengths in **per-game
descriptor-SD units**, which is why nine dictionaries land within 1.4× of each other rather
than 700×. Nothing on that axis is degrees, activation units, or comparable between games.
The column scaling is not cosmetic: R²¹ mixes turning angles with activation-scale step
lengths whose variances differ ~450× across dictionaries, and a raw norm of it would report
whichever component carries the biggest units.

What the raw view adds, on the displacement pair:

- **The excess is 100× smaller where the ratio is largest.** CosineSpaDE's R2 is the grid's
  biggest ratio (1.387) and its biggest ratio spike (4.44× at r = 0); in that space's units
  the same curve sits 0.011 above its control, against raw layer 5's 1.04. The ratio ranks
  cells by excess *relative to a divisor that itself varies 700×*, so a tall bar there is a
  small divisor as readily as a long branch.
- **The distances shrink after the branch.** The control's in-plane displacement peaks at
  r = 0 and falls to 0.68–0.76× of it by r = +24 in the seven dense cells (0.46× in
  CosineSpaDE). The R²¹ pair runs the other way — 1.25 → 2.81 in raw layer 5, 2.24× — since
  φ's window keeps absorbing post-branch steps. Both directions survive the division, but
  only the raw axis says how far anything moved.

What the ratio view does that the raw view cannot, beyond the shared axis:

- **The rule-vs-control contrast is paired per game there and unpaired here.** `ratios()`
  divides each game by *that game's* control before averaging, so it estimates E[rule /
  control]; a raw figure invites the reader to compare two population means, E[rule] against
  E[control]. Those differ, and in R²¹ they differ enough to change the verdict in **14 of
  27 spliced cells**: R1 reads 1.03–1.09 in the ratio figure and 0.92–0.98 of its control in
  population means, and GatedSAE's, TopK's, GatedAnneal's and BatchTopK's R3 cross from
  above the control to below it. On the displacement pair five cells flip, all R1, all
  within 0.04 of the null. **A raw panel is not the ratio panel rescaled** — no constant
  divisor relates them, because the divisor is per game.
- **The null has a position.** The ratio figure's y = 1 is where "no further than a legal
  alternative" lives, and an eye can check a curve against a rule. In the raw figure the
  control is one more curve among four, and every ordering has to be read off two lines that
  nearly coincide.

**r < 0 is exactly zero, not nearly zero.** Every branch is bitwise identical to the
reference over the shared prefix, so the full-space distance there is 0 and `measure.py`
asserts it; the in-plane distance is an eigenvector component of a matrix whose row is zero
and comes out below 1e-18 in absolute value. On a raw axis that region is the true
origin rather than a rescaled one, and the shaded band plus the in-panel `≡ 0` label carry
it exactly as they do in the ratio figure.

**The displacement pair is the one figure in this study with a per-panel y axis, and it
has to be.** What is drawn spans 1360× spliced (SpaDE 0.0097 against raw layer 5's 13.16)
and 1671× rollout — a wider span than the control column above, which is why the titles
quote the drawn one. On a single shared linear axis the two SpaDE panels stop existing:
SpaDE's entire rule-vs-control separation is 0.0087, 0.06% of a 15.1-unit axis, 0.07 pt at
`CELL_H` against the 1.7 pt the figure draws its lines with; CosineSpaDE's is 0.69 pt. Both
render as one flat line on zero and Standard gets 23% of the axis, so `sharey=False` is
forced by the data and not chosen. The cost is real: **panel heights are not comparable,
only within-panel shapes are**, the same disclaimer §4.6 puts on a per-panel PCA refit, and
the per-panel tick labels are what carry it. Log y was built and rejected — 3.6 decades of
axis spent on a within-panel signal of ~0.1 decade leaves the gray legal control invisible
in all nine panels, which fails §3 outright.

`--shared-y` draws the displacement pair on one autoscaled linear axis anyway. It is not a
deliverable; it exists so the collapse above can be re-checked rather than taken on trust.

**The R²¹ pair keeps one shared linear axis** and needs no deviation at all: every panel
peaks between 4.63 and 5.78 (within 1.3×, 1.5× on the rollout tree) and uses at least 58%
of it. The 11.7× between the smallest and largest value drawn is inside one panel,
CosineSpaDE, not between panels.

The autoscaled raw limits keep the ratio figure's own proportions rather than inventing new
ones: bottom at −0.045 × top, which is the −0.09/2.0 the ratio panels use, so zero sits
4.31% up the axis in every panel of every figure and the shared-prefix region reads as
visibly zero instead of resting on the spine.

CSVs are `results/violation_divergence/divergence_by_space[_r21][_rollout]_n100_raw.csv`,
the same shape as the ratio ones with `ratio*` renamed `distance*`. Ground truth computed
independently from the npz is in `verification/`.

## The rollout variant

`--variant rollout` reads the tree where every branch, the reference included, samples
its own continuation from OthelloGPT instead of replaying the source game's. A branch is
K = 4 rows there, and the reduction is `../fork_and_board`'s, not F1's: each rollout is
differenced against **the reference drawn from its own uniform stream** (rows are
K-contiguous and the sampler is keyed on (game, k)) and the K displacement *vectors* are
averaged before the plane is fit, so the plane here is the plane those fork panels draw.
F1's `reduce_k` averages the scalar δ̂ instead. The two differ: averaging vectors cancels
whatever is incoherent across a branch's four samples, so magnitudes here are
systematically smaller than a scalar mean over K would give. Dividing by the control
removes most of that and not all of it, and **the size of the residual is not measured** —
it would take a second pass with the other reduction.

That tree saturates. Past r = +8 the illegal and control δ̂ agree to within 0.5% and cross
by r = +13, so both classes are climbing toward the divergence of unrelated activations
whatever legality does (`../README.md`). The figure draws the whole window and **shades
r ≥ 8 as not-evidence** rather than stopping there, because the event-locked shape before
that point is the content and a truncated axis hides where the saturation starts.

**The branch move stops setting the distance, in every space.** Over the whole window,
pooled across the nine cells, the full-space ratios are R1 0.969, R2 1.019, R3 1.016
against the control's 1.000 — nothing more than 2% off the null, where the spliced tree
put R1 8–10% below it. In the shaded region the ordering is gone entirely (r ≥ 10: R1
0.974, R2 0.997, R3 0.992). Inside the evidence window the separation survives but is
small: r ∈ [0, 8] gives R1 0.961, R2 1.054, R3 1.056 in the full space.

The plane still inflates, and by more here: 27 of 27 cells above 1, median **1.105**,
range 1.053–1.268, and inside the evidence window the pooled inflation reaches **1.29 on
R2**. So a fork panel drawn on the rollout tree shows R2 at 1.36× the control when the
branch is really at 1.05×. That is the figure's own warning about reading a projection,
at its largest.

r = 0 is the same measurement in both trees for δ̂ (the rollout copies the prefix
verbatim), but **not for this quantity**, because the plane is refit on the rollout's
K-averaged displacements: raw layer 5, R2 at r = 0 is 1.711 in the spliced plane and
1.802 in the rollout plane. The full-space numbers move too (1.528 → 1.363), which is the
vector-vs-scalar K reduction above, not a difference in the model's behaviour at t₀.

## Choices and limits

- **Stride 2, not every ply.** `fork_plane.positions` samples at t₀'s parity because the
  board code is mover-relative and alternates each ply; the plane is the fork panels'
  plane only if it is fit on the positions they draw. F1's every-ply axis is the raw-δ̂
  figure's; this one inherits the fork rule. Cost: 17 offsets instead of 33.
- **`control_legal_2` is dropped**, from the plane fit and from the panels, exactly as
  `../fork_and_board` drops it — a second legal move is the null *for* the control and
  belongs where the control is measured. Its rows are in the npz for anyone who wants the
  legal-vs-legal null in this metric; nothing here draws it.
- **n = 1553 is `complete_games`**, the games carrying the reference, both controls and
  all six illegal categories. That fixes each rule mean's composition at 2/3/1 members in
  every game, so a rule's curve is not partly a change in which categories it averages.
  The weaker `rule_complete_games` population (2000) would let that vary.
- **The rule means hide a weak grouping.** `rule_groups` measures within- ÷ between-rule
  spread of the displacements at 0.945 [IQR 0.872–1.020] on this same population and
  field, i.e. a rule's members travel barely closer to each other than to another rule's.
  Three confident lines should not be read as three coherent classes. That number lives
  here and not on the figure, which carries no footer text: a deliberate departure from
  FIGURE_GUIDELINES 1.7, which asks for it in-panel.
- **`--plane r21` is n = 100 in both trees.** The whole tree would need t₀ ≥ 18 for a
  branch-free window to exist at all (`curvature_plane.usable`); the shared t₀ = 20
  population sidesteps that. On the rollout tree the K = 4 rows are averaged after φ,
  never before, and `measure.py` asserts Δφ collapses on both branch-free offsets.
- **The raw pair departs from FIGURE_GUIDELINES 4.2**, which says a quantity whose scale
  differs across panels goes on the axis as a paired ratio and never raw. 4.2 also asks
  for the reference's raw scale per panel; the ratio figure is the compliant one and the
  raw pair is that scale published as a figure rather than only as the table above. It is
  a companion, never the figure a cross-panel claim is read off.
- **Seed 42 only**, like the rest of the study.
- **The rollout K reduction is the fork figures' and not F1's**, and the difference is
  unmeasured. See "The rollout variant" above.
- **The y-cap is 2.0** and a few panels' r = 0 spike runs past it (two spliced, three
  rollout); each prints the value it reaches. One shared y-axis is the point of the
  normalisation, and
  the alternative — a top set by one cell's spike — leaves the plateau, where the claim
  is, in the bottom eighth of every panel.
- **Not measured here**: whether the inflation is the plane's *variance share* per branch
  (the `../fork_and_board` identity ratio_plane = ratio_full × share_illegal/share_control
  predicts it exactly, and the shares are not written out by `measure.py`). Adding the
  per-branch share would settle it in one more column.
- `fork_plane.space_features` rebuilds the SAE encoder on every call, so it costs a
  checkpoint load per game. `measure.py` builds the nine encoders once instead; the fix
  belongs in `src` and is not made here.
