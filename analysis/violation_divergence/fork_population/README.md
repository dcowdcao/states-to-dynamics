# The fork over a population

`../fork_and_board` draws one game's branches in nine representation spaces. That
game is an argmax, its *n* is 1, and neither of its panels carries an interval.
This is the same fork over **100 source games that all fork at *t0* = 20**, with
the estimator's uncertainty on both panels.

Panel **(a)** is that figure's (a) with every path a mean over games. Panel **(b)**
is its **(c)** — how far each rule runs from the reference, in the plane its cell
draws and in that cell's full space, as a multiple of the same cell's legal
control — with a bootstrap CI over games.

**There is no board here.** Every game forks at move 20 and every one has a
different position there, so the board `../fork_and_board` panels beside its fork
is the one thing a population cannot draw, and the *t0* histogram and the per-rule
member counts a composition panel would show are a spike and a constant. What the
branches *are* is that figure's panel **(b)**; read it there, on the one game.

```
python analysis/violation_divergence/fork_population/measure.py [--plane displacement|r21] [--n_games 100]
python analysis/violation_divergence/fork_population/make_figures.py \
    [--plane displacement|r21] [--n_games 100] [--camera-ready]
```

`measure.py` writes `$SPADE_DIVERGENCE_DIR/fork_population/{plane}_n100.npz`.
`make_figures.py` reads it and writes the figure plus
`results/violation_divergence/fork_population_{plane}.csv`, one row per (space,
series, plane | full space) with the mean over games, its bootstrap CI and the
median.

| file | plane (a) is drawn in |
|---|---|
| `figures/fork_population_displacement` | PCA of `D(t) = z_b(t) − z_ref(t)` — the default |
| `figures/fork_population_r21` | PCA of `Δφ(t) = φ(z_b(t)) − φ(z_ref(t))`, inside the 21-D ReStraV curvature descriptor |

Both planes are fit **once per space, on the population mean field**, and both are
uncentred, so zero displacement is still the origin and the reference is still the
vertical line through it.

## The population

The first 100 of the branch-complete games forking at *t0* = 20, ascending by game
id. The tree was extended so that 2000 games fork at that one position
(`docs/finish_notes/t0-20-population.md`), of which **1696** carry all nine branch
types. **No game-level criterion enters this figure** — not a differentiation
argmax, not a cleanliness filter, not a legality one. Game 1235, the argmax
`../fork_and_board` draws, is in it as one ordinary member of 100, 42nd by game id;
it is not excluded for being the game the single-game figure was built on.
`--n_games` cuts the prefix, so a smaller run may drop it, and the disclosure line
under the figure names it only when it is in.

One *t0* for every game is what lets the panels keep the axes they had: alignment
across games is by array index, the vertical axis is still move index *t*, and the
star is still at *t* = 20 rather than at a relative offset. It costs generality —
this is the fork at move 20 and nowhere else — and buys panels that can be read
against the single-game figure position for position.

## Captions

The figures carry only what cannot be looked up while looking at them; everything
else a reader needs in order not to over-read them is here (FIGURE_GUIDELINES §2,
§5). Inside, in full: the panel titles and letters, the space names, the axis
labels, one legend for (a) and one for (b), the three muted lines under (a)'s
legend — the within ÷ between-rule spread ratio, the definition of coherence, and
what the camera does to an ellipse lying flat in the plane — each cell's coherence
number, (b)'s estimator on its own axis, the two-line disclosure naming the
population and the cancellation, and the R1/R2/R3 key.

### `fork_population_displacement`

**Where 100 forks go on average, and how far they run.** Panel **(a)**: one 3-D
fork per space, raw layer 5 first. The plane embeds the across-game mean of
`D(t) = z_b(t) − z_ref(t)` by uncentred PCA and move index *t* is the third axis;
positions are sampled every 2 plies at *t0*'s parity; view fixed at
`elev = 18, azim = −58`. One path per rule is the mean of its member categories,
taken in the plane after projection — the projection is linear, so that is exactly
the projection of the mean displacement. **The ellipse at the end of each path is
a 95% confidence interval of that mean over the 100 games**, lying flat in the
plane at that *t*: the covariance of the 100 games' own projected positions
divided by *n*, at χ²₂(0.95) = 5.991. It is an ellipse and not a bar because the
estimator is a two-dimensional mean and has no preferred direction. The reference
has none — it is 0 at every position by construction — and the legal control's is
drawn in the null styling. **The plane is refit per space**, so the axes are arb.
units, the three ticks per plane axis are a within-panel scale bar, and cells are
not comparable across the grid in position or magnitude, only in the *shape* of
the fork. Panel **(b)**: per cell and per rule, the mean over the 13 drawn offsets
*r* = 0, 2, …, 24 of each game's own distance from its own reference, divided by
that game's own legal control, then averaged over the 100 games — a mean of
per-game ratios, whose median is in the CSV. Every rule is measured twice, in the
plane the cell above draws and in that cell's full space, and both carry a 95%
percentile bootstrap CI over source games (2000 resamples, seed 0); the panel's
own legend says which mark is which, on a log ratio ladder whose baseline is the
control at 1. A rule's member categories are drawn beside it, in the plane, each
its own mean over games and not an interval. The gray series is `control_legal_2`
scored against `control_legal` exactly as the illegal branches are — the empirical
null — and the reference at 1 is the divisor itself.

### `fork_population_r21`

**The same 100 forks with (a)'s plane taken inside the 21-D ReStraV curvature
descriptor.** φ is applied per game and the difference from that game's reference
is taken inside R²¹, before the mean over games: φ of a mean is the curvature of a
path no game takes. The drawn window is *r* = −4 … +24, two positions shorter at
the front than the displacement figure's, because a W = 8 trailing window needs
context the study window does not carry. Everything else — the ellipse, the
per-space refit and its disclaimer, (b)'s estimator, the null — reads as above.

## What it says

Raw layer 5, displacement plane, per game in the full 512-d space and in the
population plane, mean over the 100 games with a 95% percentile bootstrap CI:

| series | full space | median | in the population plane |
|---|---|---|---|
| R1 placement | **0.909** [0.885, 0.934] | 0.912 | 1.225 |
| R2 bracketing | **1.022** [0.996, 1.049] | 1.024 | 2.916 |
| R3 turn | **1.017** [0.987, 1.047] | 1.030 | 2.112 |
| null (a third legal move) | **1.042** [1.006, 1.078] | 1.044 | 1.216 |

**No rule outruns a legal alternative, and only R1 runs measurably less far than
one.** R2 and R3 sit on 1 with their intervals straddling it; R1 sits below, its
interval clear of 1, and it is below 1 in all nine spaces (0.903–0.927, every upper
bound ≤ 0.954) — an occupied-square violation moves the layer-5 activation *less*
far from the reference than a second legal move does.

**The null's margin over R2 and R3 is not a measured ordering.** Its point estimate
sits above all three, but the contrast has to be read paired, on the same games,
where the shared prefix and the common divisor cancel: null − R1 = **+0.133**
[+0.103, +0.163], null − R2 = +0.021 [−0.011, +0.052], null − R3 = +0.025
[−0.011, +0.060]. Only the R1 gap is resolved, and it is the one that replicates —
+0.104 [+0.096, +0.111] over all 1696 branch-complete games at *t*₀ = 20, and
+0.112 [+0.106, +0.117] over all 3185 branch-complete games. On that full 1696-game
pool the null falls *below* R2 and R3 (1.012 against 1.018 and 1.019), which is
inside this sample's interval and is why the ordering must not be read off the
point estimates. n = 100 gives ±0.03 on a gap of ±0.005.

**This is not what the selected game says.** `../fork_and_board` puts R3 at 1.53×
its legal control in the full 512-d space of raw layer 5 and orders R3 > R2 > R1
in all nine cells. Neither the magnitude nor the ordering survives the population:
R2 and R3 are indistinguishable from each other and from a legal control, and R1,
the smallest of the three excursions there at 1.05, is now below its own control.
Dropping game 1235 from the population moves every displacement mean by ≤ 0.006
(≤ 0.016 in R²¹), so its presence is not what drives the disagreement — being one
game of 100 is.

**The plane and the space disagree several-fold, and the plane is in-sample.**
Across the nine cells the in-plane number is 1.1–3.9× the full-space one (1.0–1.5×
in R²¹) — the range (b)'s x label carries, and the reason its axis is a log ladder.
The same per-game distance measured in the population plane reads R1 1.225,
R2 2.916, R3 2.112 against a null of 1.216. The plane is the uncentred top-2 of
these same 100 games' mean field — the direction the illegal branches share — so
a per-game distance in it is weighted by how well that game aligns with the
population's direction, and a game that runs just as far somewhere else reads as
short. The direction is real: a split-half probe put the cosine at *r* = 0 between
the mean of 50 games and the mean of the other 50 at **0.93–0.97** for R2's and
R3's member categories, 0.47–0.70 for R1's and **−0.03** for the legal control.
The magnitude is not held out: every game in the numerator also helped fit the
plane it is measured in. So read the plane column as evidence that the illegal
branches share a direction, and the full-space column as the answer to how far
they run.

**Averaging over games destroys most of a displacement.** Coherence — ‖mean over
games of *D*‖ ÷ mean over games of ‖*D*‖, the share of one game's displacement
that survives the mean — at *r* = 0 in raw layer 5, per branch:

| `empty_detached` | `empty_own_only` | `empty_no_bracket` | `opponent_only` | `occupied_own` | `occupied_opp` | legal control |
|---|---|---|---|---|---|---|
| 0.604 | 0.566 | 0.424 | 0.415 | 0.224 | 0.156 | **0.099** |

The control's 0.099 is the 1/√100 = 0.100 noise floor: a second legal move's
displacement points somewhere different in every game and cancels to nothing,
which is what it should do. The illegal branches do not: `empty_detached` and
`empty_own_only` keep more than half of a game's displacement and the two
occupied-square categories keep least. By
*r* = +24 that is gone — every branch, the control included, sits at 0.17–0.23 and
they lie on one common drift ray (pairwise cosine ≈ 0.93, illegal-to-control
included), so late in the window the paths in (a) are one shared drift and not six
directions. The rule-mean coherences the figure prints at *t* = 44 are R1 0.18,
R2 0.22, R3 0.19.

**R²¹ gives the same verdict with wider intervals.** Raw layer 5, per game in the
full column-scaled Δφ: R1 0.990 [0.922, 1.062] (median 0.985), R2 1.060
[0.994, 1.129] (1.064), R3 1.045 [0.974, 1.122] (1.011), null 1.056 [0.981, 1.129]
(1.026). CosineSpaDE is the one cell in either plane where a rule stands clear of
its space's null — R2 1.298 [1.205, 1.393] and R3 1.219 [1.128, 1.318] against
1.053 [0.974, 1.133], R2's interval clear of it and R3's lower bound (1.128) just
inside its upper (1.133). One cell of nine, on an unpaired comparison, in the
field where the skew is worst; it is a lead, not a finding.

## Reading it honestly

- **(a) averages displacement *vectors*, so it cancels, and each cell prints by how
  much.** `coherence` is ‖mean over games of *D*‖ ÷ mean over games of ‖*D*‖, taken
  on the rule mean each path draws (`coherence_rule` in the npz, not the per-branch
  `coherence`): a rule mean cancels across its member categories too, so the
  members' own coherences average to a different and larger number. In raw layer 5
  the drawn paths keep 0.20 (R1), 0.58 (R2) and 0.42 (R3) of a game's displacement
  at *r* = 0 and 0.18–0.22 by *r* = +24, so the length of a path in (a) is **not**
  how far a branch travels. That is the same estimator the fork panels have always
  used for a rule mean — it is why `../r21_run_population`'s full-R²¹ series sits
  below every one of its members — applied one level up, to games instead of
  categories.
- **(b) averages per-game distances, which cancels nothing**, and it is the panel
  the "how far" question is answered from. The two panels therefore disagree by
  construction: (a) draws R1's path out of a mean that has thrown away 80% of each
  game, (b) measures R1 on every game intact and puts it at 0.909 of its
  control. The coherence number is the bridge between them, and the gap is not a
  plotting error.
- **(b) is the same measurement as the single-game figure's (c), on a population,
  and it disagrees with it.** 1.53× and R3 > R2 > R1 on game 1235; 1.02, 1.02 and
  0.91 with no ordering over 100 games. The single-game figure is an illustration
  of the construction, was selected as the argmax of its illegal branches' mutual
  separation, and its numbers were never the study's. Quote this one. Do not
  reconcile the two by picking whichever is convenient.
- **The plane is fit once per space, on the population mean field.** The PC
  variance ratios printed on (a)'s axes are shares of the *mean field's* variance —
  74% and 13% in raw layer 5, against 38% and 13% for one game. Averaging deletes
  the per-game dimensions, so a high ratio here says the mean field is
  low-dimensional, not that the plane holds much of any game's displacement, and
  the two figures' variance ratios are **not comparable**.
- **Neither interval covers the plane.** Each cell's plane is fit on the mean field
  of the *same* 100 games the ellipse and (b)'s bootstrap are then taken over, so
  both are conditional on that fit. A different draw of 100 games would move the
  axes as well as the points on them, and nothing here bounds by how much. The
  in-plane series in (b) is in-sample and alignment-weighted for the same reason;
  the full-space series is the one to quote.
- **At this n the CI is about as wide as the fork is long, so (a)'s mean direction
  is barely resolved.** That is why the ellipse is drawn once per path rather than
  at every marker — at every marker the paths disappear into the stack of rings —
  and it is the most important caveat about the panel: read (a) for the shape of
  the construction and the ordering of the ellipses, and (b) for magnitude. Two
  rule paths whose ellipses overlap are not two measured directions.
- **The bootstrap resamples source games, never branches.** A game's nine branches
  share a prefix and are divided by that game's own control, so they are not
  independent items; resampling them would give an interval several times too
  tight. One index matrix (seed 0) is shared by every series and every cell, so
  the intervals are read off the same resamples of the same games.
- **The legal control cannot be the empirical null here.** It is the divisor, so
  its ratio is identically 1 with zero variance. `control_legal_2` scored against
  `control_legal` is the null — a third legal move, measured exactly as the
  illegal branches are, which is what the second control exists for
  (`branches.py`) — and it carries its own bootstrap CI. The reference at 1 stays
  as the by-construction divisor and is not an estimate.
- **The rule grouping is weak, and the per-game number is the one to quote.**
  Within-rule ÷ between-rule mean pairwise distance over these 100 games has median
  **0.924 [IQR 0.842–0.983], above 1 in 21%** (displacement, raw layer 5) — the
  number printed under (a)'s legend, and consistent with `rule_groups`' 0.945 over
  1550 games. Near 1 means a rule's members travel barely closer to each other than
  to another rule's, so the three drawn means are a labelling of the six categories
  and not a measured structure. The same statistic **on the mean field** is
  **0.672**, and it is a different quantity — the spread of three averaged
  trajectories, not the typical spread of a game's — so it must never be quoted as
  if it were the population's value. Both are in the npz (`spread_game`,
  `spread_mean`) and printed on every render. The R²¹ figure prints its own: median
  0.917 [IQR 0.744–1.031], above 1 in 33%.
- **Quote the median beside the mean in R²¹.** Mean − median reaches 0.11 there
  (CosineSpaDE's null, mean 1.053 against median 0.947), against ≤ 0.02 anywhere
  in the displacement plane: the per-game ratio is mildly right-skewed in R²¹,
  where a game with a barely-bending control pulls its own ratio up. The CSV
  carries both for every row.
- **Ratios, so only the within-cell contrast carries.** A plane's units are its
  space's own and they differ by ~800× across the grid, so (b) divides by the
  cell's own control and never draws a distance. That makes nine cells readable
  side by side; it does not make them nine measurements of the same thing.
- **Spliced branches only.** Every branch replays the source game's `tokens[t0+1:]`,
  the asymmetry `../fork_and_board`'s rollout variant exists to remove. No rollout
  twin of this figure has been measured.

## What the figure is for

`../fork_and_board` is an illustration: one selected game, no intervals, and it
says so. This is the same construction measured where the measurement can carry a
claim — 100 games, no selection, an interval on every number and an empirical null
beside it — and what it carries is a negative result. An illegal move's layer-5
displacement runs no further from the reference than a second legal move's; an
occupied-square violation runs less far; the shared direction that does exist
shows up as coherence and as the plane column, not as distance. A claim that an
illegal move moves the representation further has to be made against these
numbers, and it does not survive them.
