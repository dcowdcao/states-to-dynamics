# The fork over a population

`../fork_and_board` draws one game's branches in nine representation spaces. That
game is an argmax and its *n* is 1. This is the same fork over **100 source games
that all fork at *t0* = 20**: that figure's (a) with every path a mean over games,
nine cells, and nothing else.

**The figure is the shape of the construction and nothing more.** It carries no
magnitude, no interval, and no prose beyond its title, its legend and each cell's
coherence number (`PROCESS.md` 2026-08-19 for why each of those went). Everything
that used to be printed on the canvas is in this file, and **this file is not
optional reading** — a reader who takes the figure on its own will read a short
mean fork as "the branches barely move", which is the one conclusion the study
does not support.

What that means for two of the guidelines the figure used to satisfy on its own
face. **§4.8** asks a 3-D projection to be paired with a quantitative companion
carrying the number, its uncertainty and its n: that companion is now a table —
*What it says* below, and every row of
`results/violation_divergence/fork_population_{plane}.csv`, which is **how far each
rule runs from the reference, in the plane its cell draws and in that cell's full
space, as a multiple of the same cell's legal control, with a 95% percentile
bootstrap CI over 100 source games**. **§5** asks the figure's shorthand to earn a
key; `R1`/`R2`/`R3` and `coherence` are shorthand and their key is now here and
nowhere else:

| rule | members | what the branch does |
|---|---|---|
| **R1 placement** (2) | `occupied_own`, `occupied_opp` | square holds the mover's disc · square holds the opponent's disc |
| **R2 bracketing** (3) | `empty_detached`, `empty_own_only`, `empty_no_bracket` | empty, touches no discs · empty, touches only the mover's discs · empty, touches opponent discs, no line closes |
| **R3 turn** (1) | `opponent_only` | flips only for the opponent (turn confusion) |

**coherence**, printed over each cell, is ‖mean over games of *D*‖ ÷ mean over
games of ‖*D*‖ for the rule path that cell draws: the share of one game's
displacement that survives the mean, 0 when all of it cancels.

A key in a README is not what §5 asks for — it wants the key where the shorthand
is. This is a known gap and not a fix; §1.7's spread ratio, below, is the other
one. Both are the price of a canvas with nothing on it but the figure.

**There is no board here.** Every game forks at move 20 and every one has a
different position there, so the board `../fork_and_board` panels beside its fork
is the one thing a population cannot draw, and the *t0* histogram and the per-rule
member counts a composition panel would show are a spike and a constant. What the
branches *are* is that figure's panel **(b)**; read it there, on the one game.

```
python analysis/violation_divergence/fork_population/measure.py \
    [--plane displacement|r21|d7] [--n_games 100] [--variant spliced|rollout]
python analysis/violation_divergence/fork_population/make_figures.py \
    [--plane displacement|r21|d7] [--n_games 100] [--variant spliced|rollout] [--camera-ready]
```

`measure.py` writes `$SPADE_DIVERGENCE_DIR/[rollout/]fork_population/{plane}_n100.npz`.
`make_figures.py` reads it and writes the figure plus
`results/violation_divergence/fork_population_{plane}[_rollout].csv`, one row per
(space, series, plane | full space) with the mean over games, its bootstrap CI and
the median. The two variants read and write different trees and differently named
figures, so a rollout run cannot touch a spliced artifact.

| file | plane it is drawn in | branch set |
|---|---|---|
| `figures/fork_population_displacement` | PCA of `D(t) = z_b(t) − z_ref(t)` — the default | spliced |
| `figures/fork_population_r21` | PCA of `Δφ(t) = φ(z_b(t)) − φ(z_ref(t))`, inside the 21-D ReStraV curvature descriptor | spliced |
| `figures/fork_population_d7` | the same `Δφ(t)` cut to R²¹'s **7 raw step lengths** `d1…d7` — the descriptor's step-length half, without the 4 moments of *d* | spliced |
| `figures/fork_population_displacement_rollout` | the same displacement plane | rollout |
| `figures/fork_population_r21_rollout` | the same R²¹ plane | rollout |
| `figures/fork_population_d7_rollout` | the same `d1…d7` block | rollout |

Both planes are fit **once per space, on the population mean field**, and both are
uncentred, so zero displacement is still the origin and the reference is still the
vertical line through it.

## The population

**Not stated on the figure** — the canvas carries no selection disclosure, so it is
stated here and only here. The first 100 of the branch-complete games forking at
*t0* = 20, ascending by game id. The tree was extended so that 2000 games fork at
that one position (`docs/finish_notes/t0-20-population.md`), of which **1696**
carry all nine branch types. **No game-level criterion enters this figure** — not
a differentiation argmax, not a cleanliness filter, not a legality one. Game 1235, the argmax
`../fork_and_board` draws, is in it as one ordinary member of 100, 42nd by game id;
it is not excluded for being the game the single-game figure was built on.
`--n_games` cuts the prefix, so a smaller run may drop it, and the disclosure line
under the figure names it only when it is in.

One *t0* for every game is what lets the panel keep the axes it had: alignment
across games is by array index, the vertical axis is still move index *t*, and the
star is still at *t* = 20 rather than at a relative offset. It costs generality —
this is the fork at move 20 and nowhere else — and buys a panel that can be read
against the single-game figure position for position.

## Captions

The figures carry only what cannot be looked up while looking at them; everything
else a reader needs in order not to over-read them is here (FIGURE_GUIDELINES §2,
§5) — and this figure keeps almost nothing inside. Inside, in full: the title, the
nine space names, the axis labels, one legend, and each cell's coherence number.
That is all. There is no panel letter, because with one panel there is nothing to
enumerate; there is no note block under the legend, because every line of it was
the caption pasted onto the canvas and the canvas is not where a caption belongs
(the same sweep `PROCESS.md`'s `gray/gray-parent` entry made over the parent
figures). The four facts that block used to carry live above and below: the
within ÷ between-rule spread ratio and the coherence definition, the population
disclosure in *The population*, and the cancellation warning in *Reading it
honestly*.

### `fork_population_displacement`

**Where 100 forks go on average.** One 3-D fork per space, raw layer 5 first. The
plane embeds the across-game mean of `D(t) = z_b(t) − z_ref(t)` by uncentred PCA
and move index *t* is the third axis; positions are sampled every 2 plies at
*t0*'s parity; view fixed at `elev = 18, azim = −58`. One path per rule is the
mean of its member categories, taken in the plane after projection — the
projection is linear, so that is exactly the projection of the mean displacement.
**Nothing drawn is an interval and nothing drawn is a magnitude.** **The plane is
refit per space**, so the axes are arb. units, the three ticks per plane axis are
a within-panel scale bar, and cells are not comparable across the grid in position
or magnitude, only in the *shape* of the fork.

The measurement the figure used to carry as a second panel, unchanged, in the CSV
and in *What it says*: per cell and per rule, the mean over the 13 drawn offsets
*r* = 0, 2, …, 24 of each game's own distance from its own reference, divided by
that game's own legal control, then averaged over the 100 games — a mean of
per-game ratios, whose median is in the CSV beside it. Every rule is measured
twice, in the plane its cell draws (`where = plane`) and in that cell's full space
(`where = full space`, the plane-free number the claim rests on), and both carry a
95% percentile bootstrap CI over source games (2000 resamples, seed 0). The null is
`control_legal_2` scored against `control_legal` exactly as the illegal branches
are; `control_legal` itself is the divisor and is identically 1.

### `fork_population_r21`

**The same 100 forks with the plane taken inside the 21-D ReStraV curvature
descriptor.** φ is applied per game and the difference from that game's reference
is taken inside R²¹, before the mean over games: φ of a mean is the curvature of a
path no game takes. The drawn window is *r* = −4 … +24, two positions shorter at
the front than the displacement figure's, because a W = 8 trailing window needs
context the study window does not carry. Everything else — the per-space refit and
its disclaimer, the CSV's estimator, the null — reads as above.

### `fork_population_d7`

**The same 100 forks with the plane fit inside R²¹'s raw step-length block
alone.** `curvature_plane.PLANE_COLUMNS` cuts Δφ to `d1…d7` and nothing else
changes: φ per game over the W = 8 window, the difference inside the descriptor,
the column scaling, the *r* = −4 … +24 drawn window, the per-space refit, the
CSV's estimator and the null all read as the R²¹ figure above. The 7 are
the **raw** step lengths, not `fork_r21.BLOCKS["d"]`'s 11, which also match
`mean d`, `min d`, `max d` and `var d` — summaries of the same seven.

It exists because `../r21_shared_basis` panel (b) reports that R²¹'s two halves
disagree about the same branches, and reports it only as a scalar run ratio. This
is the fork drawn in the half that moves.

### `fork_population_displacement_rollout`, `..._r21_rollout`, `..._d7_rollout`

**The same three planes, the same 100 games, drawn out of the rollout branch
set** — where every branch, the reference included, samples its own continuation
from OthelloGPT instead of replaying the source game's
(`spade.violation_divergence.rollout_branches`). `branches.games_at_t0` returns
the same 100 game ids from either tree, so the rollout figures are over the same
games as the spliced ones and may be read against them cell for cell. Three
things differ, and the canvas states only the third.

**A branch is K = 4 rows there, and the reduction over K happens after every
nonlinearity.** The order is: encode into the space → (descriptor planes) φ over
the W = 8 window → subtract the reference drawn on the same rollout → mean over
the 4 rollouts → mean over games. `population.paired_mean` is that step, and
`../fork_and_board` does it the same way for the same reasons. **Paired, not
marginal**: `rollout_branches` keys the sampler on `(game_id, k)`, so all branches
of a game share the k-th uniform stream and the difference is a common-random-number
contrast rather than two independent draws. **After φ, never before**: φ of an
averaged path is the curvature of a path no rollout takes.

**The window stops at *r* = +8, not +24**, `measure.ROLLOUT_RMAX`. Past *r* ≈ +8
the rollout set's illegal and control δ̂ agree to within 0.5% over the whole
2000-game set and cross over by *r* = +13
(`../fork_and_board/README.md`, *The rollout variant*), so the extra offsets draw
separation that is not there. The drawn windows are *r* ∈ [−8, +8] on the
displacement figure and *r* ∈ [−4, +8] on the two descriptor ones; the title
carries the cut and this is why it is there. The CSV's estimator averages **5**
drawn offsets *r* = 0, 2, …, 8 instead of 13, and the same restriction applied to
the *spliced* branches leaves that figure's verdict unchanged, so nothing below is
an artifact of the shorter window (*What the rollout set says*).

**The prefix still collapses.** `tokens[:t0]` is shared in the rollout tree too,
so *D* and Δφ are exactly zero at every branch-free offset, and `measure.py`
asserts it per game per space on every run. It passes, and in this variant that is
not a formality — it is the check that the `[B, K]` row axis is aligned. Stronger
still: the K-reduced displacement at *r* = 0 is **bit-identical** to the spliced
one over all 100 games × 9 branches × 512 dims, because the activation at *t0*
depends only on `tokens[:t0+1]`, which both trees share.

**What this costs is attribution, and it is not a defect of the figure.** Two
rollout branches differ at *t0* *and* at every position after it, so a displacement
at *r* > 0 is the branch move plus everything the model chose to do about it.
`branches.pt` stays the tensor the single-token claims rest on; neither variant is
the other's replication.

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
in R²¹) — the gap between the CSV's two `where` rows for the same series.
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
included), so late in the window the drawn paths are one shared drift and not six
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

**The step-length block loses the one result the other two planes have, and
gains one they do not.** Raw layer 5, per game in the column-scaled 7-D block:
R1 **1.012** [0.925, 1.116] (median 0.925), R2 1.071 [0.992, 1.159] (1.009),
R3 1.068 [0.967, 1.175] (0.971), null 1.086 [0.991, 1.182]. Paired on the same
games — the only form this ordering may be read in — null − R1 = **+0.074**
[−0.005, +0.146], null − R2 = +0.015 [−0.065, +0.095], null − R3 = +0.018
[−0.063, +0.094]: **nothing is resolved in raw layer 5**. R1's deficit, which the
displacement plane resolves in all nine spaces (null − R1 = +0.133 [+0.103,
+0.163] here) and which `../step_length_fork` resolves on the ‖Δz‖ ratio itself
(0.936 [0.910, 0.961] at *r* = 0), does not survive the cut to `d1…d7`: over nine
spaces null − R1 clears zero only in Standard (+0.087 [+0.002, +0.171]) and SpaDE
(+0.100 [+0.004, +0.197]), both marginally. The per-game ratio is 3–4× noisier
here than in the displacement plane (±0.09 against ±0.025 on the mean), which is
most of why.

What it gains is **CosineSpaDE**, and it is the largest separation anywhere in
this study: R2 **1.564** [1.430, 1.705], R3 1.403 [1.264, 1.550] against a null
of 1.086, paired null − R2 = **−0.478** [−0.614, −0.349] and null − R3 =
**−0.317** [−0.457, −0.185]. The same paired contrasts in full R²¹ are −0.245
[−0.335, −0.163] and −0.166 [−0.261, −0.081] — the block roughly doubles them —
and in the displacement plane they are −0.020 [−0.053, +0.013] and −0.002
[−0.039, +0.035], i.e. nothing. SpaDE moves the same way and stops short of
resolving (null − R2 = −0.059 [−0.164, +0.052]). One dictionary family, in the
field where the per-game skew is worst; it is the same lead the R²¹ figure
flags, now larger and paired, and it still needs a held-out space before it is a
finding.

**Between rules, the block separates worse in seven spaces and far better in
two.** Paired R2 − R1 per game, raw layer 5: **+0.059** [−0.017, +0.134] in the
block, against +0.069 [+0.018, +0.122] in full R²¹ and +0.113 [+0.093, +0.133] in
the displacement plane — the block is the only one of the three that does not
resolve it. In CosineSpaDE it is +0.521 [+0.407, +0.640] against R²¹'s +0.261 and
displacement's +0.132. **Do not read the spread ratio as agreeing with this**:
within ÷ between-rule spread is *tighter* in the block (per-game median **0.856**
[IQR 0.697–1.032] against R²¹'s 0.917 and displacement's 0.924), because that
statistic is a geometry of the six categories in their own field and says nothing
about the control they are divided by.

## Reading it honestly

- **The drawn paths average displacement *vectors*, so they cancel, and each cell
  prints by how much.** `coherence` is ‖mean over games of *D*‖ ÷ mean over games of
  ‖*D*‖, taken on the rule mean each path draws (`coherence_rule` in the npz, not
  the per-branch `coherence`): a rule mean cancels across its member categories
  too, so the members' own coherences average to a different and larger number. In raw layer 5
  the drawn paths keep 0.20 (R1), 0.58 (R2) and 0.42 (R3) of a game's displacement
  at *r* = 0 and 0.18–0.22 by *r* = +24, so the length of a drawn path is **not**
  how far a branch travels. That is the same estimator the fork panels have always
  used for a rule mean — it is why `../r21_run_population`'s full-R²¹ series sits
  below every one of its members — applied one level up, to games instead of
  categories.
- **The CSV averages per-game distances, which cancels nothing**, and that is where
  the "how far" question is answered. It disagrees with the figure by construction:
  the figure draws R1's path out of a mean that has thrown away 80% of each game,
  the CSV measures R1 on every game intact and puts it at 0.909 of its control. The
  coherence number printed over each cell is the bridge between them, and the gap
  is not a plotting error. This is why the figure is not allowed to be read for
  magnitude — a reader who takes a path's length for a distance reads the study
  backwards.
- **The CSV is the same measurement as the single-game figure's (c), on a
  population, and it disagrees with it.** 1.53× and R3 > R2 > R1 on game 1235;
  1.02, 1.02 and 0.91 with no ordering over 100 games. The single-game figure is an
  illustration of the construction, was selected as the argmax of its illegal
  branches' mutual separation, and its numbers were never the study's. Quote this one. Do not
  reconcile the two by picking whichever is convenient.
- **R1 is the occupied-square rule, so "runs less far" may be about the branch's
  board delta and not about legality.** Its two members are the only categories that
  place no new disc — `occupied_own` changes nothing, `occupied_opp` recolors (78.6%,
  `../README.md`) — while every R2 and R3 category plays onto an empty square. That
  categorical split is perfectly confounded with R1-vs-rest and nothing in this
  taxonomy separates it. The *graded* part is testable inside R1, and here it fails:
  if board delta drove the deficit, `occupied_own` would run shorter than
  `occupied_opp`, and paired per game in the full space it does not — +0.003
  [−0.026, +0.033] in raw layer 5, −0.004 to +0.015 with every interval covering zero
  in all nine spaces. The displacement deficit is not graded by board delta, which
  weakens that reading without killing it: both members are still "no new disc".
  Separating the categorical half needs a branch this taxonomy lacks — an illegal move
  onto an empty square breaking nothing else, or a legal one that recolors. Until then
  R1's deficit is a direction with an unresolved alternative explanation, and a figure
  that upgrades it to a claim has to rule that alternative out first.
- **The plane is fit once per space, on the population mean field.** The PC
  variance ratios printed on the panel axes are shares of the *mean field's* variance —
  74% and 13% in raw layer 5, against 38% and 13% for one game. Averaging deletes
  the per-game dimensions, so a high ratio here says the mean field is
  low-dimensional, not that the plane holds much of any game's displacement, and
  the two figures' variance ratios are **not comparable**.
- **The interval does not cover the plane.** Each cell's plane is fit on the mean
  field of the *same* 100 games the bootstrap is then taken over, so the interval is
  conditional on that fit. A different draw of 100 games would move the axes as well
  as the points on them, and nothing here bounds by how much. The CSV's `plane` rows
  are in-sample and alignment-weighted for the same reason; the `full space` rows
  are the ones to quote.
- **The mean direction is barely resolved, and the figure does not show it.** At
  n = 100 the 95% interval on the 2-D mean at the end of a path is about as wide as
  the whole fork is long, so the drawn direction is not a measured direction, and
  two rule paths that end apart on the canvas are not two measured directions
  either. That interval used to be drawn, as a flat confidence ellipse at each
  path's end, and it was removed because it did not survive being looked at: lying
  flat in the plane under this figure's fixed camera it foreshortens into a lens,
  and at the width n gives it the ellipses stack and sit on top of the trajectories
  they belong to — worst in the R²¹ plane, where they read as four overlapping
  rings. **The uncertainty did not go away with the drawing.** It is the sentence
  above, and per cell it is the **coherence** number printed over each panel:
  0.15–0.35 at *r* = +24 in the R²¹ plane, i.e. 65–85% of one game's displacement
  cancels against the others' before the path is drawn. **Read the figure for the
  shape of the construction, never for a direction; take magnitude and its interval
  from the table above and the CSV.**
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
  as the by-construction divisor and is not an estimate. That the null lands a few
  percent *above* 1 is a property of this population and not of this estimator:
  `../r21_run_population` measures the same second control on the same 100 games in
  full R²¹ and gets 1.04–1.10 across the nine spaces, against 1.042 [1.006, 1.078]
  here. Two unrelated fields, one gap. On the old mixed-t₀ hundred it was 0.97–1.02.
  What it is *not* is an ordering against the rules — see the paired contrasts above.
- **The rule grouping is weak, and the per-game number is the one to quote.**
  Within-rule ÷ between-rule mean pairwise distance over these 100 games has median
  **0.924 [IQR 0.842–0.983], above 1 in 21%** (displacement, raw layer 5),
  consistent with `rule_groups`' 0.945 over 1550 games. FIGURE_GUIDELINES §1.7 wants
  this ratio on the canvas; it is not there any more, so it is here, and `main()`
  prints it on every run. Near 1 means a rule's members travel barely closer to
  each other than to another rule's, so the three drawn means are a labelling of
  the six categories and not a measured structure. The same statistic **on the mean field** is
  **0.672**, and it is a different quantity — the spread of three averaged
  trajectories, not the typical spread of a game's — so it must never be quoted as
  if it were the population's value. Both are in the npz (`spread_game`,
  `spread_mean`) and printed on every run. The R²¹ plane's own is median 0.917
  [IQR 0.744–1.031], above 1 in 33%; the `d1…d7` block's is 0.856
  [IQR 0.697–1.032], above 1 in 30%.
- **Quote the median beside the mean in R²¹.** Mean − median reaches 0.11 there
  (CosineSpaDE's null, mean 1.053 against median 0.947), against ≤ 0.02 anywhere
  in the displacement plane: the per-game ratio is mildly right-skewed in R²¹,
  where a game with a barely-bending control pulls its own ratio up. The CSV
  carries both for every row. The `d1…d7` block is worse again: mean − median
reaches 0.14 (CosineSpaDE R2, 1.564 against 1.428).
- **Ratios, so only the within-cell contrast carries.** A plane's units are its
  space's own and they differ by ~800× across the grid, so the CSV divides by the
  cell's own control and never reports a distance. That makes nine cells readable
  side by side; it does not make them nine measurements of the same thing.
- **The spliced figures are spliced.** Every branch replays the source game's
  `tokens[t0+1:]`, the asymmetry the rollout set exists to remove, and the three
  `*_rollout` figures are the twin over that set. They are not a replication of
  each other — *What the rollout set says* below.

## What the rollout set says

Same 100 games, same nine spaces, same estimator, over the 5 drawn offsets
*r* = 0, 2, …, 8; `results/violation_divergence/fork_population_{plane}_rollout.csv`.
Raw layer 5, per game in the full space, mean over the 100 games with a 95%
percentile bootstrap CI over source games (2000 resamples, seed 0):

| series | displacement, spliced | displacement, rollout | R²¹, rollout | `d1…d7`, rollout |
|---|---|---|---|---|
| R1 placement | 0.909 [0.885, 0.934] | **0.956** [0.938, 0.974] | 1.304 [1.171, 1.442] | 1.436 [1.277, 1.604] |
| R2 bracketing | 1.022 [0.996, 1.049] | **1.084** [1.059, 1.110] | 1.350 [1.186, 1.514] | 1.593 [1.394, 1.805] |
| R3 turn | 1.017 [0.987, 1.047] | **1.058** [1.029, 1.088] | 1.211 [1.076, 1.356] | 1.443 [1.228, 1.685] |
| null (a third legal move) | 1.042 [1.006, 1.078] | **1.018** [0.998, 1.037] | 1.187 [1.028, 1.359] | 1.206 [1.043, 1.372] |

**R1's deficit survives the rollout on the displacement, and it is the only
spliced conclusion that does.** R1 sits below its legal control in **9 of 9**
spaces with every upper bound under 1 (0.937–0.976), and paired per game
null − R1 = **+0.063** [+0.044, +0.080] in raw layer 5, resolved in **9 of 9**
spaces (+0.035 … +0.079). It is about half the spliced gap (+0.110 … +0.138) and
in the same direction everywhere. This is the one effect that replicated across
every spliced field, and resampling the future does not remove it.

**The spliced *negative* result does not survive.** "No rule outruns a legal
alternative" was the study's headline, and on the rollout displacement R2 and R3
do: paired null − R2 = **−0.066** [−0.088, −0.044] and null − R3 = **−0.039**
[−0.066, −0.012] in raw layer 5, both resolved and both **sign-flipped** from the
spliced +0.021 and +0.025, which were not resolved either way. Across the grid the
flip holds in 5 of 9 spaces for each (raw layer 5, PAnneal, Standard, CosineSpaDE,
SpaDE); the four TopK/Gated cells sit on zero (−0.000 … +0.016). CosineSpaDE is
the largest, null − R2 = −0.155 [−0.184, −0.130].

**It is the branch set, not the shorter window.** The rollout figure averages 5
offsets and the spliced one 13, so the comparison was re-run with the *spliced*
branches cut to the same *r* ≤ 8 (raw layer 5, full space): R1 0.906 [0.880, 0.934],
R2 1.037 [1.006, 1.069], R3 1.020 [0.987, 1.053], null 1.039 [1.000, 1.076], and
paired null − R1 = +0.133 [+0.100, +0.164], null − R2 = +0.002 [−0.032, +0.037],
null − R3 = +0.019 [−0.018, +0.058] — the spliced verdict unchanged to three
decimals on the R1 gap and still unresolved on the other two. The window is not
what moves the numbers.

**Both descriptor planes lose R1 and reverse it.** In R²¹ every series jumps above
1 in **9 of 9** spaces (R1 1.26–1.45, R2 1.23–2.01, R3 1.11–1.77) and paired
null − R1 = −0.117 [−0.271, +0.044] in raw layer 5, resolved *negative* in 7 of 9
spaces; the `d1…d7` block is the same and stronger, null − R1 = **−0.230**
[−0.405, −0.068], resolved negative in **9 of 9**. So R1's "runs less far" is a
property of the displacement field under both branch sets and of the curvature
descriptor under neither: on the rollout set, an occupied-square branch's *φ*
runs measurably *further* from the reference than a third legal move's. Read the
displacement row as the finding and the descriptor rows as what the rollout does
to a curvature statistic when the whole suffix is resampled.

**The null itself moves, and that is the attribution loss showing up as a number.**
The empirical null — a second legal move scored against the first — is 1.018
[0.998, 1.037] on the rollout displacement against 1.042 [1.006, 1.078] spliced,
but 1.19 and 1.21 in R²¹ and `d1…d7` against 1.06 and 1.09. Two legal
continuations sampled from the model diverge from each other in curvature far more
than two spliced ones do, so in the descriptor planes the divisor and the
numerator are both inflated and only the paired contrast is readable.

**Coherence is unchanged where it must be and larger where it may be.** At *r* = 0
the rule paths' coherence is identical to the spliced figure's to three decimals in
all nine spaces on the displacement plane (0.256–0.396) and on `d1…d7`
(0.378–0.795) — it has to be, since the *r* = 0 field is bit-identical between the
trees — and 0.236–0.657 against 0.287–0.681 in R²¹, where the per-column scaling
the plane fit is taken with differs between windows. At the last drawn offset the
rollout keeps *more*: 0.145–0.193 at *r* = +8 against 0.143–0.198 at *r* = +24 on
the displacement, 0.26–0.45 against 0.15–0.35 in R²¹, 0.31–0.66 against 0.14–0.44
in `d1…d7`. That is the shorter window, not a stronger signal: the spliced paths
cancel further the longer they run.

**The rule grouping is unchanged.** Within ÷ between-rule spread over the same 100
games, raw layer 5: displacement median 0.927 [IQR 0.887–0.983] against the
spliced 0.924 [0.842–0.983]; R²¹ 0.913 [0.746–1.034] against 0.917 [0.744–1.031];
`d1…d7` 0.821 [0.619–1.017] against 0.856 [0.697–1.032]. Still near 1, still a
labelling rather than a measured structure, and the rollout neither helps nor
hurts it.

Everything in *Reading it honestly* applies to these three figures unchanged — the
paths are still vector means that cancel, the plane is still in-sample and refit
per space, the CSV's `full space` rows are still the ones to quote, and the
bootstrap still resamples source games and never branches or rollouts.

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
