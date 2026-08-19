# PROCESS.md

Running log of what's being worked on and the non-obvious decisions behind it, so
humans and future agents have context the code alone doesn't show. Newest first.
Keep entries short; the code is the source of truth for *what*, this is the *why*.

---

## The population fork loses everything but the fork (2026-08-19)

Three cuts to `analysis/violation_divergence/fork_population/`, in one day and at the
user's direction, ending with a figure that is nine 3-D cells, a legend, and each
cell's coherence number. Nothing else is on the canvas.

**The 95% CI ellipses.** One per rule path, flat in the plane at the path's end. They
do not survive being looked at: the camera is fixed at `elev = 18`, so an in-plane
ellipse foreshortens into a lens, and in the R²¹ plane the orientation turns it into a
full ring — four overlapping rings sitting on the trajectories at the last drawn move.
Under it is a real fact, which is why the drawing was defensible and the *reading* was
not: at n = 100 the interval on the 2-D mean is about as wide as the whole fork is
long, so the mean direction is barely resolved. That fact is now a sentence and a
number in the study README instead of a shape on the canvas. Deleted with it:
`_ci_ellipse`, `CI_CHI2`, `ci_ellipse_proxy`, `CI_ELLIPSE_LABEL` and its `_rank`
branch, `_path`'s `open_end`, the `computed_zorder` override, and
`rule_fork_population_panel`'s `game_coords` parameter. `_path` is byte-identical to
its pre-ellipse form, so `fork_panel` and `rule_fork_panel` draw what they always drew
— checked by rendering `fork_and_board`'s ten outputs from both revisions: identical
PDF content streams, identical PNGs.

**Panel (b), the metric row.** With the ellipse gone the figure had one panel carrying
shape and one carrying magnitude, and the two disagree by construction — (a)'s vector
mean throws away ~80% of each game (`coherence`), (b)'s mean of per-game distances
throws away nothing. Cut to the grid alone; `metric_panel`, `cell_series` and the
ladder constants went with it. `table()` and the CSV stay untouched: the numbers are
the study's record and the panel was only ever a picture of them.

**The muted note block under the legend.** The spread ratio, the coherence definition,
the population disclosure, the cancellation note and the R1/R2/R3 key — the same sweep
the `gray/gray-parent` entry below made over the parent figures, on the last study that
still had one. Same keep test there and here: a line stays only if a FIGURE_GUIDELINES
section needs it *on the canvas*.

**Two guidelines this figure no longer satisfies on its own face, both deliberate.**
§4.8 wants a 3-D projection paired with a quantitative panel; the pairing is now the
README's table and the CSV, and §4.8 was amended to allow a table under two conditions
(the projection says it carries no magnitude; the caption says where the numbers are).
§1.7's spread ratio and §5's "shorthand earns a key" are **not** satisfied — `R1`,
`R2`, `R3` and `coherence` are shorthand and nothing on the canvas defines them. The
key is in the README and nowhere else. This is a real gap, not a solved one; it was not
patched by editing §5, and it is the thing to look at first if the figure ever ships
outside the repo.

The earlier ellipse entry below records the decision to draw it and is history now, not
current state. It is left as written; this is a log.

---

## Probe geometry: board-state directions against rule-legality directions (2026-08-18)

**Question:** the board-state probe and the two rule probes (legal/illegal, and
legal/R1/R2/R3) both decode `blocks.5.hook_resid_post`, so their weights are
comparable with no alignment step. Do they read the same directions? Ported from the frozen tree (`cosine-spade-gamma-sweep`); every number below was
re-derived here and the PNG is byte-identical. Study in
`src/spade/probe_geometry/`, `analysis/probe_geometry/`; everything is CPU and runs off
committed probes plus cached activations, so there is no new extraction.

**Contrasts, not class columns.** A k-way softmax is unchanged by adding one
vector to every class weight, so a raw class column is not a direction. This is
not pedantry here: the board probe's unidentified common component has norm 1.14
against a contrast norm of 8.53, and re-drawing the gauge at that magnitude moves
a raw column's cosine with the legality direction by sd 0.0098 — about 40% of the
entire pooled effect. The gauge is also an *Adam* artifact; softmax-CE gradients
sum to zero across classes, so plain SGD would have left it at its 0.131 init.
Board contrasts are `colour` = mine − theirs and `occupancy` =
½(mine + theirs) − empty, which together span everything a 3-way per-square head
identifies. The rules probes standardize, so their direction is `W_c / scale`.

**The pooled answer is near-tautological and was nearly shipped as a finding.**
Pooled mean |cos| is 0.018–0.030. The committed rules probes are *square-agnostic*
— one direction read off the residual wherever the candidate move was spliced in —
while every board direction is *square-specific*. A global marginal cannot be a
combination of per-square conjunctions, so near-orthogonality is close to forced.
An independent verifier that never read `probes.py` produced this objection; it
is the reason `square_rules_directions` exists.

**Square-matched, the alignment is real.** Refit legality per candidate square:
cos with the board read-out at its *own* square is +0.149 colour / +0.306
occupancy (ridge, matched estimator: +0.174 / +0.359), against 0.017–0.026 at
every other square. The matched-vs-other-square contrast is the right null — same
rules direction, wrong square, everything else fixed. It survives the matched
estimator, a board refit on the rules corpus (+0.177 / +0.362), per-square class
balancing, every null, and attenuation correction for a per-square split-half
reliability of only 0.525. **Projecting out the token-embedding subspace raises
it**, so the confound this repo has been burned by before is not driving it.

**Two narrowings that have to travel with the number.** (1) It is R1 only:
legal-vs-R1 gives +0.180 / +0.387 with 97%/93% of squares positive, while
legal-vs-R2 (+0.064 / +0.008) and legal-vs-R3 (+0.008 / +0.031) sit at the floor,
and that is not sample size — R3 has more rows per square at the median. So the
claim is "legality aligns with board read-out at its own square *for the square
being occupied*", not for the bracket geometry. (2) Per-move-index centering
leaves colour at +0.161 but drops occupancy to +0.077, so the occupancy half is
largely move-index-mediated (disc count grows with t).

**The honest headline is a subspace number, not a cosine.** 21.2% of each
per-square legality direction lies in the 2-d board subspace at its own square,
against a 0.39% isotropic null — but the *full* 124-direction board basis captures
only 26.4% against a 24.2% chance level for any 124-dim subspace of 512. The
board-relatedness is entirely local to the square; the other ~79% is not board
read-out at all.

**Gotchas worth keeping:**
- *The colour/occupancy ordering is shrinkage-dependent.* occupancy/colour runs
  0.344/0.142 at lambda 1, 0.333/0.152 at 10, 0.282/0.200 at 100 and 0.190/0.221
  at 1000 — the two families swap. The sign, and the matched-vs-other-square
  contrast, are the robust parts; the gap between the two board families is not.
  The square arm's lambda is pinned at 10 (held-out 0.9252) rather than the grid
  argmax of 100 (0.9269), so this is a live choice, not a rounding detail.
- *Several intervals do not contain their point estimate, by construction.* The
  point comes from the committed probes and the interval from ridge refits that
  agree with them at only 0.75-0.87, so the intervals are width, not location.
  The alternative is all-ridge tables; the committed probes were kept because
  they are the probes that were actually trained.
- *Do not read a results file without checking how it was produced.* A
  `--n-games 200 --n-boot 4` smoke run left on disk was reviewed as if it were
  the real output, and its matched occupancy of 0.107 was diagnosed as a
  square-indexing bug that did not exist. `cosine_tables.check_square` is now a
  tripwire that asserts the full-run value, but the general lesson is that a
  partial run should not be able to overwrite the real artifact's filename.
- *The isotropic null is the wrong floor for the pooled numbers.* Covariance-
  matched, it is 0.084–0.110, so the pooled 0.013–0.017 is *more* orthogonal than
  chance, and the pooled directions capture less of themselves in the board
  subspace (0.09–0.13) than a random direction does (0.24).
- *The committed 4-class probe barely finds `legal`* — test recall 0.232, with
  71.9% of true-legal rows predicted R1. Every rules direction is `W_c − W_legal`,
  so all three are anchored to that class, and `illegal·R1 = 0.923` is partly
  confusion rather than shared content.
- *The board probe's 8000/2000 split is already a game split* (rows are
  game-major, the cut falls at game 160), so its eval set is game-disjoint but is
  not an extra holdout.
- *Board `colour` reproduces itself at only cos 0.67–0.92 across disjoint game
  halves*, so per-square outliers and per-cell differences in the pooled table are
  noise and should not be discussed.
- *Memory, not compute, is the binding constraint on this box*: an 8 GB anon cap
  is shared across concurrent agents, and the corpora are float32 for that reason
  (directions stay float64; they agree to cos ≥ 0.999998).

**It lands on the same suspicion as the R1-deficit entry above, from the other
side.** That entry asks whether R1's distinctiveness across three fields is about
the branch move's *board delta* rather than about legality, and notes the two are
perfectly confounded inside this taxonomy. Here the legality read-out at a square
aligns with that square's *occupancy* direction, for R1 and for nothing else —
which is what "the probe is reading board state, not legality" looks like in
direction space. Neither result separates the confound; they agree about where it
is. Note this study was ported from the frozen tree, so its rules arm is the
committed `move_legality` probes rather than the branch populations that entry uses.

**Method.** Four agents in parallel — tables, figure, red team, and one verifier
explicitly forbidden from reading `probes.py` so a bug there could not pass into
the result twice. The verifier reproduced every number and produced the objection
that inverted the headline, which is the argument for keeping that role blind.
The integrator never delegated `probes.py`, `paper_style.py`,
`FIGURE_GUIDELINES.md` or this file. `FIGURE_GUIDELINES` gained **4.10, direction
geometry**, per its own instruction to add the rule with the figure type.

---

## The R1 deficit may be about the board delta, not about legality (2026-08-18)

Raised by the `r21_run_population` author in the round that produced 8e3481a, against the
one result that replicates across three fields: R1 placement violations move and step
*less* than a second legal move. `fork_population` gives paired null - R1 = +0.133 [+0.103,
+0.163] on the displacement, `step_length_fork` gives R1 - control = -0.060 [-0.081, -0.038]
at r = 0 and resolved negative in all nine spaces, and `r21_run_population` puts
`occupied_opp` at 0.92-0.98 of its control in all nine spaces of the curvature descriptor.

R1's two members, `occupied_own` and `occupied_opp`, are exactly the occupied-square
categories -- the only branches whose move places no new disc. Every R2 and R3 category
plays onto an empty square. So the three fields may be agreeing about the branch move's
**board delta** rather than about legality, and inside this taxonomy that categorical split
is perfectly confounded with R1-vs-rest: no contrast in the data separates it.

**What partially separates it is a test inside R1.** `occupied_own` implies no board change
at all; `occupied_opp` implies a recolor (78.6%, `analysis/violation_divergence/README.md`).
If board delta drives the effect, `occupied_own` is the more extreme of the two. Paired
`occupied_own - occupied_opp` per game over the same 100 games, 2000-draw percentile
bootstrap, seed 0:

- **Step length at r = 0** -- resolved negative in **8 of 9** spaces: raw layer 5 -0.0606
  [-0.0880, -0.0334], the rest from GatedAnneal -0.0765 to SpaDE -0.0381 [-0.0611, -0.0154];
  CosineSpaDE -0.0180 [-0.0557, +0.0216] is the exception.
- **Displacement, ratio to the control over r >= 0** -- resolved in **0 of 9**: raw layer 5
  +0.0030 [-0.0260, +0.0333], every other space between -0.0037 and +0.0150 with every
  interval covering zero.

So the two fields split. The step-length effect is graded by board delta and the account
survives there; the displacement effect is not, which weakens the account without killing
it -- both R1 categories are still "no new disc".

Settling the categorical half needs a branch the taxonomy does not have: an illegal move
that places a disc on an empty square and breaks nothing else, or a legal move that
recolors. Until one exists, R1's deficit is a direction with an unresolved alternative
explanation, and any figure upgrading it to a claim has to rule the alternative out first.
Docs only for now -- each sub-study README's "Reading it honestly" carries the confound with
its own field's number and no implication about the other's.

Arrays: `$SPADE_DIVERGENCE_DIR/step_length_fork/n100.npz` (`ratio [S, G, B, T]`, roles in
`role`, t0 index off `pos`) and `$SPADE_DIVERGENCE_DIR/fork_population/displacement_n100.npz`
(`game_space [S, G, B]`).

---


## A third fork plane: R^21's raw step-length block alone (2026-08-18)

`--plane d7` on `fork_and_board` and `fork_population`, beside `displacement` and `r21`.
The plane is the uncentred top-2 of the same `Δφ(t) = φ(z_b) − φ(z_ref)` cut to
`RESTRAV_FEATURE_NAMES[:N_RAW_D]` — the 7 raw step lengths `d1…d7`. It is deliberately
**not** `fork_r21.BLOCKS["d"]`, whose substring match also takes `mean/min/max/var d`
and so counts the same seven twice. `curvature_plane.PLANE_COLUMNS` holds the two
slices; the r21 path is otherwise untouched in both scripts, and `names_of` grew the
7-wide case it would have answered with the moment names.

Why: `r21_shared_basis`'s `run_from_reference` (b) has said since it was written that
R^21's two halves disagree about the same branches, and has only ever said it as a
scalar run ratio. Nobody had drawn the fork in the half that moves.

**The block is not a cosmetic variation of R^21, and it is not the ‖Δz‖ step-length
figure either.** Over the shared t0 = 20 population, raw layer 5, paired per game:
null − R1 = +0.074 [−0.005, +0.146], null − R2 = +0.015, null − R3 = +0.018 — nothing
resolved, where the displacement plane gives null − R1 = +0.133 [+0.103, +0.163] in all
nine spaces and `step_length_fork` resolves R1 below its control on the ‖Δz‖ ratio in
all nine. The R1 deficit is the study's one replicated result and **the d-block does not
see it**; its per-game ratios are 3–4x noisier (±0.09 against ±0.025). What the block
does see is CosineSpaDE: paired null − R2 = −0.478 [−0.614, −0.349], about twice full
R^21's −0.245 and against nothing at all (−0.020) in the displacement plane. One
dictionary family, so it is a lead, not a finding.

Two numbers that look like they agree and do not. Within ÷ between-rule spread is
*tighter* in the block (per-game median 0.856 against R^21's 0.917), while paired
R2 − R1 fails to resolve there (+0.059 [−0.017, +0.134]) and resolves in both other
planes. The spread ratio is a geometry of the six categories in their own field; it
knows nothing about the control they are divided by. Do not quote one for the other.

Verification: the eight `fork_and_board` and four `fork_population` outputs were
re-rendered. All six PNGs byte-identical; the two `fork_population` PDFs byte-identical;
the four `fork_and_board` PDFs differ from the committed copies by exactly 6 bytes, the
`/CreationDate` string, because those were committed without `SOURCE_DATE_EPOCH`. Proved
by rendering one of them from the stashed pre-change tree under `SOURCE_DATE_EPOCH=1`
and getting the same md5 as the post-change render.

---


## The step-length fork over 100 games, and what it costs the n = 1 reading (2026-08-18)

`step_length_fork/measure.py` sweeps the same ratio `s_b(t) / s_ref(t)` over the study's
shared population -- the first 100 branch-complete games forking at t0 = 20 -- and
`make_figures.py --n_games 100` draws it with a 95% percentile bootstrap CI over games
(2000 resamples, seed 0, resampling games and not branches).

**One script, not two.** The two figures share the quantity, the panel and the nine-cell
grid; they differ in the reduction, the interval, and their notes. The drift went where
it belongs: `branch_panels` grew `rule_ratio_population_panel` beside `rule_ratio_panel`,
exactly as it already carries `rule_fork_population_panel` beside `rule_fork_panel`, and
the ~35 lines of grid furniture became `panel_grid`, which both call. What is left in
each figure body is which array and which notes. A sibling script would have duplicated
the grid; a flag inside one `figure()` would have been eight `if population:` branches.
The n = 1 CSV and PNG re-render byte-identical, and the PDF differs only in
`/CreationDate` (use `SOURCE_DATE_EPOCH=1`).

**Both legal controls are drawn.** Unlike `fork_population`, the control is not the
divisor here -- the reference is -- so `control_legal` and `control_legal_2` are two
draws of one null, measured exactly as the illegal branches are. Drawing both makes the
null's own resolution visible: the c2 - c1 paired gap is +0.008 to +0.023 at r = 0 and
is never resolved, which is the yardstick every rule contrast is read against. Averaging
the two into one null curve was rejected under FIGURE_GUIDELINES 3 -- a null averaged
over k candidates has ~1/sqrt(k) the spread of the signal.

**What the population does to the n = 1 story.** The one game's headline -- *any*
different token lengthens the step -- was that game. Over 100 games the legal control
sits at 0.989-1.007 at r = 0 with every interval straddling 1, against 1.11-1.31 on game
1235. R3's 1.53 in raw layer 5 becomes 1.076 [1.044, 1.108] and goes *below* 1 in the
four gated/TopK spaces. The ordering R3 > R2 > R1 does not survive: R2 - R3 paired at
r = 0 is +0.023 [+0.001, +0.047] in raw layer 5 -- R2 is the larger excursion -- and the
population orders {R2, R3} > control > R1 with R2 and R3 indistinguishable. What does
survive is **R1**: 0.890-0.957 at r = 0 with every interval clear of 1, and R1 - control
resolved negative in all nine spaces (-0.032 to -0.117). That is the same rule and the
same direction `fork_population` found to be its only resolved contrast on the
displacement, off the same 100 games.

**Do not carry the n = 1 band test across.** The one-game figure judged "past r = +4
nothing leaves the control's own excursion band", where that band was +-0.18 to +-0.31 --
one game's noise. Over 100 games the same band is +-0.016 to +-0.075, ten times narrower,
so the two statements are not comparable and the population figure uses the paired
contrast instead.

Game 1235 is in the population as one ordinary member (42nd by id) and is an outlier in
it -- R3 1.53 and control 1.23 against population means of 1.08 and 1.00 -- but dropping
it moves every raw-layer-5 mean by <= 0.005. Being one game of 100 is the whole
difference, not which game.

Cost: 63 s for the sweep on the login node. The reduction is inside the loop -- a chunk
of 5 games is encoded, turned into 16 step lengths and dropped -- so nothing larger than
one chunk of features is ever resident, and the npz is 1 MB.

---


## The step length at the fork, drawn on its own axis (2026-08-18)

`analysis/violation_divergence/step_length_fork/` draws the literal step-length norm
`s_b(t) = ||z_b(t) - z_b(t-2)||` per branch, in the nine spaces, event-locked at t0.
Nothing drew it before: `rule_curvature` is the turning angle only, and the R^21 figures
carry step length as 7 of 21 components inside a descriptor.

**The panels draw the paired ratio `s_b(t) / s_ref(t)`, not the length.** The nine spaces
differ ~580x in absolute step length (raw layer 5 18.6, SpaDE 0.032), so a raw length is
not a quantity nine panels share; and drawn absolutely (checked, from the CSV, before
committing) every branch tracks the same per-position swing -- trough at t ~ 24, peak at
t ~ 38, ~2x deep in raw -- which is the game, not the branch. The within-game division
cancels both. The absolute lengths stay in `results/violation_divergence/step_length_fork.csv`
beside every ratio and the per-space scale is in the sub-study README.

What it shows, on game 1235 (n = 1, selected): the step at the fork is longer for a turn
violation (R3 1.25-1.74x the reference, in all nine spaces) and *shorter* for an
occupied-square one (R1 0.80-0.91x, in all nine), and it decays -- from r = +6 on nothing
leaves the legal control's own excursion band (+-0.18 to +-0.31 by space) except one
offset in two of the nine panels. **A second legal move does most of it too** (control
1.11-1.31x at r = 0): only R3 exceeds the control, by 1.04-1.57x. So the figure's honest
reading is "the first two steps after a branch differ, and only the turn violation differs
from what a legal alternative already does".

`branch_panels.rule_ratio_panel` is the panel, beside the fork panels rather than in the
script, because the 100-game population twin is a separate change that reuses it. The
prefix ratio is asserted `== 1` exactly, in all nine spaces, on every render -- it holds,
which is also the cheapest check that the rows are branch-aligned.

The sampling convention is on the figure, not only in the README: positions are
`fork_plane.positions` at STRIDE 2, so a step spans 4 plies, and `../rule_curvature`'s
per-ply convention reads 29.2 against 18.9 on the same legal corpus. Same trap as the
theta one; same sentence.

---


## The fork figure over 100 games, and what the mean does to it (2026-08-18)

`analysis/violation_divergence/fork_population/` is `fork_and_board`'s population twin:
the same nine 3-D forks and the same "how far does each rule run" panel, over the first
100 branch-complete games forking at *t0* = 20 instead of over one selected game. This
entry covers both halves: the sweep in `measure.py` and the figure.

Decisions worth keeping:

- **The board panel is deleted and the panels renumber.** Every game forks at one move
  index and each has its own position there, so there is no board to draw and the
  composition panel a population figure would otherwise carry (a *t0* histogram, n per
  rule) is a spike and a constant. (a) is the grid, (b) is `fork_and_board`'s (c), and
  the README cross-references that figure's (b) for what the branches are.
- **(a) and (b) measure different things on purpose.** (a) is a mean over games of the
  displacement *vector*, so games whose directions differ cancel; (b) is a mean of
  per-game *distances*, which does not. Without the coherence number each cell prints
  (‖mean D‖ ÷ mean ‖D‖) a reader takes the short fork in (a) for branches that barely
  move, which is the opposite of what (b) reports. Same cancellation
  `r21_run_population` documents one level down, over categories instead of games.
  The printed number is the *rule mean's* coherence (`coherence_rule`), not the mean
  of its member categories' (`coherence`): the drawn path cancels across categories
  as well as across games, so the members' average is a different and larger number
  than the path's. Both are in the npz; only one belongs next to the path.
- **Uncertainty in a plane is an ellipse, not a bar.** The estimator is a 2-D mean, so
  a 1-D error bar would be a choice of axis; `branch_panels._ci_ellipse` draws the
  covariance of the 100 games' projected positions ÷ n at chi2_2(0.95), lifted to the
  panel's *t* with `pathpatch_2d_to_3d`. Drawn at **every** marker it is a stack of
  rings the paths vanish into — at this n the CI is about as wide as the fork is long —
  so it is drawn once, at the end of each path. Both renders were compared.
- **The sweep is space-outer and encodes every space twice.** The plane is not known
  until every game has been seen, so one pass over a space's games gives the mean
  field, the plane is fit on that, and a second pass projects every game into *that*
  plane — which is what makes (a)'s ellipse and (b)'s in-plane ratio measure the
  plane the panel draws. Holding the 100 fields between the two passes instead is
  ~500 MB per space, and the login node caps a user at **8 GiB** across every process
  the session owns
  (`user.slice/user-*.slice/memory.max`); the held-field version was SIGKILLed four
  times, always in the ninth space, at 1.4–2.4 GB RSS with the encoders resident.
  `MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4` was needed even after the rewrite, to keep
  glibc's per-thread arenas inside the cap. A rerun at a larger `--n_games` belongs
  in Slurm; the login node is not where this sweep should be run again.
- **`src/spade/violation_divergence/population.py`** holds the three things a
  population sweep needs and a single-game figure does not: a fixed branch-row order
  (`ROLES`), so a game is one slice of a `[G, B, …]` array and the roles are
  constants rather than per-game lookups — `branches.pt` order is neither; the games
  forking at one *t0*, so a mean over games is a mean over aligned positions; and
  encoders built once, since `fork_plane.space_features` reloads the checkpoint on
  every call. `r21_run_population/measure.py` still has its own copies of the first
  two. Two callers, so they stay duplicated until a third asks for them.
- **The finding is that the population contradicts the game the design figure
  draws.** Game 1235 puts R3 at 1.53× its legal control in full raw layer 5 and
  orders R3 > R2 > R1 in all nine spaces; over 100 games R2 and R3 sit on 1
  (1.022 [0.996, 1.049], 1.017 [0.987, 1.047]) and R1 sits below it in every space
  (0.909 [0.885, 0.934] in raw layer 5). No rule outruns the empirical
  legal-vs-legal null (1.042 [1.006, 1.078]), but only the R1 gap is resolved:
  paired on the same games, null − R1 = +0.133 [+0.103, +0.163] against +0.021
  [−0.011, +0.052] for R2 and +0.025 [−0.011, +0.060] for R3, and on the whole
  1696-game pool the null falls below R2 and R3. The ordering the single-game figure
  invites is game-level noise, and so is the null's margin over R2 and R3. Leaving game 1235
  out moves every displacement mean by ≤ 0.006, so it is not the argmax pulling the
  population: it is that one game is one game.

FIGURE_GUIDELINES 1.7 wants the within/between-rule spread ratio on the canvas, and
the per-game median is the population's — `spread_mean`, the same statistic on the
mean field, is the spread of three averaged trajectories. It goes in the legend band
under (a), beside the coherence key, since that is where the reader meets the rule
entries; the band's lines wrap, because `save_figure` crops tight and an over-wide
line silently widens the canvas past the panel grid instead of being cut.

`rule_fork_population_panel` is a sibling of `rule_fork_panel`, not a flag on it: the
faint per-member traces come out at population n (a "member" here is itself a mean over
games and would read as game-to-game spread), which is the same split commit 5b36969
made in `rule_curvature`, and the two bodies have already drifted once.

**(b) is a log ratio ladder, not `fork_and_board`'s bars.** Every rule is measured
twice — in that cell's full space (filled marker, the plane-free number the claim rests
on) and in the plane the cell above draws (open marker, alignment-weighted and
in-sample) — and the two differ by 1.1–3.9× across the nine cells. On one linear axis
from zero the in-plane series sets the scale and every full-space marker lands on the
line at 1 where it cannot be read, so the panel moved onto `r21_run_population`'s ratio
rungs, dense around 1 and sparse above it. Bars do not survive that: a bar asserts a
distance from zero and the ladder's baseline is the control at 1. `SLOT_W` = 0.21, not
`fork_and_board`'s 0.26: four slots per space, because the empirical null is
`control_legal_2` (the legal control is the divisor, identically 1 with no variance),
and at 0.26 the gap between two spaces' groups closes below the gap inside one.


## Recovering the spliced move with no counterfactual, across the nine spaces (2026-08-18)

`analysis/violation_divergence/changepoint_localization/` -- one trajectory, no reference
branch, no knowledge that anything was spliced. Every other read-out of this tree scores
`phi_branch(t) - phi_ref(t)`, which is an oracle a detector does not have; this asks what is
left without it. CPU-only and under a minute: it reads the `d` that
`curvature_discriminant/branch_geometry.py` already cached for all nine spaces.

**Three findings.** Four dictionaries clear raw layer 5 on the *paired difference* -- GatedAnneal
+0.055 [+0.030, +0.080], GatedSAE +0.043, BatchTopK +0.032, TopK +0.029, none spanning zero --
while Standard and PAnneal sit significantly below it and the two sparsemax spaces are
indistinguishable from it. The ordering reproduces `curvature_discriminant`'s *supervised*
ordering from an unrelated statistic. Localization is the stronger half: GatedAnneal hits the
exact ply 0.343 of the time against 1/34 uniform and 0.038 for its own legal-splice control, and
all three false-alarm arms sit at chance in every space. And the `theta` block -- curvature
proper, the thing the descriptor is named for -- carries almost none of it (0.53-0.58), only in
the *straighter* direction, which is RECURVE's polarity and not the naive one.

**The metric was chosen against three candidates and the two obvious ones lose.** Per-position
standardization is the whole game: move index explains 81.3% of the variance of `d`, so a
pooled Gaussian scores 0.034 hit / 0.461 AUROC, i.e. chance, against 0.283 / 0.673 for the
profile baseline. It is what replaces the missing paired reference, and it absorbs the
mover-relative parity sawtooth for free (parity is a function of the move index), which is why
the scan runs at every ply rather than stride 2. **A Mahalanobis over the full 21-D descriptor
is the wrong object**: it scores 0.109 against 0.283 for one signed coordinate, because a
quadratic form is two-sided by construction and every two-sided form fires *harder* on real
human championship games -- legal in every position -- than on the violations (0.694-0.706 vs
0.616). One-sided puts championship at 0.376, the opposite tail. The sign is the finding, not
a detail. Structurally, `mean d` is an exact linear function of the seven step lengths, so the
21x21 covariance has rank <= 19 at any sample size and effective rank ~13.

**Paired beats unpaired for localization, unpaired beats paired for detection**, and the reason
is one line: with `phi_t = mu_t + b_game + eps_t` the unpaired form scores `b + eps` against
`Sigma_b + Sigma_eps` while the two-window contrast scores `eps_R - eps_L` and the game-level
term cancels. It is the MOSUM statistic (Eichinger & Kirch, *Bernoulli* 24(1), 2018) --
equivalently Truong/Oudre/Vayatis's window-sliding discrepancy with a Mahalanobis cost -- and
its equal-width windows leave `argmax` unbiased in position, which CUSUM/GLR/E-divisive's
`sqrt(c(n-c)/n)` factor does not. That property is why it is the primary and not CUSUM.

**A circular-shift surrogate is useless as a null for any argmax hit rate, and the first pass
shipped one.** `argmax(roll(s, k)) == (argmax(s) - k) mod n` *exactly*, so rolling a row does
not preserve the autocorrelation it was supposed to preserve -- it just relocates the peak
uniformly, returns ~1/n whether the curve is white noise or heavily smoothed, and comes out a
deterministic function of the hit rate it is compared against (`corr = -0.973` across the nine
spaces, and the better the detector the *lower* its own "null"). It inflated the headline from
11.6x to 16.7x before it was caught. The correct nulls were already in the study: uniform
`1/n_cand`, and the legal-splice control arm, which is measured at 0.021-0.045 everywhere. The
surrogate is deleted. Worth checking anywhere else a permutation or shift null backs a
localization rate.

**Two more corrections from the same review.** The selection and report corpora are **not**
disjoint -- 501 of the 573 scored games share their source legal game with a straightness
violation, and the source game is what carries the idiosyncrasy the MOSUM cancels -- so three
places that asserted disjointness, one of them a figure caption, now say what is actually true
(the *splices* differ; the sweep is also scored on a vs-reference-shaped contrast because the
straightness corpus has no legal-splice arm; the criterion's margin is 0.0074). And marginal
CIs cannot carry a comparative claim: "four dictionaries clear raw" needed the paired
difference on shared bootstrap draws, which is why `resamples()` is made once and handed to
every space rather than each calling its own `boot_ci`.

**`arms()` was a second copy of `localize.arms`.** Flagged by a peer session doing an unrelated
DRY pass. `spade.move_legality.localize.arms`/`one_per_game` already own the seeded
one-branch-per-game rule (pooling a rule's member categories gives it ~1/sqrt(3) of an item's
real spread), they agree with the hand-rolled version on all three deterministic arms, and they
hand back R1/R2/R3 for free -- so the rows are now column-comparable with
`localization_by_space.csv`. Migrating changed which illegal branch each game draws, which
moved every number inside its CI and swapped GatedSAE and BatchTopK in rank. Not a copy of that
session's `population.py`: this study loads no checkpoints and its row-selection rule is
different, so per the house rule they stay separate.

**Fit corpus, and why it is not the obvious one.** The per-move-index profile is fit on
straightness legal games minus `meta.held_out_legal`, i.e. 1,427 clean games disjoint from
every scored one. Splitting the 575 divergence games in half instead would have halved the
scored n for no gain; the two corpora are the same distribution to four decimals (z of the
divergence reference branches under the straightness profile: mean -0.04, sd 1.01), and the
cross-pipeline `d` agrees to 8e-6 relative.

**Open, and it is load-bearing: the norm-only baseline has not been run on this tree.** On the
straightness corpus `-z(||z_t||)` alone reaches 0.808 AUROC in GatedAnneal, `d_t` regresses on
`(||z_t||, ||z_{t+1}||)` at R^2 0.87-0.88, and partialling the norm out drops GatedAnneal from
0.587 to 0.173 while raw layer 5 *improves*. `curvature_discriminant`'s npz carries `d` and
`theta` but not `norm`, so this needs one GPU encode pass. Until it runs, the study is a
magnitude claim wearing a geometry label, and the README says so. Second extension: the
population is the 575 games that study selected for its probe-holdout guard, predating
`extend_branches`; the tree now holds 3,185 branch-complete games.



## divergence_by_space at the shared n = 100 (2026-08-18)

`--n_games N` on both scripts draws the study's shared N-game population instead of the
whole tree: the first N branch-complete games forking at t0 = 20, now
`branches.games_at_t0` so every n-game sweep in the study selects the same games rather
than each script re-deriving "which games". Separate artifact at every step
(`curves_nN.npz` -> `divergence_by_space_nN.png` -> `..._nN.csv`), so the whole-tree
figure is never overwritten by a short run. n = 100 is ~2 min.

Two things the twin forced, both of which the whole-tree figure needed anyway:

- **The bands are now drawn.** The README has always said "bands are 95% CIs ... about
  +-0.015 wide, hairline at this n", and the code computed the half-width and threw it
  away. `draw_series` fills it, and the legend carries `mean over n = N source games,
  band = 95% CI` -- the n statement FIGURE_GUIDELINES 1.2 wants and the figure did not
  have. The whole-tree PNG was regenerated; its CSV is byte-identical.
- **The 1.7 spread line is measured on the population drawn.** It was a hardcoded string
  quoting `rule_groups` over the whole tree, which would have been simply wrong under the
  twin. `measure.py` now calls `rule_groups.spread_ratio` per game on the same
  displacement field it is already holding and stores `spread [S, G]`; the figure quotes
  raw layer 5's median [IQR] over the games it drew. The whole-tree npz predates the field
  and keeps the constant.

The twin reproduces the whole-tree reading: every plane ratio over r >= 0 within 0.02
except CosineSpaDE, whose R2/R3 run 0.10-0.15 higher. Its r = 0 spike is the one place n
bites (4.4x at n = 100 against 3.1x, CI half-width 0.59 against 0.11); on the plateau the
CI is 0.05-0.09 and the claim reads the same.

---


## The R^21 run figure, over 100 games instead of one (2026-08-18)

`analysis/violation_divergence/r21_run_population/` (`measure.py` + `make_figures.py`) is
`r21_shared_basis`'s `run_from_reference` with a population behind it: same descriptor,
same four candidate planes, same anchors, same ratio to the game's own legal control, but
swept over the study's shared 100-game population with a mean and a 95% CI. Everything --
z-scoring, the per-space plane, the shared plane, the activation plane -- is refit **per
game**, which is what keeps a game one independent item; a plane fit across games would
make the interval a fiction.

The R^21 core moved to `src/spade/violation_divergence/fork_r21.py` (the study README had
flagged that it belonged there), and `PLANE_STYLE`/`FULL_STYLE` moved to `paper_style`, so
both figures draw the same series in the same colours. The single-game script imports both
and reproduces its old PNGs byte-identically -- that was the acceptance test for the move.

Three things the population says that one game could not:

- **The empirical null sits near 1** (a third legal move against the control, 1.04-1.10,
  CI +-0.06..0.08, covering 1 in six of the nine cells). On one game it scattered
  0.88-1.53, so the ratio had no calibration at all.
- **The spread across spaces was game noise.** One game put the nine cells over 0.48-4.03;
  100 games put every cell in 0.84-1.21. Nothing in that figure licensed a claim about
  which dictionary separates a violation, and the panel-(b) title claiming the two halves
  of R^21 disagree did not survive either (paired theta - d = -0.02..+0.06, CI +-0.06..0.07,
  not one space clearing its own interval).
- **Every plane inflates the run, the status-quo one most** (+0.22 paired against full
  R^21, vs +0.11..+0.17 for the three R^21 planes). That *is* the single-game claim, now
  with an interval.

One estimator asymmetry is inherited and worth knowing before quoting `full R^21 < 1`: the
illegal series is the run of the **mean displacement vector** over the six categories (the
reduction the fork panels draw), so their incoherent parts cancel and the group mean is
shorter than any member -- 0.84-0.93 against members at 0.92-1.33. The null is a single
branch and carries no such cancellation. The figure says so in-panel and the README says
it at length; the per-category reading is that an illegal move runs about as far as a
second legal one.

`--n_games` defaults to 100 so the committed figure is what a bare run reproduces.

**Superseded the same day:** this first sweep took the first 100 branch-complete games at
whatever t0 each was drawn at, which is not the population the study's other n-game figures
use. It now calls `branches.games_at_t0(meta, POPULATION_T0, n)` -- the first 100 forking at
t0 = 20 -- and the numbers above are the re-run's. Its `top2` also moved from an SVD of the
[~150, up to 8192] field to an eigendecomposition of its Gram matrix: LAPACK's gesdd does
not converge on the activation displacement of one of these games, and the 150 x 150
symmetric problem carries the same plane (the single-game figure's stdout is unchanged).

---

## The probe-head bar chart is AUROC, not TPR/TNR (2026-08-17)

`analysis/move_legality/detection_rates.py` -> `detection_auroc.py`; the figure is
`raw_detection_auroc.png` and the CSV `detection_auroc_raw.csv`. Same six-bar layout, but each
bar is now `localize.detection` — the paired AUROC at *t0* of one arm against the same game's
matched legal control — instead of an argmax hit rate.

**Why six bars became four.** A rates figure needs a false-alarm bar beside the hit-rate bar
or the hit rate means nothing, which is what the TNR bars were. AUROC is threshold-free and
already paired, so both of those bars come free. The legal-vs-legal null (`control` scored
against `control_legal_2`) briefly took the slot they left and was cut: drawn, it was two bars
sitting on the chance line to say what the chance line says. It is still computed per head and
still in the CSV — it is what makes that line empirical rather than assumed — and its value
prints in the note (0.502 both heads, CI 0.491-0.512).

The R1 leak is louder here than at the argmax: 0.892 against R2/R3's 0.999, with the 2-class
head's pooled illegal arm at 0.960.

Nothing was recomputed: `localize.detection` is the same function `localization_raw.csv`
already reports, so the figure and that CSV agree by construction rather than by luck.

## The two generation-legality panels become one figure (2026-08-17)

`analysis/generation_legality/figures/generation_legality.png` is now (a) P(at least one
illegal move by move t) beside (b) the per-move illegal mass split by rule — the pair the
2026-08-12 trim had left as two PNGs. Same reason as the rules board pair below: the study's
claim is "the model does emit illegal moves, and the ones it emits are R3-heavy", which is one
sentence and not two files. `generation_legality_by_rule.png` is gone. The rollout figure stays
standalone: it is a robustness check on (b), not a third question. No statistic moved — only
plotting code was touched.

**The panels are two different measurements, so the figure has to say so.** They share
`GENERATION_STYLE`'s "any illegal" hue, so the pair must not read as one curve continued: (a)
is cumulative per game, (b) is per move, and their prefix distributions differ — (a)'s games
the model wrote, (b)'s were handed to it. That went into the panel titles, and each panel keeps
its own note (estimator, n, which CSV the interval is in) under its own axes rather than one
note under the figure.

**Layout: a note is an axes child, so the rule key's y is only knowable after a draw.**
`tight_layout` lifts the axes to make room for a note hanging below them, so no fixed figure
fraction lands just under the deepest one; `key_under` draws the canvas, takes the lowest note
bbox and puts the key `KEY_PAD` below that. The note wrap comes from the panel's own width for
the same reason a constant cannot serve both — a panel of the pair is a third of the canvas
where the rollout figure's is three-quarters of a narrower one. Two smaller moves: (b)'s legend
went to the lower right, the only region four rules on a log axis leave empty at half width,
and both y labels broke onto two lines, because `↓ better` on one line runs into the panel
title at this panel height.

## The constraint rules get drawn, on one board (2026-08-12)

`figures/othello_rules/rules_legal_and_illegal.png` — the legal move in panel (a) and one
violation per rule in panel (b), off ONE position (game 606 of the seed-42 stream, t = 16,
black to move) so the legal move and the three violations are comparable without the board
moving under the reader. Script is `src/spade/figures/figure_othello_rules.py`;
`docs/othello_rules.md` had defined the taxonomy for five studies and never shown it.

**The two panels became one figure (2026-08-17).** They shipped as two PNGs, which made the
reader hold a board in memory across two files to see that (b) is the same position with one
square moved — and that identity is the whole point of the pair. One figure, panels side by
side, one rule key under both. Each panel keeps its own "black to move" and its own note:
they are two boards, and a panel read on its own must still say whose turn it is.

**Choices that are not arbitrary.** R2 is drawn as `empty_no_bracket`, not the shallower
`empty_detached`: the taxonomy orders R2's members by the board computation a violation takes
to detect, and only the deepest is *visible* as a rule — h3 has a real run of white discs
that dies on an empty d3. `empty_detached` would show an empty square with nothing to see.
R1 is `occupied_opp` (the study's worst probe cell) at b2 rather than the better-separated
e2, because e2 sits directly above e3, the last disc of the R2 run, and read as part of it.
The position was picked by scanning t in [14,26] of 800 games for: all three rules available,
a played move flipping several discs in >1 direction, and >= 3 cells between every drawn
square. All three co-occur at 97.8% of positions, so nothing here is a rarity.

**`boards.rays` is new, and `flip_set` now goes through it.** Drawing R2 needs the *failed*
bracket — the run and the empty square that breaks it — which the old flat `flip_set` threw
away. `rays` returns one `Ray` per direction (`run`, `terminator`, `blocker`), `flip_set` is
a one-line comprehension over the closed ones, and the rewrite was checked identical to the
pre-refactor body on 16,000 (board, square, colour) cases before anything was built on it.

`othello_board.mark_square` joins `draw_board` so there is still exactly one board renderer.
FIGURE_GUIDELINES gained §4.9: the disc on a square is usually *why* it is marked, so mark
with a cell outline and a corner-inset marker and never a fill; pure black is unavailable to
overlays because it is already the disc colour; and the side to move goes in the axes, not in
a title, because turn-dependent legality is unreadable without it and titles do not survive
camera-ready.

## Move-legality keeps two curves and loses the rest of its figures (2026-08-12)

The study shipped 24 PNGs -- six 4-panel figures per space, plus an 11-space grid per score
and a summary. All of it is deleted. What is left is two standalone single-panel event-locked
curves for `raw` only: `raw_2class.png` (P(illegal)) and `raw_4class.png` (1 - P(legal), one
line per rule), drawn at 5.5 x 3.4 in with no suptitle and no panel letter, so either drops
into a paper as-is.

**Deleting figures must not delete evidence.** Three numbers the README cites lived only
inside a deleted panel, so they moved into CSVs before the panels went: the mean 4-class
posterior at t0 (the 0.247-on-R1-for-legal leak, the mechanism sentence behind "R1 is the
hole") is now `results/rule_posterior_<space>.csv`, and the pick-a-position chance `1/(t0+1)`
that the localization rate is read against is now a `chance_hit_rate` column in
`localization_<space>.csv`. The per-rule `scalar_rows` calls were interleaved with the deleted
per-rule `draw` calls and had to be kept -- dropping the loop wholesale would have cost 12 of
21 rows of that CSV.

**The window then stopped at r = 0, and the hatched tail went with it.** Drawing it invited
two readings the study cannot separate. It is mostly the *splice*, not the illegality -- any
different move at t0 leaves the replayed suffix on a board it does not belong to, and 77.6%
of the post-event elevation over the reference is already present in a branch whose t0 move
is perfectly legal. And the probe was never fit to answer past t0 at all: every fit-set row's
tokens[:t] is a legal game, so a prefix containing an illegal move is a state it never saw,
and the suffix assumes the *reference* move (branches.py asserts it is shared bitwise), so
there is no label for the spliced board either. Adjudicating the control arm -- the only one
where a board survives -- put numbers on the first reading: 24.7% of suffixes are illegal at
r = +1, median first break r = +3, 99.8% break inside 14 moves, and where a board does
survive the probe tracks it at AUROC 0.92-0.98 on a sample collapsing 1500 -> 273. Nothing
past t0 was ever scored; now nothing past t0 is drawn.

Five arms on one axis is past what hue and marker carry: R1/R2/R3 land inside a 32-level gray
band, so the rules take dash patterns and the reference is left as the only solid heavy line.
With the window ending on the event, every arm also takes an explicit marker at r = 0 -- the
staggered `markevery` does not land there, and r = 0 is now the whole figure.
`hit_curve`'s per-`t0` bootstrap loop was the expensive half of every `scalar_rows` call and
existed only for the deleted rate-vs-t0 panel; it is gone, and `scalar_rows` is ~25x cheaper.

`space_comparison.py` is now a table-only script (`localization_by_space.csv`); it lost
`--camera_ready` along with its plots. `src/spade/figures/move_legality_panels.py` is gone --
one consumer left and two helpers surviving is not a module.

## The rollouts become a shared bank, and the by-rule picture survives self-conditioning (2026-08-12)

A third generation-legality figure, `generation_legality_by_rule_rollout`: F2 exactly, drawn
by the same function, on prefixes the model wrote for itself. **The answer is a null and it
is worth having.** Rollout ÷ teacher-forced illegal mass is median 0.996 over the move axis
and 0.993 pooled; the R3 > R2 > R1 ordering and the R1 suppression are unchanged. So the
by-rule claim, which was measured entirely on corpus prefixes, survives the distribution the
model is actually in — which F1's two arms diverging by 39% over a game said was not free.

**The late-game dip is survivorship and must not be read as the model behaving better.** At
t >= 40 the ratio is 0.907, because a rollout leaves the table at its first illegal move
(`umpire` asserts, so no board exists after it) and the games that leave carry the most
illegal mass. Everything in `rollout/positions.csv` is conditional on survival; n falls
1000 -> 891 and the panel prints it. `scan.py`'s table conditions on a legal prefix too,
which is what makes the pair comparable at all.

**The rollout population is now a bank, because it was being redrawn per study.**
`sample_rollouts` was shared but nothing else was: `straightness` and `generation_legality`
each spent a GPU pass at their own seed, so "the rollouts" named two populations and neither
was reusable. `spade.othello.rollout_bank` is that population once, at `study_dir("rollouts")`.
It **extends by prefix** -- the sampler draws in ROLLOUT_BATCH = 250 blocks from one seeded
generator, so block k depends only on the seed and on every earlier block being full -- so
growing the bank leaves every existing consumer's games untouched, which `build` enforces
and re-derives rather than trusting. `generation_legality.rollout` is deleted: after the
refactor it was a wrapper around `rollout_bank.load` with a redundant manifest.

Two reproductions worth recording. Re-running the sampler reproduced the committed
`rollouts.csv` exactly, so saving the sequences was provably non-destructive; and the bank's
1000 games are token-identical to the ones the first scan ran on, asserted in the job before
the rescan.

**Not done, deliberately: `straightness` still draws its own.** It uses a different seed, so
repointing it would silently move committed figures and geometry results. That is the
remaining half of the uniformity and it needs its own re-render, not a side effect of a
plotting change.

Also fixed here: `spread()` now takes arrays rather than the skew frame, because there are
three callers -- skew ratios (0.638), teacher-forced mass shares (0.951) and rollout mass
shares (0.888) -- and each rule figure must print the spread of its own source, never the
other's.

## generation_legality is two single-panel figures now (2026-08-12)

Cut from 2x3 panels to 2x1 on request: F1 is P(at least one illegal move by move t),
cumulative and averaged over games; F2 is P(illegal move at move t) split into R1/R2/R3.
The skew-vs-null panels, the mirrored KDE and the played-rate arms are gone from the
figures. **Their measurements are not gone** -- `rule_skew.csv`, `played_rate.csv` and
`overall.csv` still carry every number the README quotes, and the ratio-to-matched-null view
now lives there and in the prose. Only `illegal_density.csv` was deleted, with its producer
`aggregate.density`: a KDE grid is a plotting artifact, not a measurement, and the median and
IQR it was drawing are in `overall.csv`.

**The trim surfaced a number the old figures never showed.** `spread()` was computed on the
skew ratios (0.64) because that is what the panel drew. F2 draws raw mass, so it owes the
reader the spread of the mass, and that is **0.95** -- at ~1, FIGURE_GUIDELINES 1.7 says the
grouping is a label rather than a finding, and the figure now says so under the panel. It is
one rule's doing: R2's three members span 1.73 dex where R1's two span 0.11 and R3 has a
single member. Dividing by the matched null was hiding that, because a category's null
carries its own availability. `overall.csv` now carries both columns.

Two consequences worth knowing. The faint members set the y axis if you let them --
`empty_detached` decays four decades below R2 by the endgame -- so F2's floor comes from the
rule curves and a member leaves the axis instead. And `act_geometry.kde_band`'s `bandwidth`
parameter was reverted along with `density()`: it had exactly one caller and that caller is
gone, so the file is byte-identical to its pre-study state and bracket_flip's committed
figures are unaffected.

Re-running `aggregate` changed exactly one CSV (`overall.csv`, the new row); the other four
came back byte-identical, which is the check that the trim touched no statistic.

## The move-legality probe works and the logits beat it (2026-08-11)

Branch `study/move-legality-probe`, worktree `../spade_sae_worktrees/move-legality-probe`.
New: `src/spade/move_legality/`, `analysis/move_legality/`,
`src/spade/figures/move_legality_panels.py`. A plain L2 logistic regression on the layer-5
residual at the position of a *played* move -- 2-class and 4-class -- read out over every
position of the 34,546-sequence violation-divergence branch tree.

**The headline is the control, not the probe.** In raw layer 5 the probe localizes the
spliced illegal move (argmax over t <= t0) in 87.4% of games against a 5.0% legal-control
false-alarm rate and AUROC 0.960. The model's own `1 - P(the move played)`, off the logits
already sitting beside the activations, does it in **3923/3923** games at **AUROC 1.0000**,
with the illegal move above every other position in its own sequence by a median 0.072 in
100.0% of games. Every probe in every space is strictly worse than reading the output head.
So "a probe finds the illegal move" is not on its own evidence about the representation --
the same shape of finding `legality_probe`'s RED_TEAM.md reached against board argmax.
`model_baseline.py` exists so that cannot be forgotten; it is the rightmost column of
`space_summary`.

**What survives is R1, and it inverts the taxonomy.** Every violation on an *empty* square
reads off at AUROC >= 0.998; an occupied square reads off at 0.82-0.93, and `occupied_opp`
-- playing on an opponent's disc -- is 0.819 with a 0.502 hit rate, the worst cell in the
study. The logits are at 1.000 on all six, so this is the linear read-out and not the
model's knowledge. `taxonomy.CATEGORIES` sorts occupancy *first* as the shallowest check;
the residual disagrees. The 4-class posterior shows the mechanism -- 0.247 of the mass on R1
for moves that are legal, in `results/rule_posterior_<space>.csv` -- and it lines up with the
generation study's finding that the model's own illegal mass is suppressed 0.10x on R1 and
inflated 7.1x on R3.

**Balance is structural.** A fit-set item is one (game, position) pair carrying one row per
class; `probe.balanced` keeps whole items, so classes are equal in every split *and* share
their game and move-index marginals exactly. No reweighting, and a probe reading move index
scores at chance. The 2-class form draws one illegal row per item rather than averaging the
three, which would give the illegal class 1/sqrt(3) of its spread.

**Two things cost a rebuild, and both are worth knowing.**

*Position 0 is the attention sink and this was the only study reading it.* Left in the fit
set and the argmax, the probe emits a per-space **constant** there -- 1.0 in BatchTopK, 0.0
in raw -- and BatchTopK's localization came out as exactly 0.00000 with a median rank of 2
while its AUROC at t0 stayed at 0.99. An exact zero beside a near-perfect AUROC is what
caught it. `manifold_capture.build_corpus` and `straightness` already drop position 0;
`move_legality.SINK` now does too, at both ends.

*The batched rank AUROC had to go.* It is ~30x faster and cannot average ties, and the
no-probe baseline saturates `1 - P(played)` at exactly 1.0 in float32 on most of the illegal
arm -- the one arm whose estimator would then have differed. The baseline is computed in
float64 (22,732 distinct values out of 22,777, none exactly 1.0) and every bootstrap
resample now goes through the same tie-averaged `probe.auroc` as the point estimate.

**Ground truth stops at t0 and the figures had to say so.** A branch keeps the source game's
`tokens[t0+1:]`, legal on the reference's board and no other, and `umpire` asserts rather
than replaying past an illegal move. Everything is scored over t in [1, t0]; the rest is
drawn and hatched. That region turns out to carry a result of its own: after t0 the illegal
arm averages 0.422 but the *legal* control averages 0.362 against the reference's 0.156, so
six sevenths of the post-event elevation is the splice and not the illegality.

**Across spaces, width buys more than training.** `random_proj` -- a ReLU of a random
512->8192 projection -- reaches 0.894/0.790 test macro-F1 and 0.888 localization, above raw
on all three and above every trained dictionary except Standard, which clears it by 0.006
and 0.030. At matched sparsity TopK/BatchTopK do beat `random_topk` (L0 80) by 0.04/0.09, so
training buys something -- less than the 16x expansion does. PAnneal is the trap: 0.888 on
held-out fit-set positions, essentially raw's 0.889, and 0.758 localization against raw's
0.874. A fit-set accuracy table would have called it faithful.

Left open: every branch token is a uniform draw from its category, not from the model's own
illegal conditional. The model's real illegal moves are near-misses it finds likely, so a
model-sampled branch set would shrink the baseline's margin -- and that is the experiment
that decides whether the probe is redundant. Single seed (42) throughout, so no probe number
here has an across-seed interval.

## The model's own illegal moves are near-misses (2026-08-11)

Branch `study/generation-legality`, worktree `../spade_sae_worktrees/generation-legality`.
New study `spade.generation_legality` — every other study here *injects* an illegal move;
this one asks whether the model emits any, and of what kind. It does: 0.9982 of next-move
mass on legal squares, 18 illegal argmax moves in 56,000 positions, and 11.5% of
self-generated 59-ply games containing one.

**The skew needed a per-position null or it would have been a plot of the board filling up.**
R1's share of the illegal squares goes from 5/60 at ply 1 to ~55/60 at ply 56, so a pooled
square-count null measures that and nothing else. Every skew number is instead the model's
own illegal mass at each position redistributed uniformly over that position's illegal
squares. Also: the three rule shares sum to 1, so R2/R3 > 1 is partly forced by R1 < 1 —
the ordering and the per-category magnitudes survive that, the absolute ratios less so.

**Rules were the wrong grouping and the §1.7 spread caught it.** Within-rule ÷ between-rule
spread is 0.64, because R2 splits: `empty_no_bracket` 6.7× against `empty_detached` 0.45 and
`empty_own_only` 0.33. Read at category level the result is cleaner than "R3 goes" — the two
over-weighted categories (`opponent_only` 7.1, `empty_no_bracket` 6.7) are exactly the two
that look like legal moves, and everything detectable from the target square alone is
suppressed 3–10×.

**Both survival arms ship because the composed one is a modelling assumption.** ∏(1 − p) over
a game's plies assumes independence and, worse, assumes the model stays on the prefix
distribution it was measured on. The free-running arm (reusing `straightness.build_games.
sample_rollouts` unchanged) is 22% higher at 57 plies. The gap is the cost of
self-conditioning and it is the only thing here that is not teacher-forced.

**t ∈ [0, 56), not [4, 55].** Forfeits are not spread over the game: they are 13.5% at
*t* = 56 and 4.0% at *t* = 57 and ~0 before, so cutting exactly those two plies keeps
whole-game acceptance at 96% while covering the rest of the game. Over [0, 58) it drops 18%.

**Gotchas.** The divergence read-out set is 3,923 games, not 2,000, so excluding it by token
match consumes ~3,900 stream rows before this study's first game. And an sbatch run from a
worktree needs `PYTHONPATH=$SPADE_REPO/src` exported at submit time — the editable install
points at the main checkout, so `cd "${SPADE_REPO:?}"` alone silently runs main's code, or
here fails with ModuleNotFoundError because the package only exists in the worktree.

## The violation shift was R2 and R3 all along (2026-08-07)

`theta_dist_comparison` got the same treatment as `feature_pca`: the illegal move is drawn
as its three constraint rules, borrowed from the divergence branch population, and the
figure lost its in-figure prose block and its conclusion-shaped title.

**The split paid for itself immediately.** R2 and R3 straighten the whole-game median in
all nine spaces (−0.06 to −0.62°); R1 moves it at most 0.19° and lands on the *wrong* side
of legal in three. That is not a refinement of "every dictionary keeps the violation
shift" — it is the explanation. The straightness violation class injects on empty squares
only, so it is R2 and R3 by construction, and the shift it carried was never evidence about
placement violations. An illegal move detectable from the target square alone leaves
whole-game curvature where it was. Narrowed in `docs/claims_status.md`.

**No new compute.** Per-game mean θ is component `mean θ` of the ReStraV descriptor already
cached in `rule_descriptors/`, and it is the same estimator the geometry npz gives
(`moment4` is `nanmean` over the trajectory), so the two sides of the panel are one number.

`density_panel` now takes its style table as an argument, the way
`embedding_panels.class_cloud` already did — a panel draws the game classes or the rules
that replace the violation class, and those are two dicts. Both callers pass it.

Left open: the rules are median gaps with no interval. `sae_families_auroc` still scores
the violation class, so an R1/R2/R3 AUROC with a bootstrap CI does not exist. That is the
figure this one now argues for.

## feature_pca draws the three rules, and borrows them (2026-08-07)

Branch `fig/pca-rule-split`, worktree `../spade_sae_worktrees/pca-rule-split`. The
descriptor PCA now splits the illegal move into R1/R2/R3 instead of drawing one
violation class. Random tokens were already out of it (`fig/drop-random-tokens`); what
this adds is the split, and the figure had never been rendered in this tree.

**The straightness study cannot supply an R1, and regenerating it is the wrong fix.**
`build_games.EMPTY_CATEGORIES = [3,4,5,6]` injects on empty squares only, so its
violation class is R2 (n=1450) and R3 (n=550) and nothing else; `place_no_flip` asserts
the square is empty. Adding R1 there means deciding what a placement on an *occupied*
square does to the board — for `occupied_opp`, an overwrite with no flip line, which is
not an Othello operation and leaves "random legal continuation" meaning something else.
The violation-divergence branch set sidesteps all of that by never advancing the board
past the illegal move, and carries all six categories over 2000 rule-complete games. So
the rules are read from there.

**One pass, two products.** `rule_curvature.build` was already streaming those
activations through nine encoders for panel (a)'s curves; it now also returns each
game's 21-D ReStraV descriptor per rule (`rule_descriptors/<space>.npz`). Encoding is
the cost and the geometry is free — 7 min on an H100 for the spliced tree, against ~3 h
on a login node. Skip-if-exists is per product, so the pass will not rewrite a
`rule_population` npz another job just wrote.

**Mixing the two studies on one plane is legitimate and it is not free.** Both drop the
sink and measure per ply, agreeing to 1.5e-5 deg on the 1998 shared games, so the
descriptors are the same number. But a rule point is a branched move inside a game and a
class point is a whole game — which is exactly why panel (a) of `rule_curvature` refuses
to draw the straightness violation class beside its rules. Same call here, same caption.

**Result: the rules do not move *in the plane*.** Championship reaches 2.07 pooled SD
from legal; no rule exceeds 0.40 on either PC in any space. A faint R2 > R3 > R1 ordering
appears wherever the displacement is readable.

**Corrected the same day by `centroid_distances.py`: that is a fact about two components,
not about the descriptor.** In the full 21-D the rules sit 1.13-1.68 pooled SD from legal
against a legal-vs-legal null of 0.16, and rule-to-rule 0.25-0.64. In the plane,
rule-to-rule averages 0.05 over the nine spaces, which is the null exactly — the three
rules there are indistinguishable from one class sampled twice. The plane carries ~34% of
the variance and is fitted without labels, so it was never obliged to hold the
rule-discriminating direction, and it does not. Anything quoting the 0.40 has to say "in
this plane" or it is wrong; the wording in the README and claims_status was tightened.

What *is* structural is where the separation sits: `d1..d7` and `θ1..θ6` pivot before the
earliest t0 = 10 and the legal class is (1998/2000) the same games as the rules' sources,
so those 13 components differ by exactly 0.00 and all 1.68 SD is carried by the 8 moments,
led by `mean d`, `var d`, `max d`, `mean θ`.

**And roughly half of that is the spliced suffix.** Because the separation is entirely in
whole-game moments and t0 ∈ [10, 34] of 58 positions, most of the descriptor is computed
over a tail the illegal move has invalidated -- the source game's real moves replayed onto
a changed board. `--tree rollout` swaps that for OthelloGPT continuing from the
post-violation board and nothing else: rule-to-legal 1.00 -> 0.56 over the nine spaces,
worst for R1 (1.13 -> 0.40, near the 0.16 null), while rule-to-rule *rises* 0.36 -> 0.56.
The spliced tail adds a displacement common to all three rules, so it inflates the gap
from legal and compresses the rules together; removing it exposes the rule-identity signal
(R1/R2 0.64 -> 0.99). Championship moves 3.51 -> 3.41, the control that says the swap
touched only the branch classes. Every rule distance in this repo is a spliced number
unless it says otherwise.

`make_figures.py --tree rollout` now renders `restrav_pca_rollout.png` beside the spliced
one, with the suffix in the suptitle because the two are otherwise the same picture and
which one a reader is holding decides what the rules mean. The resampled panels show the
rules collapsed onto legal harder than the spliced ones do (raw L5 PC2: R2 -0.18 against
-0.40), which is the distance table drawn.

One reason the drop is conservative: a resampled rule point averages K = 4 rollouts on top
of its member categories, so its class tightens (within-class SD 0.96-0.97 -> 0.92-0.94,
game classes flat at ~0.99). Standardized distance divides by that, so the shrinking
spread inflates the resampled column and it still halves.

Left open: `theta_dist_comparison` is still spliced-only, and its "R1 barely moves" median
gaps come off the same tree that the distances say is inflating R1 most.

**The CosineSpaDE outlier cloud is a property of the game.** Its 114 legal games are,
token for token, the same 114 source games whose R1, R2 and R3 branches also land there;
82 rollouts join them and no championship game does. Delayed code recruitment (L0 13/34
at positions 2/3 against 44/114, caught up by position 4), and branching an illegal move
into one of those games does not move it out. The old README read the same cloud as a
class mixture, which it never was.

Left open: the rollout tree has no `rule_descriptors`. The figure reads the spliced tree
only, and a resampled-suffix version would be a second figure rather than a variant.

---

## The grey text on the fork figures, minus the caption (2026-08-07)

Branch `gray/gray-fork`, worktree `../spade_sae_worktrees/gray-fork`. Covers
`analysis/violation_divergence/fork_and_board` and `.../game_trajectory` only;
sibling branches sweep the other figure directories.

Every muted block that a reader could look up in the README is gone: the rollout
variant's caveat (five lines beside the board in `fork_and_board`, eleven inside
the δ̂ panel in `game_trajectory`) and `fork_and_board`'s two-line note under (c).
All of it was already in the two READMEs word for word, so nothing had to move.

**The test used for each block was not "is it prose" but "can a reader look it up
while looking at the figure".** That keeps four things in, each against a rule:
the board's reading key and the R1/R2/R3 member key (§5 — the outlined cell, the
corner marker, the dashed would-be disc and the flip ring cannot be decoded from
anything else on the page), (b)'s "0 by construction" band in `game_trajectory`
(§4.2), and the axis-direction/estimator text on (c).

**(c)'s note became (c)'s x axis rather than disappearing.** What the two-line
block uniquely carried was the estimator — the mean over the drawn offsets, and
n = 1 — which §1.2 wants stated and which changes with the variant (13 offsets
spliced, 5 rollout). It is now `set_xlabel`, which is where `game_trajectory`'s
metric panel already put the identical sentence. The rest of that block (n per
rule, one dot per member, the within-rule spread) was a second copy of the (a)
legend's `(mean of n)` and of (c)'s own glyph legend.

**The game-1235 selection disclosure was restored as one line, not deleted.** It
had been dropped entirely by `c8b6136` along with the legend titles. That was a
step too far: game 1235 is the argmax over 1550 branch-complete games of its
illegal branches' mutual separation, its apparent structure holds in 0.07–0.17 of
games, and a reader who does not know it reads a selected game as a result — the
figure changes meaning without the line, which is the bar for staying inside.
One line above the R1/R2/R3 key, drawn only when `--game` is the default, since a
non-default game is not the selected one and must not claim to be.

`SELECTION_NOTE` is duplicated in the two scripts. It belongs in
`spade.violation_divergence.branches`, where `selection_note(game)` used to live;
this branch is not allowed to touch `src/`.

**Layout.** `fork_and_board` only: the legend beside the board went back to
`bbox_to_anchor=(0.0, 0.62)`, the value it had before the rollout note was
appended under it, so the block is centred in the band again in both variants.
The selection line and the rule key each anchor off the bottom of the thing above
them (`branch_style.below`), so the (c) x label's height propagates.
`game_trajectory` needed no retuning: its note sat inside the pre-branch band,
which §4.2 wants empty anyway.

---

## Gray prose out of the four R^21 curvature studies (2026-08-07)

Branch `gray/gray-curvature`, worktree `../spade_sae_worktrees/gray-curvature`.
`fork_curvature_space`, `fork_expanding_curvature`, `r21_shared_basis` and
`curvature_discriminant` had grown multi-paragraph `color=MUTED` blocks under their
legends — the largest was `curvature_discriminant`'s, two wrapped paragraphs plus a
rollout paragraph plus a rollout caveat, ~20 printed lines. Almost all of it was
already in the study READMEs; the little that was not was moved there. It is gone.

**The test that decided every case.** Does a reader who does not have this sentence
*misread the panel in front of them*, or do they merely fail to know something they
could look up? Only the first stays. Under it, kept: the `R^21 undefined` marker on
the segment where the descriptor does not exist (§4.2 — an unannotated absent stretch
reads as an artifact), `reference = 0 by construction` and `axis not calibrated here`
(§4.2), the `chance` label on the bare chance line (§3), the jackknife/rollout ellipse
semantics in the legend — *dispersion, not a CI* (§4.6), the per-grid
"comparable/not-comparable across cells" verdict (§4.6), the R1/R2/R3 member key and
`curvature_discriminant`'s panel-header key `ctrl · rand · 1 game ×null` (§5). Removed:
every restatement of the descriptor definition, the window, the camera, where the mean
over K rollouts is taken, the null's construction, and the effective-sample-size
argument. The camera angle and the bootstrap semantics were *only* on the figures, so
they moved into the three READMEs that lacked them rather than being deleted (§4.8,
§1.2 both want them stated — they do not say on the figure).

**The game-1235 selection line stays, as one line, and it is the only prose the
`fork_curvature_space` grids carry.** Game 1235 is the argmax over 1550
branch-complete games of its illegal branches' mutual separation, and the structure it
shows holds in 0.07–0.17 of games. That is exactly the first kind of sentence: a reader
who does not know it reads a selected game's arrangement as the population's. It was
previously in the README only; it is now one muted line on the `fork_curvature_space`
grids and in `curvature_discriminant`'s legend title.

**The r ≥ 8 rollout saturation caveat came off three of the four studies, and the
reason differs per figure.** `fork_curvature_space`'s windowed grids are cut at
`r = 8`, so the region the caveat is about is not drawn — the window *is* the caveat.
Its whole grids cannot be cut (R^21 needs 8 sampled positions), but they draw the
sampling spread as an ellipse whose legend names it, and that ellipse swallows the
reference and every branch mean in all nine spaces: the reader sees what the sentence
asserted. `curvature_discriminant`'s panels already shade and label `axis not
calibrated here` over the whole `r > 0` region on both variants. It survives as one
line on `fork_expanding_curvature` and `r21_shared_basis`, because there the *entire*
drawn range is past the horizon and the axes are labelled by move index, so nothing
about the drawn window shows it.

**Layout re-tuning.** Every canvas and legend anchor here had been tuned around the
text blocks, so removing them left empty bands (§5: an empty band is a layout bug).
`fork_curvature_space` needed none — `branch_style.below` computes the key's y from the
legend's rendered extent. The rest were hardcoded and moved:
`fork_expanding_curvature`'s companion legend −0.26→−0.175 and its rule key
−0.50→−0.30, with a flat rollout drop replacing the per-newline one;
`r21_shared_basis` grid key −0.105→−0.065 and `separation_by_plane`'s legend
−0.34→−0.26; `curvature_discriminant` rule key −0.178→−0.082 and its figure legend
−0.030→−0.056, since the two-line legend title now grows up where the caption block
used to sit and would otherwise land on (j)/(k)'s x-axis labels.

**Panel (l)'s known overlap is fixed.** `reference = 0 by construction` sits in the
band above the first space row and its tail ran under the legend, which was `upper
right` in the same band. The legend moved to `lower right` — the bottom rows' data
stops well short of the axis — and the x headroom went 0.32→0.60 of the range so
TopK's R2 marker, the one bottom-row point that reaches out, clears the legend box.
The band now belongs to the annotation alone.

Every figure in the four directories was re-rendered for both variants: 8 in
`fork_curvature_space`, 4 in `fork_expanding_curvature`, 12 in `r21_shared_basis`,
2 in `curvature_discriminant`.

**Flagged, `src/` untouched.** `spade/figures/branch_style.ordered_legend`'s docstring
states the opposite policy — that "which game was selected" and "cells are not
comparable" belong in the README — which contradicts FIGURE_GUIDELINES §4.6 and the
judgement above. It should be reworded to name the misreads-the-panel test.

---

## The fork panels get their magnitude, over the population and along r (2026-08-07)

Branch `nfig/divergence-by-space`, worktree `../spade_sae_worktrees/divergence-by-space`.
New directory `analysis/violation_divergence/divergence_by_space/`; nothing existing
changed. It is the companion the fork grids were missing (FIGURE_GUIDELINES §4.8): a 3-D
projection shows the shape of a fork and never its size, so the reader could not tell how
far a branch went. `../fork_and_board`'s panel (c) answers that for one game as one number
per rule; this adds the **r axis and the population**, 3x3 over the nine spaces, mean over
1553 branch-complete games with 95% CIs.

**The plane inflates, but by a tenth, not by several times.** In-plane ÷ full-space ratio
to the cell's control is above 1 in **27 of 27** illegal cells, median **1.108**, range
1.077-1.217. The single-game figures report 2.7x on the same contrast (game 1235, raw
layer 5, R3: 4.08 in the plane against 1.53 in the space) — that game is an argmax over
branch separation, not a draw, and the population number is the one to quote when reading
a fork panel. The mechanism is visible at r = 0 in raw layer 5: the plane keeps 51% more
of R2's displacement than of the control's (1.711 vs 1.133) where the full space keeps
10% more (1.528 vs 1.384), so the in-plane ratio is a difference in retained *shares*.

**In the full space nothing outruns the legal control, in any of the eight dictionaries.**
0.895-0.920 (R1), 0.971-1.026 (R2), 0.983-1.013 (R3). That re-derives the study's
`ratio_illegal_control_L5` = 0.979 through a different estimator and extends it off layer
5. What varies across the grid is how much the plane exaggerates, not what the branches do.

**Two kernels worth keeping.** One Gram matrix G = D Dᵀ gives both distances — full space
is sqrt(diag G), in-plane is the top-2 eigenvector rows scaled by their eigenvalues — so
the 8192-wide basis is never formed and 1553 games x 9 spaces is ~35 min on one core.
And `measure.py` writes **raw** distances, normalising at figure time: the raw scales are
themselves a result (the control's displacement runs 13.19 in raw layer 5 to 0.0225 in
SpaDE, **585x**), and they are what makes the ÷control form mandatory rather than tidy.

**The rollout twin (n = 1553 as well) removes the ordering and keeps the inflation.**
Full-space ratios pooled over the nine cells go to R1 0.969, R2 1.019, R3 1.016 against
the control's 1.000 — where the spliced tree put R1 8-10% below — and past r = 10 nothing
is more than 0.3% off 1. The plane still inflates in 27/27 cells (median 1.105), and
inside the evidence window r in [0, 8] the pooled inflation on R2 hits **1.29**: a fork
panel there shows R2 at 1.36x the control when the branch is at 1.05x. Drawn to the full
window with r >= 8 shaded as not-evidence rather than truncated, as F1 does.

Its K reduction is `fork_and_board`'s and not F1's: the K displacement **vectors** are
averaged (paired on the sample index) before the plane is fit, because that is the
trajectory the fork panels draw, while F1's `reduce_k` averages the scalar δ̂. Vector
averaging cancels whatever is incoherent across a branch's four samples, so magnitudes
here run smaller; ÷control removes most of it and **the residual is not measured**.

Gotcha for anyone else sweeping spaces: `fork_plane.space_features` calls `sae_encoder`
on every invocation, so it reloads a checkpoint per game — 57 s each here. Build the nine
encoders once. The fix belongs in `src` and was not made from this worktree.

## The fork panels' excursion is mostly the plane (2026-08-07)

Branch `nfig/fig-plane-metric-r21`, worktree `../spade_sae_worktrees/fig-plane-metric-r21`.
Additive: the two R^21 fork studies now answer "how far does each branch run from the
reference", which their 3-D panels had only ever invited the reader to guess at
(FIGURE_GUIDELINES §4.8 asks for the companion; it was missing). `plane_displacement`
in `src/spade/figures/branch_panels.py` is the statistic.

**Every number is a ratio to the panel's own legal control, and that is not
cosmetic.** The plane is refit per space and holds part of Δφ (44% + 18% on raw layer
5), so a raw in-plane distance is in units the nine cells do not share — and nine
panels side by side get compared whatever the disclaimer says. Dividing by the
control's run in the same field cancels the cell's units and leaves a question with a
meaning: does this branch run further from the game as played than another legal move
does? 1.0 is "no" and is drawn as the null line. Absolute magnitude is still not
comparable and nothing claims it is.

**The finding.** In `r21_shared_basis`, full R^21 puts the illegal branches at
0.59–1.49 of the control's run — below 1 in six of nine spaces and **below a third
legal move's own ratio in all nine** — while every 2-D plane raises it, in 25 of 27
plane × space cells, by 0.81–2.70× (median 1.64). In `fork_curvature_space` the plane
inflates the far rule and deflates the near one (plane ÷ full R^21, 0.60–2.12, median
1.31), so the panels' visual *ordering* is real and their visual *spacing* is the fit.
That direction agrees with the parent study's `ratio_illegal_control_L5` = 0.979.

**The two halves of R^21 disagree and the figures draw it.** The 11 step-length
components and the 10 turning-angle ones give different answers about the same branch
(raw layer 5: 0.87 vs 1.04 in `r21_shared_basis`; R2 bracketing 0.98 vs 1.50 in
`fork_curvature_space`), and which one the in-plane scalar lands nearer to is a coin
flip — 14 of 27 rule × space cells on the PCA plane, 17 of 27 on the fixed one. So the
strip carries the split as a bar on the R^21 marker rather than letting one averaged
number pass for a statement about curvature.

**Where the read-out was refused.** Not on the two `whole` grids: one point per branch
over r = 0…24 makes "averaged over r >= 0" what the point already is, the in-plane
distance is a segment the ticks give, and printing it beside the windowed strips would
invite a comparison between φ of one long trajectory and a mean over 13 short windows.
Not on the Isomap grid: it is not a linear image of R^21 and its reference is not at
the origin, so a ratio in it has no denominator worth the name.

**Fixed-plane subtlety.** `fork_r21_windowed_fixed` draws `Δ mean d` against
`Δ mean θ` in named units, which do not add — a Euclidean distance there would be a
statistic about θ alone. The read-out measures that pair after each axis is put in its
own column SD (as `fit_plane` and `measure.py` do, and as the panel's independent x/y
autoscale already shows). The number is the distance in the scaled fixed plane, not
the millimetres between two marks; the README says so.

**Rollout.** Both variants ship. The windowed strips average over exactly the offsets
the panel draws — r = 0…8, five of them under rollout — and the legend title states
which, so a strip can never summarise a window its panel does not show.

Two loose ends. `plane_displacement`'s docstring says `[n, T, 2]`; it is used here on
21 columns too, and the shape line should read `[n, T, k]` (no code change needed,
this is `src/` and was left alone). And the two `space_fork_grid_r21_isomap*` figures
do not reproduce byte-for-byte across runs of an unmodified script — `sklearn`'s
Isomap is nondeterministic here — so those two files are the only ones whose diff
carries no information.

## Karvonen reproduced exactly; the failure was our eval, not our SAEs (2026-08-06)

Branches `karvonen/*`, worktrees `../spade_sae_worktrees/karvonen-*`. **Nothing is
integrated** — no analysis, figure or result in this tree was touched. This entry is the
record so the work is not lost between review and integration.

**The verdict.** Running Karvonen's own `full_pipeline.analyze_sae_groups` on his released
checkpoints reproduces his published table: across 15 checkpoints spanning all five trainer
types, both expansion factors, both the original and `-v2` sweeps, L0 10–571, board
reconstruction F1 and coverage match to **max |diff| 2.98e-07 and 8.64e-07** on mine/yours,
5.96e-07 and 2.68e-07 on blank-mask (median 0; bit-exact on 10–12 of 15), Spearman
ρ = 1.0000 on board F1, median signed L0 deviation +0.0000. `valid_moves` is looser at
1.61e-06 — quote the three board functions separately, not as one bound.

**Why it failed before.** Not the SAEs, not the configs, not the model or data — all of
those were right. `run_eval.py` is a *reimplementation* of his board-state protocol and
diverges four ways: it omits his ≥0.95-precision / ≥10-firings feature gate
(`analysis.py:145,155`), assigns exactly one label per feature where he assigns many,
picks a per-feature threshold where he sweeps a single global one, and macro-averages over
180 cells where he micro-pools over 3.78M square-labels. The gate is the dominant term:
his released precision is 0.948–0.998 on all 240 rows, ours 0.42–0.75. At the one matched
point (TopK k=80, L0 80.0 vs 79.9) his F1 is 0.902 and our old pipeline gave 0.684.
**Decision: stop reimplementing. Run his code.** His `Dictionary` interface is implemented
by his SAEs and ours alike, so both go through one pipeline and exactness is structural.

**A second, independent breakage: the vendored trainer is not the one that trained his
SAEs.** His released SAEs are dated 2024-05-23 at `dictionary_learning@31b2828`/`fbd9ee4`;
his repo pins `43421f5` (2025-01-16), which is what we vendored and ran. Commit `8b95ec9`
(2024-12-26, "Use the correct standard SAE reconstruction loss") changed `standard.py` and
`p_anneal.py` from `‖e‖₂` to `‖e‖₂²` — and left `gdm.py`/`gated_anneal.py` alone, which
were already squared. The blast radius is exactly {Standard, PAnneal}, and that is exactly
which of our baselines missed:

| trainer | recon loss rewritten | his L0 | ours |
|---|---|---|---|
| Standard | yes | 131.7 | 1094.9 |
| PAnneal | yes | 48.0 | 339.5 |
| Gated | no | 94.4 | 49.7 |
| GatedAnneal | no | 47.4 | 32.3 |
| TopK | n/a (k architectural) | 80.0 | 79.9 |

The un-squared form is scale-invariant, so λ *is* the trade-off; the squared form behaves
like λ_eff = λ/(2‖r‖), ≈11× weaker at his operating point once standardization claws back
2×. The Gated pair moved only by the standardization factor 1/scale = 1.98 (observed
1.90×). Also new since his release and silently active: `sparsity_warmup_steps=2000` and a
changed `AutoEncoder` init. At the pin, **his own sweep script cannot run** its `standard`
or `gated` branches (`TypeError`, missing `steps`) — proof the pin is not the release code.

**The config selection was right all along.** Verified against his real config files:
`p_anneal/trainer33` = 0.042105, `standard/trainer30-v2` = 0.058684,
`gated/trainer27-v2` = 1.368421, `gated_anneal/trainer38` = 0.955263, `top_k/trainer_47`
= k 80 — all dict_size 8192. `docs/karvonen_reference.md:556-557` (trainer33 as 8× at
≈0.032) is wrong; `docs/pipeline.md:258` is right.

**His metrics are n=25.** `get_eval_results.get_evals` calls `evaluate` without
`n_batches`, so every published SAE-quality column (`l0`, `frac_variance_explained`,
`cossim`, `l2_ratio`, `frac_alive`) is ONE batch of 25 activations. Every published `l0` is
an exact multiple of 1/25. Consequence: his L0 carries ~6–9% relative noise (except TopK
where L0 = k), and FVE is biased low below L0 ~100. His own two evaluation runs disagree
with each other by median 6.5% on L0 and up to 0.76 on FVE. **Board-state columns are
unaffected** — those use 1000 games × 59 positions. The harness therefore has two modes:
`repro` (n=25, the only mode comparable to his table) and `measurement`.

**We now hold his entire sweep.** 320 trained Othello SAEs downloaded to
`/n/netscratch/ba_lab/Lab/$USER/karvonen_released` (39 GB). Each ships his own
per-checkpoint `results.csv`: 280 match the published aggregate, **40 are TopK configs he
never published**. Coverage is absent from those files and was regenerated for all 320 from
his intermediate pickles via his `analysis.get_all_f1s` + `f1_analysis.best_f1_average`,
matching his published coverage to median 6e-08. A new TopK winner falls out of the
unpublished 40: `trainer_53`, k=140, coverage 0.7704 vs 0.7676 for the best published.

**His published aggregate is a mix of two runs.** On the 40 TopK rows it *is* the
per-checkpoint output (max disagreement 2.8e-14); on the other 240, 53 of 69 columns
differ. Coverage came from the 05-27 pickles, SAE-quality and reconstruction from the 06-10
re-eval.

**His figure is reproduced byte-for-byte** (`karvonen/pareto-figure`): max per-channel diff
0 against his rendered PNGs. Three things had to be invented, all documented — markers and
colours for our 3 extra classes (continuing his `plt.cm.Set1` order, skipping index 5's
unreadable yellow); class matching by equality, because his `str.contains` would plot every
CosineSpaDE point twice given `"SpaDETrainer"` is a substring of `"CosineSpaDETrainer"`;
and skipping empty classes, since his loop raises on an empty scatter. Note **his committed
notebook cannot produce his published figures** — `zip` truncates to a 4-marker list and
`label_lookup` has no `TrainerTopK` entry, so TopK is silently dropped; the published
figures came from an edited working copy not in the repo.

**Canonical SpaDE/CosineSpaDE through his pipeline.** Defaults confirmed against the code
(`spade.py:23-24`, `spade_cosine.py:20`) and `ARCHITECTURE.md:265-278`: SpaDE λ=1e-3/γ=1e-4,
CosineSpaDE γ=1.0. Measured: CosineSpaDE γ=1.0 sits at L0 172–184 across 5 seeds with
coverage 0.523 / recon F1 0.646 (seed 42 has two independently trained checkpoints, which
agree to 0.004 on F1 — average within seed before taking spread); SpaDE λ=1e-3/γ=1e-4 at L0
1841 with coverage 0.541 / recon F1 0.590. In his matched L0 band [140, 220] across 36 of
his SAEs, coverage runs 0.379–0.768 (median 0.736) and recon F1 0.594–0.936 (median 0.839)
— **CosineSpaDE is below his minimum on both in that band**, and drops from the 18.9th
percentile unmatched to the 2.1st under L0 matching, so quoting it unmatched flatters it.
Both clear the random-weight floor and fall short of raw MLP neurons, which beat every SAE
in his sweep on coverage: **layer 5 is 0.8037**, layer 4 is 0.8335 (the higher number is
layer 4 — do not attribute it to layer 5). **The floor differs by metric family and it is
easy to quote the wrong one**: on the unmasked `mine_yours` columns the random-weight model
reaches coverage 0.4316 / recon F1 0.4267; on the **blank-masked** family the figures
actually plot, the floor is coverage **0.2723** / recon F1 **0.0816**.

**β approved for CosineSpaDE — and γ turns out never to have been the sparsity knob.**
The cosine score is bounded in [−1,1], so the score spread is bounded and L0 has an
architectural ceiling: fitted `L0(γ) = 19.38 + 159.04/(1 + (γ/19.44)^1.136)`, RMSE 0.79,
saturating at 178.4. `s_i(x) = β·cos(x, Ae_i)` restores the inverse temperature that
`spade_cosine.py`'s docstring deliberately removed — so β≠1 is arguably a different
architecture, not a wider sweep of the same one, and its checkpoints are named
`CosineSpaDE-beta<b>-gam<g>`, outside the `CosineSpaDE-gam` namespace the γ-sweep
aggregator parses, so nothing can silently merge them. β is **fixed, not trainable** (a
trainable β drifts to its own optimum and stops being a sweepable knob) and defaults to
1.0, proven bit-identical: `beta` is a plain float attribute, `state_dict()` is still
exactly `['Ad','Ae']`, and encode/decode digests on 4096 real activations are byte-identical
before and after on two checkpoints (L0 173.230225, MSE 10.766513824).

An eval-time β probe on the frozen seed-42 γ=1.0 checkpoint (β only rescales scores, so it
can be measured without training) gives β 0.01/0.1/0.25/0.63/**1**/2.5/10/100 →
L0 1758/667/493/271/**170**/53/10.6/2.7. **Karvonen's whole 10–571 axis is covered by
β ∈ [0.17, 10] at a single γ.** The mechanism is exact rather than fitted: sparsemax keeps
atom k iff `Σ_{j≤k}(c_(j) − c_(k)) < 1/β`, so β is a threshold on the trained score gaps.

**The consequence for the existing γ sweep is the finding.** Across all nine trained
γ ∈ [0.05, 5], L0 at β=1 moves only 175.7 → 146.2, while β moves it 60×. γ only bites past
~10 and does so *destructively* — at γ≥500 the model sits at L0 ≈ 21 with MSE 66 against
10.9 at γ=1. **γ sparsifies by wrecking reconstruction; β sparsifies by sharpening
assignment.** Most of the 19-point γ sweep's apparent L0 range was bought with damage, and
any prose describing γ as CosineSpaDE's sparsity dial needs rewriting.

The probe is an indicator, not a prediction: at β<1 the code is smeared over atoms never
fit for it (VE 0.979 → 0.899 at β=0.2), and training re-fits them while the same gradient
widens the cosine gaps, pushing L0 back down. Bounded at ~5× by the high-γ checkpoints, so
**expect realized L0 ∈ [predicted/5, predicted]**; the launched ladder carries headroom past
571 for exactly that.

**λ below ~1e-6 is meaningless, exactly.** `softplus_inverse` in fp32 maps every
`lambda_init` in [5.96e-8, 1.19e-7] to the identical `lambda_pre = −15.94238472`; below
5.96e-8 (= fp32 eps/2) λ = 0 → uniform sparsemax → L0 8192. λ=1e-6 already carries 4.6%
quantization error. Confirmed concretely: `seed_*/autoencoders/trainer_8` and `trainer_9`
are **md5-identical `ae.pt` in all four seeds** despite configs claiming λ=8e-8 vs 1e-7.
Anyone treating them as two λ points double-counts one checkpoint.

**Gotchas found along the way, all still true of this tree:**
- `checkpoints/main/standardization_params.json` is *itself* smoke-test output — its mean
  is a 20–50× outlier against the four real GPU runs (‖Δμ‖ 0.335 vs 0.007–0.020). Every
  checkpoint under `checkpoints/main/` is evaluated with a mis-calibrated standardization.
- `checkpoints/spade_sweep/trainer_{4,6}` sit beside a params file from a later run
  (scale 0.50577 vs their configs' 0.50532); the new loader refuses them.
- The blank-mask coverage column is mechanically capped at 128/192 = 0.6667 because
  `one_hot_mask_idx` is unset for Othello — rankings are safe, absolute values are not
  comparable to unmasked coverage without a ×1.5 rescale.
- `l1_loss` is **identically 1.0** for any simplex code, at every hyperparameter. It lands
  in his CSV and reads as a suspiciously good sparsity number. Drop it from SpaDE tables.
- `frac_recovered` is saturated near 1.000 across his whole sweep and cannot rank anything;
  `frac_alive` is a degenerate accumulator (use `num_alive_features`).
- `seed_42/autoencoders/` does not exist, so distance-SpaDE families are inherently 4-seed
  while CosineSpaDE families are 5-seed. Not a missing path; not fixable without retraining.
- The same-config/different-standardization pair is **`checkpoints/cosine_spade/trainer_0`
  (L0 175.96, buffer scale 0.506043) vs `seed_42/autoencoders_cosine_spade_sweep/trainer_4`
  (L0 175.24, scale 0.505616)** — different weights, same nominal γ=1.0 config. Resolved in
  favour of the seed-sweep run so γ=1 is one internally consistent 5-seed config.
  (`seed_*/autoencoders/trainer_7` vs the sweep's `trainer_4` are byte-identical weights and
  their params files now agree, so they measure the *same* L0 — not a hazard.)
- Karvonen's eval calls `ae.encode()` **inside an nnsight trace**; the `sparsemax` package
  is a `torch.autograd.Function` branching on `input.dim()` and raises on a proxy tensor.
  SpaDE-only — his Linear+ReLU SAEs trace fine. Fixed with plain tensor ops, L0 unchanged.

**Live compute at time of writing:** eval of the 123 existing SpaDE/CosineSpaDE sweep
checkpoints; a 5-seed × 12-config L0-coverage training array (SpaDE λ=1e-3, γ 7e-4…5e-1;
CosineSpaDE γ 35, 75); and a γ×β CosineSpaDE grid being designed off an eval-time β probe
(β only rescales scores, so its L0 response can be measured on trained checkpoints before
spending GPU-hours — an indicator, not a substitute, since training at β changes what the
atoms learn).

**Any new checkpoint destined for the Pareto figure must be evaluated through
`karvonen_repro.evaluate`, never `spade.evaluation.run_eval`.** The L0-coverage sweep was
launched with its eval arrays chained to `run_eval` — our reimplementation, the one that
drifts ~0.2 absolute F1 — and emits `coverage`/`reconstruction_f1` rather than his column
names. Those eval arrays were cancelled; the training array was left running because the
checkpoints are fine. This is the easiest mistake to repeat: the two pipelines look
interchangeable and are not.

**BatchTopK is the one empty class of the eight.** It has a marker and a label but no
points — absent from his CSV (he never swept it) and from ours (`seed_*/autoencoders/
trainer_6` was never mirrored into a `karvonen_repro` run). Either evaluate it or say in
the caption why the legend carries a class with no data.

**What this costs the paper.** The baselines were trained under a loss Karvonen changed
after publishing and evaluated with a protocol missing his precision gate. The
SpaDE-vs-baseline numbers in the current draft must be **regenerated wholesale, not
adjusted**, and the baselines will move more than SpaDE does.

---

## Fig 2 moves onto the Karvonen-fit probes; ARENA becomes the control (2026-08-05)

Branch `bracket/karvonen-probes`. Fig 1 and Fig 2 were reading different instruments.
`probe_space` fits both probe families on the study's own probe set — 400 Karvonen games,
20,000 positions, mover derived by engine replay — and Fig 1 used those. Fig 2 went
through `act_geometry.retain_directions`, which loaded A1's ARENA fits: 200 games, 10,000
positions, labels from `t % 2`, 53 of them wrong (DATA_REPORT). The events come from the
Karvonen stream, so Fig 2 was applying an off-distribution probe. VERDICT §7f called this
out and it is now closed.

**Both families moved together, on purpose.** Fig 2's headline is TPR direction against
linear direction; moving only the TPR would have confounded probe class with training
corpus and handed the TPR the home-field advantage. `retain_directions(source, layer,
corpus)` and `tpr_intervene.probes(corpus)` each take one corpus for both families, so
that confound cannot be reintroduced by a partial edit.

The effect got **stronger**, on the same 1596 strict events: `cos_flip_after_control`
median 0.1748 → 0.1883 (TPR), 0.1628 → 0.2049 (linear); after-control `var_flip`
0.020544 → 0.023925 (TPR), 0.019131 → 0.029711 (linear); still positive in 1596/1596 under
both. The single-disc null did not move (median −0.0025 → −0.0030, sd 0.0550 → 0.0582), so
the separation widened rather than the scale drifting. **The ordering of the two arms
flipped** — the linear direction now beats the TPR's, where it trailed under the ARENA
fits — which is a finding, not noise: on Karvonen held-out rows the Karvonen linear probe
reaches 0.9909 against ARENA's 0.9872 while the TPRs sit at 0.9829 / 0.9813, so the linear
arm gained the most from the better fit. Prose quoting "0.173–0.184" was A5's independent
ARENA-fit re-derivation and is now labelled as such.

`redteam.py` is deliberately left on the ARENA checkpoints: it is the adversarial
re-derivation and its whole value is not sharing a code path with the analysis modules.

**ARENA is kept as the off-corpus control**, not deleted: `retain_directions(...,
corpus="arena")` and `tpr_intervene --corpus arena`, which suffixes its CSVs `_arena` so a
control run cannot overwrite the committed primary ones. `probe_space` grew `path(kind,
layer, basis)` so the intervention manifest can name the exact checkpoints it used instead
of restating the cache-key format.

Two things still ARENA-derived, both flagged in the docs: the M6 intervention CSVs
(`+1.496` etc.), pending the SLURM rerun, and therefore `fig2_direction_and_cause`, which
is left un-regenerated — half-Karvonen, half-ARENA would be worse than consistently stale.
Redraw it after the sweep lands.

**Unrelated blocker fixed in the same commit:** `submit_bracket_intervene.sbatch` ended
with two calls to `analysis/bracket_flip/intervention_figures.py`, a file that was never
ported. Under `set -e` the job failed after writing its CSVs. `make_figures.py` owns Fig 2
now and runs off the GPU, so the sbatch is CSVs only.

## The corpus is missing an opening, and `stream_games` reads a window (2026-08-05)

Branch `bracket/corpus-provenance`. Nothing in the repo said what
`adamkarvonen/othello_45MB_games` is, and three things about it condition every study
that streams it. Written up in `docs/game_corpus.md`, pointed at from
`spade/othello/games.py` and the bracket-flip design doc. No code behaviour changed —
touching the sampler would invalidate every committed result.

**It is synthetic random-legal play.** Every one of the 990,623 games is exactly 60
plies with no pass token, and the played move is uniform over `regular_moves` at
essentially every prefix (at 5 plies, 0 of 1,185 prefixes with n ≥ 400 reject at
p < 1e-3). No opening book: 791,514 of 792,498 train games have a distinct 12-ply
prefix. Worth stating because `docs/restrav_summary.md` sets "synthetic" against
"real human championship play" for the ResTrav arm — this is the synthetic side, for
everything.

**The `e6` opening holds 10.8% of games, not 25%, and it is a truncation rather than a
bias.** The four first moves are related by the symmetry group that fixes the starting
position, so random legal play owes them 25% each; observed 0.2968 / 0.2973 / 0.2982 /
0.1078 (χ² = 85,412, 3 df). The deficit is one cut in lexicographic order of the token
sequence: down the `41 40 19 28 42 …` branch each prefix has full-strength children,
then one partial child, then zeros, and walking the largest surviving child to the
bottom lands on exactly the lexicographically largest game in the corpus. The
pre-truncation size implied by the full children, ~1.18M, is consistent at every ply.
A sorted list cut to the 45 MB the dataset is named for.

**`stream_games` is a window on the corpus, not a sample of it.** The split is one
shard, so `shuffle(seed, buffer_size=10000)` cannot reorder shards and only permutes a
10,000-row reservoir: drawing n games reads the first n + 10,000 rows and no further.
Three seeds' 2,400-game draws overlap pairwise at 0.197–0.205 — 2,400/12,400 = 0.194,
the value for three draws from one window — and a 200,000-row buffer drops that to
0.013. Every study defaults to `--seed 42`, so `rules`, `straightness`,
`manifold_capture`, `legality_probe`, `violation_divergence` and `bracket_flip` walk
one list in one order; their game sets are nested prefixes, not independent draws, and
`bracket_flip`'s probe and event sets are rows 1–400 and 401–2,400 of it — a fresh
seed-42 draw reproduces all 400 rows of `probe/tokens.pt` and all 2,000 `events.csv`
prefixes, in order. A different seed is not a replication: it swaps ~80% of the games
but stays in the same 1.6% of the corpus. `IterableDataset.skip` is not the way out
either; on `datasets` 4.8.4 it lands at row 0 for `skip(1000)` and row 1,000 for both
`skip(5000)` and `skip(300000)`. `extraction.extract_activations.collect_games` has its
own copy of the same idiom, so this is not only `stream_games`.

**Which explains which.** The sampling defect is real but it does not cause the opening
skew — the `e6` share is 0.10–0.12 in every draw tried, including one taken from row
300,000 with no shuffle, and 0.1078 over the whole corpus. The skew is the corpus; the
window is a separate caveat, and the one that reaches beyond this study.

---

## The R^21 figures get `--variant rollout`, and the suffix confound turns out to be unreachable (2026-08-05)

Branch `rfig/figs-r21-basis`. `r21_shared_basis` and `curvature_discriminant` now take
`--variant {spliced,rollout}` (default spliced, rollout figures get a `_rollout` suffix,
intermediates go to `$DIR/rollout/...`). The question was whether
`curvature_discriminant`'s headline — illegal-vs-control AUROC 0.859 in GatedAnneal
against raw layer 5's 0.634 — was reading the spliced tree's broken suffix rather than
the move at *t0*.

**It was not, and the rollout is not what shows that.** The descriptor at *r* = 0 is a
**trailing** 8-ply window ending at *t0*, so it sees `tokens[:t0+1]` and nothing after,
and `rollout_branches` copies exactly those tokens. The r = 0 column is therefore the
*same measurement* in the two trees, not a replication of it — and it comes out equal to
three decimals in all nine spaces (0.634 / 0.803 / 0.635 / 0.859 / 0.824 / 0.631 / 0.609
/ 0.553 / 0.812), along with the random-direction 95th percentile, ‖Δφ‖ alone, the flip
ρ and the permutation null. `evaluate.variant_agreement` returns the residual instead of
asserting it away: 1e-3 (raw L5) to 1e-1 (GatedSAE), which is two separate extraction
passes' fp16 noise amplified by a latent sitting on its threshold, not the suffix.

**What the rollout could test, it also clears.** The *r* = 2 inversion — the four sparse
dictionaries at 0.32–0.37, illegal branches landing *below* the legal controls — is the
shape a broken replayed suffix would produce. It survives resampling at 0.35–0.41, moved
0.02–0.04 toward chance and no further; by *r* ≥ 4 both variants are at chance
everywhere. The inversion belongs to the axis, not to the inherited future.

**Three decisions worth knowing.**

- *The K = 4 mean is taken after φ and before the difference from the reference.* φ is
  nonlinear, so averaging the four trajectories first reports the curvature of a path no
  rollout took; the axis projection and the plane are linear and agree on either side of
  the mean, so this is the only place the choice exists, and `evaluate.fold_rollouts` is
  the one place it is made. `axis.fill` imputation happens per rollout, before the mean.
- *The straightness mapping is read from the SPLICED branches.pt even on a rollout run.*
  A rollout reference has resampled its own suffix, so its tokens are no longer in the
  straightness token table; the lookup would return -1 for every game and silently empty
  `held_out_legal`, which is the exclusion list the whole honesty guard rests on. game_id
  carries over, so the mapping is looked up in the spliced tensor and applied by id.
- *Do not read `r21_shared_basis`'s rollout grids as a result.* W = 8 at stride 2 is 14
  plies of span, so the first anchor it can draw is *r* = +6 — deep inside the regime the
  rollout divergence read-out found uninformative (δ̂ saturating near 1.23 against a ~1.41
  ceiling, illegal/control pinned to 1.00 by *r* ≥ 8, only *r* = 0 separating at 1.032,
  sign opposite the spliced 0.979). The grids are a shape check on the reduction. The
  claim in `separation_by_plane` does invert there — the activation plane goes from
  0.45–1.05× its own null to 2.5–4.7×, while R^21 stays flat — because with every branch
  resampling, the divergence *is* in the activations and a step-length/turning-angle
  summary discards it. Both figures carry the caveat in-panel, and the suptitle is
  variant-dependent rather than asserting the spliced claim over rollout data.

**A stale-figure trap found on the way.** The committed spliced PNGs under
`r21_shared_basis/figures/` do not match what the script renders, and did not before this
branch: commit 94a7dc2 moved the study under `analysis/violation_divergence/` and updated
the path string inside the figure's own caption without re-rendering. The four grids
therefore differ from a fresh render by that one line of text. This branch leaves them at
their committed bytes so the change stays additive; a re-render is a one-line commit for
whoever wants it. `space_fork_grid_discriminant.png` needed no such care — it re-renders
**byte-identical**, which is the proof that the variant plumbing is a no-op on the default
path. `separation_by_plane`, `..._named` and `..._refit` also re-render byte-identically
run to run; `..._isomap` does not, by 1 grey level on 1e-5 of its pixels, because
`Isomap(n_jobs=-1)`'s eigensolver is not bit-reproducible. That is pre-existing and worth
knowing before anyone byte-compares that one file.

**Not done:** `branch_geometry.py --variant rollout` is 20700 rows through nine spaces and
was killed twice on the login node around 12 minutes in. It was run under SLURM instead
(`kempner_h100`, 8 CPU, 96 GB, ~20 min). There is no committed sbatch for it — the study's
scripts live in `scripts/slurm/studies/violation_divergence/` and adding one there is the
study owner's call, not this branch's.

---

---

## The fork figures learn a second branch set, and it saturates (2026-08-05)

Branch `rfig/figs-fork-board`. `fork_and_board` and `game_trajectory` grew
`--variant {spliced,rollout}`, defaulting to spliced, and now also render out of
`$SPADE_DIVERGENCE_DIR/rollout/` (17,527 branches × K=4, 2000 games, temp 1.0, seed
42), where every branch samples its own continuation instead of replaying the
reference's. Rollout outputs carry a `_rollout` suffix; nothing existing was
overwritten. Purely additive, and the five spliced outputs re-render byte-identical
before and after.

**The K reduction is explicit and its position matters.** Rows are K-contiguous per
branch and a rollout's reference is the one drawn from the same uniform stream, so
`paired_mean` differences within a sample and averages after — common random numbers.
For R²¹ the average is taken **after φ**: φ is nonlinear, so the descriptor of an
averaged trajectory is the curvature of a path no rollout takes. The plane projection
is linear, so either side of it agrees, and the rule mean over member categories stays
where it was. In `game_trajectory` the two panels reduce *differently* on purpose: (a)
averages Δ, (b) averages δ̂ after the norm, because ‖mean Δ‖ cancels sampling noise
and is a smaller, different quantity.

**Rollout δ̂ saturates, so the drawn window stops at r = +8.** Full set, layer 5,
per-branch mean over K: δ̂ climbs 0.79 → 1.236 by r = 20 (96% of the 1.293
two-unrelated-games level RED_TEAM.md measures in this space) where spliced δ̂ decays
0.79 → 0.46. The illegal ÷ control ratio holds ~0.93 across the spliced window; under
rollout it is 0.970 at r = 1, 0.9957 [0.9926, 0.9990] at r = 8, and crosses *above* 1
by r = 13 (95% bootstrap over games, 2000 resamples). `ROLLOUT_RMAX = 8` is that cut;
both READMEs carry the table and both figures state the cut in-panel.

**r = 0 is bit-identical between the variants** — 0.7915 illegal, 0.7672 control,
ratio 1.0316 in both — because the activation at t0 depends only on `tokens[:t0+1]`.
Worth recording, because the task that commissioned this read r = 0 as the one offset
where rollout still separates, "opposite in sign to the spliced study's 0.979": 0.979
is the spliced study's *plateau* ratio, not its r = 0 value, and the rollout adds
nothing at r = 0 by construction. It is also the cheapest check that the rollout
activations are right, and it passes.

**The prefix still collapses.** Over game 1235's 36 rollout rows, raw layer 5, max
|z − z_row0| = 0 at every position before t0; `project()`'s `max |Δφ| = 0` assertion
over the branch-free offsets passes in all nine spaces on every R²¹ render.

**PDF byte-identity needs `SOURCE_DATE_EPOCH`.** Two renders of one figure differ in
exactly 3 bytes — the seconds of `/CreationDate`. With it pinned, all five spliced
outputs (2 planes × PNG+PDF, plus `game_1235_fork_pca.png`) are byte-identical across
the change. Separately: **the two committed spliced PDFs are stale.** Commit a12fa37
grew `ROW_GAP` and re-rendered only the PNGs, so a fresh PDF is 1094.4 pt tall against
the committed 1047.0. Left untouched here — this branch is additive — but they need
one re-render on their own commit.

## Fig 1 titles each accuracy against its own occupancy ceiling (2026-08-05)

Branch `bracket/basis-ceiling`. Panel (c) said "held-out board accuracy 0.735" and stopped
there, which reads as partial colour ability. It is the opposite: 0.7349 is 0.004 points
*below* the 0.73494 the labels give for free — perfect empty/occupied, then always the
majority colour — while panel (b)'s 0.9829 clears its own 0.75181 by 23.1 points. Both
titles now carry accuracy, ceiling and the gap, and the shared legend defines "ceiling"
once for both. Neither number is the claim; the pair is.

The rule lived only inside `robustness.occupancy_split`. It is now
`probe_space.occupancy_ceiling(labels)` — `probe_space` owns the labels and the accuracy
table, and `robustness` already imports half its namespace from there — and `m1_basis.csv`
carries `ceiling_relative` / `ceiling_absolute`. They are a property of the labels, so they
are constant down the layer column, and they are written to five decimals where the
accuracies get four: the absolute basis's whole gap to its ceiling lives in the fifth.

**`robustness.py`'s four CSVs do not byte-reproduce on this machine, and it is not the
refactor.** Running HEAD's code here produces the same four files my version does, bit for
bit; both differ from the committed ones by one float32 ulp on the mean reductions
(5.96e-08) plus exactly one flipped argmax out of the 768,000 eval squares in the absolute
decode (Δacc 1.3e-6 = 1/768,000). CPU here against whatever the committed run used. The
committed CSVs are left as they are. `probe_space`'s side reproduces exactly: `m1_basis`'s
six pre-existing columns came back identical, its other seven CSVs byte-identical.

## The bracket flip's geometry tables get a writer, and a null that is not a median (2026-08-05)

Branch `bracket/regen-driver`. Four CSVs under `analysis/bracket_flip/results/` had no
writer: `act_geometry.py` is a library, and the A4 driver that called it was never ported
(it survives only as a copy under the frozen repo's `.claude/worktrees/`). Its
table-writing half is now `spade.bracket_flip.geometry_tables`; the plotting half stays
dropped, since `make_figures.py` owns the two surviving figures. `probe_space.py` gained the
`f2_event_meta.csv` write beside its `f2_event.csv`, same canonical event, with `move` and
`control_move` converted from vocab tokens to squares.

**The null Fig 2(a) draws was a median of ~29 draws.** `cos_null_*` is a per-event
`nanmedian` over that event's unflipped discs while `cos_flip_*` is a single cosine, so the
panel set one draw against an average of thirty and the null came out several times tighter
than an unflipped disc really is. `act_geometry.null_direction()` picks exactly one
unflipped disc per event — it exists for this reason and the table never called it. It now
feeds four `cos_null1_*` columns. Same centre (median −0.0025 vs −0.0024, TPR,
after-control), 6.5× the spread (sd 0.0550 vs 0.0085, IQR 0.0426 vs 0.0105). The finding
survives: the flip median +0.175 is 3.2 single-disc-null sds out and no null event reaches
it. But 3.2 is the honest number, not 21, and 0.5% of single-disc nulls clear the flip's
2.5th percentile where none of the medians did. The median columns stay — where the null
*centres* is a real question and they answer it — so both live in the file and the module
docstring says which is which.

Regeneration reproduced everything committed: `variance_shares.csv`, `summary.csv`,
`sanity.csv` and all seven of `probe_space`'s CSVs came back byte-identical,
`f2_event_meta.csv` reproduced its committed row (game 1882, *t*=25), and
`flip_cos_per_event.csv` differs only by the four new columns.

The seed is 42, this table's `SEED` and the source driver's. That is not free choice: the
same draw feeds `var_flip_null` in `variance_shares.csv`, so the two files' nulls name the
same discs, and a different seed would have broken that file's byte-identical reproduction.
The Fig 2(a) agent's independent implementation at seed 0 gives median −0.0025, sd 0.0525
against this column's −0.0025 / 0.0550; running their seed here reproduces their two numbers
exactly, and sd across seeds 0/1/2/7/42/1111 spans 0.0497–0.0550. Same statistic, different
draw — not a disagreement.

**Repaired since:** `tpr_probes/tpr_r52_f2_layer{0..7}.pt` on netscratch were saved before
`basis` entered the cache key, so `tpr_probe.load(5)` raised `FileNotFoundError` looking for
`tpr_r52_f2_relative_layer5.pt`. The loader was right — the infix is what stops a relative
and an absolute fit from overwriting each other — and the file names were stale. But two
conventions were live at once: `redteam.tpr_checkpoint` and `tpr_intervene`'s manifest line
each spelled out `tpr_r52_f2_layer{L}.pt` with the dims hardcoded, so renaming alone would
have broken them. `_path` is now public as `tpr_probe.path(layer, basis="relative")` and is
the only place the rule lives; the eight files are renamed to match. Refitting layer 5
reproduces the on-disk weights bit-for-bit, so the rename is the cheap half of a lossless
repair.

Unrelated but in the same files: `paths.linear_probe_activations()` now owns the
`$SPADE_LINEAR_PROBE_ACTIVATIONS`-else-repo-relative rule that `tpr_probe`, `act_geometry`
and `tpr_intervene` each spelled out separately.

## Fig 2(a)'s null gets the signal's estimator, and the panel states the share (2026-08-05)

Branch `bracket/fig2-specificity`. The specificity density compared **one cosine per
event** against a **per-event median over the ~29 unflipped discs**. A median of 29 draws
carries ~1/√29 of a single draw's spread, so the tight spike at 0 beside the broad hump at
+0.175 was partly an estimator artifact. The null is now one randomly chosen unflipped
disc per event (`cos_null1_after_control_tpr`): median −0.003 against the old −0.002, but
**sd 0.053 where the median-null had 0.009** — six times wider, and now marginally wider
than the signal's own 0.045.

The claim survives at close to full strength: all 1,596 strict events are positive, no
null event reaches the signal's median, and the two densities separate at AUROC 0.999. The
honest null also has a left shoulder near −0.19 that the median averaged away — on some
events a random unflipped disc's retain direction is genuinely anti-aligned with Δ′.

**The panel now shows magnitude, not only direction.** A cosine density alone reads as
"this is *the* direction the representation moves". An inset carries the after-control
decomposition already in `variance_shares.csv`: the flip direction holds **2.05%
[2.00, 2.11] of ‖Δ′‖² against the null direction's 0.42% [0.38, 0.47]** (n = 1,596,
percentile bootstrap over events), with swap 8.5% and residual 89.5% named in the inset
title. Reliably present at ~5× the null, and ~2% of the displacement — both halves of that
sentence are now on the figure.

Only the TPR-direction null is drawn. The linear-direction single-disc null is the same
distribution (median −0.001, sd 0.047), and a second gray dashed curve on top of the first
could only be told from it by dash pattern.

Also fixed: n comes from the frame instead of the literal "1,596" in the y-label, and the
legend title states the band in full (95% percentile bootstrap over events, 1,000
resamples, Gaussian KDE at the fixed full-sample Scott bandwidth).

The panel was written against `cos_null1_after_control_tpr` before that column existed and
verified on a scratch recomputation at rng seed 0; the column landed with
`bracket/regen-driver` at seed 42, so the drawn null is the seed-42 draw. The two agents'
implementations agree — see the entry above for why the seed is not free.

## Fig 1's absolute panel becomes a second probe, and its error bars move onto the arrows (2026-08-05)

Branch `bracket/fig1-field`. Four defects in `bracket_flip` Fig 1, one of them a real bug.

**Wrong event set (the bug).** `probe_space.field()` applied no `control_kind` filter, so
the dashed control arrow — the panel's only contrastive element — was averaged over all
2,000 events including the 404 whose control move touches *s*'s rays. Fig 2(a) and
`variance_shares.csv` were already on the 1,596 strict ones. Restricted, and **all three
conditions restricted, not only `control`**: `before` and `after` do not depend on *m′*,
but the claim is the distance between the control head and the after head, and reading two
heads off two populations is not a contrast. The correction is small — the flipped square's
control *x* moves 7.069 → 7.125 and the after-vs-control gap 14.557 → 14.592 logits — which
is expected, since a loose control still leaves *s* unflipped. The bug was in what the
number was computed over, not in what it says.

**The error bars were on the wrong quantity, and the fix goes the opposite way from the
guess.** Bars sat on the `after` endpoint; an arrow asserts a displacement, which is paired
within event. `m2_field.csv` now carries the paired contrasts and nothing else — after −
before, control − before, and after − control, each with its own interval, all three drawn.
The paired interval is *not* uniformly tighter than the endpoint one: on the occupancy
axis it is 1.2–2.5× narrower as expected, but on the owner axis it is **1.5–2× wider**,
because `before` and `after` are strongly anticorrelated across events — the same event's
discs read confidently `mine` before and confidently `theirs` after, so the shared
event-level confidence enters the two endpoints with opposite signs and adds instead of
cancelling. The old bars understated the arrow on the axis the figure is about. Written
into FIGURE_GUIDELINES §4.7 so the next displacement field does not repeat it.

**Panel (c) is now a second measurement.** It was `coords(..., absolute=True)`, which
re-signed *x* by whose turn it is — an exact deterministic relabel of panel (b), drawn
co-equal beside it and read as independent evidence. An absolute-basis TPR has existed all
along at `$SPADE_BRACKET_DIR/probe_space/tpr_absolute_layer5.pt` (this is a different
artifact family from `tpr_probes/`, which holds A1's committed-set relative fits), so (c)
now decodes it, oriented by the event's mover so black and white do not cancel. It puts
every role within 0.22 logits of *x* = 0 — **including empty squares, which have no colour
to read**, so what little *x* there is tracks whose turn it is, not the disc — and the flip
contrast collapses from 14.59 to 0.04 logits. The panels share axes, or (c)'s 0.2-logit
spread blows up to panel width and reads as separation.

**Framing.** Fig 1 is not demoted wholesale. Panel (b) is: its title and the README now say
it is the frame the study works in, since 63 discs crossing and *s* not crossing is what
any correct board tracker does and is exactly the claim that was retracted. But (c) is a
test the model can fail and does, so Fig 1 keeps a headline and it is finding 1's — the
board code is mover-relative — which previously had no figure. A board oracle scores 1.0 on
(b) and would nail (c).

Also: the `c{xx,xy,yy}` covariance columns were computed for a 50% dispersion ellipse no
version of `field_panel` ever drew, and at this panel's scale five of them would each cover
40% of the panel; deleted. `displacements()` and `worked_example()` lose their
`absolute_*` columns for the same reason — they were the relabel, and nothing read them.

---

---

## Branch suffixes get resampled from the model, in a tensor beside branches.pt (2026-08-05)

Branch `rollout/generate`, new `violation_divergence/rollout_branches.py`. `branches.py`
splices: every branch keeps the source game's `tokens[t0+1:]`, which is a legal
continuation of the *reference's* board and of no other. It survives for the legal
control in 76.8% of games (median 2 moves) and, after an illegal branch, is legal on no
board at all. So the reference was the only branch with an in-distribution future and
every displacement in the study was measured against it — the README's "What does not
hold" named this and nothing fixed it.

The new module keeps `tokens[:t0]` and the branch token at t0 and samples the rest from
OthelloGPT at temperature 1.0, K=4 rollouts per branch, only the pass token masked. **The
reference is resampled too.** Replaying its real continuation while the others roll out
would leave it the one in-distribution sequence and would have fixed nothing; `verify()`
checks that no reference rollout reproduces the real game.

**The trade, which is not small: single-token causal attribution is gone.** Two branches
now differ at t0 and at every position after it, so a displacement is the branch move plus
everything the model then chose to do about it. K=4 separates sampling variance from the
branch effect; it does not separate the move from its consequences. `branches.pt` stays
the tensor for the single-token claim and this is an addition to it, not a replacement —
hence a parallel tree under `$DIR/rollout/`, with nothing under `branches/` or `acts/`
touched.

Two decisions worth knowing. **Seeding is per (game_id, k), not per row**: all branches of
a game share the k-th uniform stream, so the K rollouts of two branches are paired common
random numbers, and a row's continuation does not depend on batch composition or on which
branches the game happened to yield. Sampling is inverse-CDF against those uniforms rather
than `multinomial`, which is what makes that possible. **No KV cache**: transformer_lens
1.17 builds the cache's attention mask from `self.tokenizer`, and OthelloGPT has none
(`AttributeError: 'NoneType' object has no attribute 'pad_token_id'`). Faking a tokenizer
to satisfy a mask computation is not worth it for a model this size, so each step re-runs
the whole prefix — ~40x a cached step's compute, still minutes on one H100.

---

## A `--variant` tree for the divergence pipeline, and the diagnostic that justifies it (2026-08-05)

Branch `rollout/pipeline`. The spliced branch set gives every branch the *source
game's* continuation, which is a legal game on the reference's board and nowhere
else. `suffix_legality/measure.py` measures the damage from tokens alone: a legal
control's continuation survives a median of **2** moves, and an illegal branch has no
board after *t0* at all, so its suffix legality is **undefined** rather than low.
Only the reference is ever replayed inside its own game. That asymmetry, not the
control's handicap alone, is why a rollout variant is being built.

**Two decisions in the diagnostic worth knowing.**

- *No convention is invented for the illegal branches on the spliced set.* Forcing a
  number there would hide the thing being reported. What is reported instead is the
  engine's verdict on the branch move, and the one convention-free statement
  available — the continuation later plays the branch's own square (94–99% of
  empty-square rows), which is illegal under every convention because occupancy never
  shrinks. Caveat recorded in that README: a 59-move game covers nearly the whole
  board, so the number is high partly for structural reasons.
- *`opponent_only` is the only illegal category the engine will accept*, via
  `umpire`'s forfeit fallback, and it accepts it by letting the **opponent** play. An
  engine that returns a board is not the same as a branch that has one. Any future
  code tempted to replay illegal branches must not read that fallback as the branch's
  own board.

**The 76.8% reproduces, on a different branch set.** The README's figure was measured
by the red team on the one-control set (15531 rows); the set on disk is the
two-control re-run (17527 rows). This gives **0.7515 [0.7325, 0.7710]** (1000-draw
bootstrap over the 2000 games) and median **2** exactly. The interval contains 0.768
and the set's two independent controls agree to 0.2 pp, so the difference is the
branch set and not the engine handling. Settling it exactly needs the one-control set
regenerated; HF streaming does not work offline here, so it is left open. The script
**stops** rather than printing a table if it ever misses the check.

**The `--variant` flag composes with `--smoke`.** `variant_dir(study, smoke, variant)`
in `common/study.py` is the one place the rule lives; `extract_acts` and
`compute_divergence` take `--variant rollout` and read/write `$DIR[/smoke]/rollout/`.
Both defaults are unchanged and this was checked rather than asserted: the 50-game
smoke pipeline was run end to end before and after the change into two scratch trees,
and all 10 act tensors plus all 17 divergence npz files are **bit-identical** (sha256).

Two things had to change for the rollout set to work at all, both no-ops on the
spliced set. `check_prefix_invariance` asserted no branch matches the reference at
*t0*; with K samples per branch that is false by construction, so it now asserts the
stronger and variant-independent rule that activations at *t0* agree **iff** the
branch tokens do. `pairs()` keyed on `game_id`, which finds K references per game;
it now keys on `(game_id, rollout)` when the column is present.

**Not done, deliberately:** the downstream read-outs (`probe_readout/decode.py`,
`behavioral/make_figures.py`) still hardcode the spliced tree. Plumbing `--variant`
into them means deciding how their `results/` and `figures/` names separate the
variants, which is a naming call for the study owner, not a mechanical edit.

## The real-only curve in Fig 2(b) was labelled a control; it is the effect (2026-08-05)

Branch `bracket/fig2-causal`, on top of the entry below. That entry drew `d_lp_real`
and called it "the control a merely damaging edit fails", so the orange dotted curve
diving to −4.6 nats read as degradation. It is the opposite. `real_only` is the
squares legal on the **real** board and illegal on the counterfactual one, so a
successful un-flip edit has to push it **down** while it pushes `cf_only` up. Both
curves are one switch read on its two sides, and the panel now says that: the y-axis
carries `↑ onto the un-flipped board's squares, ↓ off the real board's`, and the
legend says "one switch, two sides".

**The damage control was the corner annotation.** `d_lp_legal` — log-mass on the
whole legal set, which is what a broken edit moves — is now a gray band through zero
at its worst-case CI bound over both sweeps (±0.08 nats), with a legend entry. A band
rather than a series or a twin axis: on a y-axis spanning −4.6 to +8.1 the control is
a hairline by construction, and *that* is the reading. It replaces the dashed zero
line, so the zero reference and the control are one mark.

**The low-scale asymmetry is a base-mass artifact, not a property of the edit.** At
tpr/scale 1 `d_lp_cf` is +0.278 [0.267, 0.288] while `d_lp_real` is −0.0005 [−0.0015,
+0.0003] — mass apparently added to the counterfactual squares without being taken
from the real-only ones. It is not:

- **Not the event sets.** 507 of the 1,370 flipped rows have both sets non-empty
  (986 have `cf_only`, 891 have `real_only`, none have neither). Restricted to those
  507 the numbers barely move: `d_lp_cf` +0.278 → +0.294, `d_lp_real` −0.0005 →
  +0.0003. The differing n is bookkeeping, not the effect.
- **It is base mass, by four orders of magnitude.** Median base mass is 8·10⁻⁶ on
  `cf_only` against 0.115 on `real_only` — the sets start 9.7 nats apart. The +0.278
  nats on `cf_only` is +1·10⁻⁵ of *probability*. Had every bit of it come out of
  `real_only`, `real_only` would read −0.00008 nats, an order of magnitude inside its
  own CI. The measurement cannot resolve the exchange at that scale; the flatness is
  not evidence either way, and the panel now carries that sentence in-figure so
  nobody reads the flat left end as specificity.
- **Above ‖δ‖/‖h‖ ≈ 0.07 the exchange is real and visible.** At scale 16 `real_only`
  loses 0.124 of probability and `cf_only` gains 0.096, with the whole legal set down
  only 0.075 nats: mass leaves the real board's exclusive squares, and most of it
  lands on squares the un-flipped board allows. At intermediate scales `real_only`
  loses far more than `cf_only` gains (86× at scale 4) while `d_lp_legal` sits at
  10⁻⁴ nats —
  the difference goes to squares legal on *both* boards, which is also the
  counterfactual board's legal set. Still the same switch.

The honest summary of the operating point (`‖δ‖/‖h‖` = 0.03): the edit produces a
large, tight log-scale effect on a set holding 10⁻⁵ of the mass, and nothing else in
the distribution moves enough to measure. Reading Fig 2(b)'s y-axis as a probability
budget is the mistake it is now annotated against.

Drawing only; no re-run. `causal_panel` now also reads `intervention_per_event.csv`,
for the base-mass gap and the transfer floor.

---

## Fig 2(b) stops grading each edit with its own decoder, and shows the controls (2026-08-05)

Branch `bracket/fig2-causal`. The causal panel had three problems, all of which made
the intervention look better or simpler than the run actually was.

**Collateral was scored by `collateral_tpr` for all three series**, so the TPR edit was
graded by the TPR decoder it is built from while the linear edit was graded by a
decoder it has no stake in. Each family is now scored **out of family** —
`collateral_lin` for the TPR edit, `collateral_tpr` for the probe edit. The verdict
does not change: at every magnitude the TPR curve is above and left of the probe
curve, and out-of-family scoring actually moves the TPR points slightly *left* at low
scale (0.044 vs 0.058 changed squares at scale 1). Mixing two instruments on one axis
is defensible here because the two readings of the same edit stay within ~35% of each
other; the axis label says which instrument each series used.

**The legend claimed one knob for two families.** The TPR edit is additive
(`scale · d`, 8 scales 1–16) and the probe edit projects to the class mean (`β`, 2
betas) — `tpr_intervene.edit_vectors` — so "one point per edit scale (1–16)" was false
for the probe arm and implied a shared x. Resolution: nothing is placed by knob value.
x is measured collateral, and each point carries `‖δ‖/‖h‖` (`rel_edit_norm_mean`) as a
label, which is the one magnitude both families share. The probe's betas sit at 0.11
and 0.23, inside the TPR sweep's 0.03–0.53, so the curves overlap in magnitude rather
than merely in ink. x is log-scaled because collateral spans 0.04–2.3 and the
interesting operating point is at the bottom of that range.

**The strongest controls were computed and not drawn.** `d_lp_real` and `d_lp_legal`
were in the CSV and invisible. `d_lp_real` is now a dotted, open-marker companion in
each family's own colour (hue is spent on the family, line style carries which legal
set is read — the `BRANCH_ROLE_STYLE` convention), and `d_lp_legal` is an in-panel
line stating its worst-case bound over the whole sweep. That is what answers "the edit
just damaged the model": at TPR scale 1 the counterfactual set gains +0.278 nats while
the real board's legal-only squares move −0.0005 [−0.0015, +0.0003] and whole-set
legality does not move. Specificity is a *low-collateral* property, and the panel now
shows where it ends — past `‖δ‖/‖h‖ ≈ 0.2` the control collapses.

**Three n's, not one.** `n_events` is 1,370, but a set-log-mass is NaN when its token
set is empty (`unflip_edit.set_logmass`), so the counterfactual set has n = 986 and the
real-only set n = 891. Both are constant across every scale — the sets are fixed by the
event, not by the edit — and `causal_panel` asserts that, since it prints one n per
quantity. Collateral and legality keep n = 1,370.

Not re-run: this is drawing only, off the committed `intervention_summary.csv`.
`tpr_matched` (the TPR direction at the probe edit's per-event length) is still in the
CSV and still not drawn; the per-point magnitude labels carry the same information
without a fourth curve.
## The legality read-out runs end to end, and the level is not where the signal is (2026-08-05)

Branch `legality/readout`. `src/spade/legality_probe/readout.py` fits four probe arms per
space and reads the legality arm out over the branch window; 11 spaces, 17527 branches,
4m47s on one H100. Writes `probe_fit.csv` (484 rows) and `readout.csv` (50193 rows).

**The board arm reproduces `bracket_flip/sae_arm.py` to ±0.0026.** Same architecture, same
target, a different fit corpus, 8 shared spaces: every delta is between +0.0005 and +0.0026,
all positive. That is the strongest evidence the label scatter, the frame, the encode and the
fit loop are all right. Only 7 SAEs overlap plus raw — `sae_arm.csv` carries the degenerate
`SpaDETrainer-othello` and has no row at all for the default SpaDE, which is `sae_arm.py:45`'s
roster bug showing up as missing data.

**The four arms on raw**: legality 0.9379, board 0.9820, wrong player 0.9072, shuffled 0.2856
(accuracy 0.5554). Every space's shuffled arm lands in 0.28-0.31, so the floor is a property
of the class balance, not of the space. The wrong-player arm tracks the real one within 0.03
in every space, which is why no claim here rests on the level.

**Where the signal actually is: `p_pred_correct_rule` at r = -1, on the R3 panel.** At r = 0
the illegal move has been played and every space reads that square as occupied, so R3 recall
is ~0 everywhere and the panel is degenerate. At r = -1 — the shared pre-move board, so this
is a read-out of the space and not of the branch — raw gets R3 right 0.814 of the time, the
seven ordinary SAEs 0.71-0.83, `random_topk` 0.473, and the two SpaDE spaces **0.048 and
0.006**. R1 and R2 barely separate the spaces (0.87-0.98, 0.79-0.94). Turn is the rule the
SpaDE bottleneck destroys, and it is the rule with one taxonomy member.

**`excess_macro_f1` is not scale-free, and the two SpaDE spaces make that visible.** It is
`drop(space) - drop(raw)`, drop being the r<0 to r=0 fall in macro-F1, and CosineSpaDE and
SpaDE-lam come out *negative* (-0.061, -0.079) only because their r<0 baseline is already
0.49 and 0.42 — there is less left to lose. `drop_agreement_with_raw` is the version to
trust: no ground truth, no floor, and it puts GatedAnneal (0.079) above everything else.

**Masking.** 85 of 66000 window positions are forfeits in the reference game and are dropped
from the F1 metrics; the agreement and played-square metrics need no truth and keep all of
them. Building the labels from an incrementally replayed board rather than through
`labels_at_ply` matters: that helper's `len(game) < t + 1` guard would have masked r = +24
for all 80 games with t0 = 34, since the label needs no next move but the guard demands one.

**Prefix invariance is now checked through the probe**, span by span, in all 11 spaces: for
r < 0 every branch of a game decodes to the identical 64 labels. Strictly stronger than the
encoder-level check, and it is the assumption the r<0 baseline rests on.

**Two things the frozen schemas could not carry** and now do, as values rather than columns:
rule-level rows are `category=""` with `rule` set, and R1 without the four starting squares
is `class="R1_no_start"` / `metric="f1_R1_no_start"`. Reading either CSV needs
`pd.read_csv(..., keep_default_na=False)` or those empty strings come back as NaN.

---

## The 4-way legality probe, and the filler dimension it needs (2026-08-05)

Branch `legality/probe`. `src/spade/legality_probe/probe.py` is `bracket_flip/tpr_probe.py`'s
bilinear TPR with the filler alphabet widened from the 3 board classes to
{legal, R1, R2, R3}, fit at whatever input dimension the activations carry (raw 512, SAE
8192) and saved keyed by space name.

**Upstream's class count is a module global imported by value.** `train_tpr_probe` does
`from train_probe import BOARD_LABEL_OPTIONS` (3), and that one name sizes
`filler_embeddings`, sizes `bias`, and multiplies the loss (`train_tpr_probe.py:227`).
Building a 4-class probe without rebinding it gives 3 fillers and *no error*, and leaves the
loss scaled by 3 — a 25% effective-LR shift on top. `filler_classes(n)` rebinds it for the
duration of a fit and restores it, so 3-class board fits and 4-class legality fits coexist in
one process. `test_probe.py` pins both effects, including that the same logits under n=3 and
n=4 differ by exactly 3/4.

**The loop is the committed one, verified.** Refitting the 3-class board task through
`legality_probe.fit` at layer 5 reproduces `tpr_probe._fit` *parameter for parameter*
(`torch.allclose` on all three tensors), accuracy 0.9827 — the number in
`analysis/bracket_flip/reports/PROBE_REPORT.md`.

**d_f = 3.** Sweep on raw layer 5, 8000 train / 2000 held out, 4-class labels derived from
the reconstructed mover-relative board (`taxonomy.classify` collapsed by `RULE_OF`):

| d_f | macro-F1 | accuracy | F1 legal / R1 / R2 / R3 |
|-----|----------|----------|-------------------------|
| 2   | 0.8891   | 0.9342   | 0.944 / 0.984 / 0.887 / 0.742 |
| 3   | 0.9374   | 0.9631   | 0.974 / 0.993 / 0.932 / 0.851 |
| 4   | 0.9378   | 0.9635   | 0.974 / 0.993 / 0.933 / 0.851 |

d_f=2 costs 4.8 points, nearly all of it on R3 and R2; d_f=4 buys 0.0004 over d_f=3 and
makes the filler geometry uninformative (4 classes in 4 dimensions can be exactly
orthogonal, so the angles the bracket-flip study reads off the fillers stop meaning
anything). Default is 3.

**Two things about the metric.** Fit on shuffled labels the probe still scores **0.56
accuracy** at 0.29 macro-F1, because R1 is 54% of squares mid-game — so the floor to beat is
0.56, not chance, and accuracy alone says almost nothing. And legality computed for the
*wrong* player (the one who just moved) is still decodable at **0.9101** macro-F1 against
0.9374 for the right one, the whole difference sitting on the `legal` class (0.832 vs 0.974).
Legality is a deterministic function of a board that decodes at 0.98, so a high legality
macro-F1 is not by itself evidence that the model represents legality; the study's claims
have to rest on differences between spaces and on the r<0 / r=0 contrast, not on the level.
## Legality-probe labels and fit set (2026-08-05)

Branch `legality/labels`. New package `src/spade/legality_probe/`: `labels.py` collapses the
7-way taxonomy into the probe's 4-way filler (0 legal, 1 R1 placement, 2 R2 bracketing,
3 R3 turn), `fit_set.py` chooses the positions, `extract_acts.py` runs layer 5,
`frame_check.py` is the validation that the labels are in the right player's frame.

**The transposition is the whole risk.** `taxonomy.classify_all` returns 60 categories in
TOKEN order (entry i is token i+1); the TPR probe's roles are 64 squares in ROW-MAJOR board
order. `board_labels` is that scatter via `stoi_indices` and nothing else. `test_labels.py`
checks it three ways a transposed or shifted board would fail — the scatter must equal
classifying each of the 64 squares directly, the squares it calls legal must equal
`board.get_valid_moves()`, and the squares it calls R1 must equal the occupied ones — on
positions with no diagonal symmetry left.

**The four starting squares are R1 placement**, not a fifth "not applicable" class, because
that is what `classify` returns for them: they hold a disc at every ply of every game, so
the rule that makes any occupied square illegal covers them. All 64 squares then carry the
same alphabet and `build_square_selection_mask(exclude_center_squares=False)` — already
`bracket_flip/tpr_probe.py`'s choice — scores them like any other occupied square. The cost
is that 4/64 squares are R1 by construction and readable off the role embedding alone.

**The mover's frame is not checkable against the engine.** `classify` and `get_valid_moves`
both read `board.next_hand_color`, so they agree even when that colour is wrong. What breaks
the circle is the game record: the move played at ply t must come out legal. It does, at
10000/10000 positions in the mover frame and 3047/10000 in the other one (the rest become
R3). `verify()` asserts it per row. The frame flip is otherwise nearly invisible — it
relabels 70.3% of legal squares and 100% of R3, but **0% of R1 and 0% of R2**, because
`occupied_own`/`occupied_opp` and `empty_own_only`/`empty_no_bracket` swap *within* their
rule. A linear probe fit on the wrong frame still reaches macro-F1 0.902 against 0.939, and
the whole gap is `legal` (0.824 vs 0.984). Anything that reports only R1/R2 accuracy cannot
tell the two apart.

**Fit set: 200 games x token positions [5, 55)**, the position range of
`analysis/linear_probe/activations`, whose own token sequences are gone — that script read
ARENA's `board_seqs_id_small.npy` from a `reference/research_materials/` tree that no longer
exists, and the activations carry no game identity. A game is taken whole or not at all, so
one forfeit anywhere in the window drops it (4 games in 2204) and every accepted game gives
exactly 50 rows; the 8000/2000 split then lands on a game boundary, 160 games against 40,
with no game in both. Drawn from the branch stream (seed 42) with the 2000
violation-divergence source games excluded by exact token match — all 2000 were hit, which
is also the evidence the exclusion works — because a probe fit on a read-out game would make
the bitwise-shared r < 0 prefix look preserved for free.

**Class balance** (`analysis/legality_probe/results/class_balance.csv`): legal 14.8%, R1
53.9%, R2 21.7%, R3 9.6%. R3 is not too rare to fit. Worth correcting a claim that has been
passed around: plain accuracy for a probe that has learned nothing is ~0.54 here, not ~0.9 —
0.9 is the number for a binary legal/illegal split. Macro-F1 is still the right metric, and
the majority-class baseline for it is 0.175. The composition moves hard with ply, all of it
mechanical: R1 is exactly (p+5)/64, so R2 falls 0.65 -> 0.004 and R3 0.098 -> 0.024 across
the window. Any read-out pooled over r has to carry that.
## The legality-probe space roster, and what it takes to keep the prefix bitwise (2026-08-05)

Branch `legality/spaces`. `src/spade/legality_probe/spaces.py` holds the study's 9-space
roster and encodes both activation sets through each.

**`representative_spaces()` returns 17 entries, not 11.** raw, the two random controls, the
8 seed-42 SAEs, and the six seed-43 sweep points that appear as soon as
`study_dir("multiseed")` exists. The roster filter is therefore `n == "raw" or
entry.seed == 42`, not a name set. `bracket_flip/sae_arm.py:45`'s
`{"raw", *BASELINE_CONFIGS}` is wrong twice over — it readmits the degenerate
`SpaDETrainer-othello` that `spaces.EXCLUDED_SAE` drops, and it never names the default
SpaDE, `SpaDE-lam1e-03-gam1e-04`.

**Batch shape decides whether the r < 0 prefix survives the encode.** Causal attention makes
the pre-t0 activations of a game's branches bit-identical, which is what makes r < 0 a
*paired* in-distribution control rather than an approximate one. Two branches of one game
have to be rows of the same GEMM, or cuBLAS reduces their shared prefix differently and the
codes stop being identical for a reason that has nothing to do with the SAE.
`extract_acts.spans` already encodes that rule for the extraction pass; the encode reuses it
at `ROW_BATCH = 248`, chosen so `248 x 33 <= spaces.ENCODE_BATCH` and a span goes through
`batched_apply` as a single matmul. All 9 spaces plus both nulls then keep the prefix
bitwise on an H100.

**Measured L0 is the standardization check.** SAEs were trained on `scale*(x - mean)`, so an
unstandardized encode is silently wrong rather than merely inaccurate. Mean L0 on the branch
window lands within +0-14% of the Karvonen eval L0 for the same checkpoint, in the same rank
order, and exactly on the analytic value for the three spaces that have one (TopK 80.00,
random_topk 80.00, random_proj 4101 ≈ half of 8192). The small positive bias is the branch
set: five of its nine categories are illegal continuations, which fire more features than the
natural games the eval samples. `results/acceptance/all_results.csv` is the only eval CSV in
the tree with data — the six paths `multiseed/aggregate_seeds.SEED42_CSVS` names are empty
directories.

**The branch code is streamed, never cached.** A `[17527, 33, 8192]` fp16 tensor is 9.5 GB
per space, 76 GB for the roster, on a 90-day tier — written so it can be read once and
reduced to scalars by the read-out. `stream_branch` yields one `spans()` batch instead, so
the full code never exists and each consumer reduces `[rows, 33, d]` as it arrives. Sparse
storage would have cut the 76 GB by ~6x and was rejected for the same reason: a format
change for a tensor that should not exist. `prefix_bitwise` becomes span-local for free —
`spans` never splits a game, so a span holds every branch of every game it touches.

**The probe is not fitted on `analysis/linear_probe/activations/`.** That set holds forfeit
positions and 53 mislabeled rows, its positions cannot be recovered because the extractor
read a tree that no longer exists, and nothing ruled out overlap with the read-out games —
a probe fitted on the games it is later read out on inflates every number in the study.
`PROBE_ACTS` defaults to the label pipeline's own `[10000, 512]` layer-5 extraction, whose
200 source games exclude by exact token match all 2000 games behind the read-out branches.
It is already at layer 5, so `probe_acts` asserts `ndim == 2` instead of slicing `[LAYER]`:
hand it the old all-layer stack and it fails loudly rather than encoding the wrong positions.

**Every row records a fingerprint of that set.** `fingerprint()` is `tpr_probe._fingerprint`'s
job over floats — shape plus a blake2b digest of the activation bytes, since a float sum
collides far too easily to key a cache on. Digesting the activations rather than a manifest
means no other corpus can produce the same fingerprint, so `$SPADE_LEGALITY_PROBE_ACTS`
repointed at a set that is *not* disjoint changes the fingerprint and every cached row
asserts. Nothing measured off the branch window — L0, prefix invariance, d, everything in
`space_quality.csv` — depends on the probe set at all.

Cache: `study_dir("legality")/latents/`, resumable per space on `row_<space>.json`. Only
`probe_<space>.pt` is written, 164 MB per 8192-wide space, ~1.5 GB for all 11 — refitting a
probe is the expensive thing to redo, re-encoding a window is not.

---

## The fork figures' prose moves to their READMEs, and the axes stop naming a component (2026-08-04)

Branch `fig/strip-captions`. Every fork figure — `fork_and_board` (both planes),
`fork_curvature_space` (all four), `game_trajectory` — carried a multi-paragraph legend
title: the selection disclosure, where the rule mean is taken, the spread ratio, n=1, the
not-comparable-across-cells disclaimer. That is a caption, and §2/§5 put captions outside
the graphic. All of it now lives in each study's `README.md` under `## Captions`, one
heading per figure, and **nothing was dropped** — the selection of game 1235 above all,
which is the one a reader cannot be allowed to miss. What stays inside: panel titles and
letters, per-panel space names, axis labels, the legend, the board's reading key, and the
R1/R2/R3 key, because §5 requires domain shorthand to be readable without another document.

Two things that were in the prose became legend *entries* rather than moving out, since the
legend is in-figure by right: the jackknife ellipse's semantics (`50% Gaussian contour of 13
jackknife replicates (dispersion, not a CI)`) and the `(mean of n)` already there. An
uncertainty a reader could mistake for a CI does not belong two documents away.

**Why the axis labels lost their lead component — a correctness fix, not taste.** The R^21
paths appended the descriptor coordinate the PC loads hardest on, giving
`PC1 (44% var, Δ mean θ)`. Measured on raw layer 5, the windowed plane's PC1 loads **0.360
on `mean θ`, which is 13.0% of the direction's squared length**, over a near-tied and
same-signed `mean d` 0.328, `max θ` 0.323, `min d` 0.267, `max d` 0.249; **16 of 21
coordinates are needed to reach 90%** of it. PC1 is a broad mixture, not that named
quantity. Worse, which coordinate "wins" flips between the near-ties with the branch set and
the variant: the `whole` plane, same game and same space, leads on `min d` at 0.290 — 8.4%
of its direction, 15 of 21 for 90%. Naming one component asserted something false and would
have flipped under a `--game` change. Axes now read `PC1 (44% var)` / `PC2 (18% var)` and
nothing else, from the shared `branch_panels.axis_labels` the displacement paths already
used; Isomap keeps `arb. units` and no ratio. `Plane.lead` stays — `measure.py` and
`r21_shared_basis` print it, where a lead is a diagnostic and not a label.

**The rule key now names the members.** Three means hide six categories and the faint member
lines were unlabelled. `branch_style.rule_member_key` prints one compact line per rule —
name, member count, and each member's gloss — driven by `RULE_MEMBERS` and the new
`taxonomy.CATEGORY_GLOSS` (which replaces the comments that used to sit beside
`CATEGORIES`), so the counts cannot disagree with the legend's `(mean of n)`. It does not
replace `paper_style.RULE_TEXT`: nine other figures print that three-sentence form, and R4
and R5 have no members at all.

**Panel (b) gained the would-be disc.** An outline plus a corner marker said "a branch plays
here" but nothing said a move *puts a disc down*. Each branch square now carries the disc it
would place, in the mover's colour, at **half a real disc's radius, translucent, dashed, and
pinned to the corner opposite the category marker** — never at the centre, because
`occupied_own` and `occupied_opp` already hold a real disc whose colour is the reader's only
evidence for which of the two the branch is. It survives grayscale: real discs are solid and
full-size, the would-be disc is grey (or white) with a dashed edge and visibly smaller. It
is drawn on **all** branch squares including the reference and the control, not only the
illegal ones — a marking that appeared only on illegal squares would imply the placement is
what makes them illegal, which is false for four of the six categories. The panel title also
names the mover, derived from `board.next_hand_color` via `branch_panels.mover_name` (shared
with the board note), because the mover alternates with *t0* and a hardcoded "Black" goes
wrong the moment anyone passes `--game`; verified on game 1 (*t0*=11), which renders "White
to move".

**Layout.** Stripping ten lines of legend title from `fork_and_board` left a hole beside the
square board, which §5 calls a layout bug. The board's cell is now as wide as it is tall so
the board fills it, `BOARD_H` drops 5.4 → 3.9, and the legend block is centred in the band
rather than pinned to its top. The rule key moved out of the band and under the whole
figure, the only place with the width its three lines need. The three grid figures pinned
their key at a hardcoded `y` tuned to a ten-line title; `branch_style.below(owner, leg)` now
measures the rendered legend instead, so the key follows it however the entries wrap.

`selection_note()` and `POPULATION_SPREAD` lost their only callers and are deleted; their
content is in the READMEs and their reasoning survives as comments beside `FIGURE_GAME` and
in `rule_groups`'s docstring. Both scripts still *print* the spread ratios they drew, so the
README numbers can be re-derived without reading a figure.

---

## The R^21 plane moves into the library, and fork_and_board can draw it (2026-08-04)

Branch `r21/plane-and-combined`. `fork_curvature_space/descriptor.py` was analysis-level,
so `fork_and_board` could only have reached it by importing one analysis dir from another.
It is now `src/spade/violation_divergence/curvature_plane.py`, beside `fork_plane.py`.

**What the move deduplicated, and what it did not.** The sampling window
(`STRIDE`/`WINDOW`/`FIRST_POS`/`LAST_POS`) and the nine-cell space list are one rule, so
they live once in `fork_plane.py`; `positions(t0, back=0)` grew the `back` parameter the
R^21 window needs, and `back=0` is the displacement plane's old body exactly. The
duplicated figure-game constant is gone — `branches.FIGURE_GAME` was always the one that
mattered. The two planes' *fitting* is genuinely different (uncentred PCA of D vs
column-scaled PCA of Δφ, which has to survive NaNs) and stays separate. Verified by
re-rendering all four `fork_curvature_space` figures before and after the move:
byte-identical PNGs. Four other analysis dirs still each carry their own copy of the
nine-space list; they were out of scope here.

**fork_and_board grew `--plane {displacement,r21}`**, default `displacement`. Two figures,
`game_1235_fork_and_board_displacement_pca` and `..._r21_pca` (the old
`game_1235_fork_and_board_pca` is the former, renamed).

**The trap, which would have been silent.** φ — step lengths, turning angles, moments — is
nonlinear, so `mean over members of φ(z_b) ≠ φ(mean over members of z_b)`. A figure drawing
one mean per rule must apply φ per member and average afterwards; averaging the trajectories
first measures the curvature of a path no branch takes, and the result looks perfectly
plausible. The plane projection is linear, so mean-then-project is safe and the figure
keeps taking its mean in the plane, as the displacement version does.
`test_curvature_plane.py` pins both halves: φ does not commute with the mean, the
projection does.

**The R^21 grouping is no better: within/between rule spread = 0.902 in R^21 against 0.885
on displacements** (game 1235, raw layer 5, column-scaled Δφ over t ≥ t0). Slightly worse,
in fact. There is no population number in R^21 — the 0.945 median is a displacement figure
— so each legend names which of the two it is quoting rather than letting one stand for
the other.

**The window still costs two offsets and the prefix still collapses.** At t0=20, 15 of the
study window's 17 offsets carry a descriptor and 2 of those are branch-free, unchanged by
dropping control 2 and by the rule grouping. `project()` asserts `max|Δφ| = 0` on the
branch-free offsets in all nine spaces on every render, so the legend's "the prefix still
IS the origin" is checked rather than asserted.

**Layout gotcha.** The R^21 legend title is five lines longer than the displacement one, so
the board note and the rule key, which used to sit at a fixed y, landed on top of it. They
now hang off the legend's measured bottom edge (one `fig.canvas.draw()`), which is why the
two figures' canvases differ slightly in height.

---

## fork_curvature_space: the same three changes, and a worse number (2026-08-04)

`fork_and_board`'s three changes applied to the four R^21 grids. The measurement is the
point of the entry.

**The rule grouping is worse in R^21 than in the displacement plane, and for the whole
variant it is gone.** Within-rule ÷ between-rule mean pairwise distance, column-scaled
full R^21 (never the drawn plane), raw layer 5, game 1235: **0.90** on `Δφ(t)` at
`t ≥ t0` (0.77–1.01 over the nine spaces) and **1.03** on `φ`, one point per branch
(0.90–1.06, above 1 in three spaces). The same statistic on the full 512-d displacements
of the same game and window reproduces `fork_and_board`'s **0.885** exactly, so the
comparison is apples to apples. Mapping into the curvature descriptor does not make the
rules cohere; on the whole variant the label carries nothing, and `fork_r21_whole_pca`
shows it — the three R2 members scatter across the cell with their mean in the middle of
nothing. Both numbers ride in the legend titles, members drawn faint.

**φ is nonlinear, and that constrains where the mean is taken.** mean φ(z) ≠ φ(mean z):
each member gets its own descriptor, its own Δφ and its own projection, and only the
coordinates are averaged. The projection is linear (windowed, uncentred) or affine
(whole, centred), so plane-mean and R^21-mean agree exactly — which is the only reason
the cheap version is also the correct one. Column scaling before the distance follows
`measure.py`: the 21 components mix degrees with activation-scale lengths, so a raw
Euclidean statistic is a statistic about θ alone.

**`rule_fork_panel` is not reused, deliberately.** It takes `k` from `pos == t0` and
draws from the branch onward. This figure draws from the first *defined* offset, two
sampled positions before the branch, because those two points sitting exactly on the
origin are the visible proof that Δφ collapses over the shared prefix — one of the two
invariants the sub-study exists to keep. Everything else is shared:
`branch_style(by_rule=True)`, `MEMBER`, `legend_proxy`, `rule_rows`, `ordered_legend`
and `PLANE_TICKS`, and the local `branch_style`/`legend_order`/`ordered_legend` copies
are deleted. What is duplicated is ~8 lines of path drawing.

**Ticks** go on the windowed grids' plane axes, three symmetric about zero. The whole
grids are 2-D scatters of physical quantities (`mean d ∈ [18, 21]`), where a symmetric
locator puts every tick off-panel; they keep ordinary ticks. Also honoured the `zlabel`
argument `windowed_grid` was already passing and `fork_panel` was ignoring, so
`t − t0 (plies)` is labelled once per row as in the sibling figure.

**Left behind:** `make_figures.py`'s import block is owned by a parallel agent moving
`descriptor.py` into `src`, so the now-unused `CATEGORY_STYLE`, `NULL`, `REFERENCE`,
`CATEGORIES` (and the pre-existing `GRID`) are still imported and `LEGAL` came in on its
own line beside them. Prune on merge.

---

## fork_and_board: rules instead of categories, one control, ticks on (2026-08-04)

Three requested changes to the shipped figure. The first one produced a number worth
knowing.

**Six categories → three rules.** `othello.taxonomy` now owns `RULE_MEMBERS` (the domain
fact — which rule a category violates — belongs beside `CATEGORIES`, not in a style dict),
`branch_style(..., by_rule=True)` swaps `CATEGORY_STYLE` for the `RULE_STYLE` that was
already in `paper_style` and unused, and `branch_panels.rule_fork_panel` draws one mean per
rule. Both panels group, not just (a): if the fork drew rules and the board drew
categories, the figure's one-legend-for-both property — the whole reason the band beside
the board exists — would collapse into two legends. The cost is that the two R1 squares are
now the same colour, so the board note says a rule's members are told apart by their square.

**The grouping is weak, and that is the finding.** Within-rule ÷ between-rule mean pairwise
distance on the full 512-d layer-5 displacement trajectories: **0.885** on game 1235,
**median 0.945 [IQR 0.872–1.020]** over all 1550 branch-complete games, **above 1 in 30%**
of them (`python -m spade.violation_divergence.rule_groups`). A rule's members travel ~5%
closer to each other than to another rule's. Three confident lines would misrepresent that,
so members are drawn faint behind their mean and the ratio is in the legend title. Game
1235 is near the first quartile — its grouping is better than typical, which is selection
again, since it is the argmax of the mutual separation of those same six trajectories.
FIGURE_GUIDELINES §1.7 is this rule generalised: a grouped mean carries n, its members, and
its within/between spread.

The mean is taken **after** projection, not before. For PCA that is identical (`pca2` is an
uncentred linear map), and for Isomap it is the only thing that means anything — one rule
covers both embeddings.

**One control.** `control_legal_2` is filtered out of the figure's rows in `main()`, which
also drops it from the PCA fit, since the plane should be fit on what is drawn. The cost is
real and is written down in the README rather than swallowed: control 2 exists because *one
control gives no null* (RED_TEAM F1 — the first pass could not interpret its own direction
result), so the figure loses the reader's visual answer to "is that separation big compared
to two legal moves differing?". Nothing measured changes; the legend title now points at
`game_trajectory` and the parent README, which is where the legal-vs-legal scale lives.
`game_trajectory` keeps all nine branches and per-category styling on purpose — it is now
the figure that carries what this one dropped, and its README says so.

**Ticks on the PC axes.** Three per plane axis, `MaxNLocator(3, symmetric=True)` so the
origin — the reference, in a displacement plane — always gets one, 7 pt, negative tick pad.
This reverses FIGURE_GUIDELINES §4.6/§4.8, which said ticks off because the coordinates are
panel-local; both sections are rewritten. Within-panel scale is worth more than the
ambiguity, and the "not comparable across cells" disclaimer stays and matters more, not
less. `fork_panel` and `rule_fork_panel` share `_fork_axes`, so `game_trajectory` gets the
ticks too; its two x/y `labelpad`s went from −8 to −2 to clear the new tick labels.

**Status:** `fork_and_board` PCA re-rendered as PNG and camera-ready PDF, `game_trajectory`
PCA re-rendered; both isomap variants render (checked, not shipped — neither was shipped
before).

---

## One figure: the fork over the board it forks on (2026-08-04)

`analysis/violation_divergence/fork_and_board/` merges the nine-space fork grid and the
branch board into a single figure — (a) the 3×3 grid of representation spaces, (b) the
board at *t0* — because the two only mean something together: (b) says what the nine
branches are, (a) says where they go. Both draw the same game, **1235** (*t0*=20), and the
entry points now assert the requested game is branch-complete instead of taking
`complete[0]`.

The board is square and the grid is three wide, so the band beside the board carries the
figure's apparatus: **one legend for both panels** (the payoff of `branch_style` — both
halves already draw identically, so a second legend would repeat itself), the board's
reading key, and the R1/R2/R3 gloss.

Three extractions, each because the combined figure would otherwise have copied code:
`src/spade/figures/branch_panels.py` owns `fork_panel` / `board_panel` / `axis_labels`
(were private to the two scripts), `src/spade/violation_divergence/fork_plane.py` owns the
sampled positions and the displacement embedding, and `branches.py` — which defines what a
row of `branches.pt` means — grew the reader for it (`load_game`, `complete_games`,
`reference_row`). `ordered_legend` and `rule_key` now take the figure *or* an axes as
their owner, which is what lets both live inside the band rather than under the figure.

**Axis labels.** `space_grid` never labelled x/y, so nine panels carried their variance
ratios buried in the titles; they now read `PC1 (27% var)` / `PC2 (10% var)` on the axes
per §4.6, per panel because the plane is refit per space. `draw_board` labels `file` /
`rank (1 at the top)` alongside the ticks it already owned, so the board's reading note
moved off the xlabel to wherever the caller has room for it.

**Game 1235 is selected, and that is now on the figure.** It is the argmax over the 1550
branch-complete games of the mutual separation of its six illegal trajectories (raw layer
5, full 512-d, never an SAE space or a projection; population median 0.705, this game
1.080). It matters: on 1235 the reference and both controls separate from all six illegal
branches in all nine spaces, but over 40 games that ordering holds in 0.07–0.17 of them
against a 0.036 chance floor — selected structure that does not generalise, and a reader
who does not know it was chosen over-reads the panels. `branches.FIGURE_GAME` and
`branches.selection_note()` keep the constant and its disclosure together, so every figure
drawing the default game prints it; a non-default `--game` prints that instead.

**The two nulls were indistinguishable.** `branch_style` returned byte-identical dicts for
`is_control` 1 and 2, and on this board they land on adjacent squares (h3, h4). Control 2 —
the null *for* control 1 — now takes an open marker (`mfc`/`mec` in the style dict, which
also pulled the hardcoded `"white"` marker edges out of the three panel bodies). Same
colour, same shape, different fill: separable without reading legend text, and in grayscale.

**Status:** rendered on game 1235 with all nine spaces, working PNG and camera-ready PDF.
`branch_board/` and `game_trajectory`'s `space_grid` were subsets of it and are deleted; the
board drawing survives in `spade.figures.othello_board`. `fork_figure` stays in
`game_trajectory` — it carries the δ̂ marginal panel, the quantitative half §4.8 requires and
which this figure does not have.

---

## The branch board, and one board-drawing rule for two studies (2026-08-04)

The board the game forks on, with all nine branch moves on their squares — the companion the
fork figure needed, since a projected trajectory never says what a branch *is*. It began as
`analysis/violation_divergence/branch_board/` and is now panel (b) of `fork_and_board/`.

Two extractions, both because the same rule was about to be written twice.
`src/spade/figures/othello_board.py` owns "draw an 8×8 board with these discs" (was
private to bracket_flip's `board_panel`); `src/spade/figures/branch_style.py` owns
`branch_style` / `legend_order` / `ordered_legend` (was private to the fork figure). Each
caller keeps its own overlay — role shading, category outlines — out of the shared helper,
so neither grows a flag per caller. bracket_flip Fig 1 and the fork figure both re-render
pixel-identical after the move.

**Occupied squares.** Two illegal categories play onto a square that already holds a disc,
and the disc's colour is exactly what distinguishes `occupied_own` from `occupied_opp` —
so a centred marker would cover its own evidence. Every branch square is therefore outlined
in its category's colour and line style, with the marker in the square's **corner**, for all
nine branches. Uniform grammar, no disc obscured. Only the reference's flips get rings
(dashed, outside the disc so they read over black and white); drawing all three legal
branches' flip sets is a knot with no question behind it.

---

## The fork figure's game is now chosen, and the criteria disagree (2026-08-04)

`game_trajectory/` drew `complete[0]` — game 0, *t0*=10, the minimum *t0* over the
1550 branch-complete games. List order, not a criterion.
`analysis/violation_divergence/game_selection/` sweeps all 1550. Its pure argmax is
**game 662, *t0*=22**; the figures draw **game 1235, *t0*=20** — see the last paragraph.

Two constraints made the sweep worth trusting, and both were committed before it ran
(`score_games.py` docstring): score in **raw layer 5 only** — an SAE in the selection
loop would tilt the nine-space grid toward whichever dictionary drove the choice — and
score in **full R^512**, since a top-2 PCA criterion selects the game that flatters one
plane and the plane is being redesigned.

**The four candidate criteria have disjoint top-10s.** ρ(*S3*, *S4*) = −0.442: the games
whose branches diverge furthest are the games whose illegal categories are *least*
differentiated, because large δ̂ means everything moving along one shared direction. So
"pick the biggest divergence" is the worst rule for a category-colored figure.

**Why not *S1*, the illegal-vs-control separation.** It is the study's own scientific
contrast, and the aggregate says illegal displaces *less* than a legal control (0.978).
Its argmax is therefore the single game that most contradicts the paper, dressed as an
illustration. The pick is argmax *S3* instead — scale-free, and no claim rests on it.
Cost, stated in the sub-study README: game 662 is at the 5.8th percentile of *S1*
(conservative direction, but not representative of that contrast).

Transfer check: game 662 is rank 1/302 in Standard, TopK and SpaDE latent spaces too,
ρ = 0.935–0.990 against raw L5. The choice is branch geometry, not the selecting space.

**Why the figures draw 1235, not 662.** Argmax *S3* alone lands on a game at the 5.8th
percentile of *S1*, so the panel would understate the very contrast the study is about. The
rule was restated as *argmax S3 among games at or above the median S1* — 1235 is *S3* rank
3/1550 (1.042 vs the 1.080 max) and above median on *S1* and *S2*, with *t0*=20 against a
population median of 21. Chosen after seeing the results, which is why the rule is written
down here and the disclosure is on the figure; `branches.FIGURE_GAME` holds the constant.

---

## R^21 cannot carry the curve inside the r >= -8 window (2026-08-04)

Feasibility check on projecting `space_fork_grid_pca` through the ReStraV descriptor
instead of the raw activation displacement. New study in
`analysis/violation_divergence/fork_expanding_curvature/`; tables in its README.

**The blocker is dimensional, not statistical.** `restrav_features` needs 8 sampled
positions (14 plies at STRIDE=2) to fill d1..d7 and θ1..θ6. The fork window has 4 sampled
positions before *t0* (8 plies). A sliding window at W=8 therefore has zero curve points
free of the branch, so the shared prefix at the origin and the spray at *t0* — the figure's
entire content — are outside the descriptor's domain. W ≤ 4 keeps a prefix but returns
R^13 at most. No W does both.

**Superseded in part — the window, not the arithmetic, was the constraint.** The 4-sampled-
position limit comes from the study window `r >= -8`, not from the game. Reading the prefix
back to the start of the game dissolves it whenever *t0* >= ~16: at game 1235's *t0*=20 the
pre-branch samples are 2,4,…,20, so a trailing W=8 window ending at *t0* spans positions
6..20 and is both full-width R^21 and branch-free. `fork_curvature_space/` builds it and
measures `max|Δφ| = 0` at every *r* < 0 in all nine spaces. Everything below about the
truncation trap and the denominator still stands.

**Gotcha worth remembering: `restrav_features` truncates silently.** `d[:, :7]` and
`theta[:, :6]` on a short window return 11/13/15/17/19 columns with no error (and an
obscure `IndexError` from `moment4` at W=2), and the column *meanings* shift — at W=5,
column 4 is θ1, not d5. Anything slicing it from a variable window must assert the width.

**The expanding window anchored at *t0* was prototyped and rendered.** It is causal and
event-locked, but it starts 14 plies after the branch and 13 of its 21 coordinates are then
frozen exactly (measured 0.000e+00) — the first steps since the branch never change again,
so only the 8 cumulative moments move, and their travel is only 1.2–2.1× a null with the
temporal order shuffled away.

**A large effect size here is the denominator, not the descriptor.** Illegal-vs-control
separation goes 0.09–0.45 (the current per-position plane) → 1.2–1.6 (R^21), but the *same
raw displacement* averaged to one point per branch gives 0.89–1.87, i.e. larger, in every
space. Collapsing the position axis removes the within-group spread that dominates the
denominator. A label-permutation null over all 28 relabellings of the 8 branches clears
nothing (best p=0.18 against a 0.036 floor). Also noted: an unscaled PCA of R^21 is a plane
of turning angles (PC1 puts 1% of its energy on the step-length block, since d is in
activation-norm units and θ in degrees), so the planes are fit after a diagonal unit-RMS
rescale that keeps zero displacement at the origin.

**Recommendation:** keep the raw-displacement plane for the fork; if R^21 belongs in the
study it belongs as one point per branch over the whole post-*t0* stretch, as a §4.6
embedding scatter over the full 13,531-branch population, not at n=1.

---

## The fork grid can share one basis, if it goes through R^21 (2026-08-04)

`analysis/violation_divergence/r21_shared_basis/`, branch `curvproj/r21-shared-basis`. The nine-space fork grid
in `violation_divergence/game_trajectory` refits its plane per space and has to disclaim
that cells are not comparable, because R^512 and R^8192 share no coordinates. Projecting
the displacement through the 21-D ReStraV descriptor first gives every space the same named
coordinates, and then one plane does work for all nine cells.

**A shared plane costs nothing; not fitting at all costs less.** Illegal-vs-control centroid
gap, in pooled within-group SDs on game 0: per-space refit 0.70–1.80, one shared fitted
plane 0.70–1.64 (better in 6 of 9 spaces), fixed axes (Δ mean *d*, Δ mean θ) 1.32–2.00 —
best in every space, on 0.11–0.20 of the variance the fitted planes maximise. The status-quo
activation plane is 0.13–0.39 on the same points. Verdict and full tables in `REPORT.md`.

**The comparability claim is half, not whole.** Z-scoring is mandatory in R^21 (the
components mix degrees with activation-scale lengths, ~450x SD spread across dictionaries)
and it is per space, so a shared basis makes arrangement and direction comparable cell to
cell but never absolute magnitude. The grids say exactly that; the refit grid keeps the old
disclaimer.

**Every number is scored against an exact permutation null** — all 28 ways of relabelling
the 8 non-reference branches 6/2 — because n=1 game and 10 overlapping windows per branch
make a raw effect size meaningless. R^21 roughly doubles the null as well as raising the
gap; after dividing it out the status quo sits at 0.45–1.05x chance and the named plane at
2.32–3.10x.

**Do not quote that 2.32–3.10x.** It is a one-game number and it does not replicate. Both
`fork_curvature_space/` (40 games) and `curvature_discriminant/` re-measured the same named
pair: 1.27x over 40 games, and 0.24–1.10x on game 1235 specifically. Nothing was wrong with
the computation — the n=1 statistic is simply unstable, which is what 8 effective branches
in 21 dimensions predicts. The defensible figure for R^21 over the incumbent plane is the
40-game paired gap, **+0.08 [+0.01, +0.15]**.

**Gotcha for anyone reusing this.** A full 21-D descriptor needs 8 sampled positions = 15
plies, so on game 0 (t0=10, window clipped at the board start) every plotted window straddles
t0 and the figure loses its zero-by-construction prefix segment. Also: `restrav_features`
silently returns a *shorter* vector for a short window instead of raising — assert width 21.

---

## Wave 8: the docs catch up, and q15 narrows to one hypothesis (2026-08-04)

Three agents on `wave8/*` — the claims ledger, the archive zone, the docs sweep. Per-agent
records in `docs/finish_notes/`. Wave 8's third part, deleting the source repo's residue,
stays gated: questions 14 and 15 are open and wave 0 is blocked.

**The retraction had not propagated.** Three non-historical docs still stated that λ=9.77e-5
/ γ=0.2 were "stale defaults read off a Mac smoke config" — `pipeline.md` §5 and §7a,
`spade_vs_baselines.md` Appendix A, `straightness/sae_profiles/README.md`. Fixed. Port notes
keep the old numbers on purpose; they record what an agent believed at the time.

**q14's ±0.014 band now appears where numbers are compared.** It changes a reading:
`spade_vs_baselines.md`'s Pareto listing separates Gated (0.5673) from Standard (0.5679) on
reconstruction F1, which is well inside the band.

**q15, narrowed.** A byte-exact sweep over netscratch, the frozen repo and the backup finds
exactly two copies of `main/trainer_5/ae.pt`: itself and
`feature_viewer/hf_space/checkpoints/SpaDE/ae.pt`, the checkpoint deployed to the public
Space. Nothing in the multiseed tier matches, so it is not a stored sibling seed. But it
reruns at L0 127.76 / VE 0.973, which is what `spaces.py:28` already records for seeds 43–46
of the same config. Leading hypothesis, untested: seed 42 was re-run post-canonicalization
and overwrote the degenerate checkpoint in place, without ever landing on netscratch. Testable
by retraining that config at seed 42 and comparing. **This pins q11** — `hf_space/` holds the
only redundant copy and cannot be deleted first.

**One label is wrong in eight places.** The excluded checkpoint is called "the γ=1e-2
checkpoint" in prose; its config and its wandb record both say **γ=0.2**. `EXCLUDED_SAE` is
the config name `"SpaDETrainer-othello"`, so no code identifier is affected — but the
exclusion decision was taken under the wrong label, so the owner should confirm the rename
rather than have an agent do it silently.

---

## The port runs, and two of its reference numbers do not survive contact (2026-08-04)

Three agents on `finish/*` branches — packaging, the checkpoint copy, the flagged defects —
merged here. The tree went from "code moved, nothing verified" to installable, loaded with
its seed-42 weights, and 25/25 on `final_validation` with no skips. Per-agent records in
`docs/finish_notes/`.

**The port is faithful, and the proof is exact.** `checkpoints/{batch_topk,cosine_spade,
spade_revival}/trainer_0` reproduce their committed reference rows to 0.000000 on L0,
variance explained, coverage and reconstruction F1. Identical inputs, identical outputs. That
exercises SAE loading, standardization, the Karvonen adapter and the F1/coverage math end to
end, so nothing below is a port defect.

**Wave 3's stated acceptance number is wrong.** It gave VE = 0.943 for TopK on
`main/trainer_3`; the committed reference is 0.9867 and the rerun gives 0.98688. 0.943
appears nowhere in the tree. That wrong number has been the gate on the whole port, and
anyone checking against it would have concluded the port was broken. Corrected in
MIGRATION.md and in `final_validation`'s caveat text.

**Two provenance findings, now MIGRATION.md questions 14 and 15.** The activations behind the
five `results/karvonen/` reference rows no longer exist — those rows were committed 50
minutes before the only surviving extraction was written — so they reproduce only to ±0.014
on the assignment-protocol metrics. And `main/trainer_5/ae.pt` is not the run its config
names: it reruns at VE 0.973 where its wandb record and eval row both say ≈0.36, while
trainer_0–4 all match theirs.

That second one **retracts** the 2026-08-03 entry's reading that the λ=9.77e-5, γ=0.2
configuration never ran. It ran, 36621 steps on cuda:0, and finished degenerate, which is
what the docs reporting it say. `steps=400` and `device=mps` are stale in all six configs,
not just that one. The exclusion decision's justification is unaffected; what is in doubt is
which model the studies that loaded those weights were measuring.

Worth repeating for the next session: gitignored data must be copied into the real tree, not
into an agent worktree, or it evaporates with the worktree. And two agents prepending to
PROCESS.md conflict every time — one merge write-up, per-agent notes elsewhere.

---

## `spade` is installable now; the "101 modules import" claim never went through an install (2026-08-04)

Wave 1's stated verification — `pip install -e .` in a clean env, `import spade`,
`repo_root()` resolving here — had not been run. `spade` was installed in no environment,
`pyproject.toml` declared no dependencies at all (a commented-out placeholder), and every
line of `make setup` was a comment. The 101-module claim came from per-agent worktrees with
ad-hoc `PYTHONPATH`, which is not the thing the package has to do.

Fixed and verified in a throwaway venv built `--system-site-packages` off `spade_env`, so
the torch 2.5.1+cu124 stack is inherited and only the packaging is under test. **102 of 102
modules import cleanly** — the old count was one low, nothing was broken. Also verified from
stock `/usr/bin/python3.11` + pip 22.3.1, which is the fresh-clone path.

Do not `pip install -e` into `spade_env` itself: the editable path points at whatever
worktree ran it and dangles when that worktree goes.

Three real defects behind the missing install: `requirements.txt` named `python-chess` where
the working env has the `chess` distribution; `plotly` and `umap-learn` were pinned with no
importer anywhere we call, which is the exact defect that file's header criticises in the
source repo's version; and `pydantic` was undeclared although `viewer/server.py` imports it.
`pyproject.toml` now reads `requirements.txt` as its dependency list, so there is one list.

Still data-blocked, and reported as such rather than skipped: `final_validation` is 23/24
with the checkpoint round-trip skipped, `saes.registry.entries()` raises on the missing
`ae.pt`, and the wave-3 acceptance number needs weights this tree does not have. Details,
plus the `omegaconf` gap in the vendored TPR fork, in
[docs/finish_notes/env-install.md](docs/finish_notes/env-install.md).

---

## The defects the port agents recorded but did not fix (2026-08-04)

Swept all fifteen `docs/port_notes/` files for concrete code defects and fixed five. The
one that mattered: the sweep merge's baseline filter matched the substring `"SpaDE"`, which
also matches `"CosineSpaDE"`, so every `combined_results.csv` lost its CosineSpaDE row.
Latent in the source — `results/karvonen` had no CosineSpaDE row to lose — and live now
that `results/main` carries all eight baselines. `final_validation` pins it.

The other four are small: a `TypeError` on `--save-counts` (`repo_root` divided, not
called), `branches.py` importing `classify_prefix` through the rules study after it moved
into `othello/games.py`, and two docs that contradicted their own code.

Everything the notes flagged and I did *not* touch is written up with the reason in
`docs/finish_notes/flagged-defects.md`, including the one recommendation worth acting on:
`figure_karvonen_extension` writes its "no data" placeholder over the real figure's
filename, and six shell callers depend on that not raising.

---

## The port is done: 15 parallel agents, one branch each (2026-08-03)

Every zone ported from the frozen source repo — `common`, `othello`, `saes`, `spaces` +
`rules`, `rules/causal`, `training`, the pipeline, `figures`, the four studies, `scripts`,
`docs`, and the viewer. Each agent worked in its own worktree on its own `port/*` branch;
all fifteen are merged. Per-agent records are in `docs/port_notes/`.

**Where it stands:** 101 modules import cleanly, 28 analysis scripts compile, all 38 shell
scripts parse, `final_validation` is 23/24 with the one skip being the checkpoint
round-trip (weights are gitignored). Unit tests pass: 7 taxonomy, 11 divergence kernels,
plus the manifold-capture selftests reproducing the source numbers. What is *not* verified
is any real number — no eval has been run, because neither the checkpoints nor the
activations have been copied in. The wave-3 acceptance check (TopK on
`checkpoints/main/trainer_3` → L0 80.00, VE 0.943, coverage 0.722) is still outstanding and
is the first thing to do with a GPU.

**Seams the agents could not see from inside their own worktrees, fixed on merge:**
- *Three scripts would have destroyed the checkpoint layout.* They looped the eight
  per-type training entrypoints into one `--save_dir`, which writes `trainer_0` eight times.
  `trainer_N` is assigned by position within a single run, and `checkpoints/main/trainer_3`
  being TopK is a fact the registry, every eval CSV and every figure depend on. Now one
  `spade.training.sweep` call, which is also what the source did.
- *The SAE exclusion filtered nothing.* `EXCLUDED_SAE` was ported but never applied;
  `representative_spaces()` now drops it, which is what makes "excluded everywhere" true.
- *Two agents both ported `collect_legal` and `first_illegal_pos`* — byte-identical, since
  one took them from the other. Deduplicated.
- Four figure scripts called `results_dir()` with the old no-arg signature (`TypeError`);
  the viewer exporter used a registry field name that does not exist (`AttributeError`);
  three modules re-declared `LAYER`/`D_MODEL`/`CTX_LEN` locally, which is the drift that put
  nine copies of the dictionary size in the source repo.

**Needs David — the two that affect published claims:**

1. *The manifold-capture headline rests on the checkpoint we just excluded everywhere.*
   "Distance-SpaDE is the purest tiler, support size 0, shattering on 7/8 concepts" is a
   claim **about** the degenerate γ=1e-2 seed-42 checkpoint, and `docs/claims_status.md`
   lists it under "Holds" while retiring the geometry family's conclusion for resting on
   that same checkpoint. The study knows: it carries its own `LOW_VE` marker naming it.
   I aliased `LOW_VE` to `EXCLUDED_SAE` so there is one definition, and changed no
   behaviour. Either the shattering result is re-derived on a healthy SpaDE, or it needs
   stating as a property of a degenerate run, or the exclusion carves out an exception.

2. *A configuration in three documents never ran.* `checkpoints/main/trainer_5/config.json`
   records λ=9.77e-5, γ=0.2, steps 400, device mps. Three source docs report "SpaDE
   initial, λ=9.77e-5, γ=0.2" as a distinct over-regularized run. It is not a run at all —
   it is the γ=1e-2 production run wearing a Mac smoke config with pre-canonicalization
   defaults. That also explains the L0 15.4 / VE 0.352 attribution. The ported docs say so;
   the numbers in any older writeup do not.

Smaller items, each in the relevant `docs/port_notes/` file: BatchTopK *has* been evaluated
since 2026-07-14 (three source docs say it never was — real numbers now in the tables); four
bracket-flip CSVs have no producer in the tree because the A4 driver was never committed
upstream, so those figures redraw but cannot be re-derived; the extracted baseline-merge
helper filters `sae_type` on the substring "SpaDE", which also matches CosineSpaDE; and
`autoencoders_spade_v2` has no row in the artifact table and was mapped by the general rule.

---

## Wave 0 is blocked: there is nowhere to put the netscratch backup (2026-08-03)

**Needs a decision.** The agreed first step was to back up the netscratch tier before
porting. It cannot be done as specified. The tier is ~92 GB
(`spade_multiseed` 68 G, `spade_divergence` 9.3 G, `spade_divergence_v1_onecontrol` 8.4 G,
`spade_straightness` 5.3 G, `spade_bracket` 658 M, `spade_manifold` 471 M, `spade_rules`
69 M) and `/n/home10` has **22 GB free** against a 95 GB filesystem already 77% full.
Copying within netscratch is not a backup — the same 90-day retention applies, which is
what the existing 129 MB `spade_checkpoint_backup` there already runs into.

**Done instead, tonight.** Two things. A full file manifest of every `spade_*` dir — path,
size, mtime — committed under `manifests/netscratch/`, so if the tier expires we know
exactly what was lost. And the part that does fit: everything except the `.pt` checkpoints
is only 738 MB across 1306 files, and is now copied to
`/n/home10/dcao/spade_netscratch_backup/` (outside the repo, mirroring the netscratch
layout). That covers the per-seed CSVs and derived tensors the aggregation and figures
actually read. What remains exposed is ~91 GB of `ae.pt` weights.

**The question for David:** where does 92 GB go — lab storage, tape, cold storage — and is
it all of it, or only the parts that cannot be regenerated? The distinction that matters:
the per-seed CSVs are small and irreplaceable in practice; the checkpoints are large and
*technically* regenerable by re-running training, which is 2-day GPU jobs across 5 seeds
and 14 configurations. Also worth deciding whether `spade_divergence_v1_onecontrol` (8.4 G)
is kept at all — it is the superseded one-control branch set, but three shipped
sub-studies were computed on it and never re-derived.

**Deadline:** retention started 2026-07-22, so ~2026-10-20.

---

## The plan, re-derived against the live repo (2026-08-03)

**Goal:** the 2026-07-22 skeleton was written against the source repo as it stood that
morning, and twelve days of research landed on top of it. Fold all of that drift into the
plan so the port is planned against reality rather than against a two-week-old snapshot.

**Method:** eight parallel read-only surveys of the source repo, one per zone — `src/`,
`analysis/`, top-level artifacts, `shell/` + cluster env, the feature viewer, docs and
constraints, dependencies and packaging, and an audit of the skeleton itself. Findings are
folded into [ARCHITECTURE.md](ARCHITECTURE.md) and [MIGRATION.md](MIGRATION.md).

**What the plan was missing.** Seven code packages (`rules/` and its `causal/` subpackage,
`bracket_flip/`, `violation_divergence/`, `straightness/`, `manifold_capture/`,
`cosine_gamma_range/`), eight analysis studies, a second vendored runtime dependency
(`external/tpr_othello`), the whole figure-style layer (`paper_style.py` +
`FIGURE_GUIDELINES.md`), the feature viewer, 26 of 37 shell scripts, and the netscratch
tier that six studies depend on.

**Three findings that changed the plan rather than extending it:**

- *`rules/` is not a leaf study.* The plan modelled legality as one linear sub-pipeline.
  It is now the layer four other studies import — taxonomy, game streaming, board helpers,
  the checkpoint registry and the representation-space API all live inside it. Left as-is,
  every downstream study would import from a sibling study. The genuinely shared pieces are
  lifted into `spade/othello/`, `spade/saes/registry.py` and `spade/spaces.py`.
- *Three dirs the plan queued for deletion are live inputs.* `8_spade_configs_updated/`,
  `lambda_sweep_results/` and `gamma_sweep_results/` are hard-coded in
  `aggregate_seeds.py` and `figure_all_configs_combined.py`; dropping them removes the
  seed-42 point from every sweep config in the headline figure. The question for them
  changed from *drop?* to *where?*.
- *The per-study convention should be architecture, not repetition.* Five packages
  independently re-derived the same shape: a `study_dir.py` reading `SPADE_<STUDY>_DIR`
  with a netscratch default, numbered resumable steps, and a paired
  `analysis/<study>/make_figures.py` that only draws. That is lifted into
  `spade/common/study.py` and written down as the way a study is built.

**Decisions:**
- *All work happens in this worktree.* The source repo is frozen and read-only, ported
  from and never edited. The cost — no experiment can run here until the pipeline lands in
  wave 3 — is why the port order follows the import graph.
- *Commit at every coherent step, push at every wave boundary.* The skeleton sat
  uncommitted and unpushed for two weeks; that is the failure this rule prevents.
- *`paper_style.py` and `FIGURE_GUIDELINES.md` port in wave 1*, before any figure or
  analysis code, or the port recreates the color-literal drift the guidelines exist to stop.
- *`requirements.txt` gets regenerated, not copied.* The current file cannot build a
  working environment: four pins are wrong, `transformer_lens>=2.0` is incompatible with
  the vendored `circuits`, four hard deps are undeclared, and two pins have no importer.

**Decided with David (2026-08-03):**
- *Lift the shared layer out of `rules/`* — confirmed.
- *Back up netscratch before porting anything else*, as wave 0. ~92 GB on a 90-day clock
  that expires around 2026-10-20, holding the seeds behind every error bar.
- *Exclude the SpaDE γ=1e-2 seed-42 checkpoint everywhere*, from one place, rather than
  keeping the source repo's split where one study excluded it and three others plus the
  public viewer kept reading it. Its "SpaDE is broken for cell-state mapping" conclusion in
  the geometry family therefore has to be re-run or retired before it is quoted again.
- *The default SpaDE configuration is λ=1e-3, γ=1e-4* (`checkpoints/spade_revival/`),
  replacing γ=1e-2. It takes the excluded checkpoint's slot in the viewer roster, which
  stays at eight. Noted for the record: this pair was given as "γ=1e-3, λ=1e-4"; it is
  assigned here the way the existing named checkpoint assigns it, and is a one-line flip if
  the literal reading was intended.
- *CosineSpaDE's default stays γ=1.0*, against the γ-sweep's own recommendation of γ=20.
  The sweep selected 20 on reconstruction F₁, the one metric with an interior maximum
  (0.547 vs 0.530). Coverage runs the other way and is the deciding metric here: 0.537 at
  γ=1.0 against 0.474 at γ=20. The rest of the trade is L0 173 → 98 and loss recovered
  0.991 → 0.984. Worth knowing that the F₁ peak is thin — 0.011 over γ=10, with seed
  dispersion reported for L0 only (n=5) — so the metric that picked 20 is also the one
  whose margin is unquantified.
- *Put a freeze notice in the source repo* so a session that starts there is redirected.

**Open:** thirteen questions remain in MIGRATION.md — the report-file locations, several
data-retention calls, whether the geometry family gets re-run, the one-control divergence
sub-studies, the tracked-figure policy, and the viewer's deployment.

**Status:** plan updated and pushed. No code ported yet.

---

## New repository architecture skeleton (2026-07-22)

**Goal:** clean-slate layout for the SpaDE / OthelloGPT SAE project, built in this
sibling worktree (`spade_sae`, orphan branch `new-architecture`) because the original
repo's top level had grown ~13 parallel artifact/snapshot dirs. The original `src/`
was already clean, so the churn is top-level artifact consolidation + light code
renames.

This is a **skeleton**: every `.py` is a header comment + pseudocode. Real logic ports
over from `../Spade_Research_David_Sumedh_Ba` incrementally. The two worktrees share one
`.git`, so content can be diffed and cherry-picked between them.

**Decisions (user-confirmed):**
- *Installable package.* `src/spade/` is a real package (`pip install -e .` via
  `pyproject.toml`), replacing the original's sys.path-injection hack; `import spade.*`
  resolves everywhere and `make setup` installs it.
- *One training entrypoint per SAE type.* Shared loop lives in `training/core.py`; each
  `train_<type>.py` hands its type's config to `core`. Keep shared logic in `core`, not
  copied into each entrypoint, or the files drift (this trades some DRY for per-type
  clarity — the tradeoff was chosen deliberately).
- *Analysis code + outputs co-located.* Each `analysis/<study>/` keeps its `make_figures.py`
  next to its figures + results, so a study is self-contained.
- *Consolidated artifacts.* 4 `autoencoders*/` → `checkpoints/<experiment>/`;
  `results/karvonen*/` → `results/<experiment>/`; one deduped `figures/`.

**Status:** skeleton committed (5ac74a5). Superseded by the 2026-08-03 entry above, which
keeps all four decisions and corrects the layout they were applied to.

---

## Invariants carried over from the source repo

Moved to [ARCHITECTURE.md](ARCHITECTURE.md) § *Invariants a port must not break* — dispatch
ordering, checkpoint state-dict contracts, hyperparameter canonicalization, the known-stale
configs, the figure-style contract, and the environment pins.

## Legality-probe figures against the real CSVs (F1-F4)

The four figures were written against synthetic data and then re-cut when
`analysis/legality_probe/results/` landed. Three things changed, and each was a
substantive finding rather than a plotting fix.

**The two untrained controls are not one control.** `random_topk` has L0 80, exactly
TopK's k; `random_proj` is a ReLU of a dense 512 -> 8192 projection at L0 4101. Drawn as
one gray dashed pair they imply the study has a null, which it does not: `random_proj`
reads legality at 0.904, above three trained dictionaries, and takes the smallest hit of
any space on `drop_agreement_with_raw`. `paper_style` now carries `CEILING` beside `NULL`,
`CONTROL_SPACES = [MATCHED_NULL_SPACE, CEILING_SPACE]` replaces `NULL_SPACES`, and F1 makes
the distinction the panel's subject.

**F3 moved from r = 0 to r = -1.** At r = 0 the illegal move has been played and the square
is occupied, so R3's ground truth becomes R1: correct-rule recall is 0.000-0.007 in all
eleven spaces including raw. Degenerate in the data, not in the probe. r = -1 is the
bitwise-shared pre-move board, which also makes the panel a read-out property of the space
rather than an effect of the violation, and it is the right phase of the mover-relative
frame -- R3 is near zero at every even offset by construction.

**`excess_macro_f1` is not plotted.** It is not scale-free: SpaDE's is negative only
because its r < 0 baseline is already 0.42. It is quoted in the README exactly where the
baselines are comparable (trained dictionaries against `random_proj`, whose baseline sits
inside their range) and nowhere else. `drop_agreement_with_raw` has a milder version of the
same problem, which is why F4 reads its bottom row against its top row. No normalisation
was invented.

The finding the re-cut surfaced: on the paired difference from raw, every trained
dictionary steps down 0.023-0.057 at r = 0 and recovers by r = 1, and neither untrained
control has a step at all. That is SAEs failing off-distribution, not rules being lost, and
it is what F2 is now drawn to show.

Read every CSV in this study with `keep_default_na=False`; `category` and `rule` are empty
strings on the aggregate rows and NaN drops every aggregate silently.

## The violation-divergence figures against the rollout branch set

`rollout_branches.py` regenerates every branch's continuation from the model, the
reference included, so the shared suffix that made `control_legal` the *largest*
perturbation in the study is gone. The four quantitative surfaces —
`violation_divergence/make_figures.py`, `sae_spaces/`, `behavioral/`,
`probe_readout/` — now take `--variant {spliced,rollout}` and write `*_rollout`
beside today's names. Nothing on the default path changed: all 21 spliced figures and
CSVs re-render byte-identically, checked with `cmp` before and after.

**The headline flips sign and stops meaning anything, in that order.**
`ratio_illegal_control_L5` goes 0.9797 [0.9715, 0.9871] spliced to 1.0016 [1.0002,
1.0033] rollout, same estimator. So the "an illegal move displaces layer 5 *less*"
ordering was the confound. But the rollout plateau is 1.23 against a measured
unrelated-activation ceiling of 1.31, the ratio is 0.996 at r = 8 and 1.004 at
r = 24, and both classes are still climbing — the interval excludes 1 because it is
tight, not because the effect is there. `divergence_layer5_rollout` shades r >= 8 and
draws the ceiling so the figure cannot be read the other way.

**r = 0 is not new evidence.** delta-hat(0) depends only on `tokens[:t0+1]`, which the
rollout copies verbatim, so the two trees agree to 1.1e-5 there (fp16 batch noise).
The r = 0 ratio is 1.0348 in *both*. Anything the rollout reports at r = 0 — SAE
feature recurrence, `p_illegal_disc_decoded` — is the spliced number again. The
spliced 0.979 was always a plateau statement.

**Two constructions break in the rollout tree and the figures say so.** The token
embedding is no longer silent after t0 (plateau delta-hat 0 -> 1.233), so F3's
"computed, not inherited" anchor does not exist there; and the reference's board is
itself sampled, so it stops existing at the model's first illegal move — 4.2% of
reference rollouts inside the window for F6, 1.1% of window positions for the probe
read-out. Those positions are NaN and dropped, never scored as zero. Never fake a
panel: F6's legal-mass and KL curves collapse onto one another in the rollout tree
(0.289 vs 0.289, 7.44 vs 7.48 nats) and the panel titles were rewritten to say that
rather than keep "legality still tracks the true board" over a curve that falls to
0.29.

**K = 4 is a trap.** The rollout tree stores K rows per source branch
(`row = source_row * K + k`, contiguous), so anything that treats a row as an item
inflates n fourfold and halves every interval. Every per-branch score is averaged over
its K before the bootstrap (`reduce_k` / `per_branch` in each script, identity at
K = 1 so the spliced path is bit-for-bit untouched); the pooled probe anchor collapses
categories and rollouts in one `per_game`; the game-level bootstraps were already
correct because a game's K rows resample together. Point estimates are unaffected —
K is balanced — so only n and the intervals were ever at risk.

`probe_readout/decode.py` now loads the layer-5 activations with `mmap=True`: the
rollout tensor is 4.2 GB and a login session's cgroup caps at 8 GiB.

Two things left open. `make_figures.py`'s `illegal_pooled` is every non-control row,
which includes `control_legal_2`, while `sae_spaces` pools only the six illegal
categories; the two `ratio_illegal_control_L5` values differ by 0.4% and should be
pinned to one scope. And the rollout tree is temperature 1.0, seed 42, K = 4 only —
nothing here separates "the branch move does not matter" from "temperature-1 sampling
swamps it".

## A number under the fork panels: how far each branch runs, in the plane and in the space

`fork_and_board` and `game_trajectory` both draw a fork and both asked the reader to
eyeball how far each branch travels from the reference. Neither answered it. Both now
carry a panel (c) that does: per branch, the mean distance from the reference over the
drawn offsets r >= 0 (`branch_panels.plane_displacement`, already in `src`), measured
in the plane the fork panel draws.

**The number is a ratio, and it had to be.** A refit plane's units are its space's
own: the same control branch sits 2.80 from the reference in raw layer 5 and 0.0036 in
SpaDE's code, a factor of ~800 for the same game and the same move. Nine such numbers
side by side in one bar chart would be nine unit systems, and a bar chart invites
exactly the comparison that makes no sense. So (c) divides every branch's distance by
the same cell's `control_legal`, which is dimensionless, is a within-cell contrast by
construction, and is the form the rest of the study already argues in. The reference is
then 0 and the control 1 in every cell, both by construction, and the divisor is in the
axis label rather than the caption because a reader who misses it misreads the panel.

**Measuring the plane is not measuring the space, and the gap is large.** The plane is
the top two PCs of the branches it draws, so it is fit where the largest of them travel
and holds a different share of each: in raw layer 5 it carries 30% of the control's
displacement and 81% of `opponent_only`'s. The ratio read off the plane is inflated by
exactly the ratio of those shares — 1.53 x (0.810/0.303) = 4.08, an identity, not an
approximation. So every bar in (c) carries an open marker: the same ratio measured in
that cell's *full* space (the SAE's 8192-d code, or the column-scaled Δφ in R^21). The
gap between the two is the part of the plane's answer that is about the plane.

That pair is also what makes the across-cell reading defensible. Displacement, spliced,
R3: the bars run 3.81-6.18 over the nine cells and the markers 1.34-1.61. Nine times
the spread, and it is the plane fit moving rather than the spaces differing. The
READMEs say to read differences between cells off the markers.

Numbers worth keeping (game 1235, layer 5, mean over r = 0, 2, ..., 24, x the cell's
control, plane | full space): raw layer 5 R1 1.22|1.05, R2 2.43|1.20, R3 4.08|1.53.
`game_trajectory`'s (c) draws all nine branches including `control_legal_2`, and that
row is the one to read: a second legal move sits 1.31x the control's distance from the
reference in the full 512-d space — further than four of the six illegal branches.
Under `--variant rollout` everything collapses toward 1 and R1 falls *below* it in
every cell, which is the saturation `../fork_and_board/README.md` documents, as one
number per rule.

**One cell sets the y scale of the R^21 rollout panel and it is not an artifact.**
CosineSpaDE's control has the smallest Δφ of the nine cells (1.72 against 2.4-4.6)
while its illegal branches are among the largest, so its ratios reach 12.3 in the plane
and 5.1 in the space and the other eight compress against them. The axis is not capped
and no bar is broken — one linear axis from zero is what a bar chart owes the reader,
and a cap would have needed a threshold constant to hide one real cell. It is the
sharpest available illustration of the panel's own rule: these are within-cell numbers.

Layout: (c) is a third full-width row under the board in `fork_and_board` (the canvas
grows by its height plus one row gap, so the fork grid and the board stay within 0.4%
of their previous size) and a third column in `game_trajectory` (the canvas grows by
its width, so (a) and (b) are untouched). Nine cells x five series is 45 numbers; none
of it goes inside the fork panels.

Verification: all ten checked-in outputs were re-rendered before the change and
compared. The six PNGs came back byte-identical; the four PDFs differed in 6 bytes
each, the `/CreationDate` string, and were identical in size and content. So the fork
geometry the addition sits under is unchanged.

Left open: the metric carries no interval over games — n = 1, like the figures it sits
in — and the only spread on it is over a rule's member categories (n = 2, 3, 1). A
population version of "how far does a rule run, relative to a legal control" would
belong in `../README.md` with a bootstrap over games, and does not exist yet.

## Gray text: the parent violation-divergence figures (`gray/gray-parent`)

Every figure in `analysis/violation_divergence/` outside `rule_curvature/` and the
fork/board directories carried a multi-line muted note block under its suptitle. Most
of it was the README's caption pasted into the canvas. Removed, with the line each
figure still owes FIGURE_GUIDELINES kept and nothing else.

The keep test was a section number, not taste. What stayed: §1.2's estimator line (what
the interval is, over what unit, and where it lives), §1.4's axis-direction annotations
(`above 0: the violation displaces further than a legal move`, theta's
`← straighter / curvier →`), §3's chance and ceiling line labels, §4.2's
zero-by-construction and shaded-region notes, and §5's `S1`/`S3` key in
`game_selection`. What went: anything a reader could look up. The saturation band on
the rollout figures kept its shading and lost its four-line paragraph — the band is the
claim, the paragraph was the README's.

Two judgement calls. **The suptitles stay.** They are claims, not prose, they are what
§5 wants the caption's first sentence to be, and they already vanish under
`--camera-ready`. **The n / CI-source lines stay**, compressed to one line: §1.2 is not
satisfied by a README, and F1's curves carry no drawn band, so where the interval lives
has to be on the figure. In `behavioral/` that line stays a legend title rather than
moving to a `fig.text` — the legend is where its n already is.

Layout re-tune: `F1_NOTE`/`F3_NOTE` were anchored at `y=0.905` with `tight_layout`
reserving 14% of the canvas; one line needs `y=0.95` and `rect` top `0.93`. Same move in
`sae_spaces` (0.92) and `theta_comparability` (0.92). `probe_readout`'s rollout note and
rule key came up from `-0.45`/`-0.52` to `-0.36`/`-0.42`. Nothing changed size except
`theta_sampling_mismatch`, which shrank to its declared 9.0 in — the old note was wider
than the panels and `bbox_inches="tight"` had been stretching the canvas to fit it.

`probe_readout.parity_panel` and `PARITY_BRANCHES` deleted: `main()` has dropped that
panel since the cosine cut and never called it. `docs/finish_notes/bands-divergence.md`
had flagged it as known rot.

`sae_spaces/make_figures.py` has no `--camera-ready` flag, so its two figures ship as
PNG only. That predates this change and is the one figure pair in the study without a
vector twin.

One thing rides along that is not about text: re-running the scripts on the current
tree picks up the three appended rule-complete games, so every regenerated figure and
CSV moves from n = 2000 to n = 2003 per category. The committed artifacts predated
`extend_branches`. No number the READMEs quote changes past the fourth decimal
(`ratio_illegal_control_L5` 0.9797 -> 0.9798 spliced, 1.0016 unchanged on the rollout
tree), but the diff on the CSVs is larger than the diff on the figure text.
