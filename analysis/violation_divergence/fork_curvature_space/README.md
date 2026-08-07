# The fork in R^21: the branch grid drawn in ReStraV descriptor space

`../game_trajectory/space_fork_grid_pca` projects the displacement
`D(t) = z_b(t) − z_ref(t)` straight onto a PCA of `D`. This sub-study asks what
happens if the game is first mapped into **R^21** — the ReStraV curvature
descriptor of `src/spade/straightness/geometry.py`: 7 raw step lengths, 6 raw
turning angles, and the 4 nan-aware moments of each — and the plane is taken
*there*.

Eight branches are drawn — the played move, one legal control, and one move per
illegal category — and the six illegal categories are grouped into the three
constraint rules they violate, so each grid carries five series.

```
python analysis/violation_divergence/fork_curvature_space/make_figures.py [--game 1235] [--w 8] [--variant spliced|rollout] [--camera-ready]
python analysis/violation_divergence/fork_curvature_space/measure.py [--n_games 40] [--w 8] [--game 1235]
```

`--variant rollout` redraws all four grids on the branch set whose suffixes are
sampled from the model, into `*_rollout` files; see "The rollout variant" below.
`measure.py` is spliced-only.

The plane itself lives in `src/spade/violation_divergence/curvature_plane.py`,
beside `fork_plane.py`, which owns the displacement plane and the sampling window
both of them share. It started here as `descriptor.py` and moved down when
`../fork_and_board` needed the same plane: two surfaces, one rule.

**Verdict: the projection works and barely outperforms the plane it replaces.**
The fork is drawable in R^21 at full width with every invariant preserved
exactly, but on illegal-vs-legal-control separation it scores 1.26–1.27x its own
permutation null against the incumbent activation-PCA plane's 1.19x — a paired
gap of +0.08 [+0.01, +0.15] over 40 games x 9 spaces. Details below; every number
is `measure.py` output.

## The dimensional accounting, which is the whole constraint

The descriptor is a property of a *whole trajectory*; the fork is a *curve over
move index*. A window of `w` sampled positions gives `w−1` step lengths and `w−2`
angles, so filling `d1..d7` and `θ1..θ6` needs **w ≥ 8**, i.e. 14 plies of
span at `STRIDE = 2`. Below that, `restrav_features` slices `d[:, :7]` and `theta[:, :6]`
and those slices silently return *fewer columns* — a shorter vector, no error,
not R^21. `descriptor()` asserts the width and falls back to the named 8-D moment
block instead of shipping a truncated one.

Inside the figure's analysis window `r ≥ −8` there are only 4 pre-branch samples
whatever the game, so no full-width window there is also branch-free. Reading
context back to the **start of the game** dissolves that: at game 1235's
`t0 = 20` the pre-branch samples are 2, 4, …, 20, and the window ending at
`r = −2` spans positions 4–18, entirely prefix.

At **w = 8**, game 1235: **21 of 21 coordinates populated**, **15 of the 17
plotted offsets defined** (`r = −4 … 24`; `r = −8, −6` run off the board), **2 of
them branch-free**, and no component NaN in any of the nine spaces. The cost is
smear: the descriptor at offset `r` reads the window `r−14 .. r`, so the branch
enters 1 of the window's 7 steps at `r = 0` and does not fill it until `r = 12`.
Measured, the median offset at which `‖Δφ‖` passes half its maximum is **2 plies**
in seven spaces and 3 in the other two — the branch step dominates its window
immediately, so the smear is much milder than the worst case.

## Trajectory, not displacement

The descriptor goes on the branch trajectory `z_b(t)`, and the figures then
difference **in R^21**: `Δφ(t) = φ(z_b(t)) − φ(z_ref(t))`.

Putting it on `D` instead does not work, and the reason is structural rather
than a tuning problem. `D` is *exactly zero* over the shared prefix, so its steps
have zero length and its angles are undefined; and for the reference branch
`D ≡ 0` at every position, so **11 of its 21 components are NaN** (`moment4` also
sends min/max to ±inf on an all-NaN window). The reference has no descriptor at
all — it cannot anchor the figure, and it is the class everything is scored
against.

| fraction undefined | value |
|---|---|
| θ on the trajectory `z_b` | **0.000** |
| θ on the displacement `D` | 0.467 |
| θ on `D`, at `r ≥ 0` only | 0.289 |
| the whole `φ(D)` field | 0.276 |
| `φ(D)` for the **reference** branch | **0.538** |

Identical in all nine spaces, because the cause is the prefix-zero construction
and not sparsity: on the trajectory `z_b` the NaN fraction is **0.000 in every
space**, the sparsest dictionary included. Two sampled positions at stride 2
never coincide, so the zero-step worry does not materialise here at all.

Differencing in R^21 keeps both invariants the incumbent figure is built on, and
they are verified exactly: `max|Δφ| = 0.00e+00` at every `r < 0` (the prefix
collapses onto the origin) and `max|Δφ_ref| = 0.00e+00` everywhere (the reference
*is* the vertical line at the origin).

## The four figures

| file | variant | plane |
|---|---|---|
| `fork_r21_windowed_pca` | Δφ(t) over the w=8 window, curve over move index | top-2 uncentred PCs of the column-scaled Δφ, refit per space |
| `fork_r21_windowed_fixed` | same | `Δ mean d` × `Δ mean θ`, the same named R^21 axes in every cell |
| `fork_r21_whole_pca` | one R^21 point per branch over `r = 0..24` | top-2 PCs of the z-scored φ, refit per space |
| `fork_r21_whole_fixed` | same | `mean d` × `mean θ` |

Each has a `_rollout` twin drawn from the rollout branch set on the same game,
same planes, same styling. The windowed twins stop at `t − t0 = 8`; the whole
twins keep `r = 0…24` because R^21 has no shorter form. "The rollout variant"
below is their caption.

The two **windowed** grids end in a read-out strip — the number the 3-D panels
make the reader guess at. The two **whole** grids do not, and "How far each
branch runs from the reference" below says why.

## Captions

The grids carry one line of prose: that game 1235 was selected. Everything else a
reader needs in order not to over-read them is here (FIGURE_GUIDELINES §2, §5);
what stays inside the figure is the apparatus that cannot be looked up while
looking at it — per-panel space names, axis labels, the legend (which carries n
per rule and what the ellipses are), the R1/R2/R3 key naming each rule's member
categories, and that one selection line. The selection line is in-figure and the
rest is not for the same reason: a reader who does not know the game was chosen
misreads the panels in front of them, while one who does not know where the mean
over K rollouts is taken reads them correctly and can look the rest up here.

**Common to all four.** Game **1235**, *t0* = 20, eight branches: the played move,
one legal control, and one move per illegal category, the six of them collapsed
onto the three rules they violate, with each rule's members drawn faint behind
their mean and every legend entry carrying n — including `R3 turn (mean of 1)`,
which is one branch and not an average. **φ is nonlinear, so it is applied to each
member's own trajectory and the rule means are taken afterwards, in the plane**;
the projection is linear (windowed, uncentred) or affine (whole, centred), so
averaging in the plane and averaging in R^21 agree exactly, while φ of an averaged
trajectory would be the curvature of a path no branch takes. Scaling is
column-wise, because the 21 components mix degrees with activation-scale lengths.

**Game 1235 is selected, not typical**: the argmax, over all 1550 branch-complete
games, of the mutual separation of its six illegal displacement trajectories,
scored in raw layer 5 in the full 512-dimensional space and never in an SAE space
or a 2-D projection (`../game_selection`). Population median 0.705; this game
1.080. A reader who does not know the game was chosen will over-read the panels —
see "One finding that did not survive" below for exactly that failure.

**One game (n = 1) and no error bars over games**: an illustration of the
construction, not a claim. The separation numbers, with bootstrap CIs over 40
games and an exact permutation null, are in the table below and come from
`measure.py`. The rule-spread ratios quoted per figure are themselves n = 1.

**One legal control is drawn, as the null.** Its own null — control 2, scored
against control 1 exactly as the illegal branches are — is not drawn here; it is
in `../game_trajectory` and in `../README.md`, which is where the legal-vs-legal
scale lives. It leaves the plane fit too, since the plane is fit on what is drawn.

### `fork_r21_windowed_pca`

**The fork drawn inside the 21-D ReStraV curvature descriptor: one path per rule
through a fitted plane of Δφ, with offset from the branch on the third axis.**
`Δφ(t) = φ(z_b(t)) − φ(z_ref(t))` over the **w = 8 sampled positions (14 plies)
ending at *t***, every 2nd ply, view fixed at `elev = 18, azim = −58`. The plane
is the top-2 uncentred PCs of the column-scaled Δφ, **refit per space**, so the
axes are arb. units and the three ticks per plane axis are a within-panel scale
bar only: **cells are not comparable across the grid**, only in the arrangement
of the classes within a cell. The grey stub below each fork is the shared prefix,
where the window runs off the board and no descriptor exists.

**The rule grouping is weak.** Within-rule ÷ between-rule mean pairwise distance
on the column-scaled Δφ at *t* ≥ *t0*, in full R^21 in raw layer 5 and never on
the 2-D plane drawn: **0.90** here, against **0.885** for the same statistic on
the full 512-d displacements of the same game and window (`../fork_and_board`) and
a displacement population of **median 0.945 [IQR 0.872–1.020] over all 1550
branch-complete games, above 1 in 30% of them**
(`python -m spade.violation_divergence.rule_groups`). R^21 has its own population:
**median 0.936 [IQR 0.827–0.998], above 1 in 25%** over 60 games with *t0* ≥ 18
(`--field r21`; the sweep needs a branch-free window, so it is a different and
smaller game set, not a subset comparison). The two fields are indistinguishable
at population scale — the single-game pair orders them the other way from the
medians, so neither ordering means anything. A rule's members travel barely closer
to each other than to another rule's, so the faint members and not the three means
carry the evidence.

The strip below the nine cells is the quantitative companion a 3-D projection
owes the reader (FIGURE_GUIDELINES §4.8): **how far each rule's mean path runs
from the reference, averaged over the 13 drawn offsets `r = 0…24`, divided by the
legal control's run in the same field**, on a log axis whose 1.0 line is the
control itself. Filled marker and line = the plane the cell draws; open marker =
the same statistic in **full R^21**, with a bar spanning its 11 step-length and
10 turning-angle components. Faint marks are the rule's member categories, the
only spread one game supports. The reference is 0 in every field by construction
and is not a point on the strip.

### `fork_r21_windowed_fixed`

**The same fork against a fixed pair of named R^21 axes instead of a fitted
plane**, so the arrangement is the same object in all nine cells: `Δ mean d
(activation norm)` × `Δ mean θ (deg, ↑ curvier)`. **θ in degrees is comparable
across cells; `mean d` is on each space's own activation scale and is not** —
SpaDE codes live on a simplex — so only its sign and the within-cell ordering
mean anything across the grid. Everything else, the spread ratio included, reads
as for `fork_r21_windowed_pca`.

Its strip differs in one respect. The two drawn axes carry *different units*, so
a Euclidean distance in the drawn coordinates would be a statistic about θ alone
(θ is in degrees; `mean d` is 0.65 in raw layer 5 and 0.001 in SpaDE). The
read-out therefore measures the fixed pair **after each axis is put in its own
column SD** — the same scaling `fit_plane` and `measure.py` apply, and the same
scaling the panel's independent x/y autoscale is already showing. The number is
the distance in the *scaled* fixed plane, not the millimetres between two drawn
marks.

### `fork_r21_whole_pca`

**One R^21 point per branch over its whole post-branch trajectory, with the only
uncertainty a single game supports.** φ over the 13 sampled positions from the
branch to *t0* + 24; the plane is the top-2 PCs of the centred, column-scaled φ,
**refit per space** (arb. units, cells comparable in arrangement only). The
ellipse around each point is the **50% contour of a Gaussian fit to its 13
leave-one-position-out jackknife replicates — a dispersion over which move enters
the descriptor, not a confidence interval**; for a rule it is the jackknife of the
rule mean.

**No read-out strip here, on purpose.** See "How far each branch runs from the
reference" below: with no move axis, "averaged over `r ≥ 0`" is what the point
already *is*, and the in-plane distance from the reference is the length of a
segment the reader can measure off the ticks.

**The rule grouping is gone in this variant.** The same within ÷ between statistic
on φ, one point per branch, is **1.03**: above 1 means a rule's members sit
*further* from each other than from another rule's, so the rule label carries no
information about where a branch lands. The panels show it directly — the three
`R2` members are scattered across the cell with their mean in the middle of
nothing. Displacement context as above: 0.885 on this game, population median
0.945 [IQR 0.872–1.020], above 1 in 30% of 1550 games; no R^21 population exists.

### `fork_r21_whole_fixed`

**The same one-point-per-branch view against the fixed named pair**, `mean d
(activation norm)` × `mean θ (deg, ↑ curvier)`, with the same jackknife ellipses.
Cells share the axes: **θ is comparable across them, `mean d` is not**. These
cells keep ordinary ticks rather than the symmetric three, because they are
scatters of physical quantities and a symmetric locator would leave a panel
spanning `mean d ∈ [18, 21]` with no tick at all. Spread ratio 1.03 as above, and
this is the panel "One finding that did not survive" is about: read it with the
selection of game 1235 in hand.

Scaling follows `analysis/straightness/feature_pca`: the PCA is fit on
column-scaled features, because the 21 components mix degrees with
activation-scale lengths whose absolute variances differ ~450× across
dictionaries, and a raw covariance PCA would report units rather than structure.
Centring differs by variant — `Δφ` is uncentred, since zero displacement is a
real point exactly as in the incumbent figure, while absolute `φ` has no origin
to preserve and is centred.

The whole variant carries the only uncertainty one game supports: a **leave-one-
position-out jackknife**, 13 replicates per branch — and, for a rule, 13
replicates of the rule mean — drawn as a 50% Gaussian contour. It is a dispersion
over which move enters the descriptor, not a confidence interval, and the legend
entry says exactly that. The windowed grids are n = 1 with no error bars; the
separation numbers with CIs over 40 games are below.

## How far each branch runs from the reference

A fork panel asks the reader to judge how far a branch travels from the vertical
line at the origin, and until now nothing answered it. The windowed grids now
carry the answer as a strip:

> mean over `r ≥ 0` of `‖coords_b(r)‖` in the plane the cell draws, for the
> **mean path** each rule is drawn as, **divided by the legal control's** value
> in the same field.

Two decisions are load-bearing.

**It is a ratio, and it has to be.** The plane is refit per cell and holds only
part of Δφ — 44% + 18% on raw layer 5's windowed PCA — so a raw in-plane distance
is in units nine different fits do not share. Nine panels side by side *will* be
compared, so the alternative to a disclaimer nobody reads is a quantity for which
the comparison is defensible. Dividing by the control's run in the same field
cancels the cell's units and its arbitrary axis scale, and leaves a question with
a meaning: **does this branch run further from the game as played than another
legal move does?** 1.0 is exactly "no", and the strip draws it as the null line.
What is still not comparable is absolute magnitude; nothing on the strip claims
it, and the axis label says "÷ the control's".

**The rule's number is the run of the mean path, not the mean of its members'
runs.** The panel draws the mean path, so the strip measures the mean path; where
the members leave in different directions the two differ, and the faint member
marks show by how much. `R3 turn` is one category, so for it there is nothing to
average and the legend says `(mean of 1)`.

### The plane exaggerates, and the two halves of R^21 disagree

Game 1235, spliced, PCA plane, 13 offsets `r = 0…24`, raw layer 5, with the
full-R^21 value in brackets:

| rule | in the drawn plane | full R^21 | 11 step-length | 10 turning-angle |
|---|---|---|---|---|
| R1 placement (mean of 2) | 0.91 | 1.04 | 0.78 | 1.34 |
| R2 bracketing (mean of 3) | 1.69 | 1.20 | 0.98 | 1.50 |
| R3 turn (mean of 1) | 2.93 | 2.15 | 2.05 | 2.33 |

Over the nine spaces the plane spans R1 0.49–1.32, R2 1.13–2.17, R3 2.24–3.94,
against full R^21's much narrower R1 0.77–1.19, R2 0.89–1.30, R3 1.69–2.47.
**Plane ÷ full R^21 runs 0.60–2.12 with a median of 1.31** (1.27 on the fixed
plane): the plane stretches the branch that is already far and shrinks the one
that is close, so the panel's visual ordering is real but its visual *spacing* is
the fit, not the geometry. That is the whole reason the open R^21 marker sits
beside the filled one instead of in this file only.

**Which of R^21's two halves the plane's number follows is a coin flip.** The 11
step-length and 10 turning-angle components disagree about the same branch —
R2 in raw layer 5 is 0.98 on `d` and 1.50 on θ — and the in-plane number lands
nearer the θ block in **14 of 27** rule × space cells on the PCA plane and 17 of
27 on the fixed one. It is not that PC1 loads on θ and drags the number with it;
it is that a single in-plane scalar averages over an opposition and which side it
lands on is not stable. The bar on the strip is that opposition drawn, so a
reader cannot take the plane's number for a statement about step length or about
curvature.

**Which block leads also flips with the offset range.** Measured the other way
round — the mean over the *six illegal branches* of their own runs, rather than
the run of their mean path — game 1235 in raw layer 5 gives illegal ÷ control of
**1.11 on the 11 step-length components and 0.92 on the 10 turning-angle ones at
`r = 0` alone**, and **1.21 vs 1.80** averaged over `r = 0…24`: the block that
sits above the control swaps. Nothing in this tree computes a population version
of that; a figure-brief for this work quotes 1.312 [1.224, 1.405] and 0.931
[0.869, 0.992] for the same `r = 0` contrast with a bootstrap over games, whose
signs the single game reproduces but whose interval this repo cannot regenerate
— `measure.py` scores separation, not run length. Treat the population numbers as
external until something here recomputes them.

### Where the read-out is refused

- **The two whole grids get no strip.** They draw one R^21 point per branch over
  `r = 0…24`, so the average over `r ≥ 0` is not a summary of the panel — it *is*
  the panel, and the in-plane distance from the reference is a segment the
  reader can measure off the axes. Printing it would also invite the comparison
  it must not support: φ of one long trajectory and a mean over 13 short windows
  are different quantities, and side by side on the same page they would read as
  one.
- **The strip carries no error bar over games.** One game, no CI; the spread it
  shows is the rule's member categories. The 40-game bootstrap with an exact
  permutation null is `measure.py`'s, on the separation statistic, below.
- **There is no legal-vs-legal scale here**, because this study draws one control
  and the strip's denominator *is* that control. `../r21_shared_basis`'s
  `run_from_reference` draws both controls and carries "a third legal move ÷ the
  control" as an empirical null; read the two together.

### Under rollout

Same strip, over the 5 drawn offsets `r = 0…8` — the window the rollout grids
actually draw, past which two rollouts of one branch differ by more than the
branch differs from the reference. Raw layer 5, PCA plane: R1 **0.31** (0.70 in
full R^21), R2 **1.07** (0.89), R3 **2.43** (2.19). Plane ÷ full R^21 is
0.32–2.65, median 1.17. R1 drops to a third of the control's run in the plane and
to 0.70 in full R^21, where the spliced strip has it at 0.91 and 1.04 — the gap
between those two is the plane again, and the drop is a fact about *this* branch
set: when every branch including the reference resamples its own future, a
resampled legal control is itself a long way from the reference, so the
denominator grows. Read it with "The rollout variant" below; every offset here is
inside the region where the branch still outweighs the sampler, but the branch
set answers a different question from the spliced one.

## The rollout variant

Every branch in the spliced set replays `tokens[t0+1:]` from the source game.
That suffix is a legal continuation of the **reference's** board and of no other
— a legal control's survives a median of 2 moves and an illegal branch has no
board for it at all — so every displacement above was measured against the one
branch guaranteed an in-distribution future.
`spade.violation_divergence.rollout_branches` resamples every branch's suffix
from OthelloGPT (temperature 1.0, K = 4, seed 42), the reference included.
`--variant rollout` draws the same four grids from it.

**Where the mean over K is taken, and why it is not negotiable.** φ is nonlinear:
`mean_k φ(z_k) ≠ φ(mean_k z_k)`, and the right-hand side is the curvature of a
path no rollout took. Every rollout row gets its own `windowed` / `descriptor` /
jackknife and the K descriptors are averaged afterwards (`phi_mean`). The plane
projection and the difference from the reference are linear, so they commute with
that average; nothing else in the pipeline does. This is the figures' caption and
lives only here: it is a fact about how the numbers were made, not about how the
panels read, so it does not belong on the page a reader is looking at.

**The prefix survives the construction — checked, not assumed.** `tokens[:t0+1]`
is copied from the source branch, so the activations before *t0* should still be
bit-identical across branches *and* across rollouts, and the R^21 window at
offset `r < 0` (which covers `r−14 … r`, ending at or before `t0 − 2`) should
still collapse. It does: **`max|Δφ| = 0.00e+00` at every `r < 0` in all nine
spaces**, and `max|Δφ_ref| = 0.00e+00` everywhere, exactly as on the spliced set.
Two of the four pre-branch offsets are defined (`r = −4, −2`); `r = −8, −6` run
off the board, as before.

### Where the rollout curve stops being about the branch

K = 4 buys the one diagnostic the spliced set cannot have: how far two rollouts
of the *same* branch drift apart, against how far the branch drifts from the
reference. On column-scaled R^21, game 1235:

`‖mean_k φ_b − mean_k φ_ref‖` (over the six illegal branches) ÷ mean pairwise
`‖φ_k − φ_k'‖` within a branch, per offset:

| `t − t0` | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw layer 5 | — | 1.43 | 1.04 | 1.01 | 1.02 | 0.91 | 0.87 | 0.67 | 0.66 | 0.66 | 0.61 | 0.56 | 0.56 |
| median over 9 spaces | — | 1.43 | 1.04 | 0.97 | **0.93** | 0.84 | 0.77 | 0.62 | 0.62 | 0.60 | 0.58 | 0.54 | 0.53 |
| spaces with ratio ≥ 1 | 0/9 | 8/9 | 5/9 | 2/9 | 3/9 | 1/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |

At `r = 0` the within-branch spread is **exactly zero** — the trailing window
`−14 … 0` ends at *t0*, and the rollouts only differ after it — so `r = 0` is
pure branch effect with no sampling noise at all. From `r = 12` no space has a
ratio at 1. The windowed grids are therefore **drawn to `t − t0 = 8`**, the last
offset whose median is within a rounding of 1; past it the curve is the sampler,
not the branch. This is the R^21 reading of the divergence-level caveat below,
and the two agree on the same cut.

**The saturation caveat, from the divergence read-out.** On the smoke rollout set
`delta-hat` climbs monotonically and both illegal and control saturate near 1.23
against a ~1.41 unrelated-activations ceiling; the illegal/control ratio pins to
1.00 by `r ≥ 8` and only `r = 0` still separates (**1.032**, sign opposite the
spliced **0.979**). **Long-horizon rollout geometry is not informative.** The
whole grids cannot obey that: R^21 needs 8 sampled positions and the jackknife
needs 9, i.e. `r` out to 16 at the very least, so one point per branch over the
post-branch trajectory is *by construction* a summary of the saturated region.
They are drawn over the same `r = 0…24` as the spliced twins so the two are
comparable, and their ellipses are changed to say it (below).

**Neither figure restates this caveat in gray.** It does not have to. The
windowed grids cut the axis at `r = 8`, so the region the caveat is about is not
drawn at all — the window *is* the caveat. The whole grids cannot cut it, but
they draw the sampling spread as an ellipse the legend names, and in every space
that ellipse swallows the reference and every branch mean inside it. A reader
looking at either figure sees the thing a sentence would have asserted, so the
sentence is here and not on the page.

### What changes on the rollout figures

- **The ellipses on the whole grids are the K = 4 rollout descriptors, not the
  jackknife.** On the spliced set there is one row per branch and the only
  uncertainty a single game supports is leave-one-position-out; on the rollout
  set the sampling spread exists and is the thing a reader must see. The legend
  says `50% Gaussian contour of 4 rollout descriptors (sampling spread, not a
  CI)`. It is a 4-point Gaussian fit and thin evidence on its own — but in
  `fork_r21_whole_fixed_rollout` the **reference's own ellipse swallows most of
  the panel and every branch mean inside it**, in all nine spaces, which is the
  honest picture of what resampling costs.
- The windowed grids stop at `r = 8`, and their **read-out strip averages over
  exactly the 5 offsets drawn** — `r = 0, 2, 4, 6, 8` — which its legend title
  states, so the strip can never summarise a window the panel above it does not
  show. Everything else — rule grouping with faint members, one control, three
  symmetric ticks on the windowed planes, `PC1 (NN% var)` and nothing else, named
  units on the fixed planes — is unchanged.

### The rule-spread ratio under rollout, recomputed

Not carried over: the same within-rule ÷ between-rule statistic, on the
column-scaled full R^21 field in raw layer 5, game 1235, recomputed on the
rollout set.

| field | spliced | rollout, `r ≤ 24` | rollout, `r ≤ 8` (the drawn window) |
|---|---|---|---|
| `Δφ(t)`, windowed grids | 0.902 | **0.762** | **0.688** |
| `φ`, one point per branch (whole grids) | 1.034 | **0.818** | 0.818 |
| full 512-d displacement, same game and window | 0.885 | 0.930 | 0.887 |

**The rule grouping is stronger under rollout, and in the whole variant it comes
back from nothing.** 1.034 → 0.818 moves the one-point-per-branch statistic from
*above* 1 — a rule's members further from each other than from another rule's —
to clearly below it. Read it with the mechanism in hand rather than as a win for
the rules: the K = 4 average shrinks each branch's descriptor toward its group's,
and the paired common random numbers mean two branches of the same game share a
noise draw, so some of the tightening is the estimator and not the taxonomy. It
is n = 1 on a selected game with no CI, exactly as the spliced ratios are, and
`rule_groups.py` has no rollout population to put it against. Note also that the
displacement ratio moves the *other* way over the full window (0.885 → 0.930),
so the two fields disagree about the direction — one game cannot settle that.

## One trajectory per rule, and where the mean is taken

The six illegal categories are averaged into the three rules they violate
(`spade.othello.taxonomy.RULE_MEMBERS`): **R1 placement** = `occupied_own`,
`occupied_opp`; **R2 bracketing** = `empty_detached`, `empty_own_only`,
`empty_no_bracket`; **R3 turn** = `opponent_only`. Sizes 2, 3 and 1, so every
legend entry carries n — including **`R3 turn (mean of 1)`**, which is one branch
and not an average. Styling comes from the same `spade.figures.branch_style` the
sibling fork figures use, so a colour means the same rule across the study.

**φ is nonlinear, so it is applied per member and the means are taken after.**
mean_b φ(z_b) ≠ φ(mean_b z_b): the curvature of an averaged path is not the
average curvature, and averaging trajectories first would put a quantity on the
figure that no branch has. Each member gets its own descriptor, its own Δφ, its
own projection, and only then are the coordinates averaged — the projection is
linear (windowed, uncentred) or affine (whole, centred), so averaging in the
plane and averaging in R^21 agree exactly.

**The grouping is weak — weaker here than in the displacement plane, and for the
whole variant it is gone.** Within-rule ÷ between-rule mean pairwise distance, on
the column-scaled full R^21 field in raw layer 5 and never on the 2-D plane the
figure draws:

| field | ratio, game 1235 | over the nine spaces |
|---|---|---|
| `Δφ(t)`, `t ≥ t0` (windowed grids) | **0.90** | 0.77–1.01, median 0.85; above 1 in 1 of 9 |
| `φ`, one point per branch (whole grids) | **1.03** | 0.90–1.06, median 0.96; above 1 in 3 of 9 |
| full 512-d displacement, same game and window (`../fork_and_board`) | 0.885 | — |

Above 1 means a rule's members sit *further* from each other than from another
rule's. The windowed number, 0.90, is slightly worse than the 0.885 the
displacement figure reports on the same game and window — mapping into R^21 does
not make the rules cohere. The whole variant is at 1.03, i.e. the rule label
carries no information about where a branch lands, which the panels show directly:
in `fork_r21_whole_pca` the three `R2` members are scattered across the whole
cell with their mean in the middle of nothing. Both grids draw the members faint
behind their mean and their captions print the ratio, so the three confident
means cannot stand in for six trajectories that do not travel together. The
population context is the displacement statistic's — median 0.945 [IQR
0.872–1.020] over all 1550 branch-complete games, above 1 in 30% of them
(`python -m spade.violation_divergence.rule_groups`); there is no 40-game sweep
of the R^21 ratio, so the R^21 numbers above are n = 1 on a selected game.

## One control, and where the null went

`control_legal_2` is **not drawn**. One legal control remains, in the null
styling — grey, dashed, last in the legend. It leaves the plane fit too, since
the plane should be fit on what is drawn: PCA cells shift slightly against the
nine-branch version, and nothing in them was comparable across cells anyway.

The cost is not cosmetic. `branches.py` builds two legal controls because *one
gives no null*: control 2 is scored against control 1 exactly as the illegal
branches are, and the study's first pass shipped with one control and could not
interpret its own direction result (`../RED_TEAM.md` F1). Dropping it removes the
reader's visual answer to "is that separation big compared to two legal moves
differing?" — which matters more here than in `../fork_and_board`, because the
separation table below is scored against *both* controls. Nothing measured
changes: `branches.pt` still carries both rows and `measure.py` still uses 6
illegal vs 2 legal controls. The legal-vs-legal scale lives in
`../game_trajectory`, which draws all nine branches with per-category styling,
and in `../README.md`, which carries the numbers with CIs; the captions point at
both.

## Separation, against each plane's own permutation null

`‖mean_illegal − mean_control‖ / pooled within-group RMS spread`, 6 illegal vs 2
legal controls, averaged over `r = 0…24`. Raw size is not comparable across
planes, so each is scored against its own exact null — all 28 ways of calling two
of the eight branches the control group. **Ratio to the null mean; 1.0 = the plane
sees nothing.** Percentile bootstrap over n = 40 games, pooled over 9 spaces.

| plane | ratio to its own null | null percentile |
|---|---|---|
| `PCA(D)` 2-D — the incumbent figure's plane | 1.19 [1.13, 1.26] | 0.58 |
| full D, unprojected | 1.24 [1.21, 1.27] | 0.76 |
| **Δφ PCA 2-D** — fitted plane in R^21 | **1.26 [1.21, 1.33]** | 0.61 |
| **Δφ fixed** — `mean d` × `mean θ` | **1.27 [1.21, 1.33]** | 0.61 |
| full Δφ, all 21 | 1.28 [1.24, 1.33] | 0.69 |
| **φ PCA 2-D** — whole variant, fitted | **1.43 [1.35, 1.51]** | 0.69 |
| φ fixed — whole variant | 1.30 [1.22, 1.38] | 0.63 |
| full φ, all 21 | 1.25 [1.22, 1.28] | 0.75 |

Paired within game and space, `Δφ fixed − PCA(D)` = **+0.08 [+0.01, +0.15]**. The
interval excludes zero, so R^21 is genuinely better — by about 8% of a null,
which is not a difference a reader would see in a panel. Per space the ordering is
stable and the between-dictionary spread is smaller than the CI; raw layer 5 is
not distinguishable from any SAE space here.

Everything sits between 1.19 and 1.43 against a null of 1.0, R^21 included. That
is consistent with the parent study's own headline (`ratio_illegal_control_L5`
= 0.979 — an illegal move displaces layer 5 slightly *less* than a legal
alternative): there is not much signal for any plane to find.

**One finding that did not survive.** In the drawn game, `fork_r21_whole_fixed`
puts the reference and the legal control at low `mean d` and low `mean θ` with
all six illegal branches up and to the right, in all nine spaces at once (the
panel draws one control; the number below scores both). Over 40
games, the fraction in which both controls fall below all six illegal branches is
**0.07–0.17 on `mean d`** and **0.03–0.15 on `mean θ`** against an exact chance of
**0.036** — elevated, but nowhere near what the single panel suggests. The panel
is an n = 1 illustration and behaves like one.

## What the fixed plane buys, and what it costs

A refit PCA cannot be compared cell to cell — the incumbent grid states this and
so does this one. The fixed pair can: `mean θ` is in degrees everywhere, and the
arrangement is the same object in all nine panels. `mean d` is on each space's
own activation scale (SpaDE codes live on a simplex) and is **not** comparable
across cells; only its sign and the within-cell ordering are.

## Axes and ticks

- Windowed cells: the two plane axes carry **three ticks, symmetric about zero**
  (`branch_panels.PLANE_TICKS`, shared with `../fork_and_board`) so the origin —
  where the reference sits, since the figure draws a difference from it — always
  gets one. They are a within-panel scale bar and nothing more, which is why the
  caption's "not comparable across the grid" disclaimer stays and matters
  more, not less (§4.6, §4.8). The ordered axis reads `t − t0 (plies)`, labelled
  once per row.
- PCA plane axes read `PC1 (44% var)` — **per component, never summed**, per panel
  since the plane is refit per space, and **nothing else**. They used to append
  the descriptor component the PC leads on; they no longer do, because a PC here
  is a broad mixture and naming one component asserted something false. In raw
  layer 5 the windowed plane's PC1 loads 0.360 on `mean θ` — 13% of the direction
  — over a near-tied, same-signed `mean d` 0.328 and `max θ` 0.323, and 16 of 21
  coordinates are needed to reach 90% of it. Which coordinate leads flips with the
  variant: the whole plane's PC1, same game and space, leads on `min d` at 0.290,
  8% of the direction.
- Fixed plane axes name the quantity, its unit and its direction:
  `Δ mean d (activation norm)` and `Δ mean θ (deg, ↑ curvier)` on the windowed
  grids, without the Δ on the whole grids.
- Whole cells are 2-D scatters of physical quantities, so they keep ordinary
  ticks: a symmetric locator would put the ticks at ±(max |x|) and leave a panel
  spanning `mean d ∈ [18, 21]` with none at all.

## Caveats

- The game drawn is **1235, t0 = 20**, shared with the sibling fork studies.
  `../game_trajectory`'s game 0 (t0 = 10) cannot be used at all here: it has zero
  sampled positions of prefix context, so no full-width window is branch-free.
- Everything drawn is one game. Every quantitative statement above carries a
  bootstrap CI over 40 games and an exact permutation null, from `measure.py` —
  except the rule-spread ratios, which are n = 1.
- **Three lines are six categories.** The rule means are a labelling convenience
  and, on this evidence, not much more: at 0.90 and 1.03 the members barely
  cohere, and the whole variant's ratio is above 1. Read the faint members. A
  claim about a rule has to be scored per category and aggregated with a CI.
- **Only one control is drawn.** The legal-vs-legal scale is in
  `../game_trajectory` and `../README.md`; the figures name them rather than
  showing one.
- `moment4` returns ±inf, not NaN, for an all-NaN window; `descriptor()` maps
  non-finite to NaN locally so a degenerate window cannot poison a PCA.
- **The rollout figures are not a substitute for the spliced ones.** Two rollout
  branches differ at *t0* and at every position after it, so single-token causal
  attribution — the reason this branch set exists — does not hold there. The
  `_rollout` grids answer "what does the fork look like when nobody is handed the
  reference's future", not "what did the move at *t0* do".
- `measure.py`'s separation table and its 40-game bootstrap are **spliced-only**.
  Nothing in the rollout section above carries a CI.
- **The read-out strip measures the plane, not R^21.** A branch that leaves the
  plane reads as close in the filled marker and far in the open one; that gap is
  drawn rather than argued, but it is a gap in the *figure's* plane and no
  correction is applied to the panels themselves. The strip's own numbers are
  n = 1 with no CI over games, like everything else drawn here.
