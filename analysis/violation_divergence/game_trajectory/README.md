# The fork: one game, every future

The rest of the study reports averages over 13,531 branches. This sub-study
draws the object those averages are taken over — a single game as a path, shared
up to *t0* because the branches are token-identical there, then one path per
branch. It is an illustration of the design, not a claim-supporting figure; the
numbers live in the parent `README.md`.

```
python analysis/violation_divergence/game_trajectory/make_figures.py \
    [--game 1235] [--n_games 1] [--embedding pca|isomap] \
    [--variant spliced|rollout] [--camera-ready]
```

The default is **game 1235** (*t0* = 20), picked by a sweep over the 1550
branch-complete games (`../game_selection`); the script asserts the requested
game carries all nine branch types rather than silently drawing eight. Both
branch sets carry it, and both give 1550 branch-complete games.

**`--variant spliced`** (the default) is the study's own branch set: every branch
replays the source game's `tokens[t0+1:]`. **`--variant rollout`** reads
`$SPADE_DIVERGENCE_DIR/rollout/`, where every branch — the reference included —
samples its own continuation from OthelloGPT (K = 4 per branch, temperature 1.0,
seed 42; 17,527 branches, 70,108 rows over the same 2000 games). Read *[The
rollout variant](#the-rollout-variant)* below before reading that figure.

## What is plotted

Move index is the third axis, so the plane is a 2-D embedding of the
**displacement** `Δ(t) = z_branch(t) − z_ref(t)`, never of raw activations: move
index dominates activation geometry in every geometry study in this repo, so a
raw PCA would put it on PC1 and encode time twice. With Δ, the reference is
exactly the vertical line at the origin, the shared prefix collapses onto it,
and the spray at *t0* is the whole content of the figure.

Two embeddings ship, because PCA alone cannot tell "the branches do not
separate" from "the branches do not separate *linearly*":

- **PCA**, uncentred — zero displacement is a real point, not an arbitrary
  origin. Mean-centring moves the reference off (0, 0) and detaches it from the
  prefix line and the branch marker.
- **Isomap**, following `src/spade/rules/embed.py` (PCA-50 pre-reduction, kNN
  geodesic). Δ is exactly zero for the reference at every position and for every
  branch before *t0*, and those duplicates form a disconnected clique in the kNN
  graph, so the embedding is **fit on the non-degenerate rows and transforms the
  rest**. Isomap has no origin and no variance ratio; its axes are arb. units.

Positions are sampled every **2 plies at t0's parity**. At every ply the path is
a sawtooth: the board code is mover-relative, so the frame inverts each ply and
consecutive activations alternate. That ripple is real — it is the odd/even
split in `cos(Δ(r), Δ(0))` — but it is a property of the representation's frame,
not of the trajectory's shape, and it hides everything else. One mover per curve
is the readable view.

The window is the study's own, r ∈ [−8, 24]. View angle is fixed at
`elev=18, azim=−58` so the figures are reproducible. The plane axes carry three
sparse ticks symmetric about zero (§4.6): a within-panel scale bar, refit per
game, and never comparable between figures.

This figure draws **all nine branches, one per illegal category, and both legal
controls** — including `control_legal_2`, the null *for* `control_legal`. That
is deliberate: `../fork_and_board` groups the six categories into their three
rules and drops the second control, so when a reader wants the legal-vs-legal
scale or a per-category path, this is the figure that has them.

## (c): the fork as one number per branch

Panel (a) asks the reader to judge how far each branch runs from the reference
and answers nothing; (b) answers it for the full space but as thirteen points
per branch. Panel **(c)** is the mean of each: for every branch, its mean
distance from the reference over the drawn *r* ≥ 0, taken **in (a)'s plane**
(the bar, `branch_panels.plane_displacement`) and **in the full 512-d space**
(the open marker, the mean of (b)'s δ̂ over the same offsets). Both are divided
by `control_legal`'s, which is what lets an arb.-units plane distance and a
δ̂ ratio share one axis; the reference is 0 by construction and the control is 1
by construction.

**The bar and its marker disagree, and that is the panel's job.** The plane is
the top two PCs of the displacement, so it is fit where the largest branches
travel and holds a different share of each branch: 30% of the control's
displacement in this game and 81% of `opponent_only`'s. Dividing one by the
other is exactly the factor by which the plane inflates the contrast —
1.53 × (0.81/0.30) = 4.1 — and a fork panel read as a magnitude reports the 4.1.
The identity is exact: plane ratio = full-space ratio × (branch's share ÷
control's share). Where a branch leaves the plane the error runs the other way
and the branch reads as one that stayed home.

Spliced set, game 1235, layer 5, mean over the 13 drawn offsets *r* = 0, 2, …, 24:

| branch | in (a)'s plane | in the full 512-d space |
|---|---|---|
| R1 occupied (own) | 1.29 | 1.17 |
| R1 occupied (opp) | 0.87 | 0.94 |
| R2 detached | 2.09 | 1.03 |
| R2 own-only | 2.99 | 1.53 |
| R2 no bracket | 1.19 | 1.04 |
| R3 opponent-only | 3.51 | 1.53 |
| `control_legal_2` (null) | 1.54 | 1.31 |

**Read the null row first.** A second legal move sits 1.31× the control's
distance from the reference in the full space — *further than four of the six
illegal branches*. The panel where the legal-vs-legal scale lives is this one,
and with the number on it the fork stops looking like six illegal branches
departing from a legal cluster. In the plane the same row reads 1.54, above
three of the six.

Under `--variant rollout` the six illegal branches run 0.83–1.66× the control in
the plane and 0.89–1.23× in the full space, with the null at 1.12 / 1.06 — the
saturation the section below describes, as one number per branch.

**n = 1.** One game, one draw; no interval. The dispersion a single game
supports is between branches, which the panel shows, and nothing over games,
which it does not.

## Figures

| file | what |
|---|---|
| `game_<id>_fork_pca` | one game: the 3-D fork, δ̂ vs r — the magnitude a projection cannot show — and both of those as one number per branch |
| `game_<id>_fork_pca_rollout` | the same three panels on the rollout branch set, each curve a mean over that branch's K = 4 sampled continuations, window cut to *r* ∈ [−8, +8] |
| `space_fork_grid_pca` | one game across **all nine representation spaces** — raw layer 5 and the eight seed-42 dictionaries. Planes are refit per space, so only the *shape* of the fork is comparable across cells, never position. **Superseded by `../fork_and_board`**, which draws this grid as panel (a) over the board the branches play on |

## Caption

The figure carries no prose. What a reader needs in order not to over-read it is
here (FIGURE_GUIDELINES §2, §5); what stays inside the figure is the apparatus
that cannot be looked up while looking at it — panel titles and letters, axis
labels, the legend, the in-panel "0 by construction" annotation, and the R1/R2/R3
key naming each rule's member categories.

### `game_1235_fork_pca`

**One prefix, nine futures: a single game as a path, shared up to *t0* because the
branches are token-identical there, then one path per branch.** Panel **(a)**: the
displacement `Δ(t) = z_branch(t) − z_ref(t)` in an uncentred PCA plane, **a
learned basis refit for this game**, with move index on the third axis; the
reference is exactly the vertical line at the origin and the shared prefix
collapses onto it. Positions are sampled every 2 plies at *t0*'s parity, window
*r* ∈ [−8, 24], view fixed at `elev = 18, azim = −58`. The three ticks per plane
axis are a within-panel scale bar only and **coordinates are not comparable
between figures**. Panel **(b)**: relative displacement `δ̂ = ‖Δ‖ / ‖z_ref‖`
against offset — the magnitude a projection cannot show. The shaded band is the
pre-branch window, where δ̂ is **0 by construction** because the branches share
their tokens there; the panel says so in place. Panel **(c)**: each branch's mean
distance from the reference over the 13 drawn offsets *r* = 0, 2, …, 24, in (a)'s
plane (bar) and in the full 512-d space (open marker, the mean of (b)'s δ̂), both
**÷ `control_legal`'s** — so the reference is 0 and the control 1, both by
construction, and the two measurements share an axis. The gap between a bar and
its marker is the part of the displacement the plane does not hold: the plane
carries 30% of the control's and 81% of `opponent_only`'s, which is the whole of
why (c) reads 3.51 in the plane and 1.53 in the space. `--embedding isomap` swaps
(a) for a nonlinear plane with no origin and no variance ratio, axes in arb.
units, fit on the non-degenerate rows and transforming the exact-zero
duplicates; (c)'s bars follow it and its markers do not move.

**Game 1235 is selected, not typical.** It is the argmax, over all 1550
branch-complete games, of the mutual separation of its six illegal displacement
trajectories, scored in raw layer 5 in the full 512-dimensional space and never in
an SAE space or a 2-D projection (`../game_selection`). Population median 0.705;
this game 1.080. A reader who does not know the game was chosen will over-read the
panels.

**n = 1: an illustration of the design, not a claim.** The per-position wobble in
(b) is real variation in this game, not noise the aggregate figures smoothed away.
Every number — the 0.978 ratio, the 0.4985 legal-vs-legal null, the plateau —
belongs to `../README.md`, where it carries its CI over 13,531 branches.

**All nine branches are drawn**, one per illegal category and *both* legal
controls, including `control_legal_2` — the null *for* `control_legal`, scored
against it exactly as the illegal branches are. That is the point of this figure:
`../fork_and_board` groups the six categories into their three rules and drops the
second control, so this is where the legal-vs-legal scale and the per-category
paths live.

### `game_1235_fork_pca_rollout`

**The same nine branches of the same game, drawn out of the rollout branch set:
every branch, the reference included, samples its own continuation from
OthelloGPT rather than replaying the source game's.** Panels, styling, camera and
selection read exactly as above. What is different:

**The two panels reduce over K = 4 differently, on purpose.** Panel (a) averages
the displacement `Δ` over the branch's four rollouts and then projects — the
plane is an uncentred linear map, so mean-then-project and project-then-mean
agree. Panel (b) averages **δ̂ itself**, after the norm: `‖mean_k Δ_k‖` cancels
the sampling noise the four rollouts differ by and is a smaller, different
quantity from the typical displacement. The panel says so in place. **(c)
inherits both**: its bar is a distance in (a)'s K-averaged plane and its marker
is the mean of (b)'s K-averaged δ̂, so the two marks on one row are not two
reductions of one quantity and the bar is the smaller reduction of the two.

**The rollouts are paired.** A rollout's reference is the one drawn from the same
uniform stream (`rollout_branches.py` keys the sampler on `(game_id, k)`), so
`Δ` is a within-sample difference under common random numbers, not two
independent draws differenced.

**(b) draws no spread over the K = 4 rollouts.** It carried a ±1 SD band per
branch; nine of those were dropped along with every other shaded ribbon in the
study, and nothing replaced them, so the rollout panel is means only. That is a
departure from FIGURE_GUIDELINES 1.2 — read the curves as shape.

**The window is *r* ∈ [−8, +8], not [−8, 24].** See below.

**δ̂ is on a different scale from the spliced panel.** It climbs instead of
decaying, and this game reaches ≈ 1.3 by *r* = +8 where the spliced figure is
near 0.5. The two (b) panels are not on shared axes and must not be read as one.

## The rollout variant

The spliced set gives every branch the *reference's* future: `tokens[t0+1:]` is a
legal continuation of the reference's board alone, it survives a median of 2
moves on a legal control, and after an illegal move no board exists for it at
all. The rollout set removes that asymmetry by regenerating every suffix, the
reference's included, and pays for it with single-token attribution — two rollout
branches differ at *t0* **and** everywhere after it.

The price shows as saturation. Layer 5, full set, per-branch mean over K then
mean over branches (11,527 illegal and 2,000 control branches over 2000 games,
from `$SPADE_DIVERGENCE_DIR/[rollout/]divergence/resid_post_5.npz`):

| *r* | spliced δ̂ illegal | spliced illegal ÷ control | rollout δ̂ illegal | rollout illegal ÷ control |
|---|---|---|---|---|
| 0 | 0.792 | 1.0316 | 0.792 | **identical by construction** |
| 4 | 0.590 | 0.9305 [0.9229, 0.9382] | 0.985 | 0.9855 [0.9809, 0.9904] |
| 8 | 0.554 | 0.9314 [0.9231, 0.9396] | 1.118 | 0.9957 [0.9926, 0.9990] |
| 24 | 0.461 | 0.9797 [0.9715, 0.9873] | 1.222 | 1.0034 [1.0017, 1.0053] |

95% percentile bootstrap over the 2000 games, 2000 resamples. *r* = 0 is the same
number in both sets to four decimals, because the activation at *t0* depends only
on `tokens[:t0+1]` and the two sets share those bits — so the rollout changes
nothing at *r* = 0 and everything after it. Rollout δ̂ climbs to **1.236** by
*r* = 20, 96% of the **1.293** two-unrelated-games level `../RED_TEAM.md`
measures in this space, against the spliced plateau's ~36%; and the
illegal/control separation is inside 0.5% of 1 by *r* = 8 and crosses **above**
1 by *r* = 13. Hence `ROLLOUT_RMAX = 8`: past there the figure would be drawing a
fork whose separation is not in the data. It is stated on the figure and it is
the number to revisit if the branch set changes.

The two variants are not two measurements of one thing, and neither is the
other's replication: `branches.pt` stays the tensor the single-token claims rest
on.

## Reading them honestly

- A single game is n = 1. The per-position wobble in the δ̂ panel is real
  variation in that game, not noise the aggregate figures smoothed away — the
  parent study averages 13,531 branches to remove it.
- Per-panel refits mean coordinates are not comparable across cells in either
  grid. The caption says so.
- (a) and (b) show *shape*; (c) is the only number on the figure, it is a ratio
  within this one game, and it still is not a claim. Every quantitative claim —
  the 0.978 ratio, the 0.4985 legal-vs-legal null, the plateau — belongs to the
  parent README.
- **A bar in (c) is not a distance in the space, and the marker beside it says
  by how much.** The plane is fit on the branches it draws, so it flatters the
  ones that align with it; nothing on (a) discloses that and the two marks in
  (c) are the only place it is visible. Read the marker when the question is
  "how far", the bar only when the question is "how far *in this picture*".
- The spliced and rollout figures do not share axes and are not a comparison.
  One holds the future fixed and moves one token; the other lets the model write
  the future. Read each through its own caption.
- Neither (b) panel carries an interval. The rollout one used to, over its K = 4
  samples within one game; that is sampling spread rather than an interval over
  games, and nothing in the parent README was ever scored against it.
