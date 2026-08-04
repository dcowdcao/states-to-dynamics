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
python analysis/violation_divergence/fork_curvature_space/make_figures.py [--game 1235] [--w 8] [--camera-ready]
python analysis/violation_divergence/fork_curvature_space/measure.py [--n_games 40] [--w 8] [--game 1235]
```

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
over which move enters the descriptor, not a confidence interval. The windowed
grids are n = 1 with no error bars and say so on the figure, exactly as
`../game_trajectory` does; the separation numbers with CIs over 40 games are
below.

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
behind their mean and print the ratio in the legend title, so the three confident
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
and in `../README.md`, which carries the numbers with CIs; the legend titles
point at both.

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
  legend title's "NOT comparable across the grid" disclaimer stays and matters
  more, not less (§4.6, §4.8). The ordered axis reads `t − t0 (plies)`, labelled
  once per row.
- PCA plane axes read `PC1 (44% var, mean θ)` — **per component, never summed**,
  per panel since the plane is refit per space, and naming the descriptor
  component the PC leads on.
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
