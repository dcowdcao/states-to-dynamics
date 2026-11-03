# One plane for all nine spaces, via R^21

`violation_divergence/game_trajectory/space_fork_grid_pca` draws one game's fork
in nine representation spaces and has to disclaim that the planes are refit per
space, so nothing but the *shape* of the fork survives the move from one cell to
the next. The obstacle is structural: R^512 and R^8192 share no coordinates.

The ReStraV descriptor removes it. Every space's displacement trajectory reduces
to the same 21 named components — 7 step lengths, 6 turning angles, 4 moments of
each (`spade/straightness/geometry.py`) — so "mean θ" means the same thing in raw
layer 5 and in CosineSpaDE, and one plane can be applied to all nine cells.

```
python analysis/violation_divergence/r21_shared_basis/make_figures.py \
    [--variant {spliced,rollout}] [--l0-matched] [--camera-ready]
```

The first run encodes game 0 through the eight seed-42 dictionaries and caches
the result (3 MB, `$SPADE_DIVERGENCE_DIR/[rollout/]fork_r21/`); later runs are
seconds. The descriptor and the four planes live in
`src/spade/violation_divergence/fork_r21.py`, shared with the population sweep.

`--variant rollout` draws the same game from the resampled-continuation copy of
the study, where every branch **including the reference** samples its own K=4
futures instead of replaying the source game's. Rollout figures carry a
`_rollout` suffix; the spliced names are unchanged.

## What is plotted

A point is the 21-D descriptor of the **W = 8 sampled positions ending at t**,
minus the reference's, z-scored per space. Three choices of window are defensible;
8 is the smallest that yields a *full* 21-D vector, because `restrav_features`
slices `d[:, :7]` and silently returns a shorter vector below that (the script
asserts width 21). Another sub-study owns the choice of W.

Z-scoring is not optional here: the 21 components mix degrees with
activation-scale lengths whose SDs differ ~450x across dictionaries (mean step
length is 0.65 in raw layer 5 and 0.001 in SpaDE), so an unscaled plane would
report units. It uses each space's own SD, which is exactly what bounds the
comparability claim below.

| figure | plane |
|---|---|
| `space_fork_grid_r21_refit` | top-2 of each space's own descriptors — the status quo, as the control |
| `space_fork_grid_r21_shared` | **one** plane fit on the nine spaces' pooled descriptors, applied unchanged |
| `space_fork_grid_r21_named` | no fit at all: the axes are Δ mean *d* and Δ mean θ |
| `space_fork_grid_r21_isomap` | one Isomap fit on the pooled non-degenerate rows (§4.8's nonlinear companion) |
| `separation_by_plane` | what each plane costs, against an exact permutation null |
| `run_from_reference` | how far a branch runs from the reference, and how much of that each plane keeps |

Each has a `*_rollout` twin from `--variant rollout`, an `*_l0_matched` twin from
`--l0-matched`, and a `*_rollout_l0_matched` twin from the two together.

**The caption, which is here and not on the grids.** Game 0, *t0* = 10; a point
is the ReStraV descriptor of the W = 8 sampled positions (every 2nd ply) ending
at *t*, minus the reference's, z-scored per space. The 3-D grids are viewed at
`elev = 18, azim = −58`, the parent fork figure's camera, so the two read
together. `run_from_reference` averages over the 10 drawn anchors
(`t − t0 = 6…24`); `separation_by_plane` scores anchors `t ≥ t0` against the
exact 28-way permutation null, whose floor on *p* is 1/28 = 0.04.

Each grid keeps exactly one thing in gray, and it is the one thing that differs
per grid and cannot be looked up while looking at the cells: **what this plane
does and does not make comparable across cells** (FIGURE_GUIDELINES §4.6), plus
`one game (n=1)` and, on the rollout twins, the anchor caveat below.

## What the panels license

The shared and named grids are drawn on **shared axis limits with ticks on**, and
the disclaimer changes accordingly: the arrangement and direction of the branches
*are* comparable cell to cell, because every cell carries the same 21 named
coordinates and the same linear combination of them. What is still not comparable
is absolute magnitude — the unit is each space's own descriptor SD, so a longer
excursion in one cell is not a larger absolute change. The refit grid keeps the
old disclaimer in full and keeps its ticks off.

## How far a branch runs from the reference — `run_from_reference`

The grids ask the reader to eyeball how far each branch travels from the vertical
line at the origin. `run_from_reference` is that number:

> mean over the 10 drawn anchors (`t − t0 = 6…24`) of the distance from the
> reference, for the mean path of the six illegal branches, **divided by the
> legal control's** value in the same field.

A ratio, because every field is z-scored per space and the `refit` plane is fit
per space too: only a within-space contrast carries from one cell to the next. A
mark twice as high is not twice the absolute change.
1.0 is the control itself; the empirical legal-vs-legal scale — a *third* legal
move against the control, in full R^21 — is drawn as gray triangles at
**0.88–1.53, median 1.13**, so 1.0 is roughly where a legal alternative really
lands and not just where the arithmetic puts it.

**Panel (a): every 2-D plane says the illegal branches run further than the
control. Full R^21 mostly says they do not.** Spliced, illegal ÷ control:

| space | activation refit | R²¹ refit | R²¹ shared | R²¹ named | **full R²¹** | control 2, full R²¹ |
|---|---|---|---|---|---|---|
| raw layer 5 | 1.20 | 2.10 | 1.77 | 1.82 | **0.97** | 1.13 |
| PAnneal | 1.15 | 1.78 | 1.63 | 1.96 | **0.96** | 1.13 |
| Standard | 1.19 | 2.43 | 2.50 | 3.59 | **1.42** | 1.46 |
| GatedSAE | 1.05 | 1.69 | 1.12 | 0.96 | **0.73** | 0.88 |
| TopK | 1.09 | 0.76 | 1.08 | 0.80 | **0.66** | 0.92 |
| GatedAnneal | 1.03 | 1.22 | 1.24 | 1.23 | **0.88** | 1.14 |
| BatchTopK | 1.07 | 0.48 | 0.65 | 0.55 | **0.59** | 0.94 |
| CosineSpaDE | 1.36 | 1.81 | 1.77 | 4.03 | **1.49** | 1.53 |
| SpaDE | 1.34 | 1.57 | 1.29 | 1.69 | **0.82** | 1.11 |

Full R^21 is **below 1 in six of the nine spaces**, and — the sharper statement —
**below the third legal move's own ratio in all nine**: over these anchors an
illegal branch's curvature descriptor sits *closer* to the reference's than a
second legal alternative does. The three R^21 planes move it upward in 25 of the
27 plane × space cells, by 0.81–2.70× with a median of 1.64 — the fitted planes
because a top-2 of the pooled displacement is chosen where the illegal rows have
their variance, the named pair because two of 21 coordinates is a small and
non-representative sample of the object. The direction matches the parent study's
headline (`ratio_illegal_control_L5` = 0.979, an illegal move displacing layer 5
slightly *less* than a legal alternative), and it is why the grids now carry a
line pointing here: **the excursion the panels show is largely the plane.**

**Panel (b): the two halves of R^21 disagree.** The 11 step-length components and
the 10 turning-angle ones give different answers about the same branches — raw
layer 5 is 0.87 on `d` and 1.04 on θ, TopK 0.34 against 0.94, CosineSpaDE 1.91
against 1.31. The step-length half is the one that moves; the turning-angle half
stays near 1 in every space. So a single R^21 number is an average over an
opposition, and which side it lands on is decided by whichever half happens to
carry the variance in that space. The six illegal categories are drawn faint
behind their mean, which is the only spread one game supports.

**Isomap is not in either panel.** Its embedding is not a linear image of R^21,
its reference is not at the origin, and its distances are not a share of anything
— a ratio in it would be a number with no denominator's worth of meaning. The
`space_fork_grid_r21_isomap` grid keeps its shape-only reading.

Under `--variant rollout` the picture inverts, as it does for `separation_by_plane`:
full R^21 goes to 0.83–1.54 (above 1 in six of nine) and the activation plane to
1.29–2.11. Every anchor is at `r ≥ +6`, so read it with the caveat below.

## Reading them honestly

- One game, n = 1. Per space there are 9 branches × 10 windows = 90 rows in R^21,
  but the windows overlap, so the effective sample is **8 non-reference branch
  trajectories**. A plane fit from that in 21 dimensions is under-supported; that
  is why the shared and named planes, which do not spend the sample on fitting,
  are the interesting ones.
- Every number is scored against the **exact permutation null**: all 28 ways of
  relabelling the 8 non-reference branches 6 illegal / 2 control. The smallest
  attainable p is 1/28 = 0.04.
- Routing through R^21 raises the null too (median 0.4–0.8 SDs, against 0.2–0.4
  for the activation plane). Panel (b) of `separation_by_plane` divides it out.
- `run_from_reference` is the same n = 1 and carries no interval either; its only
  spread is the six illegal categories drawn faint. It measures **run length**,
  which is a different question from `separation_by_plane`'s **class gap**: a
  plane can move the illegal branches further out and separate them from the
  controls no better, and here it does exactly that.
  `../r21_run_population` is that panel over 100 branch-complete games, with a mean
  and a 95% CI, and it does not reproduce this figure's spread across spaces: read
  the two together before quoting a number off this one.
- Numbers and the verdict: `REPORT.md` at the repo root.

## The rollout variant, and where the mean is taken

A branch is K = 4 rows. The mean over them is taken **after φ and before the
difference from the reference** — φ is nonlinear, so averaging the four
trajectories first would report the curvature of a path no rollout took. The
plane projection is linear, so it does not matter which side of it the mean sits
on; the descriptor does, and that is the only place the choice is available. The
activation-space comparison folds the K rollouts on the trajectories instead,
which is the same thing there because every step of it is linear. The z-scoring
SD is the SD of the plotted object, i.e. of the K-mean descriptor.

**The claim inverts.** Gap ÷ that plane's own exact permutation-null median, at
anchors t ≥ t0 (all of which are r ≥ +6, see the caveat below):

| space | activation refit, spliced | activation refit, rollout | shared R²¹, spliced | shared R²¹, rollout |
|---|---|---|---|---|
| raw layer 5 | 0.83 | **3.23** | 1.92 | 1.91 |
| PAnneal | 0.93 | **3.35** | 1.99 | 1.88 |
| Standard | 0.94 | **4.67** | 1.93 | 2.51 |
| GatedSAE | 0.62 | **2.82** | 1.60 | 1.47 |
| TopK | 0.76 | **2.46** | 2.16 | 1.05 |
| GatedAnneal | 0.92 | **2.53** | 1.07 | 1.64 |
| BatchTopK | 0.45 | **3.16** | 1.92 | 1.12 |
| CosineSpaDE | 0.97 | **2.69** | 1.75 | 3.40 |
| SpaDE | 1.05 | **4.58** | 1.93 | 2.18 |

Spliced, the activation plane sits at its own null (0.45–1.05×) and R²¹ is the
plane that separates. Rollout, the activation plane clears its null by 2.5–4.7×
in every space and R²¹ is flat or worse. That is not R²¹ failing: with every
branch resampling its own future the branches diverge in the activations
themselves, and a descriptor that summarises step length and turning angle
discards exactly that. The two variants measure different objects.

**Long-horizon rollout geometry is not informative, and this figure is all
long-horizon.** With every branch resampling, δ̂ climbs monotonically; illegal
and control both saturate near 1.23 against a ~1.41 unrelated-activations
ceiling, and the illegal/control ratio is pinned to 1.00 by r ≥ 8. Only r = 0
separates (1.032, sign opposite the spliced 0.979) — and the first anchor this
figure can draw is r = +6, because W = 8 sampled positions at stride 2 is 14
plies of span. So the rollout grids are a **shape check on the reduction**, not
a result: they show that the K-mean descriptor still fits one plane across nine
spaces. Read the separation from the spliced figures.

## The L0-matched roster — `--l0-matched`

Every grid above compares eight dictionaries that sit wherever their own default
hyperparameters left them on the sparsity axis (L0 31 to 1808), so a cell-to-cell
difference confounds the architecture with its L0. `--l0-matched` swaps in the
`l0_operating_point` picks instead: one checkpoint per type, each its own grid's
nearest-L0 member to the shared target (`results/l0_operating_point/picks.csv`,
`target_l0` 219.2), leaving a 1.4x spread instead of a 58x one.

| panel | checkpoint | L0 |
|---|---|---|
| PAnneal | `p_anneal/trainer22` | 260.1 |
| Standard | `standard/trainer25-v2` | 208.7 |
| GatedSAE | `gated/trainer32` | 219.9 |
| TopK | `top_k/trainer_61` | 220.0 |
| GatedAnneal | `gated_anneal/trainer25` | 221.7 |
| BatchTopK | `BatchTopK-k220` | 226.7 |
| CosineSpaDE | `CosineSpaDE-beta1-gam0.05` | 183.8 |
| SpaDE | `SpaDE-lam0.001-gam0.005` | 232.1 |

Cells keep the seed-42 grid's family order, so the two grids read panel for panel,
and each panel name carries its own L0. **Karvonen's five are fed the raw residual
stream and ours the standardized one**, because that is how each was trained
(`spade.l0_operating_point.checkpoints`); the two families' latents are therefore
not on one input scale, and this is a comparison of architectures at matched L0
rather than of one quantity measured eight ways.

Every artifact carries an `_l0_matched` tag and nothing seed-42 is written over.
`analysis/l0_operating_point/README.md` still describes the earlier min-max target
(L0 116.7); `picks.csv` is the file of record and holds the consensus rule's 219.2.

Nothing about the plane changes, so every claim above can be re-read off the twins.
Panel (a) of `run_from_reference_l0_matched`, illegal ÷ control, against the
seed-42 table in *How far a branch runs from the reference*:

| space | activation refit | R²¹ refit | R²¹ shared | R²¹ named | **full R²¹** | control 2, full R²¹ |
|---|---|---|---|---|---|---|
| raw layer 5 | 1.20 | 2.10 | 1.80 | 1.82 | **0.97** | 1.13 |
| PAnneal (L0 260) | 1.17 | 1.35 | 1.55 | 2.02 | **0.98** | 1.10 |
| Standard (L0 209) | 1.09 | 2.14 | 1.88 | 3.05 | **1.11** | 1.18 |
| GatedSAE (L0 220) | 1.06 | 0.86 | 0.80 | 0.98 | **0.68** | 1.00 |
| TopK (L0 220) | 1.16 | 1.46 | 1.59 | 1.56 | **0.83** | 1.09 |
| GatedAnneal (L0 222) | 1.15 | 0.48 | 0.83 | 1.22 | **0.52** | 0.79 |
| BatchTopK (L0 227) | 1.13 | 1.26 | 1.43 | 1.41 | **0.78** | 1.01 |
| CosineSpaDE (L0 184) | 1.32 | 2.15 | 2.23 | 2.88 | **1.79** | 1.87 |
| SpaDE (L0 232) | 1.20 | 0.54 | 0.72 | 1.36 | **0.56** | 0.89 |

**The headline survives the L0 control and gets slightly stronger.** Full R^21 is
below 1 in **seven** of the nine spaces (seed-42: six) and below the third legal
move's own ratio in **all nine**, as before. The three R^21 planes move it upward
in the same **25 of 27** plane × space cells, by 0.92–2.75× with a median of 1.69
(seed-42: 0.81–2.70×, median 1.64). Which cells lead does move: at matched L0 the
largest excursion is CosineSpaDE (1.49 → 1.79) and Standard drops out of the top
(1.42 → 1.11), while GatedAnneal becomes the *smallest* (0.88 → 0.52) — so the
ordering of spaces in the seed-42 grid was partly reading their L0.

### The two controls together — `--variant rollout --l0-matched`

`*_rollout_l0_matched`, one per figure above: game 0 at *t0* = 10, the same W = 8
descriptor and the same four planes, drawn from the rollout branch set through the
eight picks. n = 1 game — 9 branches × K = 4 rollouts = 36 rows per space, 10 anchors
*t − t0* = 6…24, the mean over K taken after φ as everywhere in this study. The anchor
caveat above applies unchanged, so these are a **shape check on the reduction and not
a separation**. Rendered on one H100 (`scripts/slurm/submit_gpu.sbatch`).

**The rollout inversion is a property of the branch set, not of the roster.** Gap ÷
that plane's own exact permutation-null median, anchors *t* ≥ *t0*: the activation
refit clears its null in all nine cells, 2.3–3.3× (seed-42 rollout: 2.5–4.7×), where
spliced it sat *at* its null; shared R²¹ is 1.3–3.5× (seed-42 rollout: 1.1–3.4×). No
plane reaches significance against the exact 28-way null, whose floor on *p* is 0.04.
`run_from_reference_rollout_l0_matched` panel (a), illegal ÷ control 1 in full R²¹:
0.85 (SpaDE) to 1.72 (CosineSpaDE), raw layer 5 at 1.19 — above 1 in seven of nine
cells, against below 1 in seven of nine on the spliced L0-matched twin. Resampling the
future is what moves that column; matching L0 is not.

## Figure notes

The lines the figures used to print under their panels, in their own wording.

- `figures/separation_by_plane` — one game (n=1): 6 illegal vs 2 control branches,
  10 windows each, per space.
- `figures/run_from_reference` — one game (n=1), 10 anchors *t−t0* = 6…24: ratios, so
  only the within-space contrast carries across cells — a mark twice as high is not
  twice the absolute change. The reference is 0 in every field by construction.
- Both `_rollout` twins also carried: rollout: every anchor drawn is at *r* ≥ +6,
  where illegal and control have already saturated — shape, not a separation. The
  `space_fork_grid_r21_*_rollout` grids still carry that line in their legend title.
