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
python analysis/violation_divergence/r21_shared_basis/make_figures.py [--camera-ready]
```

The first run encodes game 0 through the eight seed-42 dictionaries and caches
the result (3 MB, `$SPADE_DIVERGENCE_DIR/fork_r21/`); later runs are seconds.
That encode belongs in `src/spade/violation_divergence/` under this repo's
division of labour, and should move there if this study is kept.

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

## What the panels license

The shared and named grids are drawn on **shared axis limits with ticks on**, and
the disclaimer changes accordingly: the arrangement and direction of the branches
*are* comparable cell to cell, because every cell carries the same 21 named
coordinates and the same linear combination of them. What is still not comparable
is absolute magnitude — the unit is each space's own descriptor SD, so a longer
excursion in one cell is not a larger absolute change. The refit grid keeps the
old disclaimer in full and keeps its ticks off.

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
- Numbers and the verdict: `REPORT.md` at the repo root.
