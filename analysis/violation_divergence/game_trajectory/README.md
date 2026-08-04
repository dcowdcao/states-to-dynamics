# The fork: one game, every future

The rest of the study reports averages over 13,531 branches. This sub-study
draws the object those averages are taken over — a single game as a path, shared
up to *t0* because the branches are token-identical there, then one path per
branch. It is an illustration of the design, not a claim-supporting figure; the
numbers live in the parent `README.md`.

```
python analysis/violation_divergence/game_trajectory/make_figures.py \
    [--game 1235] [--n_games 1] [--embedding pca|isomap] [--camera-ready]
```

The default is **game 1235** (*t0* = 20), picked by a sweep over the 1550
branch-complete games (`../game_selection`); the script asserts the requested
game carries all nine branch types rather than silently drawing eight.

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

## Figures

| file | what |
|---|---|
| `game_<id>_fork_pca` | one game: the 3-D fork, beside δ̂ vs r — the magnitude a projection cannot show |
| `space_fork_grid_pca` | one game across **all nine representation spaces** — raw layer 5 and the eight seed-42 dictionaries. Planes are refit per space, so only the *shape* of the fork is comparable across cells, never position. **Superseded by `../fork_and_board`**, which draws this grid as panel (a) over the board the branches play on |

## Reading them honestly

- A single game is n = 1. The per-position wobble in the δ̂ panel is real
  variation in that game, not noise the aggregate figures smoothed away — the
  parent study averages 13,531 branches to remove it.
- Per-panel refits mean coordinates are not comparable across cells in either
  grid. The legend title says so on the figure itself.
- These panels show *shape*. Every quantitative claim — the 0.978 ratio, the
  0.4985 legal-vs-legal null, the plateau — belongs to the parent README.
