# Probe geometry: board-state directions against rule-legality directions

Do the directions OthelloGPT's rule probes read out live in the same part of
activation space as the directions its board-state probe reads out?

Both probes decode `blocks.5.hook_resid_post` of
`Baidicoot/Othello-GPT-Transformer-Lens` (d = 512), so their weights are
comparable without any alignment step. Nothing here is retrained: the primary
directions come straight off the two committed probes, and the re-fits exist
only to put error bars on them.

## Directions

A k-way softmax head is unchanged by adding one vector to every class weight, so
a raw class column is not a point in activation space — only class contrasts
are. The board probe's unidentified common component has norm ≈ 1.1 against a
contrast norm of ≈ 8.5, so this is load-bearing rather than pedantic.

| type | contrast | n | source |
|---|---|---|---|
| board colour | mine − theirs, at one square | 64 | `linear_probe.pt`, 200 ARENA games |
| board position | ½(mine + theirs) − empty, at one square | 60 | same |
| legal, R1, R2, R3 | each class − the mean of the four | 60 each | 4-class rule probe refit on one candidate square |
| illegal | illegal − legal | 60 | 2-class rule probe refit on one candidate square |

Each is one point per board square in the figure; the arrow of the same colour is
the committed probe's own direction, drawn from the same origin. The board probe
has no pooled direction, so its arrow is the mean of its per-square points.

The colour and position contrasts span everything a 3-way per-square head identifies, so
there is no third board axis to have missed. The four centre squares are never
empty (R7), so occupancy has no support there.

The rules probes standardize their inputs; a direction in activation space is
`W_c / scale`, verified against the probes' own logits.

## Why `rules_square` exists

The committed rules probes are square-agnostic — one direction, read off the
residual wherever the candidate move was spliced in — while every board
direction is square-specific. Pooled against 64 per-square directions, a single
global direction is near-orthogonal almost by construction, because the
per-square feature is a conjunction of square-identity and board state while the
global one is the marginal. Refitting legality separately per candidate square
is the comparison the board arm can actually be matched against, and the natural
null for it is the same rules direction against a *different* square's board
direction, which holds everything else fixed.

## Layout

- `src/probe_geometry/probes.py` — corpora, committed probes, contrasts, the
  closed-form ridge re-fit the bootstrap runs on.
- `src/probe_geometry/cosine_tables.py` — every cosine table.
- `src/probe_geometry/embed_directions.py` — PCA coordinates, the arrows, and
  the cosine matrix between types.
- `src/probe_geometry/robustness.py` — the red-team battery.
- `make_figures.py` — draws `results/`, recomputes nothing.
