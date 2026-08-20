# Probe geometry: board-state directions against rule-legality directions

Do the directions OthelloGPT's rule probes read out live in the same part of
activation space as the directions its board-state probe reads out?

Both probes decode `blocks.5.hook_resid_post` of
`Baidicoot/Othello-GPT-Transformer-Lens` (d = 512), so their weights are
comparable without any alignment step. Nothing here is retrained: the primary
directions come straight off the two committed probes, and the re-fits exist
only to put error bars on them. Both were fit on `adamkarvonen/othello_45MB_games`,
so the two arms no longer differ by corpus.

## Two things to know before reading a number here

**The committed board probe has seen most of the rules corpus.** Both probes stream
`adamkarvonen/othello_45MB_games` at seed 42, and all 800 rules-corpus games sit inside
the board probe's 20000 — 576 of them in its training split. So threat 4's *committed*
board row (99.36% on the rules corpus' legal rows) is a fit statistic, not a transfer
one; the transfer test is the ridge row beneath it, which fits on games held out from
itself. This is the price of putting both arms on one corpus, and it is the one thing
the old ARENA board probe did give for free. Detail in `ROBUSTNESS.md` threat 4.

**The `legal`-confusion caveat is retracted.** Earlier revisions of this study, of
`ROBUSTNESS.md` and of `results/TABLES.md` warned that the committed 4-class probe
"barely finds `legal`" — recall 0.2324, with 71.9% of true-legal rows called R1 — and
told the reader to discount `illegal·R1 = 0.923` as a shared anchor that is itself
confused with R1. That was an artifact of scoring the probe on the bare residual
instead of on the `load_space("raw")` features it was fit on. Scored correctly the
probe recovers `legal` at recall **0.8238** with 15.6% called R1, and beats every ridge
shrinkage on the same test games. `illegal·R1 = 0.923` is unchanged to the last bit —
the bug never touched a direction — so it is a fact about the geometry and cannot be
explained away by a broken anchor.

## Directions

A k-way softmax head is unchanged by adding one vector to every class weight, so
a raw class column is not a point in activation space — only class contrasts
are. The board probe's unidentified common component has norm ≈ 0.67 against a
contrast norm of ≈ 4.29, so this is load-bearing rather than pedantic.

| type | contrast | n | source |
|---|---|---|---|
| board colour | mine − theirs, at one square | 64 | `spade.probes.load.board()`, layer 5, test acc 0.9937 |
| board position | ½(mine + theirs) − empty, at one square | 60 | same |
| legal, R1, R2, R3 | each class − the mean of the four | 60 each | 4-class rule probe refit on one candidate square |
| illegal | illegal − legal | 60 | 2-class rule probe refit on one candidate square |

Each is one point per board square in the figure; the arrow of the same colour is
the committed probe's own direction, drawn from the same origin. The board probe
has no pooled direction, so its arrow is the mean of its per-square points.

The colour and position contrasts span everything a 3-way per-square head identifies, so
there is no third board axis to have missed. The four centre squares are never
empty (R7), so occupancy has no support there.

The rules probes were fit on `load_space("raw")` features, so their input map is
a global shift and one positive scalar ahead of the probe's own per-feature
standardization. The scalar cancels under unit-normalization, so a direction in
activation space is `W_c / scale`; scoring the raw residual is not free to skip
it, and `probes.rules_proba` is the only path that scores.

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
  closed-form ridge re-fit the bootstrap runs on. The ridge re-fits of the board
  arm still run on the 200 ARENA games, which is the only place ARENA is left.
- `src/probe_geometry/cosine_tables.py` — every cosine table.
- `src/probe_geometry/embed_directions.py` — PCA coordinates, the arrows, and
  the cosine matrix between types.
- `src/probe_geometry/robustness.py` — the red-team battery.
- `make_figures.py` — draws `results/`, recomputes nothing.
