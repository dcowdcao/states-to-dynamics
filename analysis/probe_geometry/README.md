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
| illegal | illegal − legal | 60 | 2-class rule probe refit on one candidate square; not drawn in panel (a) |

Panel (a) draws 364 of these — one point per board square, six families, every
family but per-square `illegal` — with the committed probe of the same colour as
a ringed marker among them. The board probe has no pooled direction, so its
marker is the mean of its per-square points.

`illegal@q` is exactly `−legal@q`: both are ridge onto centred indicators over
the same rows at the same λ, and 1[rule>0] = 1 − 1[rule==0] forces
`w2[1] = −w4[0]`. Drawing both put one 60-point set on the panel twice, reflected
through the origin, and its 180° pairs inflated every fidelity score below. The
*committed* illegal probe is a separate fit — its cosine with the committed
`legal` is −0.829, not −1 — so it is not redundant and stays in panel (b)'s
cosine matrix.

The colour and position contrasts span everything a 3-way per-square head identifies, so
there is no third board axis to have missed. The four centre squares are never
empty (R7), so occupancy has no support there.

The rules probes were fit on `load_space("raw")` features, so their input map is
a global shift and one positive scalar ahead of the probe's own per-feature
standardization. The scalar cancels under unit-normalization, so a direction in
activation space is `W_c / scale`; scoring the raw residual is not free to skip
it, and `probes.rules_proba` is the only path that scores.

## What panel (a) is and is not evidence for

The panel is a nonmetric (ordinal) MDS of the angular distances
θ_ij = arccos(x_i·x_j), not a PCA. These directions are near-mutually-orthogonal
— mean pairwise angle 89.99° ± 3.96 against 90.00° ± 2.53 for isotropic random
vectors in 512 dimensions — the spectrum's participation ratio is 121, and it
takes 50 components to reach half the variance. No plane holds their variance:
the uncentred PCA puts 3.21% and 2.47% on its two axes (4.24% and 2.51% on
the 424 points the replaced panel drew). What a
plane can hold is the *order* of the angles, which is what nonmetric MDS fits.
So distance in the panel is monotone in angle and not proportional to it, the
axes carry no variance ratio, and — SMACOF fixing a configuration only up to
rigid motion — there is no canonical origin to draw arrows from. Orthogonal
Procrustes onto the PCA basis pins the orientation, so a refit cannot rotate or
mirror the panel.

Rank fidelity — Spearman between the 512-d angles and the drawn distances, over
every pair — is **0.504**, against **0.395** for the PCA and **0.525** for direct
gradient descent on the exact objective. Metric MDS, Isomap and UMAP were
surveyed and none led; the survey ran on the old 424-point set, where the
duplicated `illegal` points lift every method (nonmetric MDS 0.551, metric MDS
0.511, Isomap k=100 0.502, UMAP nn=100 0.499, PCA 0.394), so those numbers rank
methods and are not comparable to the 0.504 above.

**The panel is family-level evidence and nothing finer.** A structure-free null —
the same six family mean directions, residuals replaced by isotropic noise at the
observed within-family spread — scores 0.574 ± 0.007 (SD over 5 draws), *above*
the real embedding, and a predictor knowing only which family a point belongs to
reaches 0.523 of the 0.525 ceiling. Square identity does not survive: the
fraction of a point's 6 nearest neighbours sharing its board square is 0.001 in
the panel against 0.111 in the real 512-d space. Family identity goes the other
way: 0.917 of a point's 10 nearest neighbours share its family in the panel
against 0.836 in 512-d, so the layout is *more* separated than the space it
draws. The square-matched claim rests
on the cosine tables in `results/TABLES.md`, never on this panel, and any actual
angle is read off panel (b).

## Figure notes

What the panels do not print, so a reader does not have to infer it:

- `fig1_direction_geometry.png` / `.pdf` panel (a) — the rule codes are R1 placement,
  R2 bracketing, R3 turn (`docs/othello_rules.md`). The null beside the fidelity is
  0.574 ± 0.007, SD over 5 draws, and it is *above* the embedding's own 0.504: the
  layout is evidence about families and never about individual squares. The committed
  probe is drawn open because it is not the family centroid — for R1 the two are 1.3
  within-family radii apart — and for the board families that "committed" direction is
  itself the mean of the per-square vectors, since the board probe has no pooled
  direction of its own.
- `fig1_direction_geometry.png` / `.pdf` panel (b) — cosines are point estimates off the
  committed probes, with no interval; the game-level bootstrap for the same quantities is
  in `results/cross_probe_cosine.csv` and `results/TABLES.md`. Random unit directions in
  512 dimensions meet at |cos| = 0.035, so the cells reading ±0.03 and below are inside
  the floor. `illegal` appears here and not in (a): per square it is exactly `-legal`,
  but the *committed* illegal probe is a separate fit and sits at cos −0.83 from
  committed `legal`.

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
- `src/probe_geometry/embed_directions.py` — the nonmetric-MDS coordinates
  (`embed_points.csv`, `embed_probes.csv`), the rank fidelity and its null
  (`embed_fidelity.csv`), and the cosine matrix between types.
- `src/probe_geometry/robustness.py` — the red-team battery.
- `make_figures.py` — draws `results/`, recomputes nothing.
