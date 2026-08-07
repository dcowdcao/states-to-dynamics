# PCA of the 21-D ReStraV descriptor, per representation space

Every figure elsewhere in the straightness study reduces a game to one number
(mean turning angle, mean step length) and scores it. This one keeps ReStraV's
whole 21-D vector — 7 step lengths, 6 turning angles, and the 4 moments of each
(`src/spade/straightness/geometry.py`) — and asks what the descriptor space
looks like before any label is used: one PCA per representation space, games
projected onto its top-2 principal components.

It is the unsupervised counterpart of the `auc_lr21` column: same 21 features,
no labels.

```
python analysis/straightness/feature_pca/make_figures.py
```

Reads geometry two studies already wrote — no re-extraction. Writes
`figures/restrav_pca.png` and prints the table below.

## Where the six classes come from

Three are whole games out of the straightness study's committed geometry npz
(`src/spade/straightness/compute_geometry.py`): legal, OthelloGPT rollout,
championship. The illegal move is drawn as the three constraint rules it can
break, and **that study cannot supply them** — its violation class injects only
empty-square categories, so it holds R2 and R3 and no R1 at all. The rules come
off the violation-divergence branch population instead
(`spade.violation_divergence.rule_curvature`, cached to
`$SPADE_DIVERGENCE_DIR/rule_descriptors/`), where all 2000 rule-complete games
carry a branch for every rule.

Three things follow, and the caption carries all three:

- **The two sides are the same measurement.** Both drop position 0 (the
  attention sink) and take one angle per ply over positions 1..58, both call
  `spade.straightness.geometry.temporal_geometry`, and the agreement was checked
  game for game: max 1.5 × 10⁻⁵ deg on the 1998 reference branches that are
  token-identical to a straightness legal game
  (`analysis/violation_divergence/rule_curvature/README.md` § θ is comparable).
- **They are not the same construction.** A rule point is a branched move inside
  a game — one illegal token at *t0* ∈ [10, 34], then the source game's real
  suffix. A game-class point is a whole game. This is the same call panel (a) of
  `rule_curvature` makes, and for the same reason: the straightness violation
  class is a no-flip placement with the turn passed, and putting it under one
  word with these would name two different things.
- **The comparison against legal is nearly matched anyway.** 1998 of the 2000
  straightness legal games are a divergence source game token for token, so the
  black class is, to 99.9%, the unbranched version of the games the rules branch.

A rule's point is the mean over its member categories of each member branch's own
21-D vector — R1 over 2 categories, R2 over 2.77 on average, R3 over 1. The
members are not drawn; one point per game per rule is the whole rule population.

## Method notes

- **Each space gets its own PCA, fit on its own z-scored features.** The 21
  components mix degrees with activation-scale lengths, and the absolute scale
  of the step-length block differs ~450× across dictionaries (SpaDE ~0.05, raw
  layer 5 ~22.8). PCA on raw covariance answers a different question: it lands
  PC1 on turning-angle variance in all nine spaces, reporting units rather than
  structure.
- **The fit runs over the six classes drawn**, never over all of `games.pt`. The
  undrawn random-token block would otherwise set the axes and the variance ratios
  of a plane it does not appear in (`analysis/straightness/README.md` § Design
  notes).
- **13 of the 21 components cannot see the rule.** `d1..d7` are the steps out of
  positions 2..8 and `θ1..θ6` pivot at positions 3..8, all strictly before the
  earliest *t0* = 10, so they are identical across a game's three branches and
  identical to its source game. Only the 8 moments, which run over the whole
  trajectory, carry the violation at all. That is most of why the three rules
  land on top of each other below, and it is a property of ReStraV's descriptor
  rather than of these games.
- **Component signs are pinned**, since they are otherwise arbitrary: raw layer
  5 puts its largest loading positive, and every SAE space takes the sign that
  agrees with raw's, so panels are not mirrored against each other for no
  reason.
- The cloud is drawn in one shuffled pass, so no class sits on top of another;
  identity rides on the 50% contour and the centroid marker. Ellipses are a
  dispersion summary, not a confidence interval.

## Caption

The figure carries no prose block — only its panel titles, axis labels and
legend. What a reader needs in order not to over-read it is here, and this is the
caption it is published with (FIGURE_GUIDELINES §5).

**Championship games are the only class that moves in the 21-D descriptor's
principal plane, and the three constraint rules do not.** One point per game in
the top-2 principal plane of that space's z-scored ReStraV features, drawn in one
shuffled pass so no class sits on top of another; ellipses enclose 50% of a
Gaussian fit per class (a dispersion summary, not a confidence interval) and large
markers are class means. n per class is in the legend: 2000 everywhere. Legal,
rollout and championship are whole games; **R1/R2/R3 are branched moves inside
games** — one point is one game's mean over that rule's member categories, on a
branch that keeps its source game's real suffix, forking at *t0* ∈ [10, 34]. Each
space fits its own basis on the six classes drawn, so **panels compare only in
within-panel arrangement, never point-for-point**; the alignment between bases is
the table below. Axis labels carry each panel's own variance ratio. PCA is
unsupervised — the supervised LR-21 probe separates classes this plane does not.
Raw (uncentered) variant, layer 5, seed 42, spliced branch tree.

## Results

**PC1 is a step-length axis in all nine spaces**, carrying 16–23% of the
variance against PC2's 11–14% — a shallow spectrum, as befits six classes that
mostly overlap. Its largest loading is in the `d` block in 9 of 9; PC2 leads on a
turning-angle component in 7 of 9, the exceptions being SpaDE and Standard. The
panels do not name their components — nine of those labels crowd a grid that is
already carrying six classes — so `make_figures.py` prints them instead.

**Championship games are the only class that moves in the plane, and the three
rules do not.** As a class mean minus legal's, in pooled SD along each PC
(`class_shift`), championship reaches 2.07; no rule exceeds 0.40 on either
component in any space, and rollout never exceeds 0.27. Unsupervised structure
says nothing here about information: the supervised LR-21 probe reaches 0.66 on
violations and 0.96 on championship games in this same 21-D space (see
`analysis/straightness/README.md`).

**The rules do order themselves, faintly and in one direction.** Wherever the
displacement is large enough to read — raw L5, TopK, GatedAnneal, GatedSAE,
Standard — it runs R2 > R3 > R1 on PC2, i.e. bracketing moves the descriptor
most and an occupied-square placement least. Treat it as suggestive: the three
means average over 2, 2.77 and 1 member categories, so they are not equally
smoothed, and in the four spaces where every rule sits under 0.07 the ordering
is noise.

|cos| between each space's principal component and raw layer 5's, with each
class's displacement from legal alongside (PC1, PC2):

| space | PC1 var | PC2 var | \|cos\| PC1 | \|cos\| PC2 | rollout | championship | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|
| raw L5 (reference) | 21% | 13% | 1.00 | 1.00 | +0.00, −0.08 | +2.07, −0.06 | −0.07, −0.28 | −0.14, −0.40 | −0.11, −0.34 |
| BatchTopK | 22% | 13% | 0.93 | 0.83 | −0.02, −0.04 | +1.29, −0.73 | −0.07, −0.01 | −0.13, +0.04 | −0.11, +0.03 |
| TopK | 21% | 13% | 0.91 | 0.95 | +0.01, −0.08 | +1.84, +0.07 | −0.01, −0.26 | −0.05, −0.38 | −0.03, −0.32 |
| PAnneal | 21% | 13% | 0.92 | 0.83 | −0.05, +0.12 | +1.76, +0.02 | −0.01, −0.06 | −0.03, −0.04 | −0.02, −0.03 |
| GatedSAE | 18% | 13% | 0.88 | 0.82 | −0.21, +0.02 | +0.01, −0.34 | −0.03, +0.04 | −0.11, +0.10 | −0.07, +0.08 |
| GatedAnneal | 17% | 11% | 0.85 | 0.70 | −0.20, −0.02 | +0.60, −0.10 | −0.10, +0.18 | −0.16, +0.35 | −0.13, +0.31 |
| Standard | 18% | 14% | 0.69 | 0.32 | −0.05, +0.04 | +0.10, +0.48 | −0.03, +0.02 | −0.07, +0.07 | −0.06, +0.06 |
| SpaDE (γ=1e-4) | 16% | 14% | 0.31 | 0.54 | +0.27, −0.22 | +0.67, +0.35 | −0.07, −0.12 | −0.10, −0.15 | −0.08, −0.11 |
| CosineSpaDE (γ=1.0) | 23% | 14% | 0.20 | 0.06 | −0.11, −0.02 | +0.78, −0.06 | −0.00, −0.03 | +0.02, −0.02 | +0.01, −0.02 |

**Only the hard-sparsity dictionaries reproduce raw layer 5's plane.** The two
sparsemax dictionaries are the outliers, and CosineSpaDE's plane is essentially
unrelated to raw's (0.20 / 0.06) — it re-expresses the descriptor rather than
passing it through, which is the same story the rest of the study tells about
sparsemax spaces. Standard is a milder version of it. These alignments are much
lower than a fit over all the stored classes gives: one gross out-of-distribution
direction then dominates every space's PC1, and the spaces agree about that
direction rather than about the descriptor. Restricting the fit to the drawn
classes is what exposes the disagreement.

## What the CosineSpaDE cluster is

CosineSpaDE's panel has a second cloud up and to the right, at PC1 > 2, PC2 > 3:
540 of the 12000 drawn points. **It is a property of the game, not of the class
or the branch.** The three rules put 115, 115 and 114 points there, and 114 of
them are the same source games under all three; those 114 games are exactly the
114 legal games in the cluster, matched token for token. 82 rollouts land there
and no championship game does.

The mechanism is **delayed code recruitment**. Those games activate far fewer
codes in the opening than the rest (mean L0 at board positions 2 and 3: 13 and
34, against 44 and 114), then catch up by position 4 (116 against 121). The
catch-up is a large displacement: the step out of position 3 is 0.385 against
0.197, +7.6 pooled SD, which is what the plane loads on. By mid-game the two
groups are indistinguishable (step-length ratio 0.99 past position 10).

This explains an entry left open in PROCESS.md: CosineSpaDE's curvature profile
has a sharp dip near t=3 that no other space shows, guessed there to be
"sparsemax code turnover." It is that, it is confined to a 3% subpopulation
rather than being a shift in the whole distribution — a per-position mean is the
wrong summary for it — and branching an illegal move into one of those games does
not move it out.

Seed 42 only, raw (uncentered) variant, spliced branch tree.
