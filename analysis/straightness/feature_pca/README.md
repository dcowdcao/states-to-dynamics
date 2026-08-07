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
python analysis/straightness/feature_pca/make_figures.py        # the figure
python analysis/straightness/feature_pca/centroid_distances.py  # how far apart the classes are
```

Reads geometry two studies already wrote — no re-extraction. `make_figures.py`
writes `figures/restrav_pca.png` and prints the basis table below;
`centroid_distances.py` draws nothing and prints the distances, because the plane
is a poor place to read a distance off by eye and a worse place to read its
absence.

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

**Championship games are the only class that moves *in this plane*, and the three
rules do not.** As a class mean minus legal's, in pooled SD along each PC
(`class_shift`), championship reaches 2.07; no rule exceeds 0.40 on either
component in any space, and rollout never exceeds 0.27.

**That is a fact about two components, not about the descriptor**, and it must not
be quoted as one — `centroid_distances.py` measures the same classes in the full
21-D and the rules are 1.13–1.68 pooled SD from legal there against a legal-vs-legal
null of 0.16 (§ *How far apart the classes actually are*). The plane holds ~34% of
the variance and is fitted without labels, so it is under no obligation to contain
the direction that separates a rule from a legal game — and it does not. This is
the same gap the supervised LR-21 probe exploits when it reaches 0.66 on violations
and 0.96 on championship games in this same 21-D space (see
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

## How far apart the classes actually are

```
python analysis/straightness/feature_pca/centroid_distances.py
```

Centroid gaps in units of the pooled within-class SD, so 1.0 means the two
centroids sit one within-class SD apart. Legal split even/odd — the study's own
train mask, 1000 games a side — is the empirical floor: two centroids that differ
only by sampling. At n = 2000 a centroid's standard error is ~0.02 SD, so nothing
here is about significance and everything is about size.

Raw layer 5, with a percentile bootstrap over games (1000 resamples) on the 21-D
column:

| pair | top-2 plane | full 21-D | 21-D 95% CI |
|---|---|---|---|
| legal ↔ championship | 1.79 | 3.51 | [3.42, 3.62] |
| legal ↔ R2 bracketing | 0.39 | 1.68 | [1.61, 1.78] |
| legal ↔ R3 turn | 0.33 | 1.42 | [1.33, 1.51] |
| legal ↔ R1 placement | 0.26 | 1.13 | [1.06, 1.23] |
| R1 ↔ R2 | 0.13 | 0.64 | [0.56, 0.74] |
| R1 ↔ R3 | 0.07 | 0.42 | [0.35, 0.53] |
| R2 ↔ R3 | 0.06 | 0.25 | [0.20, 0.38] |
| legal ↔ rollout | 0.07 | 0.17 | [0.15, 0.29] |
| legal vs legal (null) | 0.05 | 0.16 | — |

Averaged over the nine spaces: in the plane, rule-to-legal 0.16 against a null of
0.05, and **rule-to-rule 0.05, which is the null exactly** — in this plane the
three rules are not merely close, they are indistinguishable from one class
sampled twice. In the full 21-D the same comparisons are 1.00 and 0.36 against a
null of 0.18.

**All of the rule separation is carried by the 8 moments.** Per component, the
legal↔rule centroid gap is exactly 0.00 on every one of `d1..d7` and `θ1..θ6`: a
branch shares its source game's tokens before *t0* ≥ 10, and 1998 of the 2000
legal games are those source games, so the two classes are the same trajectories
there. What is left carries it all — `mean d` 0.68/0.95/0.80 for R1/R2/R3, then
`var d`, `max d` and `mean θ`. Read `min d` and `min θ` as inert (≤ 0.03).

R2 > R3 > R1 holds in the full space as it does in the plane and in
`../sae_profiles/`'s median gaps, which is three estimators agreeing. The rules
still have no AUROC with a CI anywhere — these are distances, not detection rates.

### Half of it is the suffix, not the illegal move

Everything above is the **spliced** tree, which is what the figure draws: after
*t0* a branch replays the source game's real moves onto a board the illegal move
has already changed. That continuation is off-distribution for reasons unrelated
to which rule was broken, and since the separation lives entirely in whole-game
moments with *t0* ∈ [10, 34] of 58 positions, most of the descriptor is computed
over it. `--tree rollout` is the control: same games, same *t0*, same rules, but
OthelloGPT continues from the post-violation board (K = 4). The suffix is then the
only difference between the two columns.

| pair (raw L5, 21-D) | spliced | resampled |
|---|---|---|
| legal ↔ R1 placement | 1.13 | 0.40 |
| legal ↔ R2 bracketing | 1.68 | 1.14 |
| legal ↔ R3 turn | 1.42 | 0.72 |
| R1 ↔ R2 | 0.64 | 0.99 |
| R2 ↔ R3 | 0.25 | 0.43 |
| legal ↔ championship | 3.51 | 3.41 |
| legal vs legal (null) | 0.16 | 0.16 |

Over the nine spaces, rule-to-legal falls **1.00 → 0.56** and rule-to-rule rises
**0.36 → 0.56**. So about half the distance from legal was the spliced tail, worst
for R1 (1.13 → 0.40, which is not far off the 0.16 null).

The two moving in opposite directions is the informative part: the spliced suffix
adds a displacement **common to all three rules**, which inflates every rule's gap
from legal while compressing the rules against each other. Remove it and the
rule-identity signal comes out — R1 ↔ R2 goes 0.64 → 0.99. Championship is
near-invariant (3.51 → 3.41), which is the check that the swap touched only the
branch classes, and R2 > R3 > R1 survives on both trees.

**Quote the resampled column for anything about violations, and the spliced column
only when the figure is what is being described.** The figure is drawn on the
spliced tree because that is the tree the rest of the divergence study's fork
panels read; a resampled-suffix version of it does not exist yet.

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
