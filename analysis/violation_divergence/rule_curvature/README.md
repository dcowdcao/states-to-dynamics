# Curvature at the fork: one game's three rule violations, against the population

The straightness study measures trajectory curvature over thousands of games and
reports it per *class*. This study measures the same quantity on a single game's
**fork** — game 1235 branched at *t0* = 20 into the move as played, two legal
controls, and one move per illegal category — and groups the six illegal
categories into the three constraint rules they violate.

The fork is n = 1, so panel (a) carries the same three rules over **every source
game**, beside the class baselines and on the same axis. That is the panel a
claim can rest on; (b)–(d) show what one instance of the construction looks like.

The figure is rendered from **two tensor trees, one code path**. The *spliced*
tree (`$DIR/branches/`, `$DIR/acts/`) gives every branch the source game's
`tokens[t0+1:]`. The *resampled* tree (`$DIR/rollout/...`) samples each branch's
continuation from OthelloGPT, K = 4 times. Same game, same *t0*, same three
rules, so the difference between the two figures is attributable to the suffix
and to nothing else. Two scripts would be two implementations of one rule and
would drift.

```
python -m spade.violation_divergence.rule_curvature --tree branches   # panel (a)'s rules
python analysis/violation_divergence/rule_curvature/make_figures.py \
    [--tree branches|rollout] [--game 1235] [--space raw|<family>|all] [--camera-ready]
```

Measurement: `spade.violation_divergence.rule_curvature`. The script only draws,
and prints the numbers quoted below so they can be re-derived without measuring a
figure by eye. Panel (a)'s rule curves are 2000 games × ~9 branches × 9 spaces
per tree, cached to `$DIR/[rollout/]rule_population/`; that is ~3 h on a
one-core login node under its 8 GB cap, so
`scripts/slurm/studies/violation_divergence/submit_rule_population.sbatch` runs
it on a GPU instead (~7 min on an H100 for the spliced tree). Skip-if-exists,
like every other step of this study.

Encoding those activations is what the pass costs, not the geometry, so it
answers a second question on the way past: each game's 21-D ReStraV descriptor
per rule, to `$DIR/[rollout/]rule_descriptors/`. That is the one population in
the repo with an R1 in it, and `analysis/straightness/feature_pca/` draws the
rules as classes from it. Skip-if-exists is per product — a space is re-encoded
only for the npz it is missing.

| file | what it is |
|---|---|
| `curvature_rules_{tree}_raw_L5` | the four panels in raw layer 5 |
| `curvature_rules_{tree}_{family}` | the same four panels in one SAE's latents (8 families) |
| `curvature_rules_{tree}_combined` | panel (c) alone in all nine spaces, shared axes, over the board |

`{tree}` is `branches` (spliced) or `rollout` (resampled).

## θ is comparable, and only because of the sampling convention

Panel (a)'s three class profiles are the straightness study's committed geometry
npz; its three rule curves and panels (b)–(c) are computed here. Putting them on
one scale is legitimate **only** under the per-ply convention — positions 1..58,
one angle per point, no stride and no trailing window, for the many-game rules
exactly as for the single game — and that was checked rather than assumed:

- Both sides call `spade.straightness.geometry.temporal_geometry` on raw
  `resid_post_5`, cast to float32, with position 0 (the attention sink) dropped,
  so θ column *k* is the turn at board position *k* + 2 in both.
- The two studies stream the same dataset at the same seed. **1998 of this
  study's 2000 reference branches are token-identical to a straightness legal
  game**, and on those games the two pipelines' θ agree to **max 1.5 × 10⁻⁵ deg**
  against a 142 deg level — float noise, not a convention difference. Over the
  whole class the mean profiles differ by at most **0.012 deg** against a typical
  SEM of 0.094 deg.
- The prefix invariant holds exactly and independently: Δθ against the reference
  branch is **0.0** at every turn up to position *t0* − 2, and the first nonzero
  value is the turn at *t0* − 1, which is where the straightness study's own
  alignment says it must be.

**The convention is not free, and the sibling audit is the reason this section
exists.** The fork figures of this study (`fork_plane.positions`) sample every
`STRIDE = 2` plies at *t0*'s parity, because layer 5's board code is
mover-relative and the per-ply path is a near-reversal sawtooth. The same nine
branches of the same game read **107.9 deg** under that window and **142.9 deg**
here; the audit on `fig/theta-comparability` measures the two at −0.508
correlation over 10,000 games, with every class contrast changing sign
(violation 0.397 → 0.533, championship 0.509 → 0.782). No rescale reconciles
them. This study therefore recomputes the branches per ply rather than
re-deriving the class baselines at stride 2 — the latter would be a new
measurement to which the straightness study's committed AUROCs do not apply. The
R²¹ trailing-window variant (`curvature_plane.windowed`) is not used here at all:
at W = 8 a point plotted at *t* is a 14-ply average whose vertex centroid is
*t* − 7, which an event-locked panel cannot carry.

## Captions

The figures carry no prose block. What a reader needs in order not to over-read
them is here, per figure, and this is the caption each is published with
(FIGURE_GUIDELINES §2, §5). What stays inside the figure is the apparatus a
reader cannot look up while looking at it: panel titles and letters, per-panel
space names, axis labels, the legend, the board's reading key, and the R1/R2/R3
key naming each rule's member categories.

### `curvature_rules_branches_*` — the spliced tree

**Illegal moves straighten the layer-5 trajectory: by about a degree across the
whole population, and by ten to twenty degrees at the branch in the one game
drawn here — where everything after the branch is confounded with an
out-of-distribution continuation.** Panel **(a)**: mean turning angle against
board position, for two populations on one axis. *Solid*, the straightness
study's three legal-by-construction classes — legal games (the baseline),
OthelloGPT temperature-1 rollouts, and human championship games — each a mean ±
95% CI over **n = 2000 games**, read from that study's committed
`resid_post_5_raw.npz`. Its own `violation` class is deliberately absent: that is
a different construction (a no-flip placement with the turn passed) and would put
two unrelated violations under one word. *Dashed*, the three constraint rules,
measured here on the same tree the fork panels read, over every source game that
carries them: **n = 2000 (R1), 2000 (R2) and 1997 (R3) games**, mean ± 95% CI
over games at the markers.
`CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are the same blue, so
the dash and the rule marker are what keep six series apart; the dash also says
these are branched moves inside games rather than whole games. Their *t0* is not
one position — it is round-robin over **[10, 34]**, shaded — so the dashed
vertical line is game 1235's alone and the rule curves have no single fork.
Panel **(b)**: the same quantity for game 1235's nine branches, on (a)'s x and y
axes, with **n = 1 game and no uncertainty display**. One path per rule is the
mean over its member categories of each member's own θ curve; every legend entry
carries n, including **`R3 turn (mean of 1)`**, which is one branch and not an
average. Panel **(c)**: the violation-locked Δ turning angle against the game's
own reference branch — the move as played, which is this figure's "matched source
game" — with both legal controls in the null styling, again at **n = 1 game and
with no uncertainty display**. Δ ≡ 0 for *t* − *t0* ≤ −2 by construction
(shaded): every branch shares the prefix, and the turn at *t0* − 2 is the last
whose window excludes *t0*. Panel **(d)**: the
position at the end of the shared prefix, discs in absolute colour, Black to
move; an outlined cell is a branch square with its rule marker in the corner so a
disc underneath stays visible, the small dashed translucent disc in the opposite
corner is **the disc that branch would place — never a disc on the board**, and
the dashed ring marks a disc the played move flips.

**Panel (a)'s rules are measured here because the straightness study cannot
supply them.** That study's violation class is built from `EMPTY_CATEGORIES =
[3, 4, 5, 6]`, empty-square violations only, so it contains **no R1 at all** and
no R1/R2/R3 split at n = 2000 exists there to read. The rules come off
`branches.pt` instead, one branch per illegal category per game, under the same
per-ply θ as everything else in this figure. Three consequences a reader should
carry:

- **n is per rule and the three do not match: 2000, 2000, 1997.** A rule takes
  every game with at least one of its member categories available at that game's
  *t0*. Every game affords an R1 and an R2 branch; three afford no
  `opponent_only` square, so an R3 violation cannot be constructed there at all —
  a property of those positions, not a filter. Cross-rule comparisons therefore
  rest on sets differing by 3 games in 2000, which is not worth correcting for.
- **A rule's member composition varies by game**, unlike the 2, 3 and 1 that the
  branch-complete numbers elsewhere in this study are built on: R1 averages
  exactly 2 categories in every game, R2 averages **2.77** and R3 always 1. The
  order is fixed regardless — rollouts, then member categories inside a game,
  then games — because φ does not commute with the mean
  (`test_curvature_plane.py`).
- **The rule curves belong to their tree and the class curves do not.** The
  classes are the straightness study's and know nothing about either tree; the
  rules are rebuilt per tree, so the spliced and resampled figures differ in (a)
  as well as in (b)–(c), by the suffix and nothing else.

**What (a) now shows, in numbers, because it is not a gap you can see.** One
illegal move straightens the whole population's layer-5 trajectory, by a fraction
of a degree on an axis that (b) forces to span thirty. Raw layer 5, spliced tree,
from the script's own output: the largest deviation from the legal class is **R1
−0.74° at *t* = 36, R2 −1.31° at *t* = 22, R3 −1.05° at *t* = 21**, each against a
95% CI half-width near 0.19°, so 4–7 CI widths. The ordering **R2 > R3 > R1**
holds in all nine spaces; CosineSpaDE is the extreme (R2 −3.74° at *t* = 21) and
GatedSAE the mildest (R2 −0.57°). For scale, the championship class sits +1.06°
above legal over positions 10–20 and −0.98° below it over 46–57 (that study's
committed npz), so a single illegal move shifts the population's curvature by
about what swapping the whole corpus for human games does.

**Left of the first *t0*, (a) is its own sanity check.** No branch exists there,
so the rule curves and the legal class are near-identical game sets measured by
two pipelines; they agree to **≤ 0.25°** in every space. That residual also
bounds how much of the gap above could be pipeline rather than violation, and it
is a fifth of the effect.

**The post-*t0* half of this figure conflates the violation with an
out-of-distribution continuation, and that is not a caveat but a property of the
tree.** Every branch replays the source game's `tokens[t0+1:]`, which is a legal
continuation of the *reference's* board and of no other. It survives for the
legal control in only **76.8% of games (median 2 moves)**, and after an illegal
branch there is no board for it to be legal on at all — its suffix legality is
undefined, not low (`../suffix_legality`). The reference branch is the one
sequence with a legal future by construction, so **any post-*t0* difference in
this figure is the violation plus an OOD suffix, in unknown proportion**. Only
the response at *t* − *t0* = −1 is clean: its window is positions *t0* − 2 ..
*t0*, which the tree fixes. Read the resampled figure beside it, and the section
below for what moves.

**Game 1235 is selected, not typical.** It is the argmax, over all 1550
branch-complete games, of the mutual separation of its six illegal displacement
trajectories, scored in raw layer 5 at full 512-d width and never in an SAE space
or a projection (`../game_selection`); population median 0.705, this game 1.080.
It is held fixed across both trees because the two figures must share a game or
the spliced-vs-resampled difference cannot be attributed.

**The rule grouping is weak, and this number and not the three means is the
evidence.** Within-rule ÷ between-rule mean pairwise distance, on each
branch's Δ turning angle over *t* ≥ *t0*, is **0.906** in raw layer 5 and
0.858–0.956 across the nine spaces. A rule's members travel barely closer to each
other than to another rule's; the same statistic on the full 512-d displacements
of this game is 0.885, against a population median of 0.945 [IQR 0.872–1.020]
over all 1550 branch-complete games, above 1 in 30% of them
(`python -m spade.violation_divergence.rule_groups`).

**Both legal controls are drawn, and they are the point.** `../fork_and_board`
draws one; this figure keeps both, because in panel (c) a legal alternative move
produces a Δ of the same order as the illegal branches (raw layer 5: at
*t* − *t0* = −1, controls −7.99 and −12.38 deg against −19.11 .. +5.11 across the
six illegal categories; at *t* − *t0* = 0, controls −5.14 and −9.25 against
−13.60 .. −3.46). The separation this study is about is not
visible at n = 1, and a figure showing only the illegal branches would imply it
is. The population numbers with CIs over 13,531 branches are in `../README.md`.

### `curvature_rules_rollout_*` — the resampled tree

**The same four panels with every branch's continuation sampled from OthelloGPT
instead of spliced.** `tokens[:t0]` and the branch token are kept and the
remaining tokens are sampled at temperature 1.0, K = 4 per branch, with the
reference resampled on the same footing — replaying its real continuation would
have left it the one in-distribution sequence and fixed nothing
(`rollout_branches`). Panel (a)'s three class curves are unchanged — they belong
to the straightness study and know nothing about either tree — but its three rule
curves are not: they are rebuilt on this tree, so they carry the resampled
suffix, and their legend entries read **games × 4 rollouts**.

Three differences from the spliced captions. A rule now averages over **member
categories × 4 rollouts**, and the legend says so; a branch's curve is the mean
over its rollouts. **R3 averages 4 sequences here** where on the spliced tree it
averages 1, so its spread is sampling variance of the continuation rather than
spread across categories. And
Δ is paired **per rollout**: the k-th sample of a branch and the k-th of the
reference are drawn from one uniform stream, so they are common random numbers
and the branch effect is read against a shared noise draw.

**What this tree buys and what it costs.** It removes the OOD-suffix confound
above. It also removes single-token causal attribution: two branches now differ
at *t0* **and at every position after it**, so a displacement is the branch move
plus everything the model then chose to do about it. K = 4 separates sampling
variance from the branch effect; it does not separate the move from its
consequences. `branches.pt` remains the tensor for the single-token claim.

## What changes between the two trees

Raw layer 5, game 1235, from the scripts' own output:

| | spliced | resampled |
|---|---|---|
| R1 at *t0*−1 | **+4.11** | **+4.11** |
| R2 at *t0*−1 | **−7.17** | **−7.17** |
| R3 at *t0*−1 | **−19.11** | **−19.11** |
| R1 tail (mean Δ, *t* ≥ *t0*+2) | −0.42 ± 0.47 | −1.09 ± 0.95 |
| R2 tail | −2.82 ± 0.90 | −3.41 ± 0.98 |
| R3 tail | −3.39 (one sequence, no SEM) | −3.89 ± 1.01 |
| corr of the two mean curves over *t* ≥ *t0* | — | R1 +0.20, R2 +0.38, R3 +0.60 |
| within ÷ between rule spread | 0.906 | 0.944 |
| panel (a), R1 peak vs. legal | **−0.74 ± 0.18** at *t* = 36 | **−0.27 ± 0.07** at 37 |
| panel (a), R2 peak | **−1.31 ± 0.20** at 22 | **−0.97 ± 0.15** at 22 |
| panel (a), R3 peak | **−1.05 ± 0.20** at 21 | **−0.64 ± 0.15** at 24 |

Four readings, and the second is the one that changes what may be claimed.

- **The response at *t0*−1 is bit-identical**, and it is the largest excursion in
  the figure. Its window is positions *t0*−2 .. *t0*, which both trees share, so
  it cannot be a suffix artifact. Asserted in the comparison, not eyeballed.
- **The spliced tree's SEMs understate the uncertainty by roughly 2×.** It has no
  continuation variance at all — every branch replays one fixed suffix — so its
  spread is only between-category. Adding the model's own sampling variance
  takes R1's mean SEM from 0.47 to 0.95. The tails that look resolved in the
  spliced panel are not resolved.
- **The sustained straightening survives; its fine structure does not.** The tail
  keeps its sign and grows slightly under resampling (R2 −2.82 → −3.41, R3 −3.39
  → −3.89), so the post-*t0* straightening is not an artifact of the shared
  suffix. But the two trees' mean curves correlate only **+0.20 to +0.60** over
  *t* ≥ *t0*, so the ripple a reader is tempted to interpret ply-by-ply is largely
  the one fixed suffix. R1 is where this shows: a localized trough of −7.51 at
  *t0*+1 in the spliced figure becomes −4.53 at *t0*+12, i.e. no localized trough.

- **The population straightening survives resampling, at 36% (R1), 74% (R2) and
  61% (R3) of its spliced size.** Some of what panel (a) shows on the spliced tree
  really is the shared out-of-distribution suffix — most of it for R1, which is
  the rule whose single-game structure the third reading already called mostly
  suffix. R2's peak stays at the same position (*t* = 22) in both trees, and the
  ordering R2 > R3 > R1 holds in all nine spaces on both. The resampled tree's CI
  is also 2× tighter, because K = 4 rollouts average down each game's own noise
  before the mean over games is taken.

One hypothesis this does **not** support: that the shared suffix is what makes
the six illegal branches' post-*t0* responses co-vary. Mean pairwise correlation
between them over *t* ≥ *t0* is **+0.536 spliced and +0.580 resampled** — the
co-variation is the game, not the splice.

## Reading it honestly

- **n = 1, and a selected game.** Panels (b)–(d) are an illustration of the
  construction, not a claim-supporting figure, and they carry **no uncertainty
  display at all** — no ribbon, no bar, nothing faint behind the means. At n = 1
  there is nothing over which an interval could be taken that a reader should
  believe; (a) is where the interval lives, over games. Every other population
  number belongs to `../README.md`, where it carries a CI over 13,531 branches.
- **Two registers, two axes.** (a) is a population and (b) is one game. They share
  x and y so the comparison is exact, but a single game's swing is not a
  population difference and the overlay would invite reading it as one. Inside
  (a) the two constructions — whole games, branched moves — are told apart by
  line style, because `CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]`
  are the same blue and colour alone would name one series twice.
- **Three lines are six categories.** The rule means are a labelling convenience;
  the spread ratio near 1 says the members barely cohere. If a claim ever rests on
  a rule it has to be scored per category and aggregated with a CI.
- **Angles are scale-free, so the nine spaces share panel (c)'s y-axis.** Step
  lengths and latent norms would not — SpaDE codes live on a simplex — which is
  why this figure draws the angle and nothing else.
- **The sampling convention is stated because it flips signs.** See the θ section
  above; a curve measured at stride 2 must never be drawn against these classes.
