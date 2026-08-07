# Curvature at the fork: one game's three rule violations, against the population

The straightness study measures trajectory curvature over thousands of games and
reports it per *class*. This study measures the same quantity on a single game's
**fork** — game 1235 branched at *t0* = 20 into the move as played, two legal
controls, and one move per illegal category — and groups the six illegal
categories into the three constraint rules they violate.

The fork is n = 1, so panel (a) carries the same three rules over a population of
games, beside the class baselines and on the same axis. That is the panel a claim
can rest on; (b)–(c) show what one instance of the construction looks like.

**That population is one fork position.** *t0* is round-robin over [10, 34]
across the 2000 rule-complete games, so pooling them puts *where* the violation
happened inside the rule curves. Panel (a) takes the **80 games that fork at
*t0* = 20** — game 1235's own position, so (a)'s dashed line is the branch point
(b) locks to. It costs a factor of 25 in n and the interval is ~5× wider;
what survives that is in the captions below, and `--t0 all` restores the pooled
panel.

The figure is rendered from **two tensor trees, one code path**. The *spliced*
tree (`$DIR/branches/`, `$DIR/acts/`) gives every branch the source game's
`tokens[t0+1:]`. The *resampled* tree (`$DIR/rollout/...`) samples each branch's
continuation from OthelloGPT, K = 4 times. Same game, same *t0*, same three
rules, so the difference between the two figures is attributable to the suffix
and to nothing else. Two scripts would be two implementations of one rule and
would drift.

```
python -m spade.violation_divergence.rule_curvature --tree branches [--t0 20|all]
python analysis/violation_divergence/rule_curvature/make_figures.py \
    [--tree branches|rollout] [--game 1235] [--t0 20|all] \
    [--space raw|<family>|all] [--camera-ready]
```

Measurement: `spade.violation_divergence.rule_curvature`. The script only draws,
and prints the numbers quoted below so they can be re-derived without measuring a
figure by eye. Panel (a)'s rule curves are the 80 games forking at *t0* = 20,
× ~9 branches × 9 spaces per tree, cached to
`$DIR/[rollout/]rule_population/<space>_t0-<t0>.npz` — *t0* is in the name
because a population at another fork position is another population and
skip-if-exists must not hand one back for the other. That pass is ~1 min on an
H100 (`scripts/slurm/studies/violation_divergence/submit_rule_population.sbatch`).

Encoding those activations is what the pass costs, not the geometry, so it
answers a second question on the way past: each game's 21-D ReStraV descriptor
per rule, to `$DIR/[rollout/]rule_descriptors/`. That is the one population in
the repo with an R1 in it, and `analysis/straightness/feature_pca/` draws the
rules as classes from it. It is **not** restricted to one *t0* — it is all 2000
rule-complete games — so a run that has to rebuild it is the expensive one (~7
min on an H100 for the spliced tree, 26 min for the resampled tree's 4 rollouts
per branch, and ~3 h on a one-core login node under its 8 GB cap) and the
population is selected inside that pass rather than costing a second one.
Skip-if-exists is per product.

| file | what it is |
|---|---|
| `curvature_rules_{tree}_raw_L5` | the three panels in raw layer 5 |
| `curvature_rules_{tree}_{family}` | the same three panels in one SAE's latents (8 families) |
| `curvature_rules_{tree}_combined` | panel (b) alone in all nine spaces, shared axes, over the board |

`{tree}` is `branches` (spliced) or `rollout` (resampled).

## θ is comparable, and only because of the sampling convention

Panel (a)'s three class profiles are the straightness study's committed geometry
npz; its three rule curves and panel (b) are computed here. Putting them on
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

**Illegal moves straighten the layer-5 trajectory: by two to four degrees across
a population that all forks at the same position, and by ten to twenty degrees at
the branch in the one game drawn here — where everything after the branch is
confounded with an out-of-distribution continuation. At n = 80 that population
effect is resolved for R2 and R3 and is not resolved for R1.** Panel **(a)**:
mean turning angle against board position, for two populations on one axis. *Solid*, the straightness
study's three legal-by-construction classes — legal games (the baseline),
OthelloGPT temperature-1 rollouts, and human championship games — each a mean ±
95% CI over **n = 2000 games**, read from that study's committed
`resid_post_5_raw.npz`. Its own `violation` class is deliberately absent: that is
a different construction (a no-flip placement with the turn passed) and would put
two unrelated violations under one word. Those classes are whole games and have
no fork, so *t0* does not apply to them at all. *Dashed*, the three constraint
rules, measured here on the same tree the fork panels read, over the source games
that **fork at *t0* = 20**: **n = 80 games for each of R1, R2 and R3**, mean ±
95% CI over games at the markers.
`CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are the same blue, so
the dash and the rule marker are what keep six series apart; the dash also says
these are branched moves inside games rather than whole games. Every rule curve
forks at the dashed vertical line, which is also game 1235's own *t0* — the
branch point panel (b) locks to and panel (c) draws the board of.
Panel **(b)**: the violation-locked Δ turning angle for game 1235, against its
own reference branch — the move as played, which is this figure's "matched source
game" — with a legal control in the null styling, at **n = 1 game and with no
uncertainty display**. One path per rule is the mean over its member categories
of each member's own Δ, and the R1/R2/R3 key under the panels names those members;
**R3 has one**, so its path is a single branch and not an average. Δ ≡ 0 for
*t* − *t0* ≤ −2 by construction (shaded): every branch shares the prefix, and the
turn at *t0* − 2 is the last whose window excludes *t0*. Panel **(c)**: the
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
per-ply θ as everything else in this figure. Four consequences a reader should
carry:

- **n = 80 per rule, and it is 80 and not 2000 on purpose.** *t0* is round-robin
  over [10, 34] across the 2000 rule-complete games, ~80 each; pooling them means
  averaging violations committed at 25 different points in the game, and fork
  position moves curvature on its own. Holding *t0* = 20 removes that confound
  and costs a factor of 25 in n, so the CI half-widths go from ~0.19° to ~0.95°.
  **n is 80 for all three rules**, and equal because every game in the
  rule-complete set carries a member of every rule by construction — that is what
  `rule_complete_games.json` is — so the three curves are the same 80 games.
- **A rule's member composition varies by game**, unlike the 2, 3 and 1 that the
  branch-complete numbers elsewhere in this study are built on: R1 averages
  exactly 2 categories in every game, R2 averages **2.80** at *t0* = 20 and R3
  always 1. The order is fixed regardless — rollouts, then member categories
  inside a game, then games — because φ does not commute with the mean
  (`test_curvature_plane.py`).
- **The rule curves belong to their tree and the class curves do not.** The
  classes are the straightness study's and know nothing about either tree; the
  rules are rebuilt per tree, so the spliced and resampled figures differ in (a)
  as well as in (b), by the suffix and nothing else.
- **The classes stay at n = 2000 and are not restricted to anything.** They are
  whole legal-by-construction games with no injected move, so there is no fork
  position to hold fixed; the legend says `forked at t0 = 20` on the rule entries
  only, and nothing in the panel should be read as putting a *t0* on a class.

**What (a) now shows, in numbers, because it is not a gap you can see.** Raw
layer 5, spliced tree, from the script's own output: the largest deviation from
the legal class is **R1 −2.07° at *t* = 32, R2 −4.18° at *t* = 21, R3 −3.39° at
*t* = 21**, each against a 95% CI half-width of **0.95–0.99°**. Moving from all
2000 games to the 80 at *t0* = 20 changed two things at once, and they point
opposite ways:

- **The effect is about three times larger.** Pooled it read R1 −0.74°, R2
  −1.31°, R3 −1.05°. Pooling averages violations committed at 25 different
  positions, so at any given *t* most games had not forked yet or had forked long
  before; the pooled curve is a smeared version of this one, not a
  better-estimated one.
- **The interval is five times wider**, 0.19° → 0.96° — exactly the √25 the
  sample size predicts, and the reason a single *t0* is not free.

The ordering **R2 > R3 > R1** holds in 8 of the 9 spaces (GatedSAE puts R1 above
R3, by less than a CI half-width). CosineSpaDE is the extreme (R2 −13.21° at
*t* = 19) and GatedSAE the mildest (R2 −1.94°). For scale, the championship class
sits +1.06° above legal over positions 10–20 and −0.98° below it over 46–57 (that
study's committed npz).

**Left of *t0* − 2, (a) is its own empirical null, and that is the yardstick this
panel should be read against.** No branch exists there — the turn at *t0* − 1 is
the first whose window touches the branch token — so the rule curves and the
legal class are near-identical game sets measured by two pipelines, and whatever
they differ by is sampling plus pipeline. At n = 2000 that residual was ≤ 0.25°,
a fifth of the effect. At n = 80 it is **1.61° in raw layer 5 and 1.09–1.66°
across the nine spaces**, and it is the same kind of statistic as the peak — a
maximum of |Δ| over positions — which the peak's own CI is not. Against it:

- **R2 clears the residual in all nine spaces**: raw 4.18 vs 1.61 (2.6×), from
  1.4× (GatedSAE) to 8.6× (CosineSpaDE).
- **R3 clears it in eight**: raw 2.1×; GatedSAE's 1.05× is not a separation.
- **R1 clears it nowhere**: 1.0–1.5× in every space, raw layer 5 included
  (2.07 against 1.61). At n = 2000 R1's −0.74° was 3× its 0.25° residual. **The
  R1 population effect does not survive at one fork position**, and this figure
  does not support a claim about R1 that the pooled panel appeared to support.
  The resampled tree removes the remaining doubt: there R1's largest deviation is
  *itself pre-fork*, in all nine spaces.

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

**The legal control is drawn, and it is the point.** In panel (b) a legal
alternative move produces a Δ of the same order as the illegal branches (raw
layer 5: at *t* − *t0* = −1, the two legal controls read −7.99 and −12.38 deg
against −19.11 .. +5.11 across the six illegal categories; at *t* − *t0* = 0,
−5.14 and −9.25 against −13.60 .. −3.46 — the panel draws the first of each pair,
since the second said nothing the first did not). The separation this study is
about is not visible at n = 1, and a figure showing only the illegal branches
would imply it is. The population numbers with CIs over 13,531 branches are in
`../README.md`.

### `curvature_rules_rollout_*` — the resampled tree

**The same three panels with every branch's continuation sampled from OthelloGPT
instead of spliced.** `tokens[:t0]` and the branch token are kept and the
remaining tokens are sampled at temperature 1.0, K = 4 per branch, with the
reference resampled on the same footing — replaying its real continuation would
have left it the one in-distribution sequence and fixed nothing
(`rollout_branches`). Panel (a)'s three class curves are unchanged — they belong
to the straightness study and know nothing about either tree — but its three rule
curves are not: they are rebuilt on this tree, so they carry the resampled
suffix, and their legend entries read **80 games × 4 rollouts, forked at
*t0* = 20**.

Three differences from the spliced captions. A rule now averages over **member
categories × 4 rollouts**, and the legend says so; a branch's curve is the mean
over its rollouts. **R3 averages 4 sequences here** where on the spliced tree it
averages 1, so its spread is sampling variance of the continuation rather than
spread across categories. And
Δ is paired **per rollout**: the k-th sample of a branch and the k-th of the
reference are drawn from one uniform stream, so they are common random numbers
and the branch effect is read against a shared noise draw.

**What panel (a) says on this tree.** Same yardstick as the spliced caption — the
peak against the pre-fork residual, which is bit-identical between the trees.
**R2 clears it in all nine spaces** (raw 4.24 vs 1.61, 2.6×; 1.4–8.6× across the
nine). **R3 clears it in five** (raw 1.8×; in GatedSAE, TopK and BatchTopK its
extremum is itself pre-fork). **R1 clears it in none, and its extremum is pre-fork
in every space** — the resampled tree says plainly what the spliced tree only
implied, that there is no R1 population response at *t0* = 20.

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
| panel (a), R1 peak vs. legal | **−2.07 ± 0.95** at *t* = 32 | **−1.61 ± 1.00** at 5 — *pre-fork* |
| panel (a), R2 peak | **−4.18 ± 0.99** at 21 | **−4.24 ± 0.50** at 21 |
| panel (a), R3 peak | **−3.39 ± 0.96** at 21 | **−2.95 ± 1.11** at 19 |
| panel (a), pre-fork residual | **1.61** | **1.61** |

Five readings, and the second is the one that changes what may be claimed.

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

- **At one fork position, the population straightening barely notices the
  resampling — for R2 and R3.** R2 reads −4.18° spliced and −4.24° resampled, at
  the same position (*t* = 21) both times; R3 −3.39° → −2.95° (87%). That is a
  much stronger statement than the pooled panel supported, where R2 kept 74% and
  R3 61% of its spliced size: with *t0* held fixed, what panel (a) shows is not
  the shared out-of-distribution suffix.
- **R1 has no post-fork response on the resampled tree at all.** Its largest
  deviation from the legal class is at *t* = 5 — pre-fork, where no branch
  exists — in **all nine spaces**. So R1's spliced −2.07° at *t* = 32 is the
  suffix and the noise, and there is no R1 effect left to order: R2 > R3 > R1
  holds in 8 of 9 spaces spliced but in 6 of 9 resampled, and in the three that
  break it R3's own extremum is pre-fork too.
- **The pre-fork residual is identical between the trees** (1.61° in raw layer 5
  in both, and equal in all nine spaces), which is the check that this comparison
  is well-posed: the two trees share `tokens[:t0]`, so their activations left of
  the fork are the same numbers and their sampling residual must be too.

One hypothesis this does **not** support: that the shared suffix is what makes
the six illegal branches' post-*t0* responses co-vary. Mean pairwise correlation
between them over *t* ≥ *t0* is **+0.536 spliced and +0.580 resampled** — the
co-variation is the game, not the splice.

## Reading it honestly

- **n = 1, and a selected game.** Panels (b)–(c) are an illustration of the
  construction, not a claim-supporting figure, and they carry **no uncertainty
  display at all** — no ribbon, no bar, nothing faint behind the means. At n = 1
  there is nothing over which an interval could be taken that a reader should
  believe; (a) is where the interval lives, over games. Every other population
  number belongs to `../README.md`, where it carries a CI over 13,531 branches.
- **(a) is n = 80, and the confound it removes cost it the resolution to say
  everything it used to.** One fork position is the only way the panel's dashed
  line and its rule curves mean the same thing, but 80 games put the CI at ~0.96°
  and the pre-fork sampling residual at ~1.6°. R2 and R3 stand clear of that; R1
  does not. Anything read off R1 in this panel is a claim the data does not
  support at this n — `--t0 all` pools all 2000 games and gets the resolution back
  by putting fork position back into the curves, which is a different figure with
  a different weakness, not a stronger version of this one.
- **One axis, two constructions.** (a) draws whole games and branched moves
  together, and they are told apart by line style rather than colour, because
  `CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are the same blue and
  colour alone would name one series twice.
- **Three lines are six categories.** The rule means are a labelling convenience;
  the spread ratio near 1 says the members barely cohere. If a claim ever rests on
  a rule it has to be scored per category and aggregated with a CI.
- **Angles are scale-free, so the nine spaces share panel (b)'s y-axis.** Step
  lengths and latent norms would not — SpaDE codes live on a simplex — which is
  why this figure draws the angle and nothing else.
- **The sampling convention is stated because it flips signs.** See the θ section
  above; a curve measured at stride 2 must never be drawn against these classes.
