# PROCESS.md

Running log of what's being worked on and the non-obvious decisions behind it, so
humans and future agents have context the code alone doesn't show. Newest first.
Keep entries short; the code is the source of truth for *what*, this is the *why*.

---

## Fig 1 titles each accuracy against its own occupancy ceiling (2026-08-05)

Branch `bracket/basis-ceiling`. Panel (c) said "held-out board accuracy 0.735" and stopped
there, which reads as partial colour ability. It is the opposite: 0.7349 is 0.004 points
*below* the 0.73494 the labels give for free — perfect empty/occupied, then always the
majority colour — while panel (b)'s 0.9829 clears its own 0.75181 by 23.1 points. Both
titles now carry accuracy, ceiling and the gap, and the shared legend defines "ceiling"
once for both. Neither number is the claim; the pair is.

The rule lived only inside `robustness.occupancy_split`. It is now
`probe_space.occupancy_ceiling(labels)` — `probe_space` owns the labels and the accuracy
table, and `robustness` already imports half its namespace from there — and `m1_basis.csv`
carries `ceiling_relative` / `ceiling_absolute`. They are a property of the labels, so they
are constant down the layer column, and they are written to five decimals where the
accuracies get four: the absolute basis's whole gap to its ceiling lives in the fifth.

**`robustness.py`'s four CSVs do not byte-reproduce on this machine, and it is not the
refactor.** Running HEAD's code here produces the same four files my version does, bit for
bit; both differ from the committed ones by one float32 ulp on the mean reductions
(5.96e-08) plus exactly one flipped argmax out of the 768,000 eval squares in the absolute
decode (Δacc 1.3e-6 = 1/768,000). CPU here against whatever the committed run used. The
committed CSVs are left as they are. `probe_space`'s side reproduces exactly: `m1_basis`'s
six pre-existing columns came back identical, its other seven CSVs byte-identical.

## The bracket flip's geometry tables get a writer, and a null that is not a median (2026-08-05)

Branch `bracket/regen-driver`. Four CSVs under `analysis/bracket_flip/results/` had no
writer: `act_geometry.py` is a library, and the A4 driver that called it was never ported
(it survives only as a copy under the frozen repo's `.claude/worktrees/`). Its
table-writing half is now `spade.bracket_flip.geometry_tables`; the plotting half stays
dropped, since `make_figures.py` owns the two surviving figures. `probe_space.py` gained the
`f2_event_meta.csv` write beside its `f2_event.csv`, same canonical event, with `move` and
`control_move` converted from vocab tokens to squares.

**The null Fig 2(a) draws was a median of ~29 draws.** `cos_null_*` is a per-event
`nanmedian` over that event's unflipped discs while `cos_flip_*` is a single cosine, so the
panel set one draw against an average of thirty and the null came out several times tighter
than an unflipped disc really is. `act_geometry.null_direction()` picks exactly one
unflipped disc per event — it exists for this reason and the table never called it. It now
feeds four `cos_null1_*` columns. Same centre (median −0.0025 vs −0.0024, TPR,
after-control), 6.5× the spread (sd 0.0550 vs 0.0085, IQR 0.0426 vs 0.0105). The finding
survives: the flip median +0.175 is 3.2 single-disc-null sds out and no null event reaches
it. But 3.2 is the honest number, not 21, and 0.5% of single-disc nulls clear the flip's
2.5th percentile where none of the medians did. The median columns stay — where the null
*centres* is a real question and they answer it — so both live in the file and the module
docstring says which is which.

Regeneration reproduced everything committed: `variance_shares.csv`, `summary.csv`,
`sanity.csv` and all seven of `probe_space`'s CSVs came back byte-identical,
`f2_event_meta.csv` reproduced its committed row (game 1882, *t*=25), and
`flip_cos_per_event.csv` differs only by the four new columns.

The seed is 42, this table's `SEED` and the source driver's. That is not free choice: the
same draw feeds `var_flip_null` in `variance_shares.csv`, so the two files' nulls name the
same discs, and a different seed would have broken that file's byte-identical reproduction.
The Fig 2(a) agent's independent implementation at seed 0 gives median −0.0025, sd 0.0525
against this column's −0.0025 / 0.0550; running their seed here reproduces their two numbers
exactly, and sd across seeds 0/1/2/7/42/1111 spans 0.0497–0.0550. Same statistic, different
draw — not a disagreement.

**Repaired since:** `tpr_probes/tpr_r52_f2_layer{0..7}.pt` on netscratch were saved before
`basis` entered the cache key, so `tpr_probe.load(5)` raised `FileNotFoundError` looking for
`tpr_r52_f2_relative_layer5.pt`. The loader was right — the infix is what stops a relative
and an absolute fit from overwriting each other — and the file names were stale. But two
conventions were live at once: `redteam.tpr_checkpoint` and `tpr_intervene`'s manifest line
each spelled out `tpr_r52_f2_layer{L}.pt` with the dims hardcoded, so renaming alone would
have broken them. `_path` is now public as `tpr_probe.path(layer, basis="relative")` and is
the only place the rule lives; the eight files are renamed to match. Refitting layer 5
reproduces the on-disk weights bit-for-bit, so the rename is the cheap half of a lossless
repair.

Unrelated but in the same files: `paths.linear_probe_activations()` now owns the
`$SPADE_LINEAR_PROBE_ACTIVATIONS`-else-repo-relative rule that `tpr_probe`, `act_geometry`
and `tpr_intervene` each spelled out separately.

## Fig 2(a)'s null gets the signal's estimator, and the panel states the share (2026-08-05)

Branch `bracket/fig2-specificity`. The specificity density compared **one cosine per
event** against a **per-event median over the ~29 unflipped discs**. A median of 29 draws
carries ~1/√29 of a single draw's spread, so the tight spike at 0 beside the broad hump at
+0.175 was partly an estimator artifact. The null is now one randomly chosen unflipped
disc per event (`cos_null1_after_control_tpr`): median −0.003 against the old −0.002, but
**sd 0.053 where the median-null had 0.009** — six times wider, and now marginally wider
than the signal's own 0.045.

The claim survives at close to full strength: all 1,596 strict events are positive, no
null event reaches the signal's median, and the two densities separate at AUROC 0.999. The
honest null also has a left shoulder near −0.19 that the median averaged away — on some
events a random unflipped disc's retain direction is genuinely anti-aligned with Δ′.

**The panel now shows magnitude, not only direction.** A cosine density alone reads as
"this is *the* direction the representation moves". An inset carries the after-control
decomposition already in `variance_shares.csv`: the flip direction holds **2.05%
[2.00, 2.11] of ‖Δ′‖² against the null direction's 0.42% [0.38, 0.47]** (n = 1,596,
percentile bootstrap over events), with swap 8.5% and residual 89.5% named in the inset
title. Reliably present at ~5× the null, and ~2% of the displacement — both halves of that
sentence are now on the figure.

Only the TPR-direction null is drawn. The linear-direction single-disc null is the same
distribution (median −0.001, sd 0.047), and a second gray dashed curve on top of the first
could only be told from it by dash pattern.

Also fixed: n comes from the frame instead of the literal "1,596" in the y-label, and the
legend title states the band in full (95% percentile bootstrap over events, 1,000
resamples, Gaussian KDE at the fixed full-sample Scott bandwidth).

The panel was written against `cos_null1_after_control_tpr` before that column existed and
verified on a scratch recomputation at rng seed 0; the column landed with
`bracket/regen-driver` at seed 42, so the drawn null is the seed-42 draw. The two agents'
implementations agree — see the entry above for why the seed is not free.

## Fig 1's absolute panel becomes a second probe, and its error bars move onto the arrows (2026-08-05)

Branch `bracket/fig1-field`. Four defects in `bracket_flip` Fig 1, one of them a real bug.

**Wrong event set (the bug).** `probe_space.field()` applied no `control_kind` filter, so
the dashed control arrow — the panel's only contrastive element — was averaged over all
2,000 events including the 404 whose control move touches *s*'s rays. Fig 2(a) and
`variance_shares.csv` were already on the 1,596 strict ones. Restricted, and **all three
conditions restricted, not only `control`**: `before` and `after` do not depend on *m′*,
but the claim is the distance between the control head and the after head, and reading two
heads off two populations is not a contrast. The correction is small — the flipped square's
control *x* moves 7.069 → 7.125 and the after-vs-control gap 14.557 → 14.592 logits — which
is expected, since a loose control still leaves *s* unflipped. The bug was in what the
number was computed over, not in what it says.

**The error bars were on the wrong quantity, and the fix goes the opposite way from the
guess.** Bars sat on the `after` endpoint; an arrow asserts a displacement, which is paired
within event. `m2_field.csv` now carries the paired contrasts and nothing else — after −
before, control − before, and after − control, each with its own interval, all three drawn.
The paired interval is *not* uniformly tighter than the endpoint one: on the occupancy
axis it is 1.2–2.5× narrower as expected, but on the owner axis it is **1.5–2× wider**,
because `before` and `after` are strongly anticorrelated across events — the same event's
discs read confidently `mine` before and confidently `theirs` after, so the shared
event-level confidence enters the two endpoints with opposite signs and adds instead of
cancelling. The old bars understated the arrow on the axis the figure is about. Written
into FIGURE_GUIDELINES §4.7 so the next displacement field does not repeat it.

**Panel (c) is now a second measurement.** It was `coords(..., absolute=True)`, which
re-signed *x* by whose turn it is — an exact deterministic relabel of panel (b), drawn
co-equal beside it and read as independent evidence. An absolute-basis TPR has existed all
along at `$SPADE_BRACKET_DIR/probe_space/tpr_absolute_layer5.pt` (this is a different
artifact family from `tpr_probes/`, which holds A1's committed-set relative fits), so (c)
now decodes it, oriented by the event's mover so black and white do not cancel. It puts
every role within 0.22 logits of *x* = 0 — **including empty squares, which have no colour
to read**, so what little *x* there is tracks whose turn it is, not the disc — and the flip
contrast collapses from 14.59 to 0.04 logits. The panels share axes, or (c)'s 0.2-logit
spread blows up to panel width and reads as separation.

**Framing.** Fig 1 is not demoted wholesale. Panel (b) is: its title and the README now say
it is the frame the study works in, since 63 discs crossing and *s* not crossing is what
any correct board tracker does and is exactly the claim that was retracted. But (c) is a
test the model can fail and does, so Fig 1 keeps a headline and it is finding 1's — the
board code is mover-relative — which previously had no figure. A board oracle scores 1.0 on
(b) and would nail (c).

Also: the `c{xx,xy,yy}` covariance columns were computed for a 50% dispersion ellipse no
version of `field_panel` ever drew, and at this panel's scale five of them would each cover
40% of the panel; deleted. `displacements()` and `worked_example()` lose their
`absolute_*` columns for the same reason — they were the relabel, and nothing read them.

---

---

## Branch suffixes get resampled from the model, in a tensor beside branches.pt (2026-08-05)

Branch `rollout/generate`, new `violation_divergence/rollout_branches.py`. `branches.py`
splices: every branch keeps the source game's `tokens[t0+1:]`, which is a legal
continuation of the *reference's* board and of no other. It survives for the legal
control in 76.8% of games (median 2 moves) and, after an illegal branch, is legal on no
board at all. So the reference was the only branch with an in-distribution future and
every displacement in the study was measured against it — the README's "What does not
hold" named this and nothing fixed it.

The new module keeps `tokens[:t0]` and the branch token at t0 and samples the rest from
OthelloGPT at temperature 1.0, K=4 rollouts per branch, only the pass token masked. **The
reference is resampled too.** Replaying its real continuation while the others roll out
would leave it the one in-distribution sequence and would have fixed nothing; `verify()`
checks that no reference rollout reproduces the real game.

**The trade, which is not small: single-token causal attribution is gone.** Two branches
now differ at t0 and at every position after it, so a displacement is the branch move plus
everything the model then chose to do about it. K=4 separates sampling variance from the
branch effect; it does not separate the move from its consequences. `branches.pt` stays
the tensor for the single-token claim and this is an addition to it, not a replacement —
hence a parallel tree under `$DIR/rollout/`, with nothing under `branches/` or `acts/`
touched.

Two decisions worth knowing. **Seeding is per (game_id, k), not per row**: all branches of
a game share the k-th uniform stream, so the K rollouts of two branches are paired common
random numbers, and a row's continuation does not depend on batch composition or on which
branches the game happened to yield. Sampling is inverse-CDF against those uniforms rather
than `multinomial`, which is what makes that possible. **No KV cache**: transformer_lens
1.17 builds the cache's attention mask from `self.tokenizer`, and OthelloGPT has none
(`AttributeError: 'NoneType' object has no attribute 'pad_token_id'`). Faking a tokenizer
to satisfy a mask computation is not worth it for a model this size, so each step re-runs
the whole prefix — ~40x a cached step's compute, still minutes on one H100.

---

## A `--variant` tree for the divergence pipeline, and the diagnostic that justifies it (2026-08-05)

Branch `rollout/pipeline`. The spliced branch set gives every branch the *source
game's* continuation, which is a legal game on the reference's board and nowhere
else. `suffix_legality/measure.py` measures the damage from tokens alone: a legal
control's continuation survives a median of **2** moves, and an illegal branch has no
board after *t0* at all, so its suffix legality is **undefined** rather than low.
Only the reference is ever replayed inside its own game. That asymmetry, not the
control's handicap alone, is why a rollout variant is being built.

**Two decisions in the diagnostic worth knowing.**

- *No convention is invented for the illegal branches on the spliced set.* Forcing a
  number there would hide the thing being reported. What is reported instead is the
  engine's verdict on the branch move, and the one convention-free statement
  available — the continuation later plays the branch's own square (94–99% of
  empty-square rows), which is illegal under every convention because occupancy never
  shrinks. Caveat recorded in that README: a 59-move game covers nearly the whole
  board, so the number is high partly for structural reasons.
- *`opponent_only` is the only illegal category the engine will accept*, via
  `umpire`'s forfeit fallback, and it accepts it by letting the **opponent** play. An
  engine that returns a board is not the same as a branch that has one. Any future
  code tempted to replay illegal branches must not read that fallback as the branch's
  own board.

**The 76.8% reproduces, on a different branch set.** The README's figure was measured
by the red team on the one-control set (15531 rows); the set on disk is the
two-control re-run (17527 rows). This gives **0.7515 [0.7325, 0.7710]** (1000-draw
bootstrap over the 2000 games) and median **2** exactly. The interval contains 0.768
and the set's two independent controls agree to 0.2 pp, so the difference is the
branch set and not the engine handling. Settling it exactly needs the one-control set
regenerated; HF streaming does not work offline here, so it is left open. The script
**stops** rather than printing a table if it ever misses the check.

**The `--variant` flag composes with `--smoke`.** `variant_dir(study, smoke, variant)`
in `common/study.py` is the one place the rule lives; `extract_acts` and
`compute_divergence` take `--variant rollout` and read/write `$DIR[/smoke]/rollout/`.
Both defaults are unchanged and this was checked rather than asserted: the 50-game
smoke pipeline was run end to end before and after the change into two scratch trees,
and all 10 act tensors plus all 17 divergence npz files are **bit-identical** (sha256).

Two things had to change for the rollout set to work at all, both no-ops on the
spliced set. `check_prefix_invariance` asserted no branch matches the reference at
*t0*; with K samples per branch that is false by construction, so it now asserts the
stronger and variant-independent rule that activations at *t0* agree **iff** the
branch tokens do. `pairs()` keyed on `game_id`, which finds K references per game;
it now keys on `(game_id, rollout)` when the column is present.

**Not done, deliberately:** the downstream read-outs (`probe_readout/decode.py`,
`behavioral/make_figures.py`) still hardcode the spliced tree. Plumbing `--variant`
into them means deciding how their `results/` and `figures/` names separate the
variants, which is a naming call for the study owner, not a mechanical edit.

## The real-only curve in Fig 2(b) was labelled a control; it is the effect (2026-08-05)

Branch `bracket/fig2-causal`, on top of the entry below. That entry drew `d_lp_real`
and called it "the control a merely damaging edit fails", so the orange dotted curve
diving to −4.6 nats read as degradation. It is the opposite. `real_only` is the
squares legal on the **real** board and illegal on the counterfactual one, so a
successful un-flip edit has to push it **down** while it pushes `cf_only` up. Both
curves are one switch read on its two sides, and the panel now says that: the y-axis
carries `↑ onto the un-flipped board's squares, ↓ off the real board's`, and the
legend says "one switch, two sides".

**The damage control was the corner annotation.** `d_lp_legal` — log-mass on the
whole legal set, which is what a broken edit moves — is now a gray band through zero
at its worst-case CI bound over both sweeps (±0.08 nats), with a legend entry. A band
rather than a series or a twin axis: on a y-axis spanning −4.6 to +8.1 the control is
a hairline by construction, and *that* is the reading. It replaces the dashed zero
line, so the zero reference and the control are one mark.

**The low-scale asymmetry is a base-mass artifact, not a property of the edit.** At
tpr/scale 1 `d_lp_cf` is +0.278 [0.267, 0.288] while `d_lp_real` is −0.0005 [−0.0015,
+0.0003] — mass apparently added to the counterfactual squares without being taken
from the real-only ones. It is not:

- **Not the event sets.** 507 of the 1,370 flipped rows have both sets non-empty
  (986 have `cf_only`, 891 have `real_only`, none have neither). Restricted to those
  507 the numbers barely move: `d_lp_cf` +0.278 → +0.294, `d_lp_real` −0.0005 →
  +0.0003. The differing n is bookkeeping, not the effect.
- **It is base mass, by four orders of magnitude.** Median base mass is 8·10⁻⁶ on
  `cf_only` against 0.115 on `real_only` — the sets start 9.7 nats apart. The +0.278
  nats on `cf_only` is +1·10⁻⁵ of *probability*. Had every bit of it come out of
  `real_only`, `real_only` would read −0.00008 nats, an order of magnitude inside its
  own CI. The measurement cannot resolve the exchange at that scale; the flatness is
  not evidence either way, and the panel now carries that sentence in-figure so
  nobody reads the flat left end as specificity.
- **Above ‖δ‖/‖h‖ ≈ 0.07 the exchange is real and visible.** At scale 16 `real_only`
  loses 0.124 of probability and `cf_only` gains 0.096, with the whole legal set down
  only 0.075 nats: mass leaves the real board's exclusive squares, and most of it
  lands on squares the un-flipped board allows. At intermediate scales `real_only`
  loses far more than `cf_only` gains (86× at scale 4) while `d_lp_legal` sits at
  10⁻⁴ nats —
  the difference goes to squares legal on *both* boards, which is also the
  counterfactual board's legal set. Still the same switch.

The honest summary of the operating point (`‖δ‖/‖h‖` = 0.03): the edit produces a
large, tight log-scale effect on a set holding 10⁻⁵ of the mass, and nothing else in
the distribution moves enough to measure. Reading Fig 2(b)'s y-axis as a probability
budget is the mistake it is now annotated against.

Drawing only; no re-run. `causal_panel` now also reads `intervention_per_event.csv`,
for the base-mass gap and the transfer floor.

---

## Fig 2(b) stops grading each edit with its own decoder, and shows the controls (2026-08-05)

Branch `bracket/fig2-causal`. The causal panel had three problems, all of which made
the intervention look better or simpler than the run actually was.

**Collateral was scored by `collateral_tpr` for all three series**, so the TPR edit was
graded by the TPR decoder it is built from while the linear edit was graded by a
decoder it has no stake in. Each family is now scored **out of family** —
`collateral_lin` for the TPR edit, `collateral_tpr` for the probe edit. The verdict
does not change: at every magnitude the TPR curve is above and left of the probe
curve, and out-of-family scoring actually moves the TPR points slightly *left* at low
scale (0.044 vs 0.058 changed squares at scale 1). Mixing two instruments on one axis
is defensible here because the two readings of the same edit stay within ~35% of each
other; the axis label says which instrument each series used.

**The legend claimed one knob for two families.** The TPR edit is additive
(`scale · d`, 8 scales 1–16) and the probe edit projects to the class mean (`β`, 2
betas) — `tpr_intervene.edit_vectors` — so "one point per edit scale (1–16)" was false
for the probe arm and implied a shared x. Resolution: nothing is placed by knob value.
x is measured collateral, and each point carries `‖δ‖/‖h‖` (`rel_edit_norm_mean`) as a
label, which is the one magnitude both families share. The probe's betas sit at 0.11
and 0.23, inside the TPR sweep's 0.03–0.53, so the curves overlap in magnitude rather
than merely in ink. x is log-scaled because collateral spans 0.04–2.3 and the
interesting operating point is at the bottom of that range.

**The strongest controls were computed and not drawn.** `d_lp_real` and `d_lp_legal`
were in the CSV and invisible. `d_lp_real` is now a dotted, open-marker companion in
each family's own colour (hue is spent on the family, line style carries which legal
set is read — the `BRANCH_ROLE_STYLE` convention), and `d_lp_legal` is an in-panel
line stating its worst-case bound over the whole sweep. That is what answers "the edit
just damaged the model": at TPR scale 1 the counterfactual set gains +0.278 nats while
the real board's legal-only squares move −0.0005 [−0.0015, +0.0003] and whole-set
legality does not move. Specificity is a *low-collateral* property, and the panel now
shows where it ends — past `‖δ‖/‖h‖ ≈ 0.2` the control collapses.

**Three n's, not one.** `n_events` is 1,370, but a set-log-mass is NaN when its token
set is empty (`unflip_edit.set_logmass`), so the counterfactual set has n = 986 and the
real-only set n = 891. Both are constant across every scale — the sets are fixed by the
event, not by the edit — and `causal_panel` asserts that, since it prints one n per
quantity. Collateral and legality keep n = 1,370.

Not re-run: this is drawing only, off the committed `intervention_summary.csv`.
`tpr_matched` (the TPR direction at the probe edit's per-event length) is still in the
CSV and still not drawn; the per-point magnitude labels carry the same information
without a fourth curve.

---

## The fork figures' prose moves to their READMEs, and the axes stop naming a component (2026-08-04)

Branch `fig/strip-captions`. Every fork figure — `fork_and_board` (both planes),
`fork_curvature_space` (all four), `game_trajectory` — carried a multi-paragraph legend
title: the selection disclosure, where the rule mean is taken, the spread ratio, n=1, the
not-comparable-across-cells disclaimer. That is a caption, and §2/§5 put captions outside
the graphic. All of it now lives in each study's `README.md` under `## Captions`, one
heading per figure, and **nothing was dropped** — the selection of game 1235 above all,
which is the one a reader cannot be allowed to miss. What stays inside: panel titles and
letters, per-panel space names, axis labels, the legend, the board's reading key, and the
R1/R2/R3 key, because §5 requires domain shorthand to be readable without another document.

Two things that were in the prose became legend *entries* rather than moving out, since the
legend is in-figure by right: the jackknife ellipse's semantics (`50% Gaussian contour of 13
jackknife replicates (dispersion, not a CI)`) and the `(mean of n)` already there. An
uncertainty a reader could mistake for a CI does not belong two documents away.

**Why the axis labels lost their lead component — a correctness fix, not taste.** The R^21
paths appended the descriptor coordinate the PC loads hardest on, giving
`PC1 (44% var, Δ mean θ)`. Measured on raw layer 5, the windowed plane's PC1 loads **0.360
on `mean θ`, which is 13.0% of the direction's squared length**, over a near-tied and
same-signed `mean d` 0.328, `max θ` 0.323, `min d` 0.267, `max d` 0.249; **16 of 21
coordinates are needed to reach 90%** of it. PC1 is a broad mixture, not that named
quantity. Worse, which coordinate "wins" flips between the near-ties with the branch set and
the variant: the `whole` plane, same game and same space, leads on `min d` at 0.290 — 8.4%
of its direction, 15 of 21 for 90%. Naming one component asserted something false and would
have flipped under a `--game` change. Axes now read `PC1 (44% var)` / `PC2 (18% var)` and
nothing else, from the shared `branch_panels.axis_labels` the displacement paths already
used; Isomap keeps `arb. units` and no ratio. `Plane.lead` stays — `measure.py` and
`r21_shared_basis` print it, where a lead is a diagnostic and not a label.

**The rule key now names the members.** Three means hide six categories and the faint member
lines were unlabelled. `branch_style.rule_member_key` prints one compact line per rule —
name, member count, and each member's gloss — driven by `RULE_MEMBERS` and the new
`taxonomy.CATEGORY_GLOSS` (which replaces the comments that used to sit beside
`CATEGORIES`), so the counts cannot disagree with the legend's `(mean of n)`. It does not
replace `paper_style.RULE_TEXT`: nine other figures print that three-sentence form, and R4
and R5 have no members at all.

**Panel (b) gained the would-be disc.** An outline plus a corner marker said "a branch plays
here" but nothing said a move *puts a disc down*. Each branch square now carries the disc it
would place, in the mover's colour, at **half a real disc's radius, translucent, dashed, and
pinned to the corner opposite the category marker** — never at the centre, because
`occupied_own` and `occupied_opp` already hold a real disc whose colour is the reader's only
evidence for which of the two the branch is. It survives grayscale: real discs are solid and
full-size, the would-be disc is grey (or white) with a dashed edge and visibly smaller. It
is drawn on **all** branch squares including the reference and the control, not only the
illegal ones — a marking that appeared only on illegal squares would imply the placement is
what makes them illegal, which is false for four of the six categories. The panel title also
names the mover, derived from `board.next_hand_color` via `branch_panels.mover_name` (shared
with the board note), because the mover alternates with *t0* and a hardcoded "Black" goes
wrong the moment anyone passes `--game`; verified on game 1 (*t0*=11), which renders "White
to move".

**Layout.** Stripping ten lines of legend title from `fork_and_board` left a hole beside the
square board, which §5 calls a layout bug. The board's cell is now as wide as it is tall so
the board fills it, `BOARD_H` drops 5.4 → 3.9, and the legend block is centred in the band
rather than pinned to its top. The rule key moved out of the band and under the whole
figure, the only place with the width its three lines need. The three grid figures pinned
their key at a hardcoded `y` tuned to a ten-line title; `branch_style.below(owner, leg)` now
measures the rendered legend instead, so the key follows it however the entries wrap.

`selection_note()` and `POPULATION_SPREAD` lost their only callers and are deleted; their
content is in the READMEs and their reasoning survives as comments beside `FIGURE_GAME` and
in `rule_groups`'s docstring. Both scripts still *print* the spread ratios they drew, so the
README numbers can be re-derived without reading a figure.

---

## The R^21 plane moves into the library, and fork_and_board can draw it (2026-08-04)

Branch `r21/plane-and-combined`. `fork_curvature_space/descriptor.py` was analysis-level,
so `fork_and_board` could only have reached it by importing one analysis dir from another.
It is now `src/spade/violation_divergence/curvature_plane.py`, beside `fork_plane.py`.

**What the move deduplicated, and what it did not.** The sampling window
(`STRIDE`/`WINDOW`/`FIRST_POS`/`LAST_POS`) and the nine-cell space list are one rule, so
they live once in `fork_plane.py`; `positions(t0, back=0)` grew the `back` parameter the
R^21 window needs, and `back=0` is the displacement plane's old body exactly. The
duplicated figure-game constant is gone — `branches.FIGURE_GAME` was always the one that
mattered. The two planes' *fitting* is genuinely different (uncentred PCA of D vs
column-scaled PCA of Δφ, which has to survive NaNs) and stays separate. Verified by
re-rendering all four `fork_curvature_space` figures before and after the move:
byte-identical PNGs. Four other analysis dirs still each carry their own copy of the
nine-space list; they were out of scope here.

**fork_and_board grew `--plane {displacement,r21}`**, default `displacement`. Two figures,
`game_1235_fork_and_board_displacement_pca` and `..._r21_pca` (the old
`game_1235_fork_and_board_pca` is the former, renamed).

**The trap, which would have been silent.** φ — step lengths, turning angles, moments — is
nonlinear, so `mean over members of φ(z_b) ≠ φ(mean over members of z_b)`. A figure drawing
one mean per rule must apply φ per member and average afterwards; averaging the trajectories
first measures the curvature of a path no branch takes, and the result looks perfectly
plausible. The plane projection is linear, so mean-then-project is safe and the figure
keeps taking its mean in the plane, as the displacement version does.
`test_curvature_plane.py` pins both halves: φ does not commute with the mean, the
projection does.

**The R^21 grouping is no better: within/between rule spread = 0.902 in R^21 against 0.885
on displacements** (game 1235, raw layer 5, column-scaled Δφ over t ≥ t0). Slightly worse,
in fact. There is no population number in R^21 — the 0.945 median is a displacement figure
— so each legend names which of the two it is quoting rather than letting one stand for
the other.

**The window still costs two offsets and the prefix still collapses.** At t0=20, 15 of the
study window's 17 offsets carry a descriptor and 2 of those are branch-free, unchanged by
dropping control 2 and by the rule grouping. `project()` asserts `max|Δφ| = 0` on the
branch-free offsets in all nine spaces on every render, so the legend's "the prefix still
IS the origin" is checked rather than asserted.

**Layout gotcha.** The R^21 legend title is five lines longer than the displacement one, so
the board note and the rule key, which used to sit at a fixed y, landed on top of it. They
now hang off the legend's measured bottom edge (one `fig.canvas.draw()`), which is why the
two figures' canvases differ slightly in height.

---

## fork_curvature_space: the same three changes, and a worse number (2026-08-04)

`fork_and_board`'s three changes applied to the four R^21 grids. The measurement is the
point of the entry.

**The rule grouping is worse in R^21 than in the displacement plane, and for the whole
variant it is gone.** Within-rule ÷ between-rule mean pairwise distance, column-scaled
full R^21 (never the drawn plane), raw layer 5, game 1235: **0.90** on `Δφ(t)` at
`t ≥ t0` (0.77–1.01 over the nine spaces) and **1.03** on `φ`, one point per branch
(0.90–1.06, above 1 in three spaces). The same statistic on the full 512-d displacements
of the same game and window reproduces `fork_and_board`'s **0.885** exactly, so the
comparison is apples to apples. Mapping into the curvature descriptor does not make the
rules cohere; on the whole variant the label carries nothing, and `fork_r21_whole_pca`
shows it — the three R2 members scatter across the cell with their mean in the middle of
nothing. Both numbers ride in the legend titles, members drawn faint.

**φ is nonlinear, and that constrains where the mean is taken.** mean φ(z) ≠ φ(mean z):
each member gets its own descriptor, its own Δφ and its own projection, and only the
coordinates are averaged. The projection is linear (windowed, uncentred) or affine
(whole, centred), so plane-mean and R^21-mean agree exactly — which is the only reason
the cheap version is also the correct one. Column scaling before the distance follows
`measure.py`: the 21 components mix degrees with activation-scale lengths, so a raw
Euclidean statistic is a statistic about θ alone.

**`rule_fork_panel` is not reused, deliberately.** It takes `k` from `pos == t0` and
draws from the branch onward. This figure draws from the first *defined* offset, two
sampled positions before the branch, because those two points sitting exactly on the
origin are the visible proof that Δφ collapses over the shared prefix — one of the two
invariants the sub-study exists to keep. Everything else is shared:
`branch_style(by_rule=True)`, `MEMBER`, `legend_proxy`, `rule_rows`, `ordered_legend`
and `PLANE_TICKS`, and the local `branch_style`/`legend_order`/`ordered_legend` copies
are deleted. What is duplicated is ~8 lines of path drawing.

**Ticks** go on the windowed grids' plane axes, three symmetric about zero. The whole
grids are 2-D scatters of physical quantities (`mean d ∈ [18, 21]`), where a symmetric
locator puts every tick off-panel; they keep ordinary ticks. Also honoured the `zlabel`
argument `windowed_grid` was already passing and `fork_panel` was ignoring, so
`t − t0 (plies)` is labelled once per row as in the sibling figure.

**Left behind:** `make_figures.py`'s import block is owned by a parallel agent moving
`descriptor.py` into `src`, so the now-unused `CATEGORY_STYLE`, `NULL`, `REFERENCE`,
`CATEGORIES` (and the pre-existing `GRID`) are still imported and `LEGAL` came in on its
own line beside them. Prune on merge.

---

## fork_and_board: rules instead of categories, one control, ticks on (2026-08-04)

Three requested changes to the shipped figure. The first one produced a number worth
knowing.

**Six categories → three rules.** `othello.taxonomy` now owns `RULE_MEMBERS` (the domain
fact — which rule a category violates — belongs beside `CATEGORIES`, not in a style dict),
`branch_style(..., by_rule=True)` swaps `CATEGORY_STYLE` for the `RULE_STYLE` that was
already in `paper_style` and unused, and `branch_panels.rule_fork_panel` draws one mean per
rule. Both panels group, not just (a): if the fork drew rules and the board drew
categories, the figure's one-legend-for-both property — the whole reason the band beside
the board exists — would collapse into two legends. The cost is that the two R1 squares are
now the same colour, so the board note says a rule's members are told apart by their square.

**The grouping is weak, and that is the finding.** Within-rule ÷ between-rule mean pairwise
distance on the full 512-d layer-5 displacement trajectories: **0.885** on game 1235,
**median 0.945 [IQR 0.872–1.020]** over all 1550 branch-complete games, **above 1 in 30%**
of them (`python -m spade.violation_divergence.rule_groups`). A rule's members travel ~5%
closer to each other than to another rule's. Three confident lines would misrepresent that,
so members are drawn faint behind their mean and the ratio is in the legend title. Game
1235 is near the first quartile — its grouping is better than typical, which is selection
again, since it is the argmax of the mutual separation of those same six trajectories.
FIGURE_GUIDELINES §1.7 is this rule generalised: a grouped mean carries n, its members, and
its within/between spread.

The mean is taken **after** projection, not before. For PCA that is identical (`pca2` is an
uncentred linear map), and for Isomap it is the only thing that means anything — one rule
covers both embeddings.

**One control.** `control_legal_2` is filtered out of the figure's rows in `main()`, which
also drops it from the PCA fit, since the plane should be fit on what is drawn. The cost is
real and is written down in the README rather than swallowed: control 2 exists because *one
control gives no null* (RED_TEAM F1 — the first pass could not interpret its own direction
result), so the figure loses the reader's visual answer to "is that separation big compared
to two legal moves differing?". Nothing measured changes; the legend title now points at
`game_trajectory` and the parent README, which is where the legal-vs-legal scale lives.
`game_trajectory` keeps all nine branches and per-category styling on purpose — it is now
the figure that carries what this one dropped, and its README says so.

**Ticks on the PC axes.** Three per plane axis, `MaxNLocator(3, symmetric=True)` so the
origin — the reference, in a displacement plane — always gets one, 7 pt, negative tick pad.
This reverses FIGURE_GUIDELINES §4.6/§4.8, which said ticks off because the coordinates are
panel-local; both sections are rewritten. Within-panel scale is worth more than the
ambiguity, and the "not comparable across cells" disclaimer stays and matters more, not
less. `fork_panel` and `rule_fork_panel` share `_fork_axes`, so `game_trajectory` gets the
ticks too; its two x/y `labelpad`s went from −8 to −2 to clear the new tick labels.

**Status:** `fork_and_board` PCA re-rendered as PNG and camera-ready PDF, `game_trajectory`
PCA re-rendered; both isomap variants render (checked, not shipped — neither was shipped
before).

---

## One figure: the fork over the board it forks on (2026-08-04)

`analysis/violation_divergence/fork_and_board/` merges the nine-space fork grid and the
branch board into a single figure — (a) the 3×3 grid of representation spaces, (b) the
board at *t0* — because the two only mean something together: (b) says what the nine
branches are, (a) says where they go. Both draw the same game, **1235** (*t0*=20), and the
entry points now assert the requested game is branch-complete instead of taking
`complete[0]`.

The board is square and the grid is three wide, so the band beside the board carries the
figure's apparatus: **one legend for both panels** (the payoff of `branch_style` — both
halves already draw identically, so a second legend would repeat itself), the board's
reading key, and the R1/R2/R3 gloss.

Three extractions, each because the combined figure would otherwise have copied code:
`src/spade/figures/branch_panels.py` owns `fork_panel` / `board_panel` / `axis_labels`
(were private to the two scripts), `src/spade/violation_divergence/fork_plane.py` owns the
sampled positions and the displacement embedding, and `branches.py` — which defines what a
row of `branches.pt` means — grew the reader for it (`load_game`, `complete_games`,
`reference_row`). `ordered_legend` and `rule_key` now take the figure *or* an axes as
their owner, which is what lets both live inside the band rather than under the figure.

**Axis labels.** `space_grid` never labelled x/y, so nine panels carried their variance
ratios buried in the titles; they now read `PC1 (27% var)` / `PC2 (10% var)` on the axes
per §4.6, per panel because the plane is refit per space. `draw_board` labels `file` /
`rank (1 at the top)` alongside the ticks it already owned, so the board's reading note
moved off the xlabel to wherever the caller has room for it.

**Game 1235 is selected, and that is now on the figure.** It is the argmax over the 1550
branch-complete games of the mutual separation of its six illegal trajectories (raw layer
5, full 512-d, never an SAE space or a projection; population median 0.705, this game
1.080). It matters: on 1235 the reference and both controls separate from all six illegal
branches in all nine spaces, but over 40 games that ordering holds in 0.07–0.17 of them
against a 0.036 chance floor — selected structure that does not generalise, and a reader
who does not know it was chosen over-reads the panels. `branches.FIGURE_GAME` and
`branches.selection_note()` keep the constant and its disclosure together, so every figure
drawing the default game prints it; a non-default `--game` prints that instead.

**The two nulls were indistinguishable.** `branch_style` returned byte-identical dicts for
`is_control` 1 and 2, and on this board they land on adjacent squares (h3, h4). Control 2 —
the null *for* control 1 — now takes an open marker (`mfc`/`mec` in the style dict, which
also pulled the hardcoded `"white"` marker edges out of the three panel bodies). Same
colour, same shape, different fill: separable without reading legend text, and in grayscale.

**Status:** rendered on game 1235 with all nine spaces, working PNG and camera-ready PDF.
`branch_board/` and `game_trajectory`'s `space_grid` were subsets of it and are deleted; the
board drawing survives in `spade.figures.othello_board`. `fork_figure` stays in
`game_trajectory` — it carries the δ̂ marginal panel, the quantitative half §4.8 requires and
which this figure does not have.

---

## The branch board, and one board-drawing rule for two studies (2026-08-04)

The board the game forks on, with all nine branch moves on their squares — the companion the
fork figure needed, since a projected trajectory never says what a branch *is*. It began as
`analysis/violation_divergence/branch_board/` and is now panel (b) of `fork_and_board/`.

Two extractions, both because the same rule was about to be written twice.
`src/spade/figures/othello_board.py` owns "draw an 8×8 board with these discs" (was
private to bracket_flip's `board_panel`); `src/spade/figures/branch_style.py` owns
`branch_style` / `legend_order` / `ordered_legend` (was private to the fork figure). Each
caller keeps its own overlay — role shading, category outlines — out of the shared helper,
so neither grows a flag per caller. bracket_flip Fig 1 and the fork figure both re-render
pixel-identical after the move.

**Occupied squares.** Two illegal categories play onto a square that already holds a disc,
and the disc's colour is exactly what distinguishes `occupied_own` from `occupied_opp` —
so a centred marker would cover its own evidence. Every branch square is therefore outlined
in its category's colour and line style, with the marker in the square's **corner**, for all
nine branches. Uniform grammar, no disc obscured. Only the reference's flips get rings
(dashed, outside the disc so they read over black and white); drawing all three legal
branches' flip sets is a knot with no question behind it.

---

## The fork figure's game is now chosen, and the criteria disagree (2026-08-04)

`game_trajectory/` drew `complete[0]` — game 0, *t0*=10, the minimum *t0* over the
1550 branch-complete games. List order, not a criterion.
`analysis/violation_divergence/game_selection/` sweeps all 1550. Its pure argmax is
**game 662, *t0*=22**; the figures draw **game 1235, *t0*=20** — see the last paragraph.

Two constraints made the sweep worth trusting, and both were committed before it ran
(`score_games.py` docstring): score in **raw layer 5 only** — an SAE in the selection
loop would tilt the nine-space grid toward whichever dictionary drove the choice — and
score in **full R^512**, since a top-2 PCA criterion selects the game that flatters one
plane and the plane is being redesigned.

**The four candidate criteria have disjoint top-10s.** ρ(*S3*, *S4*) = −0.442: the games
whose branches diverge furthest are the games whose illegal categories are *least*
differentiated, because large δ̂ means everything moving along one shared direction. So
"pick the biggest divergence" is the worst rule for a category-colored figure.

**Why not *S1*, the illegal-vs-control separation.** It is the study's own scientific
contrast, and the aggregate says illegal displaces *less* than a legal control (0.978).
Its argmax is therefore the single game that most contradicts the paper, dressed as an
illustration. The pick is argmax *S3* instead — scale-free, and no claim rests on it.
Cost, stated in the sub-study README: game 662 is at the 5.8th percentile of *S1*
(conservative direction, but not representative of that contrast).

Transfer check: game 662 is rank 1/302 in Standard, TopK and SpaDE latent spaces too,
ρ = 0.935–0.990 against raw L5. The choice is branch geometry, not the selecting space.

**Why the figures draw 1235, not 662.** Argmax *S3* alone lands on a game at the 5.8th
percentile of *S1*, so the panel would understate the very contrast the study is about. The
rule was restated as *argmax S3 among games at or above the median S1* — 1235 is *S3* rank
3/1550 (1.042 vs the 1.080 max) and above median on *S1* and *S2*, with *t0*=20 against a
population median of 21. Chosen after seeing the results, which is why the rule is written
down here and the disclosure is on the figure; `branches.FIGURE_GAME` holds the constant.

---

## R^21 cannot carry the curve inside the r >= -8 window (2026-08-04)

Feasibility check on projecting `space_fork_grid_pca` through the ReStraV descriptor
instead of the raw activation displacement. New study in
`analysis/violation_divergence/fork_expanding_curvature/`; tables in its README.

**The blocker is dimensional, not statistical.** `restrav_features` needs 8 sampled
positions (14 plies at STRIDE=2) to fill d1..d7 and θ1..θ6. The fork window has 4 sampled
positions before *t0* (8 plies). A sliding window at W=8 therefore has zero curve points
free of the branch, so the shared prefix at the origin and the spray at *t0* — the figure's
entire content — are outside the descriptor's domain. W ≤ 4 keeps a prefix but returns
R^13 at most. No W does both.

**Superseded in part — the window, not the arithmetic, was the constraint.** The 4-sampled-
position limit comes from the study window `r >= -8`, not from the game. Reading the prefix
back to the start of the game dissolves it whenever *t0* >= ~16: at game 1235's *t0*=20 the
pre-branch samples are 2,4,…,20, so a trailing W=8 window ending at *t0* spans positions
6..20 and is both full-width R^21 and branch-free. `fork_curvature_space/` builds it and
measures `max|Δφ| = 0` at every *r* < 0 in all nine spaces. Everything below about the
truncation trap and the denominator still stands.

**Gotcha worth remembering: `restrav_features` truncates silently.** `d[:, :7]` and
`theta[:, :6]` on a short window return 11/13/15/17/19 columns with no error (and an
obscure `IndexError` from `moment4` at W=2), and the column *meanings* shift — at W=5,
column 4 is θ1, not d5. Anything slicing it from a variable window must assert the width.

**The expanding window anchored at *t0* was prototyped and rendered.** It is causal and
event-locked, but it starts 14 plies after the branch and 13 of its 21 coordinates are then
frozen exactly (measured 0.000e+00) — the first steps since the branch never change again,
so only the 8 cumulative moments move, and their travel is only 1.2–2.1× a null with the
temporal order shuffled away.

**A large effect size here is the denominator, not the descriptor.** Illegal-vs-control
separation goes 0.09–0.45 (the current per-position plane) → 1.2–1.6 (R^21), but the *same
raw displacement* averaged to one point per branch gives 0.89–1.87, i.e. larger, in every
space. Collapsing the position axis removes the within-group spread that dominates the
denominator. A label-permutation null over all 28 relabellings of the 8 branches clears
nothing (best p=0.18 against a 0.036 floor). Also noted: an unscaled PCA of R^21 is a plane
of turning angles (PC1 puts 1% of its energy on the step-length block, since d is in
activation-norm units and θ in degrees), so the planes are fit after a diagonal unit-RMS
rescale that keeps zero displacement at the origin.

**Recommendation:** keep the raw-displacement plane for the fork; if R^21 belongs in the
study it belongs as one point per branch over the whole post-*t0* stretch, as a §4.6
embedding scatter over the full 13,531-branch population, not at n=1.

---

## The fork grid can share one basis, if it goes through R^21 (2026-08-04)

`analysis/violation_divergence/r21_shared_basis/`, branch `curvproj/r21-shared-basis`. The nine-space fork grid
in `violation_divergence/game_trajectory` refits its plane per space and has to disclaim
that cells are not comparable, because R^512 and R^8192 share no coordinates. Projecting
the displacement through the 21-D ReStraV descriptor first gives every space the same named
coordinates, and then one plane does work for all nine cells.

**A shared plane costs nothing; not fitting at all costs less.** Illegal-vs-control centroid
gap, in pooled within-group SDs on game 0: per-space refit 0.70–1.80, one shared fitted
plane 0.70–1.64 (better in 6 of 9 spaces), fixed axes (Δ mean *d*, Δ mean θ) 1.32–2.00 —
best in every space, on 0.11–0.20 of the variance the fitted planes maximise. The status-quo
activation plane is 0.13–0.39 on the same points. Verdict and full tables in `REPORT.md`.

**The comparability claim is half, not whole.** Z-scoring is mandatory in R^21 (the
components mix degrees with activation-scale lengths, ~450x SD spread across dictionaries)
and it is per space, so a shared basis makes arrangement and direction comparable cell to
cell but never absolute magnitude. The grids say exactly that; the refit grid keeps the old
disclaimer.

**Every number is scored against an exact permutation null** — all 28 ways of relabelling
the 8 non-reference branches 6/2 — because n=1 game and 10 overlapping windows per branch
make a raw effect size meaningless. R^21 roughly doubles the null as well as raising the
gap; after dividing it out the status quo sits at 0.45–1.05x chance and the named plane at
2.32–3.10x.

**Do not quote that 2.32–3.10x.** It is a one-game number and it does not replicate. Both
`fork_curvature_space/` (40 games) and `curvature_discriminant/` re-measured the same named
pair: 1.27x over 40 games, and 0.24–1.10x on game 1235 specifically. Nothing was wrong with
the computation — the n=1 statistic is simply unstable, which is what 8 effective branches
in 21 dimensions predicts. The defensible figure for R^21 over the incumbent plane is the
40-game paired gap, **+0.08 [+0.01, +0.15]**.

**Gotcha for anyone reusing this.** A full 21-D descriptor needs 8 sampled positions = 15
plies, so on game 0 (t0=10, window clipped at the board start) every plotted window straddles
t0 and the figure loses its zero-by-construction prefix segment. Also: `restrav_features`
silently returns a *shorter* vector for a short window instead of raising — assert width 21.

---

## Wave 8: the docs catch up, and q15 narrows to one hypothesis (2026-08-04)

Three agents on `wave8/*` — the claims ledger, the archive zone, the docs sweep. Per-agent
records in `docs/finish_notes/`. Wave 8's third part, deleting the source repo's residue,
stays gated: questions 14 and 15 are open and wave 0 is blocked.

**The retraction had not propagated.** Three non-historical docs still stated that λ=9.77e-5
/ γ=0.2 were "stale defaults read off a Mac smoke config" — `pipeline.md` §5 and §7a,
`spade_vs_baselines.md` Appendix A, `straightness/sae_profiles/README.md`. Fixed. Port notes
keep the old numbers on purpose; they record what an agent believed at the time.

**q14's ±0.014 band now appears where numbers are compared.** It changes a reading:
`spade_vs_baselines.md`'s Pareto listing separates Gated (0.5673) from Standard (0.5679) on
reconstruction F1, which is well inside the band.

**q15, narrowed.** A byte-exact sweep over netscratch, the frozen repo and the backup finds
exactly two copies of `main/trainer_5/ae.pt`: itself and
`feature_viewer/hf_space/checkpoints/SpaDE/ae.pt`, the checkpoint deployed to the public
Space. Nothing in the multiseed tier matches, so it is not a stored sibling seed. But it
reruns at L0 127.76 / VE 0.973, which is what `spaces.py:28` already records for seeds 43–46
of the same config. Leading hypothesis, untested: seed 42 was re-run post-canonicalization
and overwrote the degenerate checkpoint in place, without ever landing on netscratch. Testable
by retraining that config at seed 42 and comparing. **This pins q11** — `hf_space/` holds the
only redundant copy and cannot be deleted first.

**One label is wrong in eight places.** The excluded checkpoint is called "the γ=1e-2
checkpoint" in prose; its config and its wandb record both say **γ=0.2**. `EXCLUDED_SAE` is
the config name `"SpaDETrainer-othello"`, so no code identifier is affected — but the
exclusion decision was taken under the wrong label, so the owner should confirm the rename
rather than have an agent do it silently.

---

## The port runs, and two of its reference numbers do not survive contact (2026-08-04)

Three agents on `finish/*` branches — packaging, the checkpoint copy, the flagged defects —
merged here. The tree went from "code moved, nothing verified" to installable, loaded with
its seed-42 weights, and 25/25 on `final_validation` with no skips. Per-agent records in
`docs/finish_notes/`.

**The port is faithful, and the proof is exact.** `checkpoints/{batch_topk,cosine_spade,
spade_revival}/trainer_0` reproduce their committed reference rows to 0.000000 on L0,
variance explained, coverage and reconstruction F1. Identical inputs, identical outputs. That
exercises SAE loading, standardization, the Karvonen adapter and the F1/coverage math end to
end, so nothing below is a port defect.

**Wave 3's stated acceptance number is wrong.** It gave VE = 0.943 for TopK on
`main/trainer_3`; the committed reference is 0.9867 and the rerun gives 0.98688. 0.943
appears nowhere in the tree. That wrong number has been the gate on the whole port, and
anyone checking against it would have concluded the port was broken. Corrected in
MIGRATION.md and in `final_validation`'s caveat text.

**Two provenance findings, now MIGRATION.md questions 14 and 15.** The activations behind the
five `results/karvonen/` reference rows no longer exist — those rows were committed 50
minutes before the only surviving extraction was written — so they reproduce only to ±0.014
on the assignment-protocol metrics. And `main/trainer_5/ae.pt` is not the run its config
names: it reruns at VE 0.973 where its wandb record and eval row both say ≈0.36, while
trainer_0–4 all match theirs.

That second one **retracts** the 2026-08-03 entry's reading that the λ=9.77e-5, γ=0.2
configuration never ran. It ran, 36621 steps on cuda:0, and finished degenerate, which is
what the docs reporting it say. `steps=400` and `device=mps` are stale in all six configs,
not just that one. The exclusion decision's justification is unaffected; what is in doubt is
which model the studies that loaded those weights were measuring.

Worth repeating for the next session: gitignored data must be copied into the real tree, not
into an agent worktree, or it evaporates with the worktree. And two agents prepending to
PROCESS.md conflict every time — one merge write-up, per-agent notes elsewhere.

---

## `spade` is installable now; the "101 modules import" claim never went through an install (2026-08-04)

Wave 1's stated verification — `pip install -e .` in a clean env, `import spade`,
`repo_root()` resolving here — had not been run. `spade` was installed in no environment,
`pyproject.toml` declared no dependencies at all (a commented-out placeholder), and every
line of `make setup` was a comment. The 101-module claim came from per-agent worktrees with
ad-hoc `PYTHONPATH`, which is not the thing the package has to do.

Fixed and verified in a throwaway venv built `--system-site-packages` off `spade_env`, so
the torch 2.5.1+cu124 stack is inherited and only the packaging is under test. **102 of 102
modules import cleanly** — the old count was one low, nothing was broken. Also verified from
stock `/usr/bin/python3.11` + pip 22.3.1, which is the fresh-clone path.

Do not `pip install -e` into `spade_env` itself: the editable path points at whatever
worktree ran it and dangles when that worktree goes.

Three real defects behind the missing install: `requirements.txt` named `python-chess` where
the working env has the `chess` distribution; `plotly` and `umap-learn` were pinned with no
importer anywhere we call, which is the exact defect that file's header criticises in the
source repo's version; and `pydantic` was undeclared although `viewer/server.py` imports it.
`pyproject.toml` now reads `requirements.txt` as its dependency list, so there is one list.

Still data-blocked, and reported as such rather than skipped: `final_validation` is 23/24
with the checkpoint round-trip skipped, `saes.registry.entries()` raises on the missing
`ae.pt`, and the wave-3 acceptance number needs weights this tree does not have. Details,
plus the `omegaconf` gap in the vendored TPR fork, in
[docs/finish_notes/env-install.md](docs/finish_notes/env-install.md).

---

## The defects the port agents recorded but did not fix (2026-08-04)

Swept all fifteen `docs/port_notes/` files for concrete code defects and fixed five. The
one that mattered: the sweep merge's baseline filter matched the substring `"SpaDE"`, which
also matches `"CosineSpaDE"`, so every `combined_results.csv` lost its CosineSpaDE row.
Latent in the source — `results/karvonen` had no CosineSpaDE row to lose — and live now
that `results/main` carries all eight baselines. `final_validation` pins it.

The other four are small: a `TypeError` on `--save-counts` (`repo_root` divided, not
called), `branches.py` importing `classify_prefix` through the rules study after it moved
into `othello/games.py`, and two docs that contradicted their own code.

Everything the notes flagged and I did *not* touch is written up with the reason in
`docs/finish_notes/flagged-defects.md`, including the one recommendation worth acting on:
`figure_karvonen_extension` writes its "no data" placeholder over the real figure's
filename, and six shell callers depend on that not raising.

---

## The port is done: 15 parallel agents, one branch each (2026-08-03)

Every zone ported from the frozen source repo — `common`, `othello`, `saes`, `spaces` +
`rules`, `rules/causal`, `training`, the pipeline, `figures`, the four studies, `scripts`,
`docs`, and the viewer. Each agent worked in its own worktree on its own `port/*` branch;
all fifteen are merged. Per-agent records are in `docs/port_notes/`.

**Where it stands:** 101 modules import cleanly, 28 analysis scripts compile, all 38 shell
scripts parse, `final_validation` is 23/24 with the one skip being the checkpoint
round-trip (weights are gitignored). Unit tests pass: 7 taxonomy, 11 divergence kernels,
plus the manifold-capture selftests reproducing the source numbers. What is *not* verified
is any real number — no eval has been run, because neither the checkpoints nor the
activations have been copied in. The wave-3 acceptance check (TopK on
`checkpoints/main/trainer_3` → L0 80.00, VE 0.943, coverage 0.722) is still outstanding and
is the first thing to do with a GPU.

**Seams the agents could not see from inside their own worktrees, fixed on merge:**
- *Three scripts would have destroyed the checkpoint layout.* They looped the eight
  per-type training entrypoints into one `--save_dir`, which writes `trainer_0` eight times.
  `trainer_N` is assigned by position within a single run, and `checkpoints/main/trainer_3`
  being TopK is a fact the registry, every eval CSV and every figure depend on. Now one
  `spade.training.sweep` call, which is also what the source did.
- *The SAE exclusion filtered nothing.* `EXCLUDED_SAE` was ported but never applied;
  `representative_spaces()` now drops it, which is what makes "excluded everywhere" true.
- *Two agents both ported `collect_legal` and `first_illegal_pos`* — byte-identical, since
  one took them from the other. Deduplicated.
- Four figure scripts called `results_dir()` with the old no-arg signature (`TypeError`);
  the viewer exporter used a registry field name that does not exist (`AttributeError`);
  three modules re-declared `LAYER`/`D_MODEL`/`CTX_LEN` locally, which is the drift that put
  nine copies of the dictionary size in the source repo.

**Needs David — the two that affect published claims:**

1. *The manifold-capture headline rests on the checkpoint we just excluded everywhere.*
   "Distance-SpaDE is the purest tiler, support size 0, shattering on 7/8 concepts" is a
   claim **about** the degenerate γ=1e-2 seed-42 checkpoint, and `docs/claims_status.md`
   lists it under "Holds" while retiring the geometry family's conclusion for resting on
   that same checkpoint. The study knows: it carries its own `LOW_VE` marker naming it.
   I aliased `LOW_VE` to `EXCLUDED_SAE` so there is one definition, and changed no
   behaviour. Either the shattering result is re-derived on a healthy SpaDE, or it needs
   stating as a property of a degenerate run, or the exclusion carves out an exception.

2. *A configuration in three documents never ran.* `checkpoints/main/trainer_5/config.json`
   records λ=9.77e-5, γ=0.2, steps 400, device mps. Three source docs report "SpaDE
   initial, λ=9.77e-5, γ=0.2" as a distinct over-regularized run. It is not a run at all —
   it is the γ=1e-2 production run wearing a Mac smoke config with pre-canonicalization
   defaults. That also explains the L0 15.4 / VE 0.352 attribution. The ported docs say so;
   the numbers in any older writeup do not.

Smaller items, each in the relevant `docs/port_notes/` file: BatchTopK *has* been evaluated
since 2026-07-14 (three source docs say it never was — real numbers now in the tables); four
bracket-flip CSVs have no producer in the tree because the A4 driver was never committed
upstream, so those figures redraw but cannot be re-derived; the extracted baseline-merge
helper filters `sae_type` on the substring "SpaDE", which also matches CosineSpaDE; and
`autoencoders_spade_v2` has no row in the artifact table and was mapped by the general rule.

---

## Wave 0 is blocked: there is nowhere to put the netscratch backup (2026-08-03)

**Needs a decision.** The agreed first step was to back up the netscratch tier before
porting. It cannot be done as specified. The tier is ~92 GB
(`spade_multiseed` 68 G, `spade_divergence` 9.3 G, `spade_divergence_v1_onecontrol` 8.4 G,
`spade_straightness` 5.3 G, `spade_bracket` 658 M, `spade_manifold` 471 M, `spade_rules`
69 M) and `/n/home10` has **22 GB free** against a 95 GB filesystem already 77% full.
Copying within netscratch is not a backup — the same 90-day retention applies, which is
what the existing 129 MB `spade_checkpoint_backup` there already runs into.

**Done instead, tonight.** Two things. A full file manifest of every `spade_*` dir — path,
size, mtime — committed under `manifests/netscratch/`, so if the tier expires we know
exactly what was lost. And the part that does fit: everything except the `.pt` checkpoints
is only 738 MB across 1306 files, and is now copied to
`/n/home10/dcao/spade_netscratch_backup/` (outside the repo, mirroring the netscratch
layout). That covers the per-seed CSVs and derived tensors the aggregation and figures
actually read. What remains exposed is ~91 GB of `ae.pt` weights.

**The question for David:** where does 92 GB go — lab storage, tape, cold storage — and is
it all of it, or only the parts that cannot be regenerated? The distinction that matters:
the per-seed CSVs are small and irreplaceable in practice; the checkpoints are large and
*technically* regenerable by re-running training, which is 2-day GPU jobs across 5 seeds
and 14 configurations. Also worth deciding whether `spade_divergence_v1_onecontrol` (8.4 G)
is kept at all — it is the superseded one-control branch set, but three shipped
sub-studies were computed on it and never re-derived.

**Deadline:** retention started 2026-07-22, so ~2026-10-20.

---

## The plan, re-derived against the live repo (2026-08-03)

**Goal:** the 2026-07-22 skeleton was written against the source repo as it stood that
morning, and twelve days of research landed on top of it. Fold all of that drift into the
plan so the port is planned against reality rather than against a two-week-old snapshot.

**Method:** eight parallel read-only surveys of the source repo, one per zone — `src/`,
`analysis/`, top-level artifacts, `shell/` + cluster env, the feature viewer, docs and
constraints, dependencies and packaging, and an audit of the skeleton itself. Findings are
folded into [ARCHITECTURE.md](ARCHITECTURE.md) and [MIGRATION.md](MIGRATION.md).

**What the plan was missing.** Seven code packages (`rules/` and its `causal/` subpackage,
`bracket_flip/`, `violation_divergence/`, `straightness/`, `manifold_capture/`,
`cosine_gamma_range/`), eight analysis studies, a second vendored runtime dependency
(`external/tpr_othello`), the whole figure-style layer (`paper_style.py` +
`FIGURE_GUIDELINES.md`), the feature viewer, 26 of 37 shell scripts, and the netscratch
tier that six studies depend on.

**Three findings that changed the plan rather than extending it:**

- *`rules/` is not a leaf study.* The plan modelled legality as one linear sub-pipeline.
  It is now the layer four other studies import — taxonomy, game streaming, board helpers,
  the checkpoint registry and the representation-space API all live inside it. Left as-is,
  every downstream study would import from a sibling study. The genuinely shared pieces are
  lifted into `spade/othello/`, `spade/saes/registry.py` and `spade/spaces.py`.
- *Three dirs the plan queued for deletion are live inputs.* `8_spade_configs_updated/`,
  `lambda_sweep_results/` and `gamma_sweep_results/` are hard-coded in
  `aggregate_seeds.py` and `figure_all_configs_combined.py`; dropping them removes the
  seed-42 point from every sweep config in the headline figure. The question for them
  changed from *drop?* to *where?*.
- *The per-study convention should be architecture, not repetition.* Five packages
  independently re-derived the same shape: a `study_dir.py` reading `SPADE_<STUDY>_DIR`
  with a netscratch default, numbered resumable steps, and a paired
  `analysis/<study>/make_figures.py` that only draws. That is lifted into
  `spade/common/study.py` and written down as the way a study is built.

**Decisions:**
- *All work happens in this worktree.* The source repo is frozen and read-only, ported
  from and never edited. The cost — no experiment can run here until the pipeline lands in
  wave 3 — is why the port order follows the import graph.
- *Commit at every coherent step, push at every wave boundary.* The skeleton sat
  uncommitted and unpushed for two weeks; that is the failure this rule prevents.
- *`paper_style.py` and `FIGURE_GUIDELINES.md` port in wave 1*, before any figure or
  analysis code, or the port recreates the color-literal drift the guidelines exist to stop.
- *`requirements.txt` gets regenerated, not copied.* The current file cannot build a
  working environment: four pins are wrong, `transformer_lens>=2.0` is incompatible with
  the vendored `circuits`, four hard deps are undeclared, and two pins have no importer.

**Decided with David (2026-08-03):**
- *Lift the shared layer out of `rules/`* — confirmed.
- *Back up netscratch before porting anything else*, as wave 0. ~92 GB on a 90-day clock
  that expires around 2026-10-20, holding the seeds behind every error bar.
- *Exclude the SpaDE γ=1e-2 seed-42 checkpoint everywhere*, from one place, rather than
  keeping the source repo's split where one study excluded it and three others plus the
  public viewer kept reading it. Its "SpaDE is broken for cell-state mapping" conclusion in
  the geometry family therefore has to be re-run or retired before it is quoted again.
- *The default SpaDE configuration is λ=1e-3, γ=1e-4* (`checkpoints/spade_revival/`),
  replacing γ=1e-2. It takes the excluded checkpoint's slot in the viewer roster, which
  stays at eight. Noted for the record: this pair was given as "γ=1e-3, λ=1e-4"; it is
  assigned here the way the existing named checkpoint assigns it, and is a one-line flip if
  the literal reading was intended.
- *CosineSpaDE's default stays γ=1.0*, against the γ-sweep's own recommendation of γ=20.
  The sweep selected 20 on reconstruction F₁, the one metric with an interior maximum
  (0.547 vs 0.530). Coverage runs the other way and is the deciding metric here: 0.537 at
  γ=1.0 against 0.474 at γ=20. The rest of the trade is L0 173 → 98 and loss recovered
  0.991 → 0.984. Worth knowing that the F₁ peak is thin — 0.011 over γ=10, with seed
  dispersion reported for L0 only (n=5) — so the metric that picked 20 is also the one
  whose margin is unquantified.
- *Put a freeze notice in the source repo* so a session that starts there is redirected.

**Open:** thirteen questions remain in MIGRATION.md — the report-file locations, several
data-retention calls, whether the geometry family gets re-run, the one-control divergence
sub-studies, the tracked-figure policy, and the viewer's deployment.

**Status:** plan updated and pushed. No code ported yet.

---

## New repository architecture skeleton (2026-07-22)

**Goal:** clean-slate layout for the SpaDE / OthelloGPT SAE project, built in this
sibling worktree (`spade_sae`, orphan branch `new-architecture`) because the original
repo's top level had grown ~13 parallel artifact/snapshot dirs. The original `src/`
was already clean, so the churn is top-level artifact consolidation + light code
renames.

This is a **skeleton**: every `.py` is a header comment + pseudocode. Real logic ports
over from `../Spade_Research_David_Sumedh_Ba` incrementally. The two worktrees share one
`.git`, so content can be diffed and cherry-picked between them.

**Decisions (user-confirmed):**
- *Installable package.* `src/spade/` is a real package (`pip install -e .` via
  `pyproject.toml`), replacing the original's sys.path-injection hack; `import spade.*`
  resolves everywhere and `make setup` installs it.
- *One training entrypoint per SAE type.* Shared loop lives in `training/core.py`; each
  `train_<type>.py` hands its type's config to `core`. Keep shared logic in `core`, not
  copied into each entrypoint, or the files drift (this trades some DRY for per-type
  clarity — the tradeoff was chosen deliberately).
- *Analysis code + outputs co-located.* Each `analysis/<study>/` keeps its `make_figures.py`
  next to its figures + results, so a study is self-contained.
- *Consolidated artifacts.* 4 `autoencoders*/` → `checkpoints/<experiment>/`;
  `results/karvonen*/` → `results/<experiment>/`; one deduped `figures/`.

**Status:** skeleton committed (5ac74a5). Superseded by the 2026-08-03 entry above, which
keeps all four decisions and corrects the layout they were applied to.

---

## Invariants carried over from the source repo

Moved to [ARCHITECTURE.md](ARCHITECTURE.md) § *Invariants a port must not break* — dispatch
ordering, checkpoint state-dict contracts, hyperparameter canonicalization, the known-stale
configs, the figure-style contract, and the environment pins.
