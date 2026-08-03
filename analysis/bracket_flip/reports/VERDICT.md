# A5 — red team verdict on the bracket-flip claim

The claim under test: *at a flanking move, in the model's mover-relative basis, the
flanked square s is the one square whose decoded class does not change, while the
rest of the board swaps.*

Everything below comes from `redteam.py`, which re-derives every board fact from the
event prefixes with the engine and writes every decode from the saved probe tensors.
It imports no analysis-agent code. Reproduce with

```
SPADE_REPO=<checkout with external/ and reference/> \
  python src/bracket_flip/redteam.py --epochs 600 --out report.json
```

Three decoders are carried side by side, because the committed ones are not from the
same corpus as the events:

| tag | what it is | trained on |
|---|---|---|
| `committed_linear_ARENA` | `analysis/linear_probe/activations/linear_probe.pt` | 160 ARENA games |
| `tpr_A1_ARENA` | A1's `tpr_r52_f2_layer{L}.pt` | 160 ARENA games |
| `redteam_linear_karvonen` | fitted here, 600 epochs, my code | 320 of A2's 400 Karvonen games |

The third exists because both committed probes were fitted on ARENA games with
ARENA's `t % 2` labels, while the events come from the Karvonen stream. Held-out
accuracy on A2's own 80 unseen Karvonen games, layer 5: committed linear **0.9873**,
TPR **0.9816**, mine **0.9923**.

## Contract keys, my values

| key | committed linear | TPR (A1) | redteam linear |
|---|---|---|---|
| `n_events` | 2000 | 2000 | 2000 |
| `p_s_unchanged_rel` | **0.9900** [0.9855, 0.9940] | 0.9845 [0.9790, 0.9895] | 0.9995 [0.9985, 1.0000] |
| `p_unflipped_swap_rel` | **0.9711** [0.9694, 0.9727] | 0.9616 [0.9600, 0.9633] | 0.9868 [0.9855, 0.9880] |
| `p_s_absolute_correct` | 0.9900 | 0.9845 | 0.9995 — **but see attack 7a** |
| `peak_layer` | n/a (layer 5 only) | **4** | **5** |
| `cos_flip_median` | — | 0.173 / 0.020 | 0.184 / 0.006 |
| `cos_null_median` | — | −0.006 / 0.106 | −0.004 / 0.101 |

95% CIs are percentile bootstrap over events (n = 2000), clustered by event for the
per-square keys. The two `cos_*` numbers are given as *control-contrast / before-after*;
attack 7b explains why one number is not enough. Two independent runs of the whole
script, including refitting all nine probes, reproduce every value above identically.

## Attack 1 — is the claim trivial?

**Tested.** What a perfect board-state oracle would produce, against what the probes
produce; and whether the observed values carry any information beyond the probes'
marginal accuracies.

**Numbers.**

A board-state oracle in the mover-relative basis scores **1.0000 on every key**:
`p_s_unchanged_rel`, `p_unflipped_swap_rel` and `p_s_absolute_correct` are all
identically 1 by the definition of the labels. Every measured value is *below* the
oracle. The gap is entirely probe error, and it is quantitatively exact:

| decoder | `p_s_correct_before` | `p_s_correct_after` | product | `p_s_unchanged_rel` |
|---|---|---|---|---|
| committed linear | 0.9950 | 0.9950 | **0.990025** | **0.9900** |
| TPR | 0.9905 | 0.9940 | **0.984557** | **0.9845** |
| redteam linear | 1.0000 | 0.9995 | **0.99950** | **0.9995** |

The headline is the product of the two marginal accuracies at s, to four decimals, for
all three probes. Errors before and after are independent; nothing is left over for a
flip-specific effect to explain. Separately, `p_s_unchanged_rel` equals
`p_s_unchanged_and_correct` exactly for all three probes — there is not a single event
where the decode is unchanged *and* wrong — so "unchanged" is just "read the board
right twice".

**What the measurement does add, stated precisely.** Only one thing: that s is decoded
*at least as reliably as any other disc*, so the model does not garble the square it
just flipped. On the same `after` activations (redteam probe):

- accuracy at s: **0.9995** [0.9985, 1.0000]
- accuracy at unflipped opponent discs: **0.9889** [0.9868, 0.9905]
- accuracy at all unflipped discs: **0.9886** [0.9867, 0.9901]

s is *easier* than an average disc, not harder. That is a real if modest result, and it
is a statement about probe accuracy at one square class, not about a special geometry.

**A second, worse problem: the sentence is false at the event level.** "s is *the one*
square whose class does not change" is an event-level claim. With ~29.2 discs per event
each swapping at p ≈ 0.97, probe noise leaves other discs unchanged too:

| decoder | mean other unchanged discs | events where s is genuinely the unique unchanged disc |
|---|---|---|
| committed linear | 0.71 | **0.6250** [0.6035, 0.6465] |
| TPR | 0.71 | **0.6025** [0.5815, 0.6250] |
| redteam linear | 0.35 | **0.7875** [0.7695, 0.8050] |

So in 37.5% of events (committed probe) the flanked square is *not* the only square
whose decoded class stayed put. The marginal 0.99 hides this completely.

**Verdict: BROKEN** as a claim about the model. The decode statistic is a restatement
of "the layer-5 residual supports accurate board-state decoding", which is Li et al. /
Nanda et al., re-expressed in a basis where accurate decoding necessarily looks like
"s unchanged, everything else swapped". The number is arithmetic from the probe's
marginals, it is strictly below what an oracle gives, and the "one square" phrasing
fails on a third of events. It survives only as an integrity check that rows, positions
and square codes line up — which is exactly how DATA_REPORT §"Alignment check" frames
it, and that framing is the honest one. The study's genuinely non-trivial result is the
control contrast in attack 7b, not this.

## Attack 2 — probe leakage

**Tested.** Whether the probes' training positions reappear as event positions. Games
being disjoint is not the relevant question: identical *prefixes* give identical
activations, so a shared prefix is the only channel by which an event row can also have
been a probe training row.

**Numbers.** A2's claim that the probe set and the event games share no games is correct
by construction — `build_probe_set` drains 400 games off the shared `stream_games`
iterator before `mine_events` touches it (`events.py:251` then `:278`), so the two
cannot overlap. I confirm the openings and the corpus match but the games do not.

Prefix collisions, however, are not zero:

| | count / 2000 events |
|---|---|
| event prefix appears in an ARENA training game (games 0–159) | **11** |
| event prefix appears in ARENA games 140–159 (A1's wd-selection slice) | **1** |
| event prefix appears in any of A2's 400 probe games | **22** |
| any of the above | **30** |
| duplicate prefixes among the events themselves | 2 |

All 30 are at **t ∈ {5, 6, 7}** — the opening tree is small enough there that two
independently sampled games collide. Dropping all 30:

| decoder | `p_s_unchanged_rel` all 2000 | uncontaminated 1970 |
|---|---|---|
| committed linear | 0.9900 | 0.9898 |
| TPR | 0.9845 | 0.9843 |
| redteam linear | 0.9995 | 0.9995 |

A1's `weight_decay = 2.0` was selected on ARENA games 140–159, which are inside the
8000-row training split and disjoint from the 2000-row ARENA eval rows; A1's account is
accurate. Exactly one event prefix touches that slice.

**Verdict: SURVIVES.** The leak exists, is 1.5% of events, is confined to t ≤ 7, and
moves the headline by 0.0002. The wording "the two sets share no games" should become
"share no games; 30 events share an opening prefix with a probe game, all at t ≤ 7".

## Attack 3 — forfeits

**Tested.** Every forfeit rule re-derived from `replay` in my own code, and whether the
mover is engine-derived rather than `t % 2`.

**Numbers**, over all 2000 events, all zero failures:

| check | failures |
|---|---|
| nominal mover has a regular move at t | 0 |
| opponent has a regular move at t+1 after m | 0 |
| opponent has a regular move at t+1 after the control m' | 0 |
| `mover` column equals `replay(P).next_hand_color` | 0 |
| played move is regular for that mover | 0 |
| `\|flip_set\| == 1` and equals `square_s` | 0 |
| s holds the opponent's colour before, the mover's after | 0 |
| board changes in exactly 2 squares | 0 |
| `control_flip_set`, `n_legal`, `control_kind`, `control_squares` reproduce | 0 |
| t ≥ 5 | 0 |

Parity is genuinely engine-derived, and it matters: `t % 2` names the wrong mover at
**2 / 2000** events. Small, but those two events would have had every one of their 64
labels inverted.

**Verdict: SURVIVES.** The forfeit rule is applied correctly, no surviving event has a
forfeit at t or t+1, the control branch is forfeit-free too, and parity comes from the
engine. This is the best-executed part of the study.

## Attack 4 — position and prefix confound

**Tested.** The confound is real in principle: `before` sits at position t−1 and `after`
at position t, so the move number and every other board change are entangled with the
flip. The study already contains the exact control the attack asks for and does not
report it. `control` is position t of `run(P + m')` — the *same* sequence position, the
same prefix, the same mover, the same next mover, and m' does not flip s. It is a
consecutive-position pair with no flip at s.

**Numbers.** If the "unchanged" reading at s were positional, it would appear in the
control branch too. It does not:

| decoder | s unchanged, before→after | s unchanged, before→**control** | s correct as *mine* in control |
|---|---|---|---|
| committed linear | 0.9900 | **0.0085** | 0.9935 |
| TPR | 0.9845 | **0.0115** | 0.9895 |
| redteam linear | 0.9995 | **0.0015** | 0.9985 |

Holding the position increment, the prefix and the perspective swap fixed and changing
only whether s was flanked moves the readout from 99% unchanged to 1% unchanged. A
no-flip pair at the same positions does not reproduce the effect; it reproduces the
opposite.

Two supporting numbers. Empty squares do not swap (`p_empty_unchanged` = 0.9983,
committed probe), so the swap is not a blanket position-parity bias on the probe output.
And the placed square m reads empty→theirs in 0.9890 of events, as the board says.

**How much of the before/after difference is position alone**, as the attack asks. Take
the same-position no-flip displacement `act(P+m') − act(P)` as the position-and-
perspective increment and project the real displacement onto it, per event:

| quantity | value |
|---|---|
| variance share of `act(P+m) − act(P)` along the per-event no-flip direction | **0.689** |
| median cos between the two displacements | **0.831** |
| mean ‖`act(P+m) − act(P)`‖ | 29.35 |
| mean ‖`act(P+m) − act(P+m')`‖ | 17.11 |

So 69% of the before→after displacement is reproduced by a move that does *not* flip s,
which is exactly the confound the attack names — and it is why the study is right to
build its geometric contrasts on `act(P+m) − act(P+m')` rather than on before→after. The
remaining 31% is where the flip lives, and attack 7b shows the flip direction is
recoverable there at cosine 0.18 against a null of −0.004.

**Verdict: SURVIVES, decisively.** This is the study's strongest internal control and it
belongs in DATA_REPORT and in the figures. Recommend the integrator promote
`p_s_unchanged_before_vs_control` to a reported key.

## Attack 5 — control quality (strict vs loose)

**Tested.** Whether the 404 `loose` events behave differently from the 1596 `strict` ones.

**Numbers.** The decode keys do not use the control move at all, so they cannot differ
except by sampling, and they do not:

| decoder | `p_s_unchanged_rel` strict / loose | `p_unflipped_swap_rel` strict / loose |
|---|---|---|
| committed linear | 0.9906 / 0.9876 | 0.9724 / 0.9659 |
| TPR | 0.9843 / 0.9851 | 0.9625 / 0.9580 |
| redteam linear | 0.9994 / 1.0000 | 0.9884 / 0.9804 |

The contrast keys do consume the control, and there the `loose` group is very slightly
weaker, as it should be — a `loose` control has the placement or one of its flipped
discs somewhere on one of s's eight rays:

| direction | strict (n=1596) | loose (n=404) |
|---|---|---|
| `cos_flip_median`, TPR pinv direction | 0.1748 | 0.1651 |
| `cos_flip_median`, linear probe direction | 0.1856 | 0.1782 |

**Verdict: SURVIVES.** The decode keys cannot depend on the control and do not; the
contrast keys move by 0.008–0.010 between strict and loose, an order of magnitude
smaller than the 0.18 effect and far inside the flip-vs-null separation. Pooling the two
groups is safe. Reporting them split, as above, costs one row and removes the question.

## Attack 6 — class collapse

**Tested.** Whether the probe's error structure makes the "swap" statistic nearly forced.

**Numbers.** Full 3×3, row-normalised, rows = true, columns = predicted, on the 2000
`before` activations at layer 5:

`committed_linear_ARENA`

| true \ pred | empty | theirs | mine | n |
|---|---|---|---|---|
| empty | 0.9992 | 0.0003 | 0.0006 | 67,566 |
| theirs | 0.0026 | 0.9830 | 0.0144 | 32,041 |
| mine | 0.0028 | 0.0165 | 0.9807 | 28,393 |

`tpr_A1_ARENA`

| true \ pred | empty | theirs | mine | n |
|---|---|---|---|---|
| empty | 0.9973 | 0.0012 | 0.0015 | 67,566 |
| theirs | 0.0088 | 0.9749 | 0.0163 | 32,041 |
| mine | 0.0111 | 0.0160 | 0.9730 | 28,393 |

`redteam_linear_karvonen`

| true \ pred | empty | theirs | mine | n |
|---|---|---|---|---|
| empty | 0.9998 | 0.0001 | 0.0001 | 67,566 |
| theirs | 0.0005 | 0.9910 | 0.0085 | 32,041 |
| mine | 0.0007 | 0.0105 | 0.9888 | 28,393 |

The concern is half-right. The probe *does* predict empty for occupied squares — 0.26%
to 1.1% of the time, so this is not literal collapse. But 85% (committed), 59–65% (TPR)
and 94% (mine) of the errors on an occupied square are theirs↔mine confusions, which makes
the disc classes effectively a binary problem. The consequence is measurable:

| decoder | `p_unflipped_changed` | `p_unflipped_swap_rel` | ratio |
|---|---|---|---|
| committed linear | 0.9757 | 0.9711 | 0.9953 |
| TPR | 0.9757 | 0.9616 | 0.9856 |
| redteam linear | 0.9879 | 0.9868 | 0.9989 |

Given that a disc's decode changed at all, it swapped between theirs and mine 99.5% of
the time. "Swap" therefore carries about half a point beyond "changed", and "changed" is
forced by the mover-relative basis plus accurate decoding.

**Verdict: WEAKENED.** Not collapsed, but the three-class framing is doing almost no
work; the statistic is a two-class one. Report the full 3×3, not accuracy, and do not
present "swaps" as if the empty class were a live alternative.

## Attack 7 — additional attacks

### 7a. `p_s_absolute_correct` is a relabelling, and the underlying claim fails

The design doc says absolute colour "is recoverable from the same probe: compose the
relative label with whose turn it is" and calls the relabelling "free and exact". It is
— which is the problem. Given the engine-derived mover, the map
(relative class) → (absolute colour) is a bijection, so `p_s_absolute_correct` computed
by relabelling is *identically equal* to "s decoded correctly before and after" in the
relative basis. I verified the identity holds element-wise on all 2000 events. It is not
a second measurement.

The design doc anticipates this and specifies the real test: fit a probe on
engine-derived absolute labels and compare. I ran it at layer 5 on A2's set, same
architecture, same optimiser, same epochs, same split:

| basis | train accuracy | held-out accuracy | accuracy at s over the 2000 events |
|---|---|---|---|
| relative {empty, theirs, mine} | 0.9997 | **0.9923** | 0.9995 |
| absolute {empty, black, white} | 0.8205 | **0.7407** | **0.3425** [0.3220, 0.3625] |

Train accuracy is 0.82, so this is a capacity ceiling, not underfitting — the absolute
labels are not linearly available at layer 5. A probe trained on absolute colour gets
the "white before, black after" statement right at s in **34%** of events, against 100%
for the relabelled version. Li et al.'s result reproduces here.

**Consequence.** "White becomes black" is something *we* compute from the relative
decode plus the mover; it is not something the model stores. Any figure or sentence
presenting the absolute panel as a second, independent confirmation is
double-counting one measurement. `p_s_absolute_correct` should either be dropped from
the contract or renamed to make the relabelling explicit, with 0.3425 reported beside
it as the model's actual absolute-colour readout.

### 7b. `cos_flip_median` / `cos_null_median` are not well defined

The design doc's M3 asks for the TPR-predicted flip term at s,
`pinv(M)[r_s (f_after − f_before)ᵀ]`, from the before→after displacement. **In the
mover-relative basis, s's class does not change, so `f_after − f_before = 0` and this
term is identically the zero vector.** The contract key as literally specified is
degenerate. Two repairs are available and they order the two medians in opposite
directions:

| reading | Δ | direction at s | `cos_flip_median` | `cos_null_median` |
|---|---|---|---|---|
| A: control contrast | `act(P+m) − act(P+m')` | mine → theirs | **0.184** (linear) / **0.173** (TPR) | **−0.004** / **−0.006** |
| B: before → after | `act(P+m) − act(P)` | theirs → mine | 0.006 / 0.020 | 0.101 / 0.106 |

Reading A is the correct one and it is a genuine result: in the m branch s reads
*theirs*, in the m' branch it reads *mine*, the two branches share a prefix, a position
and a next mover, and the displacement aligns with the predicted direction at 0.17–0.18
against an unflipped-disc null at ≈0. Reading B measures whether s swaps like the rest
of the board — it does not (0.006), while unflipped discs do (0.101), which is the same
fact as attack 1 in cosine form.

A4 must state which contrast the reported number uses. If A3 and A4 quote medians from
different readings they will differ by 0.18 and the contract's 0.05 tolerance will fire
on a definitional difference, not a disagreement.

### 7c. `peak_layer` is not a single number

| probe | by `p_s_unchanged` | by `p_s_correct_both` | by `p_unflipped_swap` | by board accuracy | by specificity margin |
|---|---|---|---|---|---|
| TPR (A1) | 4 | 4 | 4 | 4 | 4 |
| redteam linear | 5 | 5 | 5 | 6 | 6 |

The TPR peaks at layer 4 on *every* definition — including on ARENA, where A1's own
table has L4 0.9831 above L5 0.9827. Layer 5 was chosen because the design doc names it,
not because the TPR peaks there. My independent linear probe peaks at 5 for the flip
readout and 6 for raw board accuracy; its layer-6 and layer-7 numbers stay high
(`p_s_unchanged` 0.9970 at both), whereas the TPR droops to 0.9745 and 0.9720. That
droop is the capacity artefact PROBE_REPORT §6.6 predicted, and it is the reason the two
probes disagree about the peak.

The contract demands `peak_layer` match *exactly*. It will not. Pin the definition
(I recommend `p_s_correct_both`, the strictest) and read the depth curve as a gap
against the linear probe, as the design doc already instructs for M5.

### 7d. The design doc's premise about the global swap is wrong

> "the before→after displacement is dominated by a global perspective swap that is
> nearly identical across events"

The concept is right; "nearly identical across events" is not. The *mean* displacement
direction explains **0.61%** of the energy of `act(P+m) − act(P)`, so there is no shared
swap vector. The same swap taken **per event** — the direction of `act(P+m') − act(P)`,
which carries the identical position increment and perspective flip but no flip at s —
explains **68.9%**. The swap is a per-event, board-dependent operation, not a constant
offset. That is a stronger justification for the control-move design than the one the
doc gives, and it should replace it: a figure motivated by "subtract the shared swap
vector" would be subtracting something that accounts for under 1% of the displacement.

### 7e. Scope: the headline subset is the structurally easiest case

`s_not_adjacent_to_m` fails on **0 of 2000** events: with `|flip_set| == 1` the bracket
is (m, s, own disc), so s is *always* the square immediately adjacent to the token just
written. Combined with `|flip_set| == 1` rejecting 62,445 of 106,000 scanned positions
(59%), the study measures the minimal-change, adjacent-square case and says nothing
about the 59% of moves that flip more than one disc — for which "s is the one square
whose class does not change" does not even typecheck. This is a scope limit, not an
error, but it should be stated in the abstract of the write-up rather than left to the
mining table.

### 7f. Provenance: the TPR was never retrained on A2's set

The design doc says "At integration the TPR is retrained on A2's set, with A1's
validation number as the control." The checkpoints in
`$SPADE_BRACKET_DIR/tpr_probes/` are A1's, fitted on 160 ARENA games with ARENA's
`t % 2`-derived labels (which DATA_REPORT itself shows carry 53 mislabelled rows in
10,000). If A3 reports headline numbers from these, they are cross-corpus and
cross-convention. The damage is small — the TPR still scores 0.9816 on A2's held-out
Karvonen rows — but the retraining step in the design is outstanding and should either
happen or be explicitly withdrawn.

### 7g. Row alignment and tensor sanity

No `before` row equals its `after` row; mean absolute per-coordinate change at layer 5
is 1.024 against a mean activation norm of 23.3. Every one of the 2000 rows decodes to
the board my engine replay predicts for that row, at 0.99+ accuracy, which is only
possible if `events.csv` row i really is row i of each tensor. No misalignment.

## Bottom line

The study's data pipeline is sound and I could not break it: every integrity check
re-derives cleanly through my own code, the forfeit handling is correct, parity is
engine-derived, and the position confound is decisively ruled out by the control branch
that is already in the data. What I *can* break is the headline sentence. In the
mover-relative basis, "s does not change while the rest of the board swaps" is what a
perfect board oracle produces by definition — it scores 1.0 — and the measured 0.99 /
0.97 is exactly the product of the probe's marginal accuracies at those squares, to four
decimals, for all three probes I tried. It carries no information about the model beyond
"layer 5 supports accurate board-state decoding", which is prior work; and as an
event-level statement it is false in 37.5% of events, because probe noise leaves 0.71
other discs unchanged on average. The absolute-colour version is a relabelling of the
same argmax, and when a probe is actually trained on absolute labels it reaches 0.74
held-out and gets s right 34% of the time, so that framing should not be presented as
independent support. The one result here that is about the model rather than about the
label convention is the control contrast: `act(P+m) − act(P+m')` aligns with the
predicted direction at s at median cosine 0.18 against an unflipped-disc null at −0.004,
holding prefix, position and mover fixed. **That should be the study's headline. The
decode statistic should be demoted to what DATA_REPORT already calls it — an alignment
check — and the "one square whose class does not change" phrasing should be retired.**

## Numbers where I disagree with the other agents

A3 and A4 had not written `probe_space.py` or `act_geometry.py` when I finished, so
these are disagreements with the wave-1 reports and with the design doc.

1. **`p_unflipped_swap_rel` = 0.9757 (DATA_REPORT) vs 0.9711 (mine).** A2's 0.9757 is
   the fraction of unflipped discs whose decoded class *changed*, not the fraction that
   *swapped* between theirs and mine — I reproduce 0.9757 exactly under the "changed"
   definition and get 0.9711 under the swap definition. The 0.46-point difference is
   discs that moved into or out of the empty class. A2's own per-row numbers (0.9683
   theirs→mine, 0.9643 mine→theirs) are the strict both-correct version and pool to
   0.9664 in my run. Three different statistics are in play under one name; the contract
   key must say which.
2. **`peak_layer`.** I get 4 from the TPR and 5 from an independent linear probe, on
   every definition. Any single value reported without a definition and a probe is not
   checkable.
3. **`cos_flip_median` / `cos_null_median`.** Undefined as specified (the predicted flip
   term at s is the zero vector in the relative basis). My values under the control
   contrast are 0.184 (linear) / 0.173 (TPR) against nulls of −0.004 / −0.006; under
   before→after the ordering reverses to 0.006 / 0.101.
4. **`p_s_absolute_correct` = 0.99 by relabelling vs 0.3425 from an absolute-label
   probe.** These are not the same quantity and only the second is a measurement of the
   model.
5. **"the before→after displacement is dominated by a global perspective swap that is
   nearly identical across events" (design doc).** The mean displacement direction
   accounts for 0.61% of the displacement energy.
6. **"the two sets share no games" (DATA_REPORT).** True for games; 30 of 2000 events
   share an opening prefix with a probe-training game, all at t ≤ 7. Effect on the
   headline: 0.9900 → 0.9898.
