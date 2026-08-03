# Bracketing flips: what happens to a square's representation when it is flanked

Design doc for the `bracket_flip` study. Subagents implement against this;
the rule set it sits in is `docs/othello_rules.md`.

> ## Correction — three of this design's premises did not hold
>
> The study ran and was red-teamed. Three things this document asserts or
> measures are retired; it is kept as the record of what was intended, not as a
> statement of results.
>
> - **"The flanked square is the one square whose class does not change" is not
>   evidence about the model.** A perfect board oracle scores 1.0 on that key by
>   construction. The measured 0.9900 is the product of the probe's marginal
>   accuracies at the square (0.9950² = 0.990025, agreeing to four decimals), it
>   tracks probe accuracy at r = 0.999 across ranks, and it is false in 37.5% of
>   individual events. **`p_s_unchanged_rel` in the contract below is degenerate
>   as specified.** Killed independently twice.
> - **`p_s_absolute_correct` = 0.99 is a relabelling, not a second
>   measurement.** A probe actually trained on absolute labels reaches 0.7407
>   held out and gets the square right in 34.25% of events.
> - **The before→after displacement is not "a global swap nearly identical
>   across events."** The mean direction explains 0.61% of the energy; the
>   per-event matched control explains 68.9%. The M3 decomposition and the F1/F2
>   framing below both rest on that premise.
>
> What survives, and what the study is actually for (`docs/claims_status.md`):
> the model is **mover-relative** and barely encodes absolute colour (absolute
> TPR 0.7349, 0.004 *below* the label-only occupancy ceiling, while the relative
> probe clears its ceiling by +23.1); the flip has **a specific direction**
> (median cosine 0.173–0.184 against a matched non-flipping move, null ≈ 0,
> positive in 1596/1596 strict events, TPR and linear directions agree at 0.822);
> and it is **causally used** (editing along it moves +1.496 log-units against
> +0.121 for random).
>
> One narrowing to quote with its qualifier: `peak_layer` is not one number — 4
> by the TPR probe, 5–6 by an independent linear probe. That is not a conflict
> with the causal depth result: board state resolves early, legality consequence
> late.
>
> One open defect: the TPR probe was never retrained on the integrated event set.

## The question

A white disc sits on square *s*. Black plays a move *m* that flanks it. The
disc flips to black. Does OthelloGPT's layer-5 residual follow — and what does
the change look like geometrically?

The decoder is the tensor-product-representation probe of Lee, Viégas &
Wattenberg (arXiv:2605.09967, `github.com/ajyl/tpr_othello`), which factorizes
board-state decoding as `logit(s, c) = r_sᵀ B f_c` with `B = M(h)`: square roles
`R ∈ ℝ^{64×d_r}`, color fillers `F ∈ ℝ^{3×d_f}`, and a hidden-to-binding map
`M ∈ ℝ^{d_r×d_f×d_model}`. Bilinear `d_r=52, d_f=2` reaches 99% accuracy on 57.5%
of the linear probe's parameters. Their experiments use layer 7; ours use
layer 5, the best layer for our checkpoint (linear-probe accuracy 0.9896 vs
0.9870 at layer 6 — `analysis/linear_probe/activations/probe_metadata.json`).

## The subtlety that defines the design

Both probes are **mover-relative**: fillers are {Empty, Current, Opponent},
our labels are {empty, theirs, mine} with "mine" = the player to move next.
Playing a move flips the perspective as well as the disc, so:

| square | before *m* | after *m* | |
|---|---|---|---|
| flanked disc *s* | theirs | theirs | **unchanged** |
| unflipped opponent disc | theirs | mine | swaps |
| unflipped own disc | mine | theirs | swaps |
| placed square *m* | empty | theirs | — |

A flip is therefore **the absence of a swap** against a board where every other
disc swaps. "White becomes black" is the absolute-color statement of the same
event, and it is recoverable from the same probe: compose the relative label
with whose turn it is. **Derive that from engine replay, never from `t % 2`** —
forfeits are unmarked in this data and 34% of games contain one, so
move-number parity desynchronizes from the mover.

Relabeling is free and exact. Whether the model natively carries absolute color
is a separate, measurable question: train a second TPR whose fillers are
{empty, black, white} on engine-derived absolute labels and compare accuracy
against the relative one. Matching accuracy means the basis is cosmetic;
a collapse means mover-relative is the representation and absolute color is
something we compute, not something the model stores. Li et al. report the
latter for linear probes, so this is a real test rather than a formality. **This
is the part of the design that worked**: the absolute probe collapses to 0.7349,
below its own label-only ceiling.

Consequence for the geometry, **as originally reasoned**: the before→after
displacement is dominated by a global perspective swap that is nearly identical
across events, so the flip signal is the deviation from it, and every contrast
below is built to cancel that. **That premise is false** — the mean direction
explains 0.61% of the energy against the per-event matched control's 68.9%. The
contrasts below are still the right ones to compute, because a matched control
move cancels far more than a global mean ever could; what does not follow is any
claim that the displacement decomposes into "a global swap plus a small flip
term."

## Events

From the legal-game corpus, for each game and move index *t*: prefix `P =
tokens[:t]`, board `B = replay(P)`, mover *e*, played move *m*, `flip_set(B, m,
e)`.

- **Headline subset: `|flip_set| == 1`.** The boards before and after differ in
  exactly two squares — *m* (empty→*e*) and *s* (opponent→*e*). This is the
  cleanest minimal pair available and is the scenario as posed.
- **Control squares** (within the same activation): unflipped discs of both
  colors, plus empty squares. They ride along in the same forward pass, so they
  control for move number and parity by construction.
- **Control move** *m'*: another legal move for *e* at the same prefix whose
  flip set excludes *s*. Prefer *m'* whose placement and flip set avoid *s*'s
  eight rays entirely — the `strict` idea from `spade.rules.causal.pair_patch`.
  `act(P+m) − act(P+m')` cancels the shared prefix, the move-number increment
  and the parity flip, leaving flip-attributable structure.

Activations, layer-5 `resid_post` plus all 8 layers for the depth sweep:
`before` = last position of `run(P)`; `after` = position *t* of `run(P+m)`;
`control` = position *t* of `run(P+m')`.

Target ≈2000 headline events; report the mining yield and the *t* distribution
rather than silently taking what comes.

## Measurements

- **M1 — decode.** TPR readout of *s* before/after in the native basis (predict
  theirs→theirs) and relabeled to absolute color (predict white→black). Report
  per-class accuracy for flipped vs unflipped squares. Then the basis test: a
  second TPR trained on engine-derived absolute labels, same architecture and
  seed, accuracy compared against the relative one.

  **The first half of this is the degenerate measurement.** "Predict
  theirs→theirs" is what a board oracle does by construction; only the basis
  test carries information about the model. See the correction header.
- **M2 — displacement field in decode space.** Draw before→after arrows for all
  64 squares. Prediction: 63 arrows cross from one owner to the other, *s*'s
  arrow is ~zero.

  **Draw it in logit differences, not filler coordinates.** The bilinear form
  `logit = r_sᵀ B f_c` is invariant under `f → Af, B → BA⁻¹` for any invertible
  2×2 `A`, so filler coordinates carry four arbitrary degrees of freedom — A1
  measured the mine/theirs angle moving 131.3° → 113.3° → 104.5° across seeds
  while accuracy moved 0.0006. Panels drawn there are not comparable across
  probes, seeds, or label conventions. Logits are gauge-invariant, and three
  classes leave exactly two free differences, so the panel stays honestly 2-D:

  - `x = ℓ_mine − ℓ_theirs` — who owns the square; this is what swaps.
  - `y = ½(ℓ_mine + ℓ_theirs) − ℓ_empty` — whether it is occupied at all.

  The prediction reads cleanly in this frame: unflipped discs cross x at
  roughly constant y, the placed square climbs in y, and *s* stays put. Report
  filler geometry as a number (A1's angles, with the gauge stated) rather than
  as an axis anyone is asked to read.
- **M3 — activation geometry.** Paired PCA of before/after with per-*t*
  centering and pair arrows; then decompose `Δ = h_after − h_before` into the
  global-swap component, the TPR-predicted flip term for *s*
  (`pinv(M)[r_s(f_after − f_before)ᵀ]`), and the residual, reporting variance
  shares. The flip-attributable contrast is `act(P+m) − act(P+m')`.

  **The three-way decomposition is built on the false premise** — the global-swap
  component is 0.61% of the energy, so the split is not informative as designed.
  The `act(P+m) − act(P+m')` contrast is what carried the surviving result.
- **M4 — specificity.** `cos(Δ, flip direction for s)` against the same
  quantity for unflipped squares — the empirical null. Distributions, not point
  estimates. **This is the measurement that held.**
- **M5 — depth.** Repeat M1 per layer 0–7. Where the flip readout appears
  should agree with `analysis/rules_causal/pair_patch.png`, which puts flip
  consequences in mlp5/mlp6. Cross-study check on an existing result.

  **Confounded by probe capacity, and the figure must say so.** The TPR trails
  the linear probe by 0.14–0.36 points at layers 0–5 but by 1.53 and 1.87 at
  layers 6 and 7, where the linear probe stays flat. A TPR depth curve peaking
  at layer 4–5 therefore partly reports the probe, not the model. Plot the
  linear probe on the same axes as the capacity reference, and read the *gap*,
  not the TPR curve alone. In the event, the two instruments disagree — peak
  layer 4 by TPR, 5–6 by the linear probe — which is why `peak_layer` is
  reported as a pair rather than a number.
- **M6 — deferred by decision, not dropped.** Applying the paper's TPR
  intervention (`ΔB = r_s(f_ŷ − f_y)ᵀ` through `pinv(M)`) to un-flip *s* would
  cross-check `spade.rules.causal.unflip_edit`, which makes the same edit along
  linear-probe directions. Revisit once M1–M5 land. `edit_direction()` is built
  now so the hook exists.

## Figures

Per `FIGURE_GUIDELINES.md`; colors and markers from `spade.figures.paper_style`,
which the integrator extends — agents never edit it.

- **F1** filler-space displacement field (M2), native and absolute panels.
- **F2** one canonical worked event: board before/after with *s* marked, beside
  its decode. The concrete anchor for the abstract panels.
- **F3** activation geometry (M3): paired PCA with arrows, plus the
  decomposition's variance shares.
- **F4** specificity distributions (M4) with the unflipped-square null — KDE per
  §4.4, bootstrap band, null gray/dashed/last.
- **F5** depth sweep (M5) per §4.1: layers 0–7, indexing base stated, both
  anchors labeled.

Uncertainty on every claim-supporting curve: 95% CI over events, n stated,
empirical nulls preferred over analytic chance lines.

The set that shipped is smaller than this, for the reasons in the correction
header: figures whose only content is the degenerate decode key were cut rather
than captioned.

## Layout

```
external/tpr_othello/              vendored at pinned commit cc90cea
src/spade/bracket_flip/{tpr_probe,events,extract_acts,geometry}.py
analysis/bracket_flip/{make_figures.py,figures/,results/,README.md}
scripts/slurm/studies/bracket_flip/
```

Large artifacts on `$SPADE_BRACKET_FLIP_DIR` (default netscratch
`spade_bracket_flip`), resolved by `spade.common.study.study_dir()` and shared
across worktrees so extraction runs once. Committed results are per-event CSVs
small enough to read as text.

## Interfaces the agents must honor

```python
# spade.bracket_flip.tpr_probe
train(layer: int, seed: int) -> TPRProbe        # fits on our activations
load(layer: int) -> TPRProbe
TPRProbe.decode(h) -> Tensor        # [..., 64, 3] logits over {empty, theirs, mine}
TPRProbe.roles                      # [64, d_r]
TPRProbe.fillers                    # [3, d_f]
TPRProbe.edit_direction(square, from_c, to_c) -> Tensor   # [512], via pinv(M)

# spade.bracket_flip.events  -> one row per event
# game_id, t, mover, move, square_s, flip_set, control_move, control_squares
# spade.bracket_flip.extract_acts -> {before, after, control}: [n_events, 8, 512]
```

`events.csv` mixes two integer conventions, so read it with care:
**`prefix`, `move`, `control_move` are vocab tokens (1–60)**; **`square_s`,
`flip_set`, `control_squares` are engine squares (0–63)**. `to_string(token)`
converts one to the other. The give-away is that `square_s` takes values above
60, which no token does. `mover` is the engine's ±1 (black = 1).

## Acceptance criteria

- TPR layer-5 accuracy within 1 point of the linear probe. Met: **0.9827**.
  Two baselines exist and they are not the same number — 0.9896 is the layer
  sweep's quick probe (`probe_metadata.json`), 0.9884 is the committed
  `linear_probe.pt` scored on the same eval rows. Gaps: 0.69 and 0.57 points.
  Quote which one you mean.
- Every mined event re-derived against the engine: `board_after ==
  replay(P + m)`, `|flip_set| == 1`, *s* holds the opponent's color before and
  the mover's color after.
- The two analysis agents independently report the same n and the same sign and
  rough magnitude for the headline effect; the red-team agent arbitrates
  disagreements and tries to break both.
- Position 0 is never used (attention sink, norm ~300 vs ~22 — see the
  straightness study).

Worth noting against the first criterion: probe accuracy this high is exactly
what makes `p_s_unchanged_rel` uninformative. A criterion that a probe be
near-perfect and a headline metric that is a function of that same accuracy
cannot both carry weight.

## Cross-validation contract

Wave-2 agents report these keys, with these names, in a `RESULTS.md` table.
Comparing them is mechanical, which is the point — agreement is checkable
rather than asserted.

| key | definition | reported by |
|---|---|---|
| `n_events` | headline events analyzed | A3, A4, A5 |
| `p_s_unchanged_rel` | events where *s*'s decoded relative class is identical before and after — **degenerate as specified; a board oracle scores 1.0** | A3, A5 |
| `p_unflipped_swap_rel` | unflipped discs decoded `mine` before and `theirs` after, **or vice versa** — *swapped*, not merely *changed*: a disc decoded `empty` after has changed but not swapped, and the two statistics differ by ~0.5 points (0.9757 changed vs 0.9711 swapped on the same data). Three agents reported three numbers under this one name before it was pinned down. | A3, A5 |
| `p_s_absolute_correct` | events decoding white before **and** black after, relabeled — **a relabelling of the row above, not a second measurement** | A3, A5 |
| `acc_rel_L5`, `acc_abs_L5` | TPR accuracy at layer 5, relative vs absolute fillers — **the pair that carries the basis result** | A3 |
| `cos_flip_median`, `cos_null_median` | alignment of Δ with the predicted flip direction, and the unflipped-square null | A4, A5 |
| `var_swap`, `var_flip`, `var_resid` | variance shares of the Δ decomposition — **`var_swap` is 0.61%; the decomposition's premise did not hold** | A4 |
| `peak_layer` | layer where the flip read-out is strongest — **two numbers, not one: 4 by TPR, 5–6 by linear probe** | A3, A5 |

Agreement: `n_events` must match exactly, proportions within 2 points, medians
within 0.05, `peak_layer` exactly. Any mismatch is A5's to arbitrate, and the
integrator does not merge until it is explained — not averaged away.

## Agent plan

Two waves of worktree-isolated agents, each merged only after review.

**Wave 1 — foundation, parallel, disjoint files.**
- **A1 probe**: vendor `external/tpr_othello` at a pinned commit, read their
  trainer, implement `tpr_probe.py`, validate against the *already committed*
  `analysis/linear_probe/activations/` set so it is unblocked immediately, and
  report the learned filler geometry.
- **A2 data**: everything that touches the model — `events.py`,
  `extract_acts.py`, the SLURM script. Mines events, extracts their
  activations, and builds a fresh probe-training set carrying **both** label
  conventions (engine-derived, all 8 layers) for the absolute-vs-relative test.
  One cache under the study dir; reports yield and integrity checks.

The split keeps A1 off the GPU and off A2's critical path. At integration the
TPR is retrained on A2's set, with A1's validation number as the control. (That
retrain is the step that never happened — see the correction header.)

**Wave 2 — analysis, parallel, deliberately overlapping.**
- **A3 probe-space**: M1, M2, M5 → F1, F2, F5.
- **A4 activation-space**: M3, M4 → F3, F4. Uses the TPR only for the predicted
  flip direction; its core result must stand on raw activations and the existing
  linear probe, so the two agents' conclusions are method-independent. This is
  the split that saved the study: A4's specificity result survived the
  red team, A3's decode headline did not.
- **A5 red team**: re-derives the headline numbers by its own code path, attacks
  the confound handling (parity, move number, flip-set size, control matching,
  probe leakage), writes a verdict. No figures.

**Rules for every agent.** Work only inside your assigned paths. Never edit
`spade/figures/paper_style.py`, `FIGURE_GUIDELINES.md`, `PROCESS.md`, or any
existing study — request changes from the integrator. Read `CLAUDE.md` and
`FIGURE_GUIDELINES.md` first. Reuse `spade.othello.boards` and the existing
corpus loaders; do not re-implement the Othello engine. Report numbers,
not adjectives.

**Integration (not delegated).** Cross-check A3/A4/A5 numbers, resolve
disagreements, merge worktrees, unify figures and `paper_style` entries, write
`analysis/bracket_flip/README.md` and the PROCESS.md entry, commit once.

## Running things on this cluster

`python` is not on `PATH` and `module` is unavailable in non-interactive shells.
Use `/n/home10/dcao/.conda/envs/spade_env/bin/python` directly. GPU work goes
through `scripts/slurm/studies/bracket_flip/` (account `kempner_ba_lab`,
partition `kempner_h100`); the rules causal battery took 3m21s there, so this
study's extraction is minutes, not hours. Position 0 is never used.
