# Bracket flips

What happens to a square's layer-5 representation at the moment it is flanked
and flipped. Design and rule mapping: `docs/bracket_flip_design.md`; the rule
set it sits in is `docs/othello_rules.md` (R5's state update, reached through
the R2 bracketing that causes it).

Decoder is the tensor-product-representation probe of Lee, Viégas & Wattenberg
(arXiv:2605.09967), vendored at `external/tpr_othello`, trained here on our
activations: `logit(s, c) = r_sᵀ M(h) f_c` at `d_r=52, d_f=2`, 56,582
parameters — 57.6% of the linear probe's, matching the paper's 57.5%.

**Which fit every number here comes from.** Both probe families — this TPR and the
linear reference it is measured against — are the project's canonical probes,
`spade.probes.load`: 20,000 games of `adamkarvonen/othello_45MB_games`, 812k
training rows split by game, labels from engine replay, stopped on a validation
plateau. One corpus and one protocol for both, so Fig 2's head-to-head is a
comparison of probe class and of nothing else. Two arms have no canonical
counterpart and are still fit here, on the study's own 400-game probe set but
under the same protocol (`probe_space.local`, cached under
`$SPADE_BRACKET_DIR/probe_space/`): the absolute-colour basis, and layer −1,
which is the input space rather than a block output.

**The relative accuracies in `m1_basis.csv` are not held out.** The canonical
probes are fit on the same Karvonen stream at the same seed this study samples,
so most of the 240 eval games are inside their training split. The held-out
number is `acc_*_relative_canonical_test` (TPR 0.9920, linear 0.9937 at layer 5),
and that is the one to quote; the study-set column exists so the relative and
absolute arms are read off one population. The absolute arm is fit here and its
0.7315 *is* held out.

## What this study does and does not show

**Retired claim.** An earlier draft led with "the flanked square is the one
square whose class does not change." That sentence is a true description of the
board in a mover-relative basis, and it is **not evidence about the model** — a
perfect board oracle reproduces it by definition. Two agents killed it
independently. `p_s_unchanged_rel` = 0.998 tracks the probe's accuracy: it moves
with `acc_rel_L5` at Pearson 0.9997 across probe ranks (0.683 at `d_r=16`, 0.9945
at 64), which is what a decode-noise quantity does and not what a finding does.
The canonical probe is accurate enough that the event-level version is now nearly
true — *s* is the *uniquely* unchanged disc in 0.822 of events, against 0.625
under the retired 8,000-row fit — and that is the point: the proportion is a
property of the read-out, so improving the read-out moves it.

What the flip framing is good for is stating the *basis*, not proving anything
in it. Fig 1 is drawn that way: the arrow field in the native basis is the
frame, and what carries a claim is the second probe drawn beside it.

**Three findings that survive.**

1. **The model represents board state mover-relatively, and barely represents
   absolute colour at all.** An absolute-label TPR reaches 0.7315 — 0.003 points
   *below* the label-only occupancy ceiling of 0.73494, i.e. below never looking
   at colour. The relative probe clears its own occupancy ceiling by +24.2
   points. Capacity does not rescue it from `d_r` 16 to 64, and the linear probe
   collapses identically (0.7371 absolute against 0.9954 relative), so this is
   the model's basis, not the probe's rank.
2. **The flip has a specific direction in activation space.** Against the
   matched non-flipping move at the same prefix, position and next mover, the
   displacement aligns with the predicted flip direction at median cos
   **+0.183 (TPR) / +0.238 (linear)** against an unflipped-disc null at ≈0
   (median −0.003, sd 0.061 for a single drawn disc), positive in 1596/1596
   strict events under both, and the two directions agree at cos 0.835. This is
   not decode arithmetic: it is a claim about where the representation moves.
   `reports/VERDICT.md` §7f and A5's re-derivation predate the canonical probe
   and quote the ARENA-fit values.
3. **That direction is causally used, and the TPR is the cleaner lever, not the
   stronger one.** Editing layer-5 residual along it moves next-move probability
   mass onto the counterfactual board's legal set by **+3.120 nats [3.023,
   3.207]** at matched step length (`‖δ‖/‖h‖` = 0.163), against **+0.244 [0.216,
   0.275]** for a magnitude-matched random direction. The linear fit's direction
   at that same length reaches **+3.376 [3.265, 3.486]**, i.e. *more* effect,
   with 3.4× the collateral (0.112 vs 0.033 squares, out-of-family decoder).
   Under the retired 8,000-row fits the TPR won on both counts (+2.380 vs
   +2.035); on the canonical probe it wins only on collateral, so what survives
   is the cleanliness claim and not the strength claim. It is a move and not
   damage: log-mass on the whole legal set stays inside ±0.06 nats over both
   sweeps, and the real board's legal-only squares shed mass as the
   counterfactual-only squares gain it — one switch, read on both sides.

Read the contrasts, never the bare proportions. `cos_flip_median` must be
computed on `act(P+m) − act(P+m')`; the before→after reading is degenerate,
because in the relative basis the predicted flip term at *s* is the zero vector.

## The figures

Three, one per surviving claim. An earlier draft had eight; the cut panels'
numbers are below and their CSVs are still in `results/`.

- **Fig 1 — the basis** (`make_figures.py`). Panel (a) is one event's board just
  before the flanking move, every square shaded by the role whose arrow it feeds
  and three squares outlined — *s*, the disc that gets flanked, *m*, the empty
  square the mover takes, and *m′*, the matched non-flipping control. Panel (b)
  is before→after arrows per role, in **gauge-invariant logit
  differences**: `x = ℓ_mine − ℓ_theirs` (who owns it),
  `y = ½(ℓ_mine + ℓ_theirs) − ℓ_empty` (is it occupied). Not filler coordinates —
  `logit = rᵀBf` is invariant under `f → Af, B → BA⁻¹`, so that plane has four
  arbitrary degrees of freedom and its angles move 131° → 105° across seeds at
  constant accuracy.

  **(b) is definitional, and the caption has to say so.** Everything in it — 63
  discs crossing, *s* not crossing — is what any correct board tracker does in a
  mover-relative frame, which is the retracted claim above and not a fact about
  the model. The one contrastive element is the dashed arrow to where the matched
  non-flipping move *m′* leaves *s*, **13.78 logits [13.68, 13.89] in x** from
  where *m* leaves it.

  An absolute-colour companion panel was dropped from the figure. Finding 1 still
  rests on that probe — same architecture, layer, seed and training protocol,
  trained on absolute-colour labels, collapsing the flip contrast to 0.06 logits
  [0.01, 0.11] and sitting 0.003 points *under* its 0.73494 ceiling while the
  relative fit clears 0.75181 by +24.2 points. The numbers are in `m1_basis.csv`
  (`acc_tpr_{relative,absolute}`, `ceiling_{relative,absolute}`); nothing in the
  figure carries them, so a caption or the text has to.

  Arrows are means over the **1,596 strict-control events**, not all 2,000: the
  control arrow is the panel's one contrast and a control move touching *s*'s
  rays is not a clean one, so the whole panel is read off that population. Bars
  are 95% cluster-bootstrap intervals on the **paired displacement**
  (after − before within event), which is what an arrow asserts.
- **Fig 2 — the direction and its causal use** (`make_figures.py`). (a) the
  flip-attributable displacement `act(P+m) − act(P+m′)` aligns with the
  predicted flip direction at median +0.183 (TPR) / +0.238 (linear) against an
  unflipped-disc null at −0.003.

  **(b) is currently under-scaled and should not be read as a head-to-head.**
  The canonical TPR's binding map is roughly ten times larger than the retired
  fit's, so `pinv(M)` returns a roughly ten times shorter step: the committed
  scale sweep 1–16 spans `‖δ‖/‖h‖` 0.003–0.049 and tops out at a 12% target-flip
  rate, where it used to span 0.031–0.488 and saturate. The whole TPR curve now
  sits in the panel's bottom-left corner and only the linear edit's two betas
  carry visible effect. The matched-step-length comparison — the one that is
  actually a head-to-head — is finding 3 above and is in
  `intervention_summary.csv` under `method=tpr_matched`; it is not on the panel.
  Fixing the panel needs a different `--scales`, which is a change to the
  experiment rather than to the drawing, so it is left for a decision.

  Three things that panel is careful about, none of them cosmetic:
  - **Collateral is read by the decoder the edit was *not* built from** — the
    TPR edit by the linear probe, the linear edit by the TPR probe. Grading an
    edit with its own instrument flatters it, and the old panel did exactly that
    to the TPR arm. The two readings of one edit no longer agree as closely as
    they did: the in-family reading is 1.0–2.1× the out-of-family one along the
    TPR sweep but 3.6–5.1× along the linear edit, so the axis is a lower bound on
    the linear edit's damage, not a neutral measurement of it.
  - **The two knobs are not the same quantity.** The TPR edit adds
    `scale · d_TPR` (8 scales, 1–16); the probe edit projects the activation to
    the target class's mean at `β` (2 betas). So nothing is placed by knob value:
    x is measured collateral, and every point is labelled with `‖δ‖/‖h‖`, the
    fraction of the residual norm the edit moves. The probe's two betas land at
    0.16 and 0.33, *outside* the TPR sweep's 0.003–0.049, which is why the two
    curves no longer overlap on the panel.
  - **The dotted companion is not a control — it is the other side of the same
    switch.** It reads the edit on the squares legal only on the *real* board,
    which the un-flipped board makes illegal, so a successful counterfactual edit
    drives it down exactly as it drives the solid curve up, and it does: at
    `‖δ‖/‖h‖` 0.33 the real-only squares have lost 0.126 of probability mass
    while the counterfactual-only squares gained 0.067 (on the 507 events where
    both sets are non-empty, the only ones that pair). The control is the gray
    band — log-mass on the *whole* legal set, which is what a damaging edit moves
    and this one never does (|Δ| ≤ 0.06 nats anywhere). At the small end the
    dotted curve is not evidence of anything: the two sets start 9.4 nats apart
    (median base mass 9·10⁻⁶ against 0.112), so a *complete* transfer of the
    counterfactual gain at the smallest edit would move it well inside its own
    CI. n = 986 events for the counterfactual set, 891 for the real-only set (an
    event counts only where that set is non-empty), 1,370 for collateral and for
    the whole-legal-set control.
- **Fig 3 — the SAE arm** (`sae_pca_figure.py`). 3×3 per-space PCA of
  before/after/control, per-*t* centered. **Each panel refits its own basis**,
  so read the within-panel arrangement and never positions across panels
  (FIGURE_GUIDELINES §4.6 facet rule; SAE scales differ and SpaDE lives on a
  simplex). Numbers in `results/sae_arm.csv` / `reports/SAE_ARM.md`;
  quote the flip cosine and the dissociation *gap*, not the bare proportion,
  since read-out accuracy is lossy through an SAE. The panel grid still draws
  the nine spaces it was built for; `sae_arm.csv` now has eight, because
  `spade.spaces.EXCLUDED_SAE` drops the degenerate seed-42
  `SpaDETrainer-othello` checkpoint and the committed CSV predated that.

**Findings that no longer have a figure**, kept because they are load-bearing:

- *Depth.* The flanked square's **state** is written in blocks 0–2 (read-out
  centre of mass block 0.62); what mlp5–mlp6 write is its **legality
  consequence** (pair-patch centre of mass 4.68). Two measurements 4.1 blocks
  apart, so this study and `analysis/rules_causal/pair_patch.png` never
  disagreed. The read-out's own peak wanders between blocks 4 and 5 across
  probes — that is TPR capacity, not a computation stage.
  (`results/depth_reconciliation.csv` — **stale**: nothing in the tree writes
  that file, so its numbers predate the canonical probe. The read-out peak is
  still block 4 (`contract_a3.csv`), and `pair_patch.csv` re-ran byte-identical,
  so the 4.1-block gap is unaffected; the centre-of-mass figures are not
  reproducible until the file gets a writer.)
- *The swap is board-specific.* The best shared rank-1 direction is the to-move
  axis at |cos| 0.995 but only 5.5% of the displacement; matching what a single
  matched control explains (68.9%) takes **rank 73**.
  (`results/variance_shares.csv`)

## Pipeline

`spade.bracket_flip.events` → `extract_acts` → `probe_space` and `geometry_tables`,
which read `spade.probes.load` and write the CSVs in `results/` → the figure scripts
here, which only draw. Three GPU jobs under `scripts/slurm/studies/bracket_flip/`:
`submit_bracket_flip.sbatch` (mine and extract), `submit_bracket_readout.sbatch`
(`probe_space` + `geometry_tables`), and `submit_bracket_intervene.sbatch` (the M6
sweep); `submit_bracket_robustness.sbatch` and `submit_bracket_sae.sbatch` run the
two side arms. Every figure is drawn off the GPU by `make_figures.py`.
Large artifacts on `$SPADE_BRACKET_DIR`; per-event CSVs in `results/`; the A1–A6
write-ups in `reports/`.

Events are moves that flip **exactly one** disc, so before and after differ in
exactly two squares. 2000 events, one per game so they are independent, 79.8%
carrying a strict control move whose placement and flips avoid all eight of the
flipped square's rays.

## Gotchas

- `events.csv` mixes conventions: `prefix`/`move`/`control_move` are vocab
  tokens (1–60), `square_s`/`flip_set`/`control_squares` are engine squares
  (0–63). `to_string(token)` converts.
- Parity comes from engine replay, never `t % 2` — forfeits are unmarked and
  desynchronize it on 0.35% of positions. The same check shows the committed
  `analysis/linear_probe/activations/board_states.pt` carries 53 mislabeled
  rows out of 10,000.
- Events whose opponent forfeits at t+1 are dropped: the mover would play
  twice and the flipped square would read *mine*, inverting the prediction.
