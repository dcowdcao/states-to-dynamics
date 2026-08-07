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
linear reference it is measured against — are fit on the study's own probe set:
400 games of `adamkarvonen/othello_45MB_games`, 20,000 positions, labels from
engine replay, the corpus the events themselves are mined from
(`probe_space.tpr` / `probe_space.linear`, cached under `$SPADE_BRACKET_DIR/probe_space/`).
The earlier ARENA fits — 200 games, 10,000 positions, labels from `t % 2`, which
DATA_REPORT shows mislabelled on 53 rows — are the off-corpus control, reachable at
`act_geometry.retain_directions(..., corpus="arena")` and
`tpr_intervene --corpus arena`; the latter writes `*_arena.csv` so the primary
results survive the run. Moving both families together is what keeps Fig 2's
headline a comparison of probe class rather than of training corpus.

## What this study does and does not show

**Retired claim.** An earlier draft led with "the flanked square is the one
square whose class does not change." That sentence is a true description of the
board in a mover-relative basis, and it is **not evidence about the model** — a
perfect board oracle reproduces it by definition. Two agents killed it
independently. `p_s_unchanged_rel` = 0.988 is the probe's accuracy at *s*
squared (0.995² = 0.9900 against 0.9900 observed) and tracks `acc_rel_L5` at
Pearson 0.9988 across probe ranks (0.692 at `d_r=16`, 0.9985 at 64). The
event-level version is also false: *s* is the *uniquely* unchanged disc in only
0.625 of events, because decode noise leaves 0.71 other discs unchanged.

What the flip framing is good for is stating the *basis*, not proving anything
in it. Fig 1 is drawn that way: the arrow field in the native basis is the
frame, and what carries a claim is the second probe drawn beside it.

**Three findings that survive.**

1. **The model represents board state mover-relatively, and barely represents
   absolute colour at all.** An absolute-label TPR reaches 0.7349 — 0.004 points
   *below* the label-only occupancy ceiling of 0.73494, i.e. below never looking
   at colour. The relative probe clears its own occupancy ceiling by +23.1
   points. Capacity does not rescue it from `d_r` 16 to 64, and the committed
   linear probe collapses identically, so this is the model's basis, not the
   probe's rank.
2. **The flip has a specific direction in activation space.** Against the
   matched non-flipping move at the same prefix, position and next mover, the
   displacement aligns with the predicted flip direction at median cos
   **+0.188 (TPR) / +0.205 (linear)** against an unflipped-disc null at ≈0
   (median −0.003, sd 0.058 for a single drawn disc), positive in 1596/1596
   strict events under both, and the two directions agree at cos 0.833. This is
   not decode arithmetic: it is a claim about where the representation moves.
   A5's independent re-derivation reported 0.173/0.184 off the ARENA fits
   (`reports/VERDICT.md` §7f is the corpus objection those numbers predate).
3. **That direction is causally used, and the TPR is the better lever even
   though the linear direction is the better-aligned one.** Editing layer-5
   residual along it moves next-move probability mass onto the counterfactual
   board's legal set by **+2.380 nats [2.299, 2.454]** at matched step length
   (`‖δ‖/‖h‖` = 0.144), against **+0.190** for a magnitude-matched random
   direction and **+2.035 [1.970, 2.104]** for the linear fit's direction at
   that same length, with less collateral on both counts (0.123 vs 0.141
   squares, out-of-family decoder). The ordering in (2) inverts here: being
   aligned with where the representation moves is not the same property as
   being the handle that moves it. It is a move and not damage: log-mass on the
   whole legal set stays inside ±0.07 nats over both sweeps, and from
   `‖δ‖/‖h‖ ≈ 0.07` up the real board's legal-only squares shed mass while the
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
  non-flipping move *m′* leaves *s*, **14.59 logits [14.46, 14.73] in x** from
  where *m* leaves it.

  An absolute-colour companion panel was dropped from the figure. Finding 1 still
  rests on that probe — same architecture, layer and seed, trained on
  absolute-colour labels, collapsing the flip contrast to 0.04 logits
  [0.01, 0.07] and sitting 0.004 points *under* its 0.73494 ceiling while the
  relative fit clears 0.75181 by +23.1 points. The numbers are in `m1_basis.csv`
  (`acc_tpr_{relative,absolute}`, `ceiling_{relative,absolute}`); nothing in the
  figure carries them, so a caption or the text has to.

  Arrows are means over the **1,596 strict-control events**, not all 2,000: the
  control arrow is the panel's one contrast and a control move touching *s*'s
  rays is not a clean one, so the whole panel is read off that population. Bars
  are 95% cluster-bootstrap intervals on the **paired displacement**
  (after − before within event), which is what an arrow asserts.
- **Fig 2 — the direction and its causal use** (`make_figures.py`). (a) the
  flip-attributable displacement `act(P+m) − act(P+m′)` aligns with the
  predicted flip direction at median +0.188 (TPR) / +0.205 (linear) against an
  unflipped-disc null at −0.003. (b) effect against collateral, one point per
  edit magnitude: up and to the left is a clean intervention, and the TPR edit
  sits above and left of both the linear edit and the magnitude-matched random
  null everywhere. Three things that panel is careful about, none of them
  cosmetic:
  - **Collateral is read by the decoder the edit was *not* built from** — the
    TPR edit by the linear probe, the linear edit by the TPR probe. Grading an
    edit with its own instrument flatters it, and the old panel did exactly that
    to the TPR arm. The two readings of one edit stay within ~35% of each other,
    so putting them on one axis does not distort it.
  - **The two knobs are not the same quantity.** The TPR edit adds
    `scale · d_TPR` (8 scales, 1–16); the probe edit projects the activation to
    the target class's mean at `β` (2 betas). So nothing is placed by knob value:
    x is measured collateral, and every point is labelled with `‖δ‖/‖h‖`, the
    fraction of the residual norm the edit moves. The probe's two betas land at
    0.11 and 0.23, inside the TPR sweep's 0.03–0.53, which is what makes the two
    curves readable against each other.
  - **The dotted companion is not a control — it is the other side of the same
    switch.** It reads the edit on the squares legal only on the *real* board,
    which the un-flipped board makes illegal, so a successful counterfactual edit
    drives it down exactly as it drives the solid curve up, and it does: past
    `‖δ‖/‖h‖ ≈ 0.07` it falls, and at 0.53 the real-only squares have lost 0.124
    of probability mass while the counterfactual-only squares gained 0.096 (on
    the 507 events where both sets are non-empty, the only ones that pair). The
    control is the gray band — log-mass on the *whole* legal set, which is what a
    damaging edit moves and this one never does (|Δ| ≤ 0.08 nats anywhere).
    Below `‖δ‖/‖h‖ ≈ 0.07` the dotted curve is not evidence of anything: the two
    sets start 9.7 nats apart (median base mass 8·10⁻⁶ against 0.115), so a
    *complete* transfer of the counterfactual gain at the smallest edit would
    move it by −0.0001 nats, an order of magnitude inside its own CI. The panel
    says so in-figure. n = 986 events for the counterfactual set, 891 for the
    real-only set (an event counts only where that set is non-empty), 1,370 for
    collateral and for the whole-legal-set control.
- **Fig 3 — the SAE arm** (`sae_pca_figure.py`). 3×3 per-space PCA of
  before/after/control, per-*t* centered. **Each panel refits its own basis**,
  so read the within-panel arrangement and never positions across panels
  (FIGURE_GUIDELINES §4.6 facet rule; SAE scales differ and SpaDE lives on a
  simplex). Numbers in `results/sae_arm.csv` / `reports/SAE_ARM.md`;
  quote the flip cosine and the dissociation *gap*, not the bare proportion,
  since read-out accuracy is lossy through an SAE.

**Findings that no longer have a figure**, kept because they are load-bearing:

- *Depth.* The flanked square's **state** is written in blocks 0–2 (read-out
  centre of mass block 0.62); what mlp5–mlp6 write is its **legality
  consequence** (pair-patch centre of mass 4.68). Two measurements 4.1 blocks
  apart, so this study and `analysis/rules_causal/pair_patch.png` never
  disagreed. The read-out's own peak wanders between blocks 4 and 5 across
  probes — that is TPR capacity, not a computation stage.
  (`results/depth_reconciliation.csv`)
- *The swap is board-specific.* The best shared rank-1 direction is the to-move
  axis at |cos| 0.995 but only 5.5% of the displacement; matching what a single
  matched control explains (68.9%) takes **rank 73**.
  (`results/variance_shares.csv`)

## Pipeline

`spade.bracket_flip.events` → `extract_acts` → `probe_space`, which fits both probe
families on the probe set and caches them → `probe_space` and `geometry_tables` write
the CSVs in `results/` → the figure scripts here, which only draw. `tpr_probe` fits
the ARENA control set and is otherwise the module the vendored TPR is wrapped in.
Two GPU jobs under `scripts/slurm/studies/bracket_flip/`: `submit_bracket_flip.sbatch`
(mine and extract) and `submit_bracket_intervene.sbatch` (the M6 sweep; CSVs only,
the figure is drawn off the GPU by `make_figures.py`).
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
