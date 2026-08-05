# Bracket flips

What happens to a square's layer-5 representation at the moment it is flanked
and flipped. Design and rule mapping: `docs/bracket_flip_design.md`; the rule
set it sits in is `docs/othello_rules.md` (R5's state update, reached through
the R2 bracketing that causes it).

Decoder is the tensor-product-representation probe of Lee, Viégas & Wattenberg
(arXiv:2605.09967), vendored at `external/tpr_othello`, trained here on our
activations: `logit(s, c) = r_sᵀ M(h) f_c` at `d_r=52, d_f=2`, 56,582
parameters — 57.6% of the linear probe's, matching the paper's 57.5%.

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
in it.

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
   **0.173–0.184** against an unflipped-disc null at ≈0, positive in 1596/1596
   strict events, and holds whether the direction comes from the TPR or the
   linear probe (the two agree at cos 0.822). This is not decode arithmetic: it
   is a claim about where the representation moves.
3. **That direction is causally used.** Editing layer-5 residual along it moves
   next-move probability mass onto the counterfactual board's legal set by
   +1.496 log-units at matched step length, against +0.121 for a
   magnitude-matched random direction, with less collateral than the random
   control at the operating scale.

Read the contrasts, never the bare proportions. `cos_flip_median` must be
computed on `act(P+m) − act(P+m')`; the before→after reading is degenerate,
because in the relative basis the predicted flip term at *s* is the zero vector.

## The figures

Three, one per surviving claim. An earlier draft had eight; the cut panels'
numbers are below and their CSVs are still in `results/`.

- **Fig 1 — the phenomenon and the basis** (`make_figures.py`). Panel (a) is one
  event's board just before the flanking move, every square shaded by the role
  whose arrow it feeds and the two named squares outlined — *s*, the disc that
  gets flanked, and *m*, the empty square the mover takes. Panels (b) and (c) are
  before→after arrows per role, in **gauge-invariant logit differences**:
  `x = ℓ_mine − ℓ_theirs` (who owns it), `y = ½(ℓ_mine + ℓ_theirs) − ℓ_empty`
  (is it occupied). Not filler coordinates — `logit = rᵀBf` is invariant under
  `f → Af, B → BA⁻¹`, so that plane has four arbitrary degrees of freedom and
  its angles move 131° → 105° across seeds at constant accuracy. The flipped
  square's dashed arrow is the same square under the matched non-flipping move;
  the gap between the two heads is the claim. Panel (b) relabels to absolute
  colour and the picture inverts.
- **Fig 2 — the direction and its causal use** (`make_figures.py`). (a) the
  flip-attributable displacement `act(P+m) − act(P+m′)` aligns with the
  predicted flip direction at median +0.175 (TPR) / +0.163 (linear) against an
  unflipped-disc null at −0.002. (b) effect against collateral over the edit
  scale sweep: up and to the left is a clean intervention, and the TPR edit sits
  above and left of both the linear edit and the random null at every scale.
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

`spade.bracket_flip.events` → `extract_acts` → `tpr_probe` → `probe_space` and
`geometry_tables`, which write the CSVs in `results/` → the figure scripts here,
which only draw. One SLURM job: `scripts/slurm/studies/bracket_flip/submit_bracket_flip.sh`.
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
