# Violation divergence: two futures from one prefix

A legal game is branched at position *t0* into the move that was actually played,
a different legal move, and one move per illegal category of R1–R3
(`src/spade/othello/taxonomy.py`). Every branch shares the prefix **and** the
suffix, so two sequences differ in exactly one token and everything measured
after *t0* is attributable to that token. Design: `docs/violation_divergence_design.md`.

## Pipeline

```
python -m spade.violation_divergence.branches        # 2000 games -> 15531 branches
python -m spade.violation_divergence.extract_acts    # resid 0-7 + W_E + logits -> $DIR/acts/
sbatch scripts/slurm/studies/violation_divergence/submit_violation_divergence.sh        # the two steps above, on a GPU
sbatch scripts/slurm/studies/violation_divergence/submit_violation_divergence_spaces.sh # 17 spaces, 6-way H100 array
python analysis/violation_divergence/make_figures.py                # F1, F3
python analysis/violation_divergence/behavioral/make_figures.py     # F6
python analysis/violation_divergence/probe_readout/make_figures.py  # F7
python analysis/violation_divergence/sae_spaces/make_figures.py     # F2, F4, F5
python analysis/violation_divergence/fork_and_board/make_figures.py # the design, n=1
python analysis/violation_divergence/rule_curvature/make_figures.py # curvature at the fork, --tree branches|rollout
python analysis/violation_divergence/suffix_legality/measure.py     # the shared-suffix defect, tokens only
```

`$DIR` = `$SPADE_DIVERGENCE_DIR` (default `/n/netscratch/ba_lab/Lab/$USER/spade_divergence`).
t0 ∈ [10, 34], window r ∈ [−8, 24], so **every game contributes to every offset** —
n is constant across the whole x-axis, verified, with zero off-board NaN.

## The result in one paragraph

OthelloGPT does not treat an illegal move as illegal. It writes the disc onto its
board state, keeps predicting from the corrupted board, and shows no elevated
uncertainty — yet a small number of SAE features fire on the violation almost
deterministically. **The legality signal is computed and linearly available, and
the model does not act on it.** The raw residual stream is the wrong place to look
for it: an illegal move displaces layer 5 *less* than an ordinary legal
alternative does.

## What holds

**1. The model represents the illegal move as played.** At r = 0 the committed
layer-5 linear probe decodes a disc on the illegally played square in
**98.5–99.9%** of events, in every category. `occupied_opp` recolors the square to
the mover in **78.6%** of events — so it is a *recolor*, not a no-op, answering the
question the design doc left open. `occupied_own` leaves the square unchanged and is
the study's one genuine zero-board-change violation.

**1b. R3's prediction is falsified, and it was the study's one directional
prediction.** `opponent_only` is the single category whose violation *is* turn
confusion, so decoded parity was the place a rule-specific signature should have
shown up. It does not: parity agreement is **0.9930**, mid-pack, and no category
drops below 0.986 anywhere in the window. The model reads an illegal
opponent-flipping move as *the mover placing a disc*, carrying the mover's colour
96.6% of the time. Whatever the model is doing with turn order, an illegal move
for the wrong player does not perturb it.

The corruption is also not a blip at the injection site. Median unbroken plies of
disagreement with the reference board at the illegally played square:
`empty_own_only` **18**, `empty_detached` and `empty_no_bracket` **16**,
`opponent_only` **13**, `occupied_opp` **2**, `occupied_own` **0** — and ~30% of
empty-square events carry it unbroken to the end of the 24-ply window. The
`occupied_own` zero is the no-op reading confirmed by a second, independent
measurement.

Untouched discs are *not* left alone either — accuracy on them falls to
0.836–0.986, and further for categories that place into a populated
neighbourhood. That is consistent with the model computing flip consequences for
a disc that was never legally placed, which would be a stronger claim than
anything else here and is **not** settled by this measurement.

**2. It then acts on the corrupted board.** Illegal branches put 0.805 of their
next-move mass on the *reference* board's legal set against a 0.9994 ceiling — but
`control_legal` scores **lowest of all** on the reference board (0.750) while
putting 0.9990 on the board it actually played. The deficit measures board
disagreement, not degraded legality. Next-move entropy at r = 0 is 2.287–2.422 for
illegal branches against 2.364 for the reference: **no surprise signature**.

**3. SAE features detect the violation; the residual norm does not.** Standard-SAE
feature 5098, activation at the branch position, over 400 games:

| branch | fires | mean activation |
|---|---|---|
| reference (legal) | 25.0% | 0.0088 |
| `control_legal` | 26.0% | 0.0107 |
| `occupied_own` / `occupied_opp` | 48–62% | 0.20–0.43 |
| the four empty-square violations | **99.0–100%** | **1.07–1.77** |

The two *legal* branches are indistinguishable, so this is legality and not "a
different move was played". PAnneal reproduces it on its own feature 859 (98.1% vs
0.0%). Half the divergence energy sits in ~39–41 features for empty-square
violations against ~120 for the occupied ones.

**4. Depth.** The plateau — a mean, and not selection-biased — triples with depth,
0.2023 → 0.5684 from layer 0 to 7, and the never-halves fraction goes 0% → 52.9%.
The token embedding's δ̂ is a delta function: 1.4136 ≈ √2 at r = 0 and **exactly
0.000000** at every r ≥ 1 for all 13531 rows, so everything downstream is computed,
not inherited.

**5. There is no illegal-specific direction — now measured against a null, not asserted.**
The second legal control (`control_legal_2`, added after the first pass) is scored
against `control_legal` exactly as the illegal branches are, so its row *is* the
legal-vs-legal null. At layer 5, r = 0:

| branch | cos(Δ, Δ_control1) | r = 24 | plateau ÷ control |
|---|---|---|---|
| **`control_legal_2` (null)** | **0.4985** | 0.7857 | **1.0022** |
| `occupied_own` | 0.5358 | 0.8546 | 0.9138 |
| `occupied_opp` | 0.5184 | 0.8393 | 0.9231 |
| `empty_detached` | 0.4509 | 0.7829 | 1.0038 |
| `empty_own_only` | 0.4434 | 0.7802 | 1.0205 |
| `empty_no_bracket` | 0.4833 | 0.7710 | 1.0061 |
| `opponent_only` | 0.4631 | 0.7945 | 0.9971 |

The 0.44–0.54 band that the first pass could not interpret **is** the legal-vs-legal
baseline. Paired within game, the modulations are real but small: R1 occupied
violations are *more* aligned with a legal displacement than a second legal move is
(`occupied_own` +0.0373 [+0.0326, +0.0419]), and empty-square violations are *less*
(`empty_own_only` −0.0559 [−0.0612, −0.0506]) — 4–11% of a 0.50 baseline. An
illegal move does not leave along a direction legal play does not use.

The null also calibrates the magnitude claim: a second legal control gives a plateau
ratio of **1.0022**, so the scale is honest, and no illegal category exceeds it while
the two occupied ones sit clearly below at 0.914 and 0.923.

## What does not hold, and why

**`ratio_illegal_control_L5` = 0.979 [0.972, 0.985].** An illegal move displaces
layer 5 *less* than a different legal move. The design's premise is false, and so is
its stated reason for believing the contrast fair.

**`control_legal` is not a null — it is the largest perturbation in the study.**
A legal alternative genuinely updates the board and passes the turn, so the shared
suffix becomes wrong *for the control*: `g[t0+1]` stays legal on the control's board
in only **76.8%** of games, and a median of **2** suffix moves survive. An illegal
move carries far less board information, is absorbed as a near-no-op, and its suffix
stays consistent with the model's belief state. The null therefore carries the
*larger* handicap, on δ̂ plateau, KL, suffix NLL and legal mass alike.

Measured per branch in `suffix_legality/`, which re-derives the 76.8% on the current
two-control set (0.7515 [0.7325, 0.7710], median 2) and adds the half this paragraph
leaves out: an illegal branch has **no** board after *t0*, so its suffix legality is
not low, it is undefined. Only the reference is ever replayed in its own game.

So the honest effect size is: **the representational effect of an impossible move is
92–103% of the effect of an ordinary legal one.** No sentence of the form "illegal
branches diverge more than the legal control" is supported. This is not repaired by
matching on flip count — the control's plateau is flat in discs flipped (r = +0.002).

**`peak` and `peak_r` are max-of-noise statistics.** `persistence()` averages
per-row maxima over 33 offsets, which estimates E[max], not max E. A within-row
offset permutation reproduces `peak` to 0.4% at every layer and 79–83% of `peak_r`'s
0.0 → 5.8 depth drift. **The population curve peaks at r = 0 in every layer.** Quote
the curve's argmax, or `peak` with its null beside it — never `peak_r` alone.

**`peak_layer` as defined is vacuous.** δ̂ divides by ‖z_ref‖ per space, so it is a
within-space ratio and cross-space differences are not "excess over the input floor".
Report the layer maximising plateau δ̂ (7), or drop the key.

## Estimator pinning

Three keys have defensible estimators that disagree by more than the
cross-validation tolerance. Comparisons must fix one:

| key | pinned definition | value (L5, illegal pooled) |
|---|---|---|
| `delta_peak_r_L5` | argmax of the **mean curve** | **0** (per-row mean 4.19, median 3) |
| `delta_peak_L5` | value of the **mean curve** at its argmax | **0.7682** (per-row mean 0.8456) |
| `ratio_illegal_control_L5` | **paired** within game, plateau | **0.9789** [0.9719, 0.9854] |

## Verification

The red team (`RED_TEAM.md`) re-derived every key through an independent code path
without importing `src/spade/violation_divergence/divergence.py`:

- **Prefix invariance is bitwise** for all 13531 branch rows at every position before
  *t0* in all ten tensors — zero differing elements. δ̂(r < 0) is exactly 0.0 in all
  17 spaces. This is the check whose failure would invalidate the study.
- Branch set re-derives against the engine with **0 mismatches over 15531 rows**;
  t0 histogram exactly 80 per value.
- Independent kernels reproduce the shipped npz to max |Δδ̂| **4.8e-7**, with 0
  `peak_r` disagreements and 1 `half_life` disagreement (a float32/float64 tie at
  layer 4, row 12783).
- Token/square identity explain almost none of δ̂ at depth (|r| ≤ 0.16 against square
  distance, unigram frequency, and input-side δ̂ — the last one *negative*).
- δ̂ has not saturated: two unrelated games at the same position give δ̂ ≈ 1.29, so a
  plateau of 0.47–0.50 has dynamic range left.
- `empty_own_only`'s 18.2% missingness is substantially a *t0* effect (mean t0 24.76
  vs 21.39), not the board-shape confound the design doc claimed. It does not bite:
  restricting to the 1554 complete games moves the ratio 0.9793 → 0.9805. Use the
  paired ratio and report per-category n.

## Figures

`figures/divergence_layer5` (F1) · `figures/depth_sweep` (F3) ·
`behavioral/figures/f6_behavioral_divergence` (F6) ·
`probe_readout/figures/f7_probe_readout` (F7) ·
`sae_spaces/figures/sae_feature_recurrence`.

**The design itself**, in `fork_and_board/` (see its README):
`game_1235_fork_and_board_displacement_pca` puts the fork in all nine
representation spaces over the board the game forks on, so a coloured trajectory
can be traced back to the move that produced it, and
`game_1235_fork_and_board_r21_pca` is the same figure with the plane taken inside
the 21-D ReStraV curvature descriptor instead.
`game_trajectory/game_1235_fork_pca` adds the
magnitude a projection cannot show, δ̂ against offset. These are n=1
illustrations of the construction, not claim-supporting curves — every number above
comes from the aggregates.

## Known gaps

- ~~M2's illegal-vs-control cosine has no matched null.~~ **Closed** — `control_legal_2`
  supplies it; see finding 5. The answer is that no illegal-specific direction exists.
- The probe read-out (finding 1), the behavioural read-out (finding 2) and the SAE
  feature sweep (finding 3) were all computed on the **one-control** branch set. The
  raw-space and direction numbers above are from the two-control re-run; the
  headline δ̂ figures are unchanged by it (L5 plateau 0.4760 → 0.4769, ratio
  0.979 → 0.978), but the three read-outs have not been re-derived on the new rows.
- The probe was trained on legal positions, so its read-out on off-distribution
  activations is evidence, not ground truth.
- Feature 5098 is verified on Standard and 859 on PAnneal; the equivalent sweep
  across the other six dictionaries is not done.
- `kl_peak_L5` is misnamed — KL is read off the logits and is not a layer-5 quantity.
- The plateau window r ∈ [20, 24] is parity-imbalanced (three even offsets, two odd)
  against a δ̂ ripple of ±0.005. Changes no ranking; [21, 24] would be free.
- Seed 42 only.
