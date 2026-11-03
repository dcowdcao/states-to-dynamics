# Red team: violation_divergence

A5. Every number below comes from my own code path, from
`$SPADE_DIVERGENCE_DIR/acts/` and `branches/branches.pt`.
`src/spade/violation_divergence/divergence.py` is never imported. The shipped
`$DIR/divergence/*.npz` are compared against my arrays, not used to produce them.

## Verdict

**The pipeline is correct. The headline claim fails.**

The data is clean: the branch set re-derives exactly against the engine, prefix
invariance holds bitwise for all 13531 branch rows at every position before *t0*
in all ten activation tensors, and my independent kernels reproduce every shipped
npz to float32 rounding (max |Δδ̂| 4.8e-7, zero `peak_r` disagreements over
13531 × 9 spaces). A1's batch-boundary fix held.

What does not survive is the study's organising expectation, that an illegal
move drives the model further from the reference than a legal alternative does.
At layer 5 the pooled `ratio_illegal_control_L5` is **0.979** — illegal branches
diverge **less** than `control_legal`, and the paired 95% CI [0.972, 0.985]
excludes 1. The same inversion appears in every behavioural read-out:
`control_legal` has the highest KL from the reference (2.52 vs 1.86–2.43),
the highest suffix surprisal (3.03 nats vs 2.69–2.92) and puts the **least**
mass on the reference board's legal set (0.750 vs 0.760–0.858). The design's
one stated cost — "the null carries the same handicap, so the
illegal-vs-control contrast is fair" — is false in the data: the null carries a
strictly **larger** handicap on all three measures.

Three of the six categories sit below the control and three at or above it, in a
1–6% band. There is a real, significant category structure inside that band, but
"illegal moves diverge more than legal ones" is not what the numbers say, and no
figure or caption should say it.

Two reported quantities are additionally artefacts of their estimator rather
than of the model: `peak` is a max over 33 noisy offsets and is inflated ~14%
above the population curve, and the depth trend in `peak_r` (0.0 → 5.8) is
~80% reproduced by argmax-of-noise on a curve whose population maximum is at
*r* = 0 in **every** layer. See F2 and F3.

## Contract keys, re-derived

`n_games` **2000**, `n_branches` **15531** rows / **13531** non-reference branch
rows. Manifest, my count of `branches.pt`, and every npz `row` array agree
exactly.

Layer 5, `resid_post_5`, mean over branch rows. `peak`/`peak_r` are the
`persistence()` definition (mean of per-row maxima); the honest population
alternatives are in brackets and in F2.

| category | n | `delta_peak_L5` | `delta_peak_r_L5` | `delta_plateau_L5` | plateau ÷ control (paired, 95% CI) |
|---|---|---|---|---|---|
| `control_legal` | 2000 | 0.8456 | 5.26 | 0.4846 | 1 (reference) |
| `occupied_own` | 2000 | 0.7919 | 5.92 | 0.4443 | 0.9169 [0.9100, 0.9236] |
| `occupied_opp` | 2000 | 0.8030 | 5.04 | 0.4488 | 0.9262 [0.9188, 0.9335] |
| `empty_detached` | 1956 | 0.8828 | 2.93 | 0.4904 | 1.0102 [1.0017, 1.0186] |
| `empty_own_only` | 1636 | 0.8966 | 2.82 | 0.4963 | 1.0255 [1.0158, 1.0355] |
| `empty_no_bracket` | 1941 | 0.8421 | 4.47 | 0.4886 | 1.0078 [0.9986, 1.0172] |
| `opponent_only` | 1998 | 0.8694 | 3.69 | 0.4835 | 0.9978 [0.9890, 1.0063] |

| key | value |
|---|---|
| `ratio_illegal_control_L5` | **0.9793** (row-pooled) / 0.9789 paired [0.9719, 0.9854] / 0.9805 complete-case |
| `delta_peak_WE` | **1.4136** (sd 0.050; plateau exactly 0.000000, `half_life` exactly 1 for all 13531 rows) |
| `peak_layer` | **7** by every non-degenerate reading — but the key is broken, see F4 |
| `kl_peak_L5` | **2.049** nats, illegal pooled, at *r* = 1 (control 2.515 at *r* = 1) |
| `jaccard_min[space]` | see the SAE table |
| `topk10_share[space]` | see the SAE table |

Depth sweep, illegal pooled, mean of per-row maxima ‖ population curve:

| space | `peak` | curve max | curve argmax | curve max over *r* ≥ 1 | plateau |
|---|---|---|---|---|---|
| `token_embed` | 1.4132 | 1.4132 | 0 | 0.0000 | 0.0000 |
| `resid_post_0` | 1.1089 | 1.1089 | 0 | 0.4096 | 0.2026 |
| `resid_post_1` | 1.0192 | 1.0191 | 0 | 0.4252 | 0.2530 |
| `resid_post_2` | 0.9502 | 0.9494 | 0 | 0.4598 | 0.2881 |
| `resid_post_3` | 0.9017 | 0.8971 | 0 | 0.4965 | 0.3247 |
| `resid_post_4` | 0.8629 | 0.8408 | 0 | 0.5485 | 0.3843 |
| `resid_post_5` | 0.8460 | 0.7925 | 0 | 0.6586 | 0.4745 |
| `resid_post_6` | 0.8785 | 0.7976 | 0 | 0.7138 | 0.5410 |
| `resid_post_7` | 0.8962 | 0.7867 | 0 | 0.7284 | 0.5680 |

SAE spaces, all encoding `resid_post_5`. `jaccard_min` is the minimum over *r*
of the row-mean support Jaccard; it lands at *r* = 0 in every space.
`topk10_share` is at *r* = 0 and is a share of ‖Δ‖₂² — see F6 for why that has
to be said out loud.

| space | `jaccard_min` | jaccard(24) | `topk10_share` | PR at *r*=0 | ΔL0 at *r*=0 | illegal ÷ control (paired) |
|---|---|---|---|---|---|---|
| `sae_BatchTopKTrainer-othello` | 0.4112 | 0.6573 | 0.5826 | 23.4 | +11.8 | 0.9767 [0.9703, 0.9829] |
| `sae_CosineSpaDETrainer-othello` | 0.3787 | 0.6542 | 0.6880 | 12.5 | +3.0 | 0.9755 [0.9680, 0.9831] |
| `sae_GatedAnnealTrainer-othello` | 0.4900 | 0.7244 | 0.7034 | 15.8 | +1.6 | 0.9768 [0.9697, 0.9834] |
| `sae_GatedSAETrainer-othello` | 0.3959 | 0.6199 | 0.6256 | 19.7 | +4.0 | 0.9774 [0.9707, 0.9841] |
| `sae_PAnnealTrainer-othello` | 0.3896 | 0.6606 | 0.5372 | 26.1 | +31.3 | 0.9804 [0.9735, 0.9871] |
| `sae_SpaDE-lam1e-03-gam1e-04` | 0.4013 | 0.6081 | 0.1894 | 177.7 | −79.8 | 0.9811 [0.9747, 0.9878] |
| `sae_StandardTrainer-othello` | 0.2882 | 0.5558 | 0.4545 | 32.7 | +89.5 | 0.9865 [0.9797, 0.9933] |
| `sae_TopKTrainer-othello` | 0.4461 | 0.6720 | 0.5632 | 24.4 | +0.0 | 0.9774 [0.9713, 0.9840] |
| `resid_post_5` (dense reference) | — | — | 0.1450 | 166.9 | — | 0.9789 [0.9718, 0.9857] |

## Findings, ranked by whether they change a conclusion

### F1. `control_legal` is not a null. It is the largest divergence in the study. *(changes the headline)*

The design's one acknowledged cost is stated as neutral: *"the null carries the
same handicap, so the illegal-vs-control contrast is fair."* It does not carry
the same handicap. It carries a larger one, on every measure I could construct.

| | reference | `control_legal` | illegal range |
|---|---|---|---|
| δ̂ plateau, L5 | 0 | **0.4846** | 0.4443 – 0.4963 |
| KL(p_b ‖ p_ref), peak over *r* | 0 | **2.515** | 1.857 – 2.432 |
| suffix NLL, mean over *t0*+1 … 58 | 2.032 | **3.033** | 2.689 – 2.916 |
| mass on the reference board's legal set at *r* = 0 | 0.9994 | **0.7501** | 0.7605 – 0.8581 |

The mechanism is not mysterious. A legal alternative move really does update the
board and pass the turn, so the shared suffix — the reference game's own
continuation — becomes wrong: `g[t0+1]` is still legal on the control's board in
only **76.8%** of games, and only **4.08** suffix moves (median 2) stay legal
before the first illegal one. An illegal move carries much less board
information, the model largely absorbs it as a near-no-op, and the suffix stays
comparatively consistent with the model's belief state.

So the design's central contrast is confounded in the direction that *opposes*
its hypothesis, and the observed inversion cannot be read as "the model is
untroubled by violations" either — some of it is just "the control broke the
board harder". Concretely:

- Any sentence of the form "illegal branches diverge more/persist longer than
  the legal control" is unsupported. Delete it.
- `ratio_illegal_control_L5` should be reported as **0.979 [0.972, 0.985]**, and
  read as *illegal branches diverge less*, with the asymmetry above stated in
  the same breath. It is not a clean effect size.
- The claim the data does support, and it is a good one: **at layer 5 the
  representational effect of an impossible move is between 92% and 103% of the
  effect of an ordinary legal move.** Whatever the model does with a violation,
  it is not a categorically larger perturbation.

Nothing rescues this by matching on board change. `control_legal`'s plateau is
flat in the number of discs it flips (corr +0.002; 0.482 at 1 flip, 0.499 at 5),
so the gap to `occupied_own` (0.4443) is not a flip-count effect, and
`empty_own_only` (0.4963) beats the control while adding one disc and flipping
none.

### F2. `peak` and `peak_r` are dominated by selection bias; the population curve peaks at *r* = 0 at every layer. *(changes a conclusion)*

`persistence()` returns the per-row maximum over 33 offsets and its argmax.
Averaging per-row maxima of a noisy curve estimates E[max], not max E. The
population mean curve of δ̂ has its maximum at *r* = 0 in all eight resid layers
and in `token_embed` — there is no layer where the mean divergence peaks late.

Null: keep each row's residual values but permute them across *r* within the
row, so the population curve is preserved and all within-row time structure is
destroyed.

| space | real `peak_r` | null `peak_r` | real `peak` | null `peak` | curve at *r* = 0 |
|---|---|---|---|---|---|
| `resid_post_0` | 0.00 | 0.00 | 1.1118 | 1.1132 | 1.1118 |
| `resid_post_3` | 0.54 | 0.50 | 0.9068 | 0.9091 | 0.9024 |
| `resid_post_4` | 2.10 | 2.00 | 0.8627 | 0.8606 | 0.8397 |
| `resid_post_5` | 4.35 | 3.41 | 0.8459 | 0.8489 | 0.7889 |
| `resid_post_6` | 4.96 | 4.04 | 0.8800 | 0.8787 | 0.7939 |
| `resid_post_7` | 5.75 | 4.77 | 0.8971 | 0.8964 | 0.7839 |

`peak` is reproduced by the null to within 0.4% at every layer: it is a
max-of-noise statistic and sits ~14% above the curve at layer 7. `peak_r`'s
depth trend is 79–83% reproduced by the null. The residual (0.94 plies at L5,
0.98 at L7) is real but is a tenth of the reported drift.

METRICS_REPORT.md reads the 0.0 → 5.8 drift as "with depth, the peak moves
later", and the design's §Spaces reads the depth sweep as separating *a
different token went in* from *the model computed a consequence*. The correct
version of that claim rests on the plateau, which is a mean and is not
selection-biased: it triples with depth, 0.2023 → 0.5684, and that survives.
Report the curve's argmax (0 everywhere) and the curve's value at *r* = 0, or
report `peak` with the null beside it. Do not report `peak_r` alone.

`delta_peak_r_L5` also has three defensible estimators that disagree by more
than the contract's "exact" tolerance — mean per-row argmax **5.26**, median
per-row argmax **3**, argmax of the mean curve **0**. `delta_peak_L5` likewise:
mean per-row max 0.8456 vs peak of the mean curve 0.7682, a 9% gap against a
2-point tolerance. **The integrator must pin one estimator per key before
comparing A3, A4 and A5**, or the cross-validation will fire on definitions
rather than on errors. Same hazard on `topk10_share` (energy vs L2 vs L1 share;
see F6) and `jaccard_min` (min of the mean curve vs mean of per-row minima).

### F3. `occupied_opp` is a recolor, not a no-op — read off the layer-5 probe. *(answers a question the design left open)*

The design says explicitly that which reading the model takes "is a result, not
an assumption". Reading the committed layer-5 linear probe on the branch
activations at *r* = 0, mover-relative classes {empty, theirs, mine} where
"theirs" is the player who just moved:

| branch | reference decodes at that square | branch decodes at that square |
|---|---|---|
| `control_legal` | empty 0.998 | **theirs 0.992** |
| `occupied_own` | theirs 0.988 | theirs 0.980 (unchanged) |
| `occupied_opp` | **mine 0.844** | **theirs 0.786** |
| `empty_detached` | empty 0.998 | theirs 0.839 |
| `empty_own_only` | empty 0.999 | theirs 0.917 |
| `empty_no_bracket` | empty 0.999 | theirs 0.945 |
| `opponent_only` | empty 1.000 | theirs 0.965 |

`occupied_opp` flips the square from the opponent's colour to the mover's in
78.6% of events: **recolor**. `occupied_own` leaves it where it was, which
supports the design's treatment of it as the one zero-board-change violation.
And the model writes the illegally placed disc onto the board for **every**
category — 84% (`empty_detached`) to 97% (`opponent_only`). At *r* = 0 the model
does not represent the move as rejected; it represents it as played.

Two caveats on this panel. The probe was trained on legal positions, so its
read-out on off-distribution activations is evidence, not ground truth. And the
rest of the decoded board also moves: whole-board agreement with the reference
is 0.879–0.935, and it is **lowest for `control_legal` among the near-no-op
branches** (0.910 vs `occupied_own`'s 0.935) — F1 again, in probe space.

`occupied_opp` − `occupied_own`, paired within game, is +0.0045 [+0.0022,
+0.0068] on plateau and +0.0111 [+0.0071, +0.0152] on peak: significant, and
1% of level. The recolor is visible to the probe and nearly invisible to δ̂.

### F4. The "input floor" is a ceiling; `peak_layer` as defined is vacuous. *(changes how a key is reported)*

`delta_peak_WE` = **1.4136** (≈ √2, two move embeddings are near-orthogonal at
equal norm), larger than every resid layer's peak (max 0.8971) and than every
curve value. "The layer maximising δ̂ excess over the token-embedding floor" is
therefore an argmax over a set of negative numbers at *r* = 0, and over
δ̂ itself at *r* > 0 because the floor there is exactly 0. Both readings give
**layer 7**; the concept does not survive.

The reason is the normaliser, not the model: δ̂ divides by ‖z_ref‖ in each space
separately, and ‖z_ref‖ grows with depth, so δ̂ is a within-space ratio and
differences across spaces are not "excess". This is the repo's own
percent-of-level rule biting the key that was written to compare across spaces.
Report `peak_layer` as *the layer maximising plateau δ̂* (7) or drop it. The
token embedding's true role is verified and worth stating on its own: δ̂ is a
delta function — peak 1.4136 at *r* = 0 for all 13531 rows, plateau exactly
0.000000, `half_life` exactly 1 in every row — so everything at *r* > 0 in any
resid layer is computed, not inherited. That claim stands.

### F5. `empty_own_only`'s missingness is a t0 effect, not "spread across all t0" — and it does not bite. *(corrects the design doc; changes nothing else)*

The design doc says the missingness is "spread across all t0, so … a board-shape
confound". It is substantially a *t0* confound: games without `empty_own_only`
have mean t0 **24.76** vs **21.39**, and mean `n_legal` **12.53** vs **10.60**.
δ̂ plateau falls with t0 (corr −0.15; 0.510 at t0 ≤ 17 vs 0.462 at t0 ≥ 28), so
the composition difference is pointed the right way to matter.

It does not. On every other category, plateau differs between the has- and
absent- groups by at most 0.010 (2% of level). Recomputing every headline on the
1554 games that carry all eight branches moves each per-category plateau by
≤ 0.0016 and `ratio_illegal_control_L5` from 0.9793 to 0.9805. The paired
within-game ratios in the table above are immune by construction. **Report the
per-category n, use the paired ratio, and the confound is closed** — no
complete-case restriction is needed.

### F6. Estimator and documentation hazards for the integrator *(no conclusion changes; will cause false cross-validation failures)*

- **`half_life` is elapsed, not absolute.** `divergence.py` implements elapsed
  (verified: max `peak_r + half_life` over finite rows is exactly 24, never
  above), matching its docstring and commit e93f9bb. `METRICS_REPORT.md`
  Judgement 1 still documents the absolute version and says "I implemented it
  literally". That paragraph is stale and will make A3/A4 report different
  numbers if they read it. Fix the report, not the code.
- **`half_life` censoring is handled correctly.** 28.6% censored at L5, 52.5% at
  L7; `+inf`, never NaN; both `mean` and `nanmean` over the column return `inf`,
  so nothing silently drops the censored rows. Median 14.0 at L5 is a valid
  estimate at that censoring rate. Summaries must quote median + censored
  fraction, as METRICS_REPORT says.
- **One float-precision disagreement in 13531 × 9 rows.** Layer 4, row 12783:
  δ̂ = 0.4264510 against a half-peak of 0.4264505, so `<=` resolves differently
  at float32 vs float64 and half_life comes out 6 (mine) or 7 (shipped). This is
  the only non-exact cell in the whole cross-check.
- **`kl_peak_L5` is misnamed** — KL is read off the logits and is not a
  layer-5 quantity. Rename or note it.
- **The plateau window is parity-imbalanced.** δ̂ oscillates with ply parity
  (odd *r* elevated by +0.005 at L5, consistently in all seven branch types),
  and *r* ∈ [20, 24] is three even offsets and two odd. The effect on the
  plateau is +0.001 to +0.002 and changes no ranking, but a parity-balanced
  window ([21, 24] or [19, 24]) is free.

### F7. Attacks that found nothing

These are evidence, not filler.

- **Prefix invariance — the one check whose failure would invalidate the
  study.** For all **13531** branch rows against their game's reference, at
  **every** position < t0 (not just the eight in the window), I compared the
  stored fp16 activations for **bitwise** equality in all ten tensors —
  `token_embed`, `resid_post_0..7` and `logits`. **Zero differing rows, zero
  differing elements, in all ten.** δ̂(r < 0) is exactly 0.0, not
  approximately: 0 nonzero entries out of 13531 × 8 in all 17 shipped npz
  including the SAE spaces. A1's game-boundary batching fix holds. (It is an
  easy bug to reintroduce: my own first pass at the suffix NLL chunked a
  `log_softmax` over rows and I suspected the same artefact — it turned out to
  be my indexing, and `log_softmax` is bit-stable across batch shapes 37/64/
  256/512 and row-at-a-time. The activation path is the one that isn't.)
- **The branch set re-derives exactly.** Replaying every reference prefix with
  the engine and re-classifying every branch square: **0 mismatches over all
  15531 rows**. All 2000 reference games and all 2000 prefixes are distinct.
  Prefix and suffix identity hold for every row. t0 histogram is exactly 80 per
  value over [10, 34].
- **No attrition, and no off-board NaN.** Window positions span exactly
  **[2, 58]** — never position 0 (the sink, 300.7 vs 23.5) and never position 1.
  Zero NaN in any aligned array in any of the 17 spaces, so *n* is genuinely
  constant across the whole x-axis for every category. The design's claim holds
  as stated.
- **Token and square identity explain almost none of δ̂ at depth.** Within
  category, corr(Chebyshev distance from the branch square to the reference
  square, L5 plateau) is between −0.14 and +0.10; corr(log unigram frequency of
  the branch token, plateau) between −0.09 and +0.07; corr(token-embedding
  δ̂(0), plateau) between −0.05 and −0.16 — *negative*, so a larger input-side
  displacement does not produce a larger downstream one. Token-embedding δ̂(0)
  has sd 0.050 on a mean of 1.4136, so the input floor is close to constant and
  cannot carry category structure.
- **δ̂ has not saturated.** Two unrelated reference games at the same board
  position give δ̂ median **1.293** (mean 1.301 over positions 30–58). The
  plateau of 0.47–0.50 is ~37% of that, so the branches remain far more similar
  to each other than two random games — the measure has dynamic range left and
  the small category differences are not compression against a ceiling.
- **Board change does not explain the ordering.** Control plateau by flip
  count: 0.4820 (1), 0.4902 (2), 0.4874 (3), 0.4733 (4), 0.4994 (5);
  corr(n_flipped, plateau) = +0.002.
- **No surprise signature in the output distribution.** Next-move entropy at
  *r* = 0: reference 2.364, control 2.368, illegal 2.287–2.422. The model is no
  more uncertain after an impossible move than after a legal one.
- **M2 separates nothing either.** cos(Δ_b, Δ_control) at *r* = 0 is +0.44 to
  +0.54 for every illegal category and rises to +0.78–0.86 by *r* = 24; there is
  no illegal-specific direction. cos(Δ(r), Δ(0)) decays to +0.12–0.16 by
  *r* = 24 for illegal branches **and** for the control alike — the offset
  introduced at t0 rotates away completely, at the same rate, in every branch
  type. `cos_control` is exactly 1.000000 on the control rows, so the
  reference/control pairing is right.
- **My kernels reproduce the shipped npz.** Over all nine raw spaces × 13531
  rows: max |Δδ̂| 4.8e-7, max |Δpeak| 4.8e-7, max |Δplateau| 1.8e-7,
  **0** `peak_r` disagreements, **1** `half_life` disagreement (above).
  npz `row` order matches `branches.pt` in every space.

