# A7 behavioural divergence: M5 and F6

Does the representational divergence reach behaviour? Yes, immediately and
permanently — but the size of the effect tracks **how much the board changed**,
not **whether the move was legal**, and the legal control diverges further than
any illegal category.

Everything here comes from `analysis/violation_divergence/behavioral/make_figures.py`
over `$SPADE_DIVERGENCE_DIR/acts/logits.pt` (fp16 `[15531, 59, 61]`) and the
reference games replayed with `rules.causal.boards`. Numbers are read off
`results/violation_divergence/behavioral_summary.csv`; the full event-locked
curves are in `behavioral_curves.csv`.

## Three conventions, because each one can silently move the whole curve

**Alignment.** `r` indexes the board position `s = t0 + r` whose logits row it
is — the same convention `divergence.align` uses, so KL(r) and δ̂(r) share an
x-axis. `logits[:, s]` is the distribution over the move at `s + 1`, so `r = 0`
is the first prediction the model makes after the branch token entered the
context, and `r = -1` is the last one made from the shared prefix. Checked
rather than assumed: `KL(r < 0) = 0` **exactly** is asserted in the script over
all 15531 rows × 8 pre-event offsets, and so is the reference row's KL against
itself, 0 at every `r`. On the 50-game smoke set I checked the stronger form
directly — a branch's logits are **bit-identical** to its reference's at every
position `< t0` and first differ at exactly `t0` — which is what fixes `r = 0`
as the first affected prediction rather than `r = 1`. Hand-checked on game 0
(t0 = 10): `softmax(logits[:, t0-1])` puts 0.152 on the reference token and
1.2e-6 – 1.4e-5 on the six illegal ones, matching `DATA_REPORT.md`'s per-branch
means.

**Softmax.** float32, over the **whole 61-wide vocabulary, pass token
included**. The distribution stays the model's own; doctoring it would change
the KL by an unstated amount. The pass token is irrelevant either way: the
largest per-category mean `p(pass)` anywhere in the window is **5.0e-4**
(`control_legal`), and 1.8e-5 for the reference branch.

**Legal set.** `regular_moves` on the **reference** board after replaying
`tokens[:s+1]`, with the R4 forfeit fallback (`regular_moves` for the opponent
when the nominal mover has none) — that is the set the next token can come
from. 84 of the 66 000 scored game-positions are forfeits (0.13%); none is
terminal, so the legal mass is defined at every offset for every game. Mean
legal-set size over the window: 10.27 of 60 squares, so uniform-over-the-board
chance is 0.171.

## Contract keys

`kl_peak_L5` is the contract's name; KL is a read-out of the **output head**, not
of layer 5, and the key carries the layer only because the rest of the table
does. "Peak" is the maximum of the mean curve over `r ≥ 0`, pooled over the six
illegal categories (11 531 rows, 2000 games), matching `ratio_illegal_control_L5`'s
pooling convention.

| key | value | 95% CI | n |
|---|---|---|---|
| `kl_peak_L5` | **2.048 nats** at `r = 1` | [2.001, 2.099] | 11 531 rows / 2000 games |
| `kl_at_r0` (pooled illegal) | 1.957 nats — 96% of the peak | | 11 531 |
| `kl_plateau` (pooled illegal, `r ∈ [20,24]`) | 1.641 nats — 80% of the peak | | 11 531 |
| `kl_peak_L5` (`control_legal`, the null) | **2.515 nats** at `r = 1` | [2.448, 2.581] | 2000 |
| legal mass, pooled illegal, `r = 0` | 0.805 | | 11 531 |
| legal mass, `control_legal`, `r = 0` | 0.750 | | 2000 |
| legal mass, `reference` (the ceiling) | 0.9994 at `r = 0`, 0.9997 before `t0` | | 2000 |
| `legal_mass_control_own_board_r0` | **0.9990** | [0.9979, 0.9996] | 2000 |
| top-1 legality, pooled illegal, `r = 0` | 0.823 (reference: 1.000) | | 11 531 |
| `kl_fp16_floor` | 1.1e-6 median, 4.4e-6 p99 | | 20 000 sampled positions |

## Per category

Peak and offset are the maximum of the mean curve over `r ≥ 0`; `r=0` and
plateau columns are that curve at `r = 0` and averaged over `r ∈ [20, 24]`.

| branch | n | KL peak | at `r` | KL at `r=0` | KL plateau | legal mass `r=0` | legal mass plateau | top-1 legal `r=0` |
|---|---|---|---|---|---|---|---|---|
| `reference` (baseline) | 2000 | 0 | — | 0 | 0 | 0.9994 | 0.9952 | 1.000 |
| `occupied_own` (R1) | 2000 | 1.859 | 24 | 1.478 | 1.526 | 0.858 | 0.842 | 0.859 |
| `occupied_opp` (R1) | 2000 | 2.099 | 1 | 2.006 | 1.552 | 0.801 | 0.839 | 0.838 |
| `empty_detached` (R2) | 1956 | 2.014 | 24 | 1.740 | 1.678 | 0.821 | 0.826 | 0.853 |
| `empty_own_only` (R2) | 1636 | 2.143 | 1 | 2.027 | 1.696 | 0.802 | 0.822 | 0.811 |
| `empty_no_bracket` (R2) | 1941 | 2.145 | 1 | 2.070 | 1.711 | 0.787 | 0.823 | 0.797 |
| `opponent_only` (R3) | 1998 | 2.432 | 0 | 2.432 | 1.697 | 0.761 | 0.824 | 0.776 |
| `control_legal` (null) | 2000 | 2.515 | 1 | 2.494 | 1.704 | 0.750 | 0.825 | 0.759 |
| illegal pooled | 11 531 | 2.048 | 1 | 1.957 | 1.641 | 0.805 | 0.830 | 0.823 |

**Read the ordering, not the individual numbers.** Behavioural divergence runs
`occupied_own` < `empty_detached` < `occupied_opp` ≈ `empty_own_only` ≈
`empty_no_bracket` < `opponent_only` < `control_legal`, and the legal-mass
ordering is its mirror image (the middle three are within 0.04 nats and 0.02
mass of each other and swap between the two metrics; nothing else does). That is
an ordering by *how much of a board change the token can be read as*, not by how
illegal it is: the no-op (`occupied_own`, the mover's disc is
already there) moves the prediction least, and the fully legal control moves it
most. **Illegality per se does not buy extra behavioural divergence.** The
occupied-square pair the design doc singles out reads the same way:
`occupied_opp` diverges more than `occupied_own` (2.10 vs 1.86 nats; 0.801 vs
0.858 legal mass), so the model behaves as if placing on the opponent's square
does *something* — closer to a recolour than to a no-op.

**n differs by category and is never pooled silently.** The only pooled row is
the one labelled as such. `empty_own_only` is present in 1636 of 2000 games, and
`DATA_REPORT.md` shows the 18.2% missingness is a board-shape criterion spread
across all `t0`, not a depth effect — so that row is conditioned on a different
game set from the others. Its numbers land mid-pack and indistinguishable from
`empty_no_bracket`'s (n = 1941), so the confound does not visibly bite here. The
pooled row weights games by how many illegal categories they offer (4 to 6), and
its bootstrap resamples games, not rows, so a game's branches move together.

**Two peak offsets sit on a knife edge.** The KL curve is U-shaped: it drops to
a minimum near `r = 14` and climbs again into the endgame, where legal sets are
small and predictions sharp. For `occupied_own` the global maximum at `r = 24`
(1.8594) beats the `r = 1` local peak (1.8574) by 0.1%, and `empty_detached` is
the same shape. If A5 gets `r = 1` for those two rather than `r = 24`, that is
the U-shape and not a disagreement — the curve CSV shows both.

**`kl_half_life` is not usable for KL** (pooled censoring 16.0%, control 13.6%,
against δ̂'s 52.5% at layer 7). Every curve carries a one-ply sawtooth, so a row
crosses half its peak on an even ply and comes straight back. Use plateau ÷ peak
(80% pooled) for persistence, not half-life.

## The sawtooth is turn parity

At `s = t0 + r` the model predicts the move at `t0 + r + 1`, which is by the
colour that played the branch move exactly when `r` is odd — measured, not
assumed: **99.81%** of window positions at odd `r` have the branch mover to move,
against **0.26%** at even `r` (the 0.2% slack is the 84 R4 forfeits, which skip a
colour). KL is high and legal mass low on those odd plies:
the corrupted disc hurts the predictions of the colour that placed it. δ̂ carries
the same oscillation at layer 5, so this is one phenomenon seen twice, not a
read-out artefact.

`opponent_only` is the exception that fits: it is the only category whose
maximum is at `r = 0` rather than `r = 1`. It is the R3 turn-confusion category
— a move that flips only for the opponent — so it is the one violation that hits
the immediately following prediction hardest.

## Does behaviour lag the representation? No — they move together, immediately

The premise in my brief ("δ̂ peaks at `r ≈ 4.3` at layer 5, `r ≈ 5.8` at layer
7") is the **mean over branches of each branch's own argmax offset**. That
estimator is biased late on a noisy monotone-decaying curve, because each row's
argmax wanders. Re-read from A2's `$DIR/divergence/resid_post_{5,7}.npz`, the
**mean δ̂ curve itself peaks at `r = 0` at both layers** and decays from there:

| estimator | δ̂ layer 5 | δ̂ layer 7 | KL |
|---|---|---|---|
| offset of the peak of the **mean curve** | **0** (0.7925) | **0** (0.7867) | **1** (2.048); 96% of it already at `r = 0` |
| mean of **per-branch argmax** offsets | 4.19 | 5.68 | 10.93 |
| plateau ÷ peak | 0.60 | 0.72 | 0.80 |

Both quantities are read with the same estimator in each row, and both say the
same thing: **the model notices the violation on the very next prediction, and
representation and behaviour peak together.** Neither of the two stories in my
brief holds — behaviour does not lag the representation, and it does not spike
and decay while the representation keeps climbing. What differs is persistence:
KL retains 80% of its peak at `r ∈ [20, 24]` while δ̂ retains 60% at layer 5 and
72% at layer 7, so **behaviour recovers less than the representation does**.

The per-branch-argmax row is the one that looks like a lag (10.9 vs 4.2), and it
is an artefact twice over: the KL curve is U-shaped, so the late climb captures
the argmax outright, and the sawtooth adds a ply of jitter. It is reported so
the mismatch with `delta_peak_r_L5` is visible rather than buried.

**This is a live issue for the contract.** `delta_peak_r_L5` is an exact-match
key. Whether it means "argmax of the mean curve" (0) or "mean of the per-branch
argmaxes" (4.19 illegal-only, 4.3 pooled with the control) changes the answer by
four plies, and A3, A5 and I will disagree unless the definition is fixed. My
recommendation: report the mean-curve argmax as the headline and keep the
per-branch mean beside it, since only the first is a property of the curve
everyone plots.

## Is the legality loss actually a loss? No — the model is following the board it was shown

Pooled illegal branches put 0.805 of their mass on the reference board's legal
set at `r = 0`, against a 0.9994 ceiling and a 0.171 chance level. That looks
like a large legality failure until the control is read: `control_legal` scores
**lowest of all** on the reference board (0.750) while putting **0.9990** on the
legal set of the board it actually played. The control has not lost legality at
all — it has correctly followed a different board. So the reference-board
deficit measures *board disagreement*, not degraded legality, and the illegal
branches' higher score means their boards have moved **less** far from the true
one, which is what "an illegal move cannot be applied to a board" predicts.

`control_legal` is the only branch this calibration exists for: an illegal
branch has no board — updating one after a disc lands on an occupied square is
exactly what the shared-suffix design avoids — so the reference board is the only
yardstick all eight branches share.

The deficit never heals. Every branch settles near 0.82–0.84 legal mass and
0.84–0.85 top-1 legality for the rest of the game, against a reference that
holds 0.995 and 0.9995. One illegal token costs about 16 points of legality,
permanently.

## Two things that turn out not to matter

**fp16 storage of the logits.** Perturbing each stored value by its own half-ulp
and re-measuring gives a KL floor of **1.1e-6 nats median, 4.4e-6 at the 99th
percentile** (20 000 sampled row-positions) — five to six orders of magnitude
below the 2-nat signal and below the smallest per-category difference in the
table (0.02 nats). The quantisation is immaterial to every claim here. It would
matter for a KL of order 1e-4, which nothing in this window is.

**The pass token.** Max per-category mean `p(pass)` over the whole window is
5.0e-4. Including or excluding index 0 from the softmax cannot move a 2-nat KL
or an 0.8 legal mass.

## For the integrator and A5

1. Fix the estimator behind `delta_peak_r_L5` / `kl_peak_r` before comparing —
   see above. This is the one place I expect a spurious mismatch.
2. `kl_peak_L5` has no layer. If the contract table can carry a footnote, say it
   is the output head.
3. `occupied_own` and `empty_detached` peak offsets are knife-edge (0.1%)
   between `r = 1` and `r = 24`.
4. The headline for the study: representational divergence **does** reach
   behaviour, immediately and permanently — but the legal control diverges
   further than any illegal category, so the claim must be "the model acts on the
   corrupted history", not "the model reacts to illegality".
