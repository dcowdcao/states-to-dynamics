# Probe-geometry cosine tables

Point estimates are the committed probes, except the per-square rules arm, which has no
committed probe and is ridge at lambda 10 throughout.

Where each interval comes from, so no column is an unlabelled mixture:

- every `_lo`/`_hi` column in every CSV here: a 200-resample game-level cluster
  bootstrap, percentile, both corpora resampled independently and both arms re-derived by
  ridge each resample. Games are the unit; squares and rows are not independent.
- the intervals quoted in the table 1 reading below: the merged 100-replicate red-team
  bootstrap of the same statistic, `robustness_bootstrap.csv`, cited not recomputed.
- everything with no `_lo`/`_hi` column (medians, IQRs, min/max, argmax, `n_*`,
  `refit_*`, `pooled_*`): a bare point estimate, no interval.

Point estimate and interval come from different estimators — committed probe against ridge
refit, agreeing at median cos 0.75 (rules4) to 0.88 (colour) per family, in bootstrap.json.
Several intervals therefore do not contain their own point estimate. That gap is estimator
disagreement plus the attenuation below, not a computation error; the `refit_*` columns give
the all-ridge value the intervals are actually centred on.

Random unit vectors in 512-d have E|cos| = 0.0353, but see the covariance-matched floor in
table 2.

## 1. Square-matched: the rules direction for a square vs the board direction for that square

The committed rules probes are square-agnostic — one direction, read wherever the candidate
move was spliced in — so a cosine against a square-specific board direction pools 60
candidate squares and cannot come out large. Here the legal-vs-illegal probe is refit per
candidate square. The null is that same rules direction against the board directions of the
other squares: it holds probe, corpus and shrinkage fixed and varies only the square. Every
pair is matched on `Direction.square`, never on list position — `board_directions` returns
64 colour but only 60 occupancy directions, so position and engine square differ above
square 27.

| board_family | n_squares | matched_mean_cos | matched_mean_abs_cos | matched_abs_lo | matched_abs_hi | other_mean_abs_cos | excess_mean_abs_cos | excess_lo | excess_hi | pooled_illegal_mean_abs_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| colour | 60 | 0.1677 | 0.1725 | 0.1470 | 0.1622 | 0.0203 | 0.1522 | 0.1215 | 0.1374 | 0.0215 |
| occupancy | 60 | 0.3120 | 0.3419 | 0.3088 | 0.3280 | 0.0212 | 0.3207 | 0.2861 | 0.3052 | 0.0227 |

Reading: occupancy at the matched square reaches mean |cos| =
0.3419 against 0.0212 at the other 59
squares, and colour 0.1725 against
0.0203. 54/60 and
57/60 squares are positive, so the signed means
(+0.3120, +0.1677) track the unsigned ones —
a consistent direction, not an alignment of arbitrary sign. Both other-square nulls sit at
the 0.0353 random-vector scale, so a square's legality direction overlaps that square's
board directions and nothing else's.

The merged 100-replicate bootstrap of the same statistic (`robustness_bootstrap.csv`) gives
matched colour [0.1450, 0.1573], matched occupancy [0.3012, 0.3192], off-square colour
[0.0191, 0.0208], off-square occupancy [0.0171, 0.0179], with its own caveat, carried
verbatim: the resample mean sits below the point estimate on every matched row because a
resample carries ~63% distinct games and this statistic attenuates with noise, so the
interval is spread, not location. The same applies to the columns above.

Four qualifications on this table, all from the merged analyses in `ROBUSTNESS.md`:

- The effect is **R1-only**. Legal-vs-R1 per square gives colour +0.1802 / occupancy
  +0.3866 (positive at 97% / 93% of squares); legal-vs-R2 gives +0.064 / +0.008 and
  legal-vs-R3 +0.008 / +0.031, both at the 0.0353 floor. Not a sample-size artifact: R3 has
  more rows per square at the median than R1.
- Per-move-index centering leaves colour at +0.161 but drops occupancy from +0.359 to
  +0.077, so the occupancy half is largely move-index-mediated.
- 21.2% of each per-square legality direction lies in the 2-d board subspace at its own
  square (isotropic null 0.39%), but the full 124-direction board basis captures only 26.4%
  against a 24.2% chance level: the board-relatedness is local to the square, not a general
  overlap with the board code.
- The lambda is pinned at 10 rather than swept (probe_quality.csv: 0.9252 held-out against
  0.9269 at the grid argmax of 100). The matched cosine is shrinkage-sensitive and the two
  families' ordering reverses by lambda 1e3, so read the colour/occupancy gap as
  shrinkage-dependent and the sign and the matched-vs-other contrast as the robust part.

## 2. Pooled: board direction vs the square-agnostic rule directions

`null_mean_abs_cos` is the rules arm refit on permuted labels; `null_random_mean_abs_cos`
replaces the rule direction with a random unit vector.

| board_family | rule_direction | mean_cos | mean_abs_cos | abs_cos_lo | abs_cos_hi | max_abs_cos | argmax_square | null_mean_abs_cos | null_random_mean_abs_cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| colour | illegal | -0.0182 | 0.0215 | 0.0333 | 0.0469 | 0.0591 | C1 | 0.0262 | 0.0351 |
| colour | R1 | -0.0148 | 0.0197 | 0.0434 | 0.0615 | 0.0514 | G1 | 0.0283 | 0.0351 |
| colour | R2 | -0.0210 | 0.0247 | 0.0309 | 0.0432 | 0.0750 | A7 | 0.0282 | 0.0351 |
| colour | R3 | -0.0054 | 0.0164 | 0.0204 | 0.0283 | 0.0548 | A0 | 0.0286 | 0.0351 |
| occupancy | illegal | -0.0046 | 0.0227 | 0.0285 | 0.0397 | 0.0579 | D1 | 0.0459 | 0.0351 |
| occupancy | R1 | -0.0105 | 0.0286 | 0.0417 | 0.0561 | 0.0596 | A7 | 0.0581 | 0.0351 |
| occupancy | R2 | -0.0061 | 0.0181 | 0.0306 | 0.0396 | 0.0541 | A3 | 0.0616 | 0.0351 |
| occupancy | R3 | -0.0116 | 0.0198 | 0.0184 | 0.0245 | 0.0565 | D5 | 0.0567 | 0.0351 |

Reading: the largest cell is occupancy vs R1 at mean |cos| =
0.0286, with signed means running from -0.0210 to
-0.0046. The honest floor for a pooled number is not the isotropic
0.0353 but the covariance-matched null of 0.084-0.110 reported in `ROBUSTNESS.md`, against
which these cells are *more* orthogonal than chance rather than at it. Either way the
comparison to make is with table 1: the same probe family, measured square by square, is
12x higher. Read this table as the cost of
pooling over 60 squares, not as evidence that legality and board state are unrelated.

## 3. Within-board: colour vs occupancy

| pair_kind | square_relation | n_pairs | mean_cos | mean_abs_cos | median_cos | q25_cos | q75_cos | mean_ci_lo | mean_ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| colour-colour | different_square | 2016 | 0.0184 | 0.0317 | 0.0049 | -0.0108 | 0.0280 | 0.0101 | 0.0126 |
| occupancy-occupancy | different_square | 1770 | 0.0463 | 0.0514 | 0.0392 | 0.0162 | 0.0721 | 0.0170 | 0.0183 |
| colour-occupancy | same_square | 60 | 0.0214 | 0.0360 | 0.0229 | -0.0049 | 0.0416 | 0.0630 | 0.0716 |
| colour-occupancy | different_square | 3780 | 0.0004 | 0.0174 | 0.0027 | -0.0108 | 0.0156 | -0.0011 | -0.0001 |

Reading: at one square, "which disc" and "is there a disc" sit at cos =
0.0214 (median 0.0229, IQR -0.0049 to
0.0416), at the 0.0353 random-vector scale — the two board axes are close to
independent readouts of the same square. The CI is on the mean; per-Chebyshev-distance rows
are in within_board_cosine.csv.

## 4. Within-rules

| direction_a | R1 | R2 | R3 | illegal |
| --- | --- | --- | --- | --- |
| R1 | 1.0000 | 0.3086 | 0.4379 | 0.9226 |
| R2 | 0.3086 | 1.0000 | 0.5037 | 0.4611 |
| R3 | 0.4379 | 0.5037 | 1.0000 | 0.5868 |
| illegal | 0.9226 | 0.4611 | 0.5868 | 1.0000 |

Reading: the four square-agnostic rule directions are far from orthogonal to each other, so
they are not four independent readouts. illegal.R1 = 0.923
is a fact about the geometry and not about a confused anchor: every rule direction is
W[c] - W[legal], and the committed 4-class probe recovers `legal` at recall
0.8238, macro-F1 0.7667, with 1444 of
9255 true-legal rows in the split.pt test games predicted R1. This is the
table where the two estimators disagree most: the ridge intervals sit far from the
committed cosines they accompany (R1.R2 is
0.309 committed against a ridge interval of
0.615
to 0.663),
so treat the inter-rule geometry as estimator-dependent and read the intervals as width only.

## 5. Replication on the injected-move pairs corpus

Rules directions refit on the 4000-prefix pairs corpus, board arm unchanged: the pooled
maximum is mean |cos| = 0.0264 (colour vs
R2) against a shuffled-label null of 0.0277, so
the pooled table replicates on an independent corpus. The per-square arm is not replicated
here: 28189 rows over 60 squares is ~470 rows per square against 512 dimensions.

## Probes

probe_quality.csv carries held-out-GAME accuracy for every probe above, over the whole
lambda grid. The committed board probe is 0.9965 against a
0.5449 per-square majority baseline, and all 200 ARENA games are held out
for it: it was fit on 20000 Karvonen games. The 8000/2000 row cut that splits this arm falls
exactly at game 160, so the ridge refit's eval games are disjoint too. The committed 4-class
rules probe is 0.7657 on the split.pt test games with macro-F1 0.7667
and legal-class recall 0.8238, scored through `load_space("raw")` — the
feature map it was fit on.
