# Red team of the probe-geometry claims

`python src/probe_geometry/robustness.py` writes every number below to
`analysis/probe_geometry/results/robustness_*.csv` (committed copies archived there under
`archive/`), except `robustness_cells.csv`, which A.4 quotes and sits here. Layer 5,
`blocks.5.hook_resid_post`, d = 512. Both committed probes were fit on
`adamkarvonen/othello_45MB_games`; the board arm's *re-fits* run on 200 ARENA games, 10000 rows,
which is the only corpus difference left. Rules arm: the move-legality fit set, 800 games,
176262 rows, 60 candidate squares with 2482-4544 rows each.

## The two claims, and why only one of them is a claim

**Pooled.** The committed rules probes are square-agnostic: one "illegal" direction read off
the residual wherever the candidate move was spliced in. Every board direction is
square-specific. Mean |cos| between them is 0.016-0.029, against an isotropic floor of
sqrt(2/(pi*512)) = 0.0353. This is close to forced. A single global direction cannot be a
combination of per-square contrasts, because the per-square feature is a conjunction of
square identity and occupancy while the global one is the marginal. **The pooled number is
not evidence about how the model computes legality.**

**Square-matched.** Legality refit separately on each candidate square
(`probes.square_rules_directions`) aligns with the board directions at its *own* square:

| board arm | matched mean cos | matched mean abs cos | positive at n/60 squares | other-square mean abs cos |
|---|---|---|---|---|
| committed probe | **+0.1677** colour, **+0.3120** occupancy | 0.1725 / 0.3419 | 57 / 54 | 0.0203 / 0.0212 |
| ridge, matched estimator | **+0.1742** colour, **+0.3592** occupancy | 0.1796 / 0.3825 | 57 / 54 | 0.0177 / 0.0168 |

n = 60 matched pairs; n = 3780 (colour) and 3540 (occupancy) off-square pairs. Every threat
below is measured against the square-matched row, with the pooled row alongside.

### Game-level uncertainty

100 bootstrap replicates over games, refitting the board arm and all 60 per-square
directions inside each replicate (`robustness_bootstrap.csv`).

| | point | resample mean | 2.5-97.5% |
|---|---|---|---|
| matched mean cos, colour | +0.1742 | +0.1508 | [0.1450, 0.1573] |
| matched mean cos, occupancy | +0.3592 | +0.3113 | [0.3012, 0.3192] |
| off-square mean abs cos, colour | 0.0177 | 0.0199 | [0.0191, 0.0208] |
| off-square mean abs cos, occupancy | 0.0168 | 0.0175 | [0.0171, 0.0179] |
| capture in the matched board pair | 0.2119 | 0.1609 | [0.1538, 0.1666] |
| cross-corpus matched mean cos, colour | +0.1766 | +0.1556 | [0.1514, 0.1605] |
| cross-corpus matched mean cos, occupancy | +0.3616 | +0.3174 | [0.3065, 0.3256] |

Read the interval as a spread, not a location. A resample of 800 games carries ~63% of them
distinct, the per-square fits get correspondingly noisier, and this statistic attenuates
with noise (section 5) — which is why the resample mean sits below the point estimate on
every matched row and above it on every off-square row. What the interval says is that
game-level sampling moves the matched number by under +/-0.01 and never brings it within an
order of magnitude of the off-square number.

## Verdicts

| # | threat | measurement | verdict |
|---|---|---|---|
| 1 | estimator artifact | matched cos 0.16-0.26 (colour) and 0.21-0.36 (occupancy) across five decades of shrinkage where both probes stay accurate | survives |
| 2 | token identity | projecting out the whole 59-dim `W_E` span moves matched cos 0.174 → 0.179 and 0.359 → 0.384 | dies |
| 3 | move index | colour holds at +0.161; occupancy falls +0.359 → +0.077 | **partially kills the occupancy half** |
| 4 | corpus | both arms on the rules corpus: +0.177 / +0.362; independent 4000-game corpus: +0.130 / +0.178 | survives |
| 5 | power | per-square directions reproduce at split-half cos 0.525 only; attenuation correction raises the number to +0.218 / +0.434 | survives, but the per-square arm is noisy |
| 6 | the null | every null (isotropic, sampled, covariance-matched, label-shuffled) puts matched at the same level as off-square | survives |
| 7 | what the basis spans | 21.2% of a per-square legality direction lies in the 2 board dimensions at its own square (isotropic null 0.39%); the other ~79% is not in the board basis at all | **this is the headline** |
| 8a | "it is only R1" | legal-vs-R1 alone gives +0.180 / +0.387; legal-vs-R2 +0.064 / +0.008; legal-vs-R3 +0.008 / +0.031 | **largely true — narrow the claim** |
| 8b | per-square class imbalance | equalizing legal/illegal counts inside every square: +0.188 / +0.310 | dies |

## 1 Estimator

Both arms refit by one ridge on centered one-hot targets, shrinkage swept 1e0 … 1e5, with
held-out-game accuracy so an over-shrunk fit is not mistaken for a geometric result
(`robustness_cells.csv`, variants `ridge_lam*`).

| lambda | board acc (n=2000 rows, 40 held-out games) | square legal-vs-illegal acc (n=35k rows, 160 games) | matched cos colour | occupancy | other-square abs cos |
|---|---|---|---|---|---|
| 1e0 | 0.9950 | 0.9358 | +0.159 | +0.358 | 0.024 / 0.024 |
| 1e1 | 0.9950 | 0.9369 | +0.172 | +0.351 | 0.023 / 0.022 |
| 1e2 | 0.9949 | 0.9354 | +0.228 | +0.307 | 0.023 / 0.020 |
| 1e3 | 0.9930 | 0.9151 | +0.259 | +0.210 | 0.025 / 0.023 |
| 1e4 | 0.9685 | 0.8785 | +0.171 | +0.108 | 0.031 / 0.032 |
| 1e5 | 0.8295 | 0.7398 | +0.127 | +0.049 | 0.032 / 0.070 |

The committed arms were fit by two different estimators (board: AdamW on a per-square
3-way softmax, early-stopped on held-out-game validation accuracy; rules: L2 multinomial
logistic on z-scored features). Matching them changes the number by less than the
shrinkage sweep does. Survives.

## 2 Token identity

The rules rows sit at a spliced move token, so the direction could be that token's
embedding. A per-square fit sees one token only, so the embedding is constant within the
fit and cannot drive the contrast — but the *board* direction at square q could itself be
aligned with `W_E[token_q]`, and the matched-vs-other structure would then be inherited
from embedding geometry. Measured (`robustness_token_energy.csv`):

- mean |cos| with the square's own `W_E` row, n = 60: colour 0.0847, occupancy 0.0968,
  per-square legality 0.0599. Isotropic floor 0.0353.
- squared-norm fraction inside the full 59-dim `W_E` span: colour 0.091, occupancy 0.072,
  per-square legality 0.143. Chance for a 59-dim subspace of a 512-d space is 0.115.
- inside the 59-dim span of the per-square *mean activations* (the empirical token
  subspace): colour 0.106, occupancy 0.177, per-square legality 0.139.

Projecting both direction sets orthogonal to those subspaces (`token_out_*` variants):

| projection | matched cos colour | occupancy |
|---|---|---|
| none | +0.1742 | +0.3592 |
| `W_E` rank 59 | +0.1788 | +0.3838 |
| activation square means rank 59 | +0.1984 | +0.4063 |
| the square's own `W_E` row only | +0.1708 | +0.3667 |

The alignment goes *up* when the embedding subspace is removed. The token confound is dead.

## 3 Move index

The one threat that bites. Each move index's legal-class mean is subtracted from its rows
and the rules arm is refit (`position_centered`).

| | matched cos colour | occupancy | matched abs cos | positive fraction | matched-pair capture |
|---|---|---|---|---|---|
| uncentered | +0.1742 | +0.3592 | 0.180 / 0.383 | 0.95 / 0.90 | 0.212 |
| centered by move index | +0.1614 | **+0.0774** | 0.165 / 0.134 | 0.92 / 0.73 | 0.063 |

Consistent with the direction geometry: |cos| with the leading principal direction of the
58 per-move-index legal-class means is 0.0064 for colour, **0.1527 for occupancy**, 0.0517
for the per-square legality directions.

Read it as: the colour half of the square-matched alignment is not a move-index effect; a
large part of the occupancy half is. "This square is occupied" and "we are late in the
game" share variance, and the occupancy probe direction carries the shared part. The
control is one-sided (only the rules arm is centered), so it removes any alignment that
runs through the move-index axis while leaving within-index variation untouched — which is
exactly the diagnosis, not an over-correction, but it does not separate "the model encodes
fill level" from "the occupancy probe is partly a fill-level probe".

One oddity worth flagging: under the same centering the *pooled* occupancy number rises
from 0.0132 to 0.1522 (R1) and 0.1276 (illegal). Removing the move-index axis moves the
square-agnostic rules directions toward the occupancy family.

## 4 Corpus

Both committed probes are fit on `adamkarvonen/othello_45MB_games`; what is still
ARENA-only is the board arm's ridge re-fit corpus, which is what this threat now tests.
The swap costs one thing: all 800 rules-corpus games are inside the board probe's 20000,
576 of them in its training split, because both stream the same dataset at seed 42. That
is 4% of the probe's games, and it makes the committed board number below a fit statistic
rather than a transfer one — the transfer test is the ridge row under it.

The rules corpus' legal rows carry the move the game actually played, so replaying
`fit_set/tokens.pt` with `circuits.othello_engine_utils` gives them mover-relative board
labels (perspective from the engine's `next_hand_color`, never from move-index parity).

- Committed board probe on those rows: **99.36%** 3-way accuracy, n = 46256 rows
  x 64 squares, matching its own held-out test accuracy of 0.9937 — but see the overlap
  above.
- Board arm refit on rules-corpus games, scored on held-out games: **99.25%**, n = 9253 rows.
- Per-direction cos between the ARENA-fit and rules-corpus-fit board arms: colour mean
  0.960 (min 0.936), occupancy mean 0.973 (min 0.948), n = 124.
- Both arms on that one corpus: matched cos **+0.1766** colour, **+0.3616** occupancy;
  off-square 0.0192 / 0.0186.
- Rules arm refit on the independent 4000-game `pairs` corpus (28189 rows, ~470 per square,
  different sampling and different injected-move construction): **+0.1297** / **+0.1776**,
  off-square 0.0293 / 0.0226. Lower, as its ~6x smaller per-square fits predict.

Survives, and this is the strongest single control.

## 5 Power

Ten random half-splits of each arm's games; per-direction cos between the two halves, then
Spearman-Brown for a whole-corpus fit (`robustness_reliability.csv`, n_splits = 10).

| family | n | split-half cos (mean, min) | reliability |
|---|---|---|---|
| colour | 64 | 0.917, 0.770 | 0.957 |
| occupancy | 60 | 0.953, 0.902 | 0.976 |
| rules4 | 3 | 0.973, 0.971 | 0.986 |
| rules2 | 1 | 0.956 | 0.977 |
| **per-square legality** | 60 | **0.525, 0.409** | **0.687** |

The per-square directions are the noisy ones — ~2800 rows each. Correcting the matched
cosine for attenuation on both sides raises it: colour **+0.2180**, occupancy **+0.4343**
(abs 0.2246 / 0.4636). The correction moves the number up, so noise is not manufacturing
the effect.

For the pooled arms the same logic runs the other way: they reproduce themselves at
0.96-0.99, so their near-zero cosine is not an estimation floor. Attenuation-corrected
pooled mean |cos| is 0.011-0.017.

## 6 Is the null the right null

Mean |cos| at the matched square, colour / occupancy, against four nulls (10 redraws each
for the sampled and shuffled nulls; `robustness_cells.csv`):

| | matched | off-square |
|---|---|---|
| measured (ridge) | **0.1796 / 0.3825** | 0.0177 / 0.0168 |
| analytic isotropic floor | 0.0353 | 0.0353 |
| sampled isotropic unit vectors | 0.0345 / 0.0365 | 0.0354 / 0.0350 |
| directions drawn from the activation covariance | 0.0932 / 0.0892 | 0.0921 / 0.0920 |
| rules labels shuffled within each square | 0.0358 / 0.0550 | 0.0220 / 0.0163 |
| board labels shuffled across rows | 0.0362 / 0.0349 | 0.0362 / 0.0364 |

Every null puts matched at the same level as off-square, which is the whole point: the
null for this claim is not "cos = 0", it is "matched = other", and that is what the nulls
deliver.

The covariance-matched null also settles the pooled question. Two directions drawn from the
layer-5 activation covariance sit at |cos| 0.084-0.110, not 0.0353. The pooled rules
directions at 0.013-0.017 are therefore *more* orthogonal to the board arm than chance in
this space, not less. (That null is generous: it is dominated by the leading activation PC,
which probe *contrasts* are largely free of.)

## 7 What the board basis actually spans

Legality is a nonlinear function of the board, so a small cosine between linear readout
directions rules out very little. The honest quantity is how much of a rules direction lies
in the span of the board directions, against the null for a generic subspace of that rank —
which is not zero (`robustness_captured.csv`).

| | captured fraction | null |
|---|---|---|
| per-square legality inside the **2** board dims at its own square, n = 60 | **0.2119** | isotropic 0.0039 [0.0001, 0.0143]; covariance-matched 0.0087 [0.0002, 0.0375]; shuffled rules labels 0.0070 |
| per-square legality inside the **124**-dim board basis, n = 60 | 0.2640 (0.072-0.447) | isotropic 0.2415 [0.192, 0.295]; covariance-matched 0.4756 [0.368, 0.584] |
| pooled rules directions inside the 124-dim basis | committed 0.053-0.068; ridge 0.033-0.040 | same |

Two things at once, and they are the result:

1. **21% of a per-square legality direction lives in the two board dimensions at that same
   square** — 54x the isotropic null, 24x the covariance-matched null, far outside its
   97.5th percentile.
2. **The remaining ~79% is not board-readout at all.** Adding the other 122 board
   directions buys only ~5 more points and lands the total at the chance level for a
   124-dim subspace (0.264 vs 0.2415). Whatever else the legality direction is made of, it
   is not in the span of the board probe.

The pooled directions sit *below* the isotropic null, so the pooled near-orthogonality is
about a global direction versus a per-square basis and nothing more.

## 8 Two deflationary readings

**8a. It is R1.** R1 is "the target square is already occupied", which is the occupancy
contrast almost by definition. Refit per square against one rule at a time (squares with at
least 200 rows of that rule):

| fit | squares | matched cos colour | occupancy | positive fraction |
|---|---|---|---|---|
| legal vs R1 | 60 | **+0.1802** | **+0.3866** | 0.97 / 0.93 |
| legal vs R2 | 45 | +0.0639 | +0.0076 | 0.80 / 0.58 |
| legal vs R3 | 60 | +0.0078 | +0.0311 | 0.52 / 0.72 |

R1 alone reproduces the whole effect; R2 and R3 sit at the 0.035 floor. This is not a
sample-size artifact: R3 has 474-951 rows per square (median 720) against R1's 217-2109
(median 644), so R3 has *more* data at the median and still shows nothing. R2 is the thin
one (30-2655 rows, median 513), hence the 200-row cut.

The honest claim is therefore narrow: **the per-square legality direction is aligned with
the board readout at its own square for the "square is occupied" violation, and not
detectably for the bracket-geometry violations R2 and R3.**

**8b. Class balance.** The legal share varies across squares (0.16 to 0.32). Across the 60
squares, the matched cosine correlates with that share at r = -0.026 (colour) and r = +0.298
(occupancy), n = 60. Refitting with legal and illegal counts equalized inside every square
gives +0.1877 colour and +0.3102 occupancy — the effect is not the imbalance.

## Caveats inherited from the probes

- The committed 4-class probe recovers `legal` at recall **0.8238** on its own test split
  (n = 9255 true-legal rows), with **15.6%** of them predicted R1. Scored in the wrong
  input space it read 0.2324 / 71.9%; the probe's features are `load_space("raw")`, not
  the bare residual. So illegal·R1 = 0.923 is not explained away by an anchor confused
  with R1: the anchor works.
- The committed board probe was fit on 20000 Karvonen games, so all 200 ARENA games are
  held out for it — but the 800 rules-corpus games are not (threat 4). The board arm's
  *ridge* accuracies use this module's own 80/20 split by game over the 200.
- Activations are held in float32: this box caps the user at 8 GB shared with other jobs and
  a float64 corpus plus the ridge's own copies does not fit. Fitted directions still come
  out in float64 and agree with a float64 corpus to cos >= 0.999998 over the whole lambda
  grid. The CSVs here come from one SLURM pass at 96 GB; on the 8 GB login node the
  entrypoint has to be split in two (everything through threat 8, then the bootstrap).
- The bootstrap covers the board arm and the 60 per-square directions, which is what the
  square-matched claim rests on. The pooled square-agnostic arms carry point estimates only;
  refitting them inside 200 replicates needs 700 MB of working set this box does not have.

## Files

| file | contents |
|---|---|
| `robustness_cells.csv` | every variant x statistic: matched/off-square cosines, pooled cosines, capture fractions, held-out accuracies |
| `robustness_square_detail.csv` | per-square matched cosine, rows and legal share, for each variant |
| `robustness_token_energy.csv` | threat 2 subspace energies |
| `robustness_reliability.csv` | per-direction split-half cosine and Spearman-Brown reliability |
| `robustness_captured.csv` | threat 7 capture fractions and their sampled nulls |
| `robustness_corpus_transfer.csv` | threat 4 accuracies, per-direction board transfer, 4-class confusion |
| `robustness_bootstrap.csv` | game-level percentile intervals on the square-matched statistics |
