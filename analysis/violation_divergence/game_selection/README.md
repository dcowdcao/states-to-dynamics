# Which game the fork figure is drawn on

`game_trajectory/` draws one game forking at *t0* into nine branches, across nine
representation spaces. It drew `complete[0]` — game 0, *t0*=10, the **minimum**
*t0* over the 1550 branch-complete games (median 21). That is list order, not a
criterion. This sub-study sweeps every complete game, scores how distinct its
nine branch trajectories are, and hands the fork figure an argmax it can disclose.

```
python analysis/violation_divergence/game_selection/score_games.py   # ~4 min, cpu
python archive/analysis/violation_divergence/game_selection/make_figures.py   # archived
python archive/analysis/violation_divergence/game_selection/score_rules.py    # archived; ~15 min, cpu
```

`score_games.py` answers the **category** question in raw layer 5 and is what
`branches.FIGURE_GAME` was picked under. `score_rules.py` answers the **rule**
question — R1/R2/R3, three means over 2/3/1 members — in SpaDE latents, for a
figure that draws rules rather than categories; the two are separate criteria
with separate answers, and the second is at the bottom of this file.

## Answer: game 662, *t0* = 22

| | value | where it sits among the 1550 |
|---|---|---|
| *S3* category differentiation | **1.0802** | **1 / 1550** (max; median 0.7045) |
| *S1* illegal vs. control | 0.8037 | 1460 / 1550 (5.8th percentile; median 1.0133) |
| *S2* branch distinctness | 0.4756 | 586 / 1550 (median 0.4612) |
| *S4* δ̂ at *r*=24 | 0.2785 | 1473 / 1550 (median 0.4454) |
| *t0* | 22 | 56.8th percentile (min 10, median 21, max 34) |

The incumbent, game 0: *S3* 0.7030 (787/1550, essentially the median), *S1* 0.9380
(1079/1550), *S2* 0.5515 (31/1550), *S4* 0.5159 (439/1550), *t0*=10 (the minimum).
The current pick is not a bad game — it is an *average* game on the criterion that
governs readability, and an extreme one on *t0*.

**Caption line, usable verbatim:**

> Game 662 (*t0* = 22) is the argmax, over all 1550 branch-complete games, of the
> mutual separation of its six illegal displacement trajectories, scored in raw
> layer 5 in the full 512-dimensional space and never in an SAE space or a 2-D
> projection; it is the clearest available illustration of the construction, not a
> typical draw (population median 0.705, this game 1.080).

## Method

Everything is computed on the displacement vectors **D(t) = z_branch(t) −
z_ref(t) in full R^512**, over the study window sampled at *t0*'s parity and
restricted to *r* ≥ 0 (before *t0* every displacement is exactly zero by
construction) — 13 positions per game. Each branch's displacements are stacked
into one vector X_b ∈ R^(13·512).

- **Scored in raw layer 5 only.** The grid compares nine spaces; selecting with an
  SAE in the loop would tilt that comparison toward whichever dictionary drove the
  choice. Raw layer 5 is the common input to all nine. The dictionaries enter
  exactly once, in `transfer()`, and never in the decision.
- **Scored in full R^512.** A top-2 PCA criterion would select the game that
  flatters one particular plane, and the plane is what is being redesigned.
- **Criterion fixed before the sweep ran** (commit `7d91857`, `score_games.py`
  docstring) and not revised afterwards.

| key | definition | units |
|---|---|---|
| *S1* | ‖μ_illegal − μ_control‖ ÷ pooled within-group RMS radius (n=6 vs n=2, dof 6) | pooled SDs |
| *S2* | mean pairwise ‖X_i − X_j‖ over all 9 branches ÷ ‖Z_ref‖_F | dimensionless |
| *S3* | mean pairwise ‖X_i − X_j‖ over the 6 illegal ÷ RMS ‖X_illegal‖ | [0, √2] |
| *S4* | mean over the 6 illegal of δ̂ = ‖D‖/‖z_ref‖ at *r* = 24 | dimensionless |

*S3* = 0 means the six illegal branches displace identically; √2 is six
equal-length mutually orthogonal displacements.

## Why *S3* decides

1. **It is what "readable" means here.** The fork colors branches by illegal
   category; the six categories separating from *each other* is what makes those
   colors carry information.
2. **It is scale-free, and so is the figure.** The plane is refit per game and per
   space, so overall divergence magnitude is normalised away by the drawing —
   only directions survive into the picture. *S2* and *S4* carry magnitude and
   would re-select the largest-δ̂ game for a property the panel cannot show.
3. **It cannot be a cherry-pick, and *S1* can.** *S1* is the study's actual
   scientific contrast, and the aggregate finding is that an illegal move
   displaces layer 5 *less* than a legal alternative (ratio 0.978 [0.972, 0.985]).
   Maximising *S1* selects the one game that most contradicts the paper's own
   headline and dresses it as an illustration. No claim rests on *S3*.

**The honest cost.** Game 662 sits at the 5.8th percentile of *S1*: its illegal
branches are *less* separated from its legal controls than in 94% of games. That
is the conservative direction — the panel under-states the illegal-vs-legal
contrast rather than over-stating it — but it must be stated. The
anti-correlation is partly mechanical: six illegal branches spraying in mutually
different directions have a centroid near zero and a large within-group spread,
which is small *S1* by construction.

## The criteria disagree, sharply

Spearman ρ over all 1550 complete games:

| | *S1* | *S2* | *S3* | *S4* |
|---|---|---|---|---|
| *S1* | 1.000 | −0.025 | **−0.317** | −0.054 |
| *S2* | −0.025 | 1.000 | 0.025 | **+0.328** |
| *S3* | −0.317 | 0.025 | 1.000 | **−0.442** |
| *S4* | −0.054 | 0.328 | −0.442 | 1.000 |

**The four top-10 lists are disjoint — no game appears in two of them.** A finding
about the data, not a defect in the criteria:

- **ρ(*S3*, *S4*) = −0.442.** The games whose branches diverge *furthest* are the
  games whose illegal categories are *least* differentiated from each other: large
  δ̂ means all branches moving together along one shared direction. A "biggest
  divergence" pick would be the worst pick for a category-colored figure.
- **ρ(*S1*, *S2*) = −0.025.** Overall branch spread says nothing about whether
  illegal and legal separate.

There is no single clearest game; there is a clearest game *per question*, and the
caption has to name which question it answered.

### Top 10 under each criterion

*S3* — category differentiation (the recommendation):

| game | *t0* | *S1* | *S2* | *S3* | *S4* |
|---|---|---|---|---|---|
| **662** | **22** | 0.8037 | 0.4756 | **1.0802** | 0.2785 |
| 755 | 15 | 0.9124 | 0.4766 | 1.0472 | 0.3833 |
| 1235 | 20 | 1.1325 | 0.5259 | 1.0416 | 0.3364 |
| 1532 | 17 | 0.7565 | 0.4405 | 1.0330 | 0.2753 |
| 1175 | 10 | 1.0484 | 0.4637 | 1.0218 | 0.2585 |
| 1033 | 18 | 0.7946 | 0.4521 | 1.0162 | 0.4099 |
| 1294 | 29 | 0.9627 | 0.4273 | 1.0104 | 0.2394 |
| 1499 | 34 | 0.9204 | 0.4716 | 1.0090 | 0.2554 |
| 1629 | 14 | 0.9441 | 0.4781 | 1.0040 | 0.3441 |
| 1269 | 29 | 0.8829 | 0.4511 | 1.0000 | 0.3241 |

*S1* — illegal vs. control:

| game | *t0* | *S1* | *S2* | *S3* | *S4* |
|---|---|---|---|---|---|
| 1487 | 22 | 2.0195 | 0.4716 | 0.6970 | 0.3606 |
| 101 | 11 | 1.8009 | 0.5089 | 0.4791 | 0.4378 |
| 1094 | 29 | 1.7573 | 0.4576 | 0.7157 | 0.3468 |
| 456 | 16 | 1.7193 | 0.4271 | 0.3766 | 0.6856 |
| 1378 | 13 | 1.6966 | 0.4651 | 0.6367 | 0.4824 |
| 182 | 17 | 1.6956 | 0.5267 | 0.7515 | 0.3351 |
| 278 | 13 | 1.6882 | 0.5140 | 0.5071 | 0.4673 |
| 1978 | 13 | 1.6402 | 0.5561 | 0.5008 | 0.7623 |
| 735 | 20 | 1.6177 | 0.4964 | 0.7351 | 0.3474 |
| 667 | 27 | 1.6095 | 0.4515 | 0.7363 | 0.3308 |

*S2* — overall branch distinctness:

| game | *t0* | *S1* | *S2* | *S3* | *S4* |
|---|---|---|---|---|---|
| 1377 | 12 | 0.8187 | 0.6048 | 0.7606 | 0.6815 |
| 365 | 25 | 1.0005 | 0.5785 | 0.7782 | 0.4072 |
| 154 | 14 | 1.0370 | 0.5782 | 0.7082 | 0.5134 |
| 1525 | 10 | 1.1214 | 0.5780 | 0.5988 | 0.5469 |
| 1259 | 19 | 0.8700 | 0.5771 | 0.7148 | 0.4870 |
| 681 | 16 | 1.3463 | 0.5763 | 0.6256 | 0.5948 |
| 1231 | 16 | 1.0602 | 0.5741 | 0.8859 | 0.5692 |
| 65 | 25 | 1.1590 | 0.5728 | 0.6193 | 0.4439 |
| 862 | 22 | 1.0720 | 0.5724 | 0.7943 | 0.4331 |
| 443 | 28 | 1.0485 | 0.5721 | 0.7081 | 0.5413 |

*S4* — δ̂ at *r* = 24:

| game | *t0* | *S1* | *S2* | *S3* | *S4* |
|---|---|---|---|---|---|
| 1842 | 27 | 0.9148 | 0.4566 | 0.5852 | 1.0346 |
| 702 | 12 | 0.8827 | 0.5688 | 0.5383 | 0.9040 |
| 1250 | 10 | 1.2649 | 0.4869 | 0.4853 | 0.8811 |
| 982 | 17 | 1.1368 | 0.5101 | 0.7493 | 0.8761 |
| 204 | 14 | 1.1570 | 0.4822 | 0.6453 | 0.8742 |
| 554 | 14 | 0.8028 | 0.4838 | 0.7114 | 0.8602 |
| 843 | 28 | 1.0455 | 0.4439 | 0.5306 | 0.8573 |
| 1196 | 31 | 0.8710 | 0.4896 | 0.7320 | 0.8567 |
| 1208 | 18 | 0.7759 | 0.5316 | 0.6829 | 0.8534 |
| 955 | 15 | 0.9005 | 0.5090 | 0.5708 | 0.8518 |

Population spread, all 1550 games:

| key | min | q25 | median | q75 | max |
|---|---|---|---|---|---|
| *S1* | 0.6345 | 0.9177 | 1.0133 | 1.1143 | 2.0195 |
| *S2* | 0.3149 | 0.4312 | 0.4612 | 0.4914 | 0.6048 |
| *S3* | 0.3240 | 0.6251 | 0.7045 | 0.7844 | 1.0802 |
| *S4* | 0.1659 | 0.3721 | 0.4454 | 0.5301 | 1.0346 |

## Transfer check: the choice is not raw-L5-specific

Game 662 and game 0, plus a random sample of 300 complete games (seed 0),
rescored with *S3* computed inside three SAE latent spaces from three families:

| space | game 662's rank | its *S3* there | game 0's rank | Spearman ρ vs raw L5 |
|---|---|---|---|---|
| Standard (L1 0.059) | **1 / 302** | 1.1098 | 174 / 302 | 0.966 |
| TopK (k=80) | **1 / 302** | 1.1068 | 147 / 302 | 0.935 |
| SpaDE (λ=1e-3, γ=1e-4) | **1 / 302** | 1.0859 | 168 / 302 | 0.990 |

Zero rank shift: the game selected in raw layer 5 is also top-ranked in every SAE
space tested, and the whole ordering transfers at ρ = 0.935–0.990. The choice is a
property of the branch geometry, not of the selecting space — which is what makes
it safe in a figure whose point is to compare spaces.

## *t0*

Game 662's *t0* = 22 is at the 56.8th percentile (min 10, median 21, max 34); the
incumbent's *t0* = 10 was the minimum. This matters beyond typicality: *t0* is
round-robin over [10, 34] and late-*t0* games are conditioned on pass-free,
non-forfeit sources, so both extremes of the range are unrepresentative on more
than one axis.

## Caveats

- **n = 1 stays n = 1.** A maximum over 1550 games is a maximum over 1550 games.
  Nothing about game 662 supports a claim; the parent README's aggregates do.
- **Game 662 is an *S1* low-outlier** (5.8th percentile) — conservative direction,
  but a reader comparing this panel to the study's *S1* numbers should know the
  panel is not representative of that contrast. A game that is differentiated
  *and* not an *S1* outlier: **game 1235** (*S3* rank 3, *S1* rank 351/1550,
  *t0*=20), recorded as an observation, not as the pre-registered pick.
- **The criterion is one of several defensible ones**, and they disagree (disjoint
  top-10s). The choice of *S3* is argued, not measured.
- **Three of eight dictionaries** in the transfer check, not all eight. Seed 42
  only, like the rest of the study.

## Outputs

| file | what |
|---|---|
| `results/ranking.csv` | one row per complete game: *t0*, all four scores, rank under each |
| `results/transfer_check.csv` | the same scores in three SAE spaces, on 302 sampled games |
| `results/summary.txt` | top-10 tables, Spearman agreement, *t0* placement, transfer ranks |
| `figures/selection_disclosure` | the population the pick came from, the criteria disagreeing, the transfer check |
| `results/rule_ranking.csv` | the rule-level sweep below: one row per game per space |
| `results/rule_summary.txt` | its anchors, population spreads, null, and top-10s |

## Consuming it

`game_trajectory/make_figures.py` still hardcodes `complete[:n_games]`; it needs a
`--game` flag before it can draw 662. That change is not made here.

---

# The rule-level question: R1/R2/R3 in SpaDE latents

A figure that draws **one mean per constraint rule** asks something the sweep
above does not. R1 placement has 2 member categories, R2 bracketing 3, R3 turn 1,
so three lines stand in for six trajectories, and the question is whether those
three lines are separable — in the **distance-SpaDE latent space**, since that is
the architecture the paper argues for. `score_rules.py` sweeps all 1550 games in
SpaDE and, as a reference, in raw layer 5.

Field and window are unchanged from above: `D(t) = f_b(t) − f_ref(t)` over the 13
positions *r* = 0, 2, …, 24 at *t0*'s parity, at full width (8192 in SpaDE, 512
raw), never a projection. A rule mean is formed **per member and averaged last**;
the pooled within-rule RMS radius carries **dof = Σ(n_r − 1) = 3**, so R3's single
member contributes zero to the sum *and* zero to the dof rather than a silent inf.

| key | definition | units |
|---|---|---|
| `sep_min` | **min** over rule pairs ‖M_i − M_j‖ ÷ pooled within-rule RMS radius | pooled SDs |
| `sep_mean` | mean over rule pairs, same denominator | pooled SDs |
| `cone_min` / `cone_mean` | the same numerators ÷ RMS ‖M_r‖ | [0, √2] |
| `null_beats` | fraction of the 59 other 2/3/1 partitions of the same six branches that the true R1/R2/R3 grouping beats on `sep_min` | [0, 1] |

`sep_min` decides, because a reader has to tell **all three** lines apart and the
closest pair is the binding constraint; `sep_mean` hides a collapsed pair behind a
distant third.

## Answer: game 1558, *t0* = 18

| | SpaDE | where it sits among the 1550 |
|---|---|---|
| `sep_min` | **2.070** | **1 / 1550** (median 1.007) |
| `sep_mean` | 2.205 | 3 / 1550 (median 1.204) |
| `cone_min` | 0.840 | 6 / 1550 (median 0.527) |
| `cone_mean` | 0.895 | 33 / 1550 (median 0.636) |
| `null_beats` | 1.00 | beats all 59 alternative partitions |
| *S1* (from `score_games.py`) | 1.044 | 656 / 1550 — **passes** the at-or-above-median-*S1* guard |
| `spread_ratio` | 0.605 | **0.1st percentile** — see the caveat |
| *t0* | 18 | 40th percentile |

The two anchors, same sweep:

| game | `sep_min` (rank) | `cone_min` (rank) | `cone_mean` (rank) | `spread_ratio` (pct) | `null_beats` |
|---|---|---|---|---|---|
| **1558** (this pick) | **2.070** (1) | 0.840 (6) | 0.895 (33) | 0.605 (0.1) | 1.00 |
| **1235** (`FIGURE_GAME`) | 1.121 (391) | 0.790 (25) | 0.968 (**9**) | 0.904 (33.8) | 0.86 |
| **662** (the *S3* argmax) | 0.946 (1010) | 0.851 (**3**) | 0.963 (10) | 1.031 (79.4) | 0.66 |

The incumbent is not a bad rule game: it is 25th and 9th on the *direction*
criteria and only middling on `sep_min`. Game 662 is the sharpest split of all
three by direction and a below-median one by `sep_min` — the same property that
made it the *S3* argmax. Six categories spraying in mutually different directions
give large mutual angles between rule means **and** a large within-rule scatter,
so the two criteria pull apart.

## Does SpaDE separate the rules at all?

**Yes, but weakly, and no better than raw layer 5.**

- The taxonomy is not arbitrary: the true R1/R2/R3 grouping beats a median of
  **0.86** of the 59 shape-matched partitions (chance 0.50), is the argmax in 11%
  of games, and falls below chance in 11%. The exact 60-partition null is the test
  that matters — six trajectories split 2/3/1 look "separated" for free.
- The margin is small in absolute terms. Median `sep_min` is **1.01 pooled SDs**:
  in the typical game the closest two rule means are one within-rule SD apart, so
  the members of one rule reach the other rule's mean. Median `cone_min` **0.53**
  against a √2 ceiling.
- **SpaDE is not the reason.** Spearman(`sep_min` in SpaDE, in raw layer 5) =
  **0.989** over all 1550 games; population medians 1.007 (SpaDE) against 1.001
  (raw); the SpaDE argmax is also the raw argmax. On direction SpaDE is ~5% better
  (`cone_min` median 0.527 vs 0.502) — a real but marginal edge. Rule separation
  here is a property of the branch geometry, not of the dictionary.
- **R1 and R2 are the pair that collides**, in 69.5% of games. Pairs involving R3
  are ~20% wider (median separation 1.31 and 1.23 SDs against R1–R2's 1.05). Part
  of that is mechanical: R3's "mean" is one branch and is not shrunk toward the
  illegal centroid the way a 2- or 3-member mean is. A figure should not read R3's
  distinctness as evidence that turn confusion is represented apart.

## The caveat that has to go in the caption

Unlike *S3*, **this criterion is not independent of a claim the study makes.**
`rule_groups.spread_ratio` (within-rule ÷ between-rule mean pairwise distance,
population median 0.945, above 1 in 30%) is a reported result, and
Spearman(`sep_min`, `spread_ratio`) = **−0.82**. Game 1558 sits at the **0.1st
percentile** of `spread_ratio` (0.605): it is very nearly the tightest rule
grouping in the entire population. A figure drawn on it shows the grouping working
far better than it typically does, and saying so is not optional.

Games that are strong on `sep_min` without being `spread_ratio` outliers, above
median on *S1*, and beating every null partition — recorded as observations, not
as the pre-registered pick:

| game | *t0* | `sep_min` (rank) | `cone_min` (rank) | `spread_ratio` (pct) | *S1* rank |
|---|---|---|---|---|---|
| 706 | 16 | 1.415 (45) | 0.789 (28) | 0.834 (14.9) | 563 |
| 1202 | 12 | 1.424 (42) | 0.759 (46) | 0.813 (12.0) | 554 |
| 1285 | 20 | 1.456 (30) | 0.585 (484) | 0.804 (10.5) | 176 |

## Caveats

- **n = 1 stays n = 1.** An argmax over 1550 games supports no claim; the
  population numbers above do.
- **One dictionary.** Seed-42 `SpaDE-lam1e-03-gam1e-04`, the non-degenerate
  distance-SpaDE checkpoint (`SpaDETrainer-othello` is excluded repo-wide). The
  raw-L5 column is the control, not a second dictionary.
- **Displacement field only.** The R^21 curvature descriptor is a different field
  with a different population (*t0* ≥ 18, 60 games); `rule_groups.r21_ratios`
  already covers the grouping statistic there, and nothing here was recomputed in it.
