# Red team: legality_probe

Every number below comes from my own code path in `redteam/`:
own rule classifier (`labels_indep.py`, written from `docs/othello_rules.md`), own TPR
probe module and fitting loop, own metrics, own bootstrap.
`src/spade/legality_probe/` is imported in exactly two places, both times because
reproducing it is the point: `checks_cpu.labels_at_ply_gap` and `seed_probe.py`. The
shipped CSVs are compared against my arrays, never used to produce them.

Inputs: `study_dir("legality")/{acts,fit_set,latents}`, `study_dir("divergence")/branches`.

## Verdict

**The pipeline is correct. The study measures board reconstruction, and one of its two
headline readings of that is misattributed.**

The data is clean to a standard the precedent study did not have to meet. My independent
labeller agrees with the shipped fit set on **0 of 640000 squares** in disagreement,
prefix invariance holds bitwise in all **11** spaces at the code level *and* at the
prediction level over all 2000 games, the 85/66000 forfeit mask reproduces exactly, and
my independently written probe reproduces `legality_probe.fit` **to the last printed
digit** at matched seeds (random_proj seed 0: 0.917862 both loops).

What does not survive is that this is a study of legality. Decoding the 3-class board arm
and computing legality from the predicted board with a rules engine beats the 4-way
legality probe in **11 of 11 spaces** — on raw, 0.9838 against 0.9382 — and the two agree
across spaces at pearson **0.9981**. A 2-class *occupancy* probe, which knows nothing about
rules or about whose turn it is, orders the ten encoded spaces at spearman **+0.976** with
the legality arm. There is no per-space legality result here that is not a per-space board
result, and `analysis/bracket_flip/results/sae_arm.csv` published that.

The one between-space finding that no information-preservation model reproduces is the
SpaDE collapse, and the coordinator is right that `linear_r2` kills the simple version of
attack 1: SpaDE-lam has linear R² **0.9725** at legality **0.4287**, while random_topk has
linear R² **0.5786** at legality **0.7739**. Information is preserved and legality is not.
But the collapse is not about turn-taking. In the SpaDE spaces the 4-way probe predicts
`legal` on **1.9%** of squares (prevalence 14.8%) and `R3` on **0.08%** (prevalence 9.6%):
it has degenerated onto two of four classes. The board arm shows the mechanism without any
rule taxonomy — mover-relative colour is gone (`mine` recall **0.384**, colour accuracy
given both occupied **0.737** against raw's 0.986) while occupancy survives at 0.830.
`legal` and R3 are the two classes that relabel under a frame flip (0.703 and 1.0 in
`frame_check.json`); R1 and R2 relabel at 0.0 and are structurally immune. "R1 and R2
survive, R3 dies" is the signature of any colour failure, not evidence about a turn rule.

## Contract keys, re-derived

| key | shipped | mine |
|---|---|---|
| branch rows / source games | 17527 / 2000 | **17527 / 2000** |
| illegal branch rows | — | **11527** |
| window positions | 66000 | **66000** |
| masked as forfeit | 85 | **85**, over 68 games |
| fit-set rows | 10000 | **10000**, 0 label mismatches |
| prefix bitwise, 11 spaces | True | **True**, code *and* prediction, every span |
| raw legality / board macro-F1 | 0.9379 / 0.9820 | **0.9382 / 0.9822** |

Held-out macro-F1, and the four quantities the study does not report. `board_oracle` is
the 3-class board arm's argmax run through my rules engine and scored on the 4 legality
classes — same alphabet, same metric, so its gap to `legality` carries no class-count
confound. `occupancy` is a 2-class {empty, occupied} probe. `linear_r2` is the best
held-out R² over a ridge grid for a least-squares map code → standardized raw.

| space | L0 | legality (shipped) | legality (mine, seed 0) | seed spread | board_oracle | occupancy | linear_r2 | frameless |
|---|---|---|---|---|---|---|---|---|
| GatedAnnealTrainer | 36.2 | 0.8854 | 0.8844 | 0.0018 | 0.9381 | 0.9664 | 0.8456 | 0.9116 |
| GatedSAETrainer | 57.4 | 0.9119 | 0.9110 | 0.0033 | 0.9552 | 0.9803 | 0.8865 | 0.9304 |
| TopKTrainer | 80.0 | 0.9235 | 0.9249 | 0.0026 | 0.9717 | 0.9886 | 0.9242 | 0.9356 |
| `random_topk` | 80.0 | 0.7742 | 0.7739 | 0.0009 | 0.8224 | 0.9196 | 0.5786 | 0.8272 |
| BatchTopKTrainer | 85.0 | 0.9259 | 0.9255 | 0.0022 | 0.9706 | 0.9888 | 0.9280 | 0.9387 |
| CosineSpaDETrainer | 178.6 | 0.4896 | 0.4836 | 0.0041 | 0.6045 | 0.8469 | 0.8796 | 0.7030 |
| PAnnealTrainer | 390.8 | 0.9288 | 0.9305 | 0.0004 | 0.9787 | 0.9933 | 0.9905 | 0.9423 |
| raw | 512 | 0.9379 | 0.9382 | 0.0001 | **0.9838** | 0.9951 | 1.0000 | 0.9441 |
| StandardTrainer | 1207.2 | 0.9466 | 0.9466 | 0.0022 | 0.9789 | 0.9910 | 0.9841 | 0.9551 |
| SpaDE-lam1e-03-gam1e-04 | 1821.7 | 0.4297 | 0.4287 | 0.0076 | 0.5217 | 0.8299 | 0.9725 | 0.6749 |
| `random_proj` | 4104.7 | 0.9036 | **0.9251** | 0.0037 | 0.9719 | 0.9864 | 0.9776 | 0.9358 |

## F1. The legality arm carries nothing the board arm does not. *(kills the framing)*

**DEAD.**

The board oracle beats the direct legality probe in **11/11** spaces, by 0.0323 to 0.1210
(mean 0.0568). Over the eleven spaces, pearson(oracle, legality) = **0.9981**, spearman
+0.964. The 2-class occupancy probe alone gives spearman **+0.976** with legality macro-F1
over the ten encoded spaces.

The residual (legality − oracle) is not literally noise — excluding the SpaDE pair it spans
0.0214, against a maximum seed spread of 0.0037 — but it does not *order* anything: it is
negative everywhere, roughly constant, and its extremes are StandardTrainer (−0.0323) and
GatedAnneal (−0.0537), which is the same ordering the board arm already gives. The two
spaces where the residual is large (SpaDE −0.0930, Cosine −0.1210) are large because the
direct probe has collapsed two classes, which is F3, not a legality result.

The class-count control the brief asked for is built into this comparison: the oracle
scores 4 classes with the same `macro_f1`, so unequal difficulty between a 4-class and a
3-class target cannot manufacture the residual. Independently, the frame-invariant 3-class
`legality_frameless` arm ({legal ∪ R3, R1, R2}) tracks the 4-class arm at +0.006…+0.053 in
every space except the SpaDE pair.

Nothing here says the pipeline is wrong. It says the legality framing adds no measurement.
Any sentence of the form "space X preserves legality better than space Y" should be
"space X reconstructs the board better than space Y", which `sae_arm.csv` reports.

## F2. Between-space ranking, and whether the nulls beat the dictionaries. *(WEAKENED)*

**WEAKENED**, in both directions.

*The ordering among ordinary dictionaries is L0 and nothing else.* Over the six ordinary
trained SAEs (all but the SpaDE pair), legality macro-F1 is a **perfectly monotone**
function of L0: spearman **+1.000**, n=6 (p = 1/720). Over the eight non-SpaDE encoded
spaces, spearman(linear_r2, legality) = **+0.952** and spearman(occupancy, legality) =
**+0.952**. Within that set the study is an information-preservation measurement.

*But the whole study does not reduce to it.* SpaDE-lam preserves more linearly recoverable
information than any dictionary except PAnneal and Standard (linear R² **0.9725**) and
scores **0.4287**; random_topk preserves the least (**0.5786**) and scores **0.7739**.
Across all ten encoded spaces spearman(log L0, legality) is **+0.207** and
spearman(linear_r2, legality) **+0.673**. No information model orders this roster.

*Training beats the sparsity-matched null, decisively, where one exists.* At k=80,
TopK 0.9249 and BatchTopK 0.9255 against `random_topk` **0.7739** — +0.151 and +0.152.
Fitting the log-L0 trend of the six ordinary SAEs and extrapolating, `random_topk` sits
**−0.139** below it and `random_proj` **−0.041** below it. Both nulls underperform trained
dictionaries at their own sparsity.

*`random_proj` is not a null for anything in the roster.* Its L0 is 4105 — 48× TopK,
113× GatedAnneal, 8× the raw residual's dimension. "A null beats GatedAnneal by 0.018"
compares across a 113× sparsity gap. Six of the eight dictionaries (L0 32.5 … 1808) have
no matched null in the study at all; there are two nulls, at L0 80 and 4105.

*And the 0.904 does not replicate.* The encodings are fine (cached vs fresh: max |Δ| 9.8e-4,
fp16 rounding). The fit is not: `random_proj` has a seed sd of **0.0094** over six seeds
(0.8999 … 0.9255, mean 0.9158), against raw's 0.0017 and TopK's 0.0043. 0.9036 is the
shipped default seed 1111, near the bottom of its own distribution, and the reported CI
(±0.0036, over 40 games) is 2.6× too narrow to cover it. At the six-seed mean, `random_proj`
0.916 still beats GatedAnneal 0.885 and still sits 0.022 below raw — the direction holds,
the number does not.

`drop_agreement_with_raw`, the metric PROCESS calls "the version to trust", is an L0
readout: spearman(log L0, drop) = **−0.934** over the eight non-SpaDE encoded spaces,
−0.839 over all ten. It also has the floor it is said not to have — CosineSpaDE and SpaDE
start from r<0 agreement of 0.686 and 0.663, so there is 0.31–0.34 less room to fall than
for the spaces at 0.94–0.96.

## F3. `p_pred_correct_rule` at r ≥ 0 is scored against a target that is wrong. *(changes how a panel is read)*

**DEAD**, for any r ≥ 0 reading of that metric.

`readout.readout_rows` compares the prediction at the branch square against
`w.rule_class`, the class the branch's move carries on the **pre-move** board, at every
offset. Re-deriving the reference game's labels myself:

| category | n | truth at r=−1 equals `rule_class` | truth at r=0 equals `rule_class` |
|---|---|---|---|
| occupied_own | 2000 | **1.0000** | 1.0000 |
| occupied_opp | 2000 | **1.0000** | 1.0000 |
| empty_detached | 1951 | **1.0000** | 0.9667 |
| empty_own_only | 1630 | **1.0000** | 0.9620 |
| empty_no_bracket | 1949 | **1.0000** | 0.7245 |
| opponent_only | 1997 | **1.0000** | **0.0045** |
| all illegal | 11527 | **1.0000** | 0.7699 |

At r = −1 the metric is exact — 11527/11527 — so the panel the study's claim rests on is
correctly targeted, and that is worth stating as a positive. At r = 0 the target is right
for 77.0% of illegal branches and for **0.45%** of `opponent_only`. PROCESS reads the r=0
R3 panel as "the illegal move has been played and every space reads that square as
occupied, so R3 recall is ~0 everywhere". That reading is unavailable: recall is ~0 there
because the label being scored against is wrong 99.5% of the time, for reasons that have
nothing to do with any space. The r ≥ 0 columns of `p_pred_correct_rule` are not recall
and should not be plotted as if they were.

## F4. SpaDE's R3 collapse is real, and it is not about turn. *(changes a conclusion)*

**WEAKENED.**

Is it selective? Two explicit models, both fitted on the nine non-SpaDE spaces:

| model | fit | SpaDE-lam predicted | actual | Cosine predicted | actual |
|---|---|---|---|---|---|
| R3 ~ the space's own macro-F1 at r=−1 | slope +2.100, R² 0.993 | **−0.261** | 0.0060 | −0.118 | 0.0481 |
| R3 ~ the space's own R2 recall at r=−1 | slope +2.470, R² 0.936 | **+0.618** | 0.0060 | +0.558 | 0.0481 |

The two models disagree by 0.88 because the first conditions on a macro-F1 that already
contains the collapse. Conditioning on the frame-invariant R2 gives a shortfall of
**−0.612** — selective by any reading. Conditioning on overall degradation gives a
prediction *below zero* — not selective at all. Neither is the honest answer on its own.

What is: **`legal` dies with R3, and `legal` is not rare.** Fit-set per-class F1 —

| space | legal (14.8%) | R1 (53.9%) | R2 (21.7%) | R3 (9.6%) |
|---|---|---|---|---|
| raw | 0.9779 | 0.9918 | 0.9326 | 0.8493 |
| random_topk | 0.8100 | 0.9225 | 0.8294 | 0.5349 |
| CosineSpaDE | **0.2514** | 0.8398 | 0.7745 | **0.0929** |
| SpaDE-lam | **0.1185** | 0.8264 | 0.7628 | **0.0113** |

The probe predicts `legal` on **1.92%** of held-out squares in SpaDE-lam and **3.92%** in
Cosine (prevalence 14.8%), and R3 on **0.08%** and **0.56%** (prevalence 9.6%). Every other
space predicts both at their prevalence (0.133–0.150, 0.072–0.093). In those two spaces the
4-way probe has collapsed onto {R1, R2}.

In the read-out this is visible without any rule: at r = −1, on the shared pre-move board,
the probe calls the reference game's **own next move** legal at **0.0830** (SpaDE) and
**0.1875** (Cosine), against **0.9765** for raw. A turn failure would call a legal move R3.
It does not: at the `opponent_only` square the SpaDE probe predicts R1 (= occupied) **54.6%**
of the time and R2 42.9%, at a square that is empty.

The mechanism is in the board arm, where there is no legality alphabet:

| space | board `mine` recall | colour accuracy given both occupied | occupancy macro-F1 |
|---|---|---|---|
| raw | 0.9720 | 0.9859 | 0.9951 |
| random_topk | 0.7848 | 0.8924 | 0.9196 |
| CosineSpaDE | 0.4771 | 0.7954 | 0.8469 |
| SpaDE-lam | **0.3839** | **0.7366** | 0.8299 |

SpaDE-lam sends `mine` to `theirs` 32.7% of the time and to `empty` 29.0%. Merging the two
frame-sensitive classes (`legality_frameless` = {legal ∪ R3, R1, R2}) recovers **+0.246**
for SpaDE and **+0.220** for Cosine, against +0.006 … +0.053 for every other space.

Two structural problems make the R1/R2-vs-R3 contrast unable to carry the claim:

- **R1 and R2 cannot fail this way.** `frame_check.json` reports that a frame flip relabels
  0.0 of R1 and 0.0 of R2, 0.703 of `legal` and 1.0 of R3, because `occupied_own`/
  `occupied_opp` and `empty_own_only`/`empty_no_bracket` swap *within* their rule. Any
  colour failure produces "R1 and R2 survive, R3 dies" for free.
- **Rarity is perfectly confounded.** Prevalence orders R3 (9.6%) < legal (14.8%) <
  R2 (21.7%) < R1 (53.9%) and the damage ordering is exactly that, in every space
  including the untrained `random_topk`. The panel cannot separate "rare classes go first"
  from "frame-sensitive classes go first".

Not a sparsemax artefact: CosineSpaDE and SpaDE-lam use different sparsity operators and
show the same pattern, and the pattern appears in the 3-class board probe, which puts no
mass on a legality simplex. Also worth stating plainly, since RULE_MEMBERS gives R3 one
member: R3 is `opponent_only` and nothing else, so "the R3 rule" and "the `opponent_only`
category" are the same 1997 branches with no internal replication.

The claim that survives, and it is a good one: **the SpaDE bottleneck destroys the
mover-relative colour of the board.** State it on the board arm, where it needs no rule
taxonomy and no 1-member rule group.

## F5. r < 0 is bitwise exact, is one measurement, and is not a control for the violation. *(SURVIVES / WEAKENED)*

**SURVIVES** as a bitwise-exact paired baseline. **WEAKENED** as "the excess at r = 0 is
what the violation costs".

Verified in my own encode, span by span, all 2000 games, all 11 spaces: the r < 0 columns
are bitwise identical across a game's branches **at the code level and at the prediction
level**, in every space including both nulls. No exceptions.

That has a consequence the study reports around. Because it is bitwise, every one of the
13 branch sets yields the *same* r < 0 number: the max spread of `macro_f1` at r = −1
across the 13 sets is **0.00173** in the worst space, and that residue is only which games
carry which category. The r<0 baseline is one per-space scalar printed 13 times with n
between 1603 and 2000. Nothing about a category can be read off it, and the per-category
r<0 rows in `readout.csv` are not 13 measurements.

Is the variance understated? No. The bootstrap unit is the source game, so duplicated rows
inside a game cannot buy precision. Measured design effect (game CI ÷ row CI)² over the
illegal-pooled rows is **1.41 … 3.97** — the game bootstrap is 1.2–2.0× wider than a naive
row bootstrap, exactly as it should be. Two smaller things:

- The r<0 point estimate is **branch-weighted**, not game-weighted: identical rows are
  summed, so games are weighted by how many categories they admit. Branch- vs game-weighted
  r<0 macro-F1 differs by +0.0034 (raw) to +0.0084 (SpaDE), same sign in every space.
- The r<0 block pools counts over eight *different* plies (t0−8 … t0−1) whose class balance
  drifts, so it is not the r = −1 baseline it is sometimes read as. Raw: 0.9469 at r=−8,
  0.9328 at r=−1, 0.9403 pooled.

What r < 0 is not a control for is the illegal move. The study already contains the right
null and does not use it: the **reference role** is the reference game itself, so its
r<0 → r=0 drop contains no violation at all.

| space | drop F1 reference | drop F1 control_legal | drop F1 illegal | drop agree reference | drop agree illegal |
|---|---|---|---|---|---|
| raw | 0.0076 | 0.1513 | 0.1253 | 0.0000 | 0.0000 |
| PAnneal | 0.0080 | 0.1491 | 0.1430 | 0.0015 | 0.0392 |
| BatchTopK | 0.0101 | 0.1519 | 0.1515 | 0.0048 | 0.0466 |
| GatedAnneal | 0.0133 | 0.1449 | 0.1657 | 0.0101 | 0.0788 |
| SpaDE-lam | 0.0127 | 0.0235 | 0.0460 | 0.0191 | 0.0471 |
| CosineSpaDE | 0.0211 | 0.0446 | 0.0647 | **0.0233** | 0.0479 |
| random_topk | 0.0267 | 0.1185 | 0.1276 | **0.0209** | 0.0646 |

Two readings follow. First, `drop_macro_f1` is not a violation effect at all in raw: the
**legal** control drops 0.1513 against illegal's 0.1253, the same inversion
`analysis/violation_divergence/RED_TEAM.md` F1 found, and for the same reason — a legal
alternative genuinely changes the board, so it disagrees with the reference-frame target
more than a near-no-op violation does. Second, `drop_agreement_with_raw` *is* a real
illegal-specific effect (control 0.0022–0.0238 against illegal 0.0308–0.0788, 5–10×), but
the drift term is not negligible where it matters: the reference role accounts for **49%**
of CosineSpaDE's illegal drop and **32%** of random_topk's. `excess_macro_f1` for
random_topk is +0.0023 while its reference-role drift excess over raw is +0.0191, eight
times larger and same-signed, so that entry's sign is not established. Subtract the
reference role before ranking spaces.

## F6. The bootstrap. *(SURVIVES, three caveats)*

**SURVIVES.**

The resampling unit is genuinely the source game everywhere it is claimed:
`resample_counts(n)` is called with n = 40 eval games in `fit_rows` and n = 2000 source
games in `readout_rows`, and the reported `n` column is games (40; 1603–2000), not rows.
The ±0.002 CIs on `drop_agreement_with_raw` are credible: implied per-game sd 0.028–0.047
on n = 2000 gives half-widths 0.0012–0.0020, my independent bootstrap (different seed,
2000 draws) reproduces every interval to ≤ 0.0004, and the design effect above shows the
game unit is the conservative one.

- One `lru_cache`d multinomial design (seed 0) is shared by every space and every metric.
  That is what makes `excess_macro_f1` and `macro_f1_minus_raw` paired, and it also makes
  the Monte Carlo error perfectly correlated: no two intervals in `readout.csv` are
  independent, and a table read as 11 independent estimates is being misread.
- The CI covers games only. It excludes probe-fit randomness, which for `random_proj` is
  **2.6× larger** (seed sd 0.0094 against a CI half-width of 0.0036). Every `probe_fit.csv`
  interval understates total uncertainty by that term, and the term is space-dependent.
- The `n` column is inconsistent inside one file. For the F1 curves it is
  `(counts.sum((1,2,3)) > 0).sum()`, computed across all 33 offsets at once, so it is
  constant in r and does not reflect the mask: at r = +24, 40 of 2000 games are masked and
  n still reads 2000. `series()`-based metrics do report n per r.

## F7. The two known open items, and the label pipeline. *(SURVIVES)*

**SURVIVES**, both.

*The `labels_at_ply` workaround is correct.* Over every (game, offset) the read-out labels,
the shipped helper and my incremental replay agree on **65859/65859** positions with
**0** label-row disagreements. The helper returns None at **96** game×offsets, all at
r = +24: 80 from the `len(game) < t + 1` guard (the 80 games with t0 = 34) and the rest
forfeits. Of those, **56** are positions the incremental replay validly labels — so the
shared path would have cost 56 rows, not the 80 PROCESS quotes, the difference being games
that are forfeit-masked anyway.

*The forfeit mask is 85/66000, over 68 games,* reproduced exactly. Its distribution is not
uniform: 40 at r = +24, 13 at +23, 10 at +22, 6 at +21, and only **2** at r < 0 (one each
at r = −4 and −3). Masking is applied through `keep` to the F1 metrics and to nothing else;
`agreement_with_raw`, `p_pred_legal_at_played_square` and `p_pred_correct_rule` compare
predictions against predictions or against branch metadata and need no ground truth. That
is exactly the right set — with the separate F3 caveat that `p_pred_correct_rule`'s
metadata target is only valid at r = −1.

*The labels themselves.* Over all 10000 fit-set rows, against a classifier written from the
rules document with its own line scan and its own scatter: **0** square mismatches out of
640000, the move the record actually plays next is labelled legal at **10000/10000**, and
my legal set equals the engine's regular-move set at **10000/10000**. The
token-order → row-major scatter is right.

*The fitting loop.* My independently written probe reproduces `legality_probe.fit` **to the
last printed digit** at matched seeds, in three spaces × three seeds (random_proj seed 0:
0.917862 / 0.917862; seed 1: 0.925466 / 0.925466; seed 2: 0.899917 / 0.899917). The
`filler_classes` rebinding of upstream's by-value `BOARD_LABEL_OPTIONS` works as
documented. The 8000/2000 split lands on a game boundary.

## Not tested

An honest list, not an implied all-clear.

- **The fit set's disjointness from the read-out games** was not re-verified by exact token
  match. `fit_set.py` asserts it and `manifest.json` records it; I took it on faith.
- **The 200 fit-set games' provenance** — that they are drawn from the same stream at seed
  42, and that the ~1% forfeit-game rejection does not select boards. Not examined.
- **`legality_shuffled` and `legality_wrong_player`** were not refit. I used the shipped
  values for the frame argument in F4.
- **Layer 5 only.** Nothing here says whether any of this is layer-specific.
- **The activation extraction** (`extract_acts.py`, `study_dir("legality")/acts`) was taken
  as given; I re-encoded from it but did not re-run the model to produce it.
- **The Karvonen eval columns** in `space_quality.csv` (L0, variance explained,
  reconstruction F1) were used as reported, not recomputed.
- **`macro_f1_minus_raw`** and the full 33-offset curves were re-derived only at
  r = −1 and r = 0, not at every offset.
- **`board_oracle_legality` at more than one seed.** It uses the seed-0 board probe; the
  board arm's own seed spread is ≤0.0154 (SpaDE) and ≤0.0041 elsewhere, so the F1 verdict
  is safe by a wide margin, but the oracle number itself is one draw.
- **Whether the SpaDE colour collapse is a property of these two checkpoints or of the
  SpaDE objective.** One seed each, and `docs/claims_status.md` records that a previous
  SpaDE conclusion died with its checkpoint.

## Code

`analysis/legality_probe/redteam/`

    labels_indep.py    independent rule classifier, own line scan and scatter
    checks_cpu.py      labels, forfeit mask, the r=-1 identity, labels_at_ply
    rederive.py        own probe module + fitting loop + metrics + bootstrap, 11 spaces
    frame_probe.py     board confusion, occupancy arm, frame-invariant legality arm
    seed_probe.py      six seeds through my loop, four through the shipped one
    csv_attacks.py     attacks needing only the shipped CSVs
    verdict_tables.py  joins both and prints every table above
    results/           rt_fit.csv, rt_window.csv, rt_rules.csv, rt_frame.csv,
                       rt_seeds.csv, rt_checks.json, checks_cpu.json

`submit_rederive.sbatch`, `submit_frame_probe.sbatch`, `submit_seed_probe.sbatch` run the
three GPU steps; total 11 spaces × (4 probe fits + a full window decode) is under 20 min on
one H100.
