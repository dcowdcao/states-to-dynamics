# Legality probe: is a rule violation still readable after an SAE?

A 4-way bilinear (TPR) probe — role = 64 squares, filler = {legal, R1, R2, R3} — is fit
independently in each of nine representation spaces at OthelloGPT layer 5, then read out
over the violation-divergence branch set, which runs a game 0 → *t0* and injects one
illegal move at *t0*. The four classes collapse the seven-way taxonomy of
`src/spade/othello/taxonomy.py` onto the three constraint rules R1–R3 via `RULE_MEMBERS`.

The question is not whether legality is decodable. It is whether a sparse dictionary
that keeps the *board* also keeps the *rules*.

## Read RED_TEAM.md first

`RED_TEAM.md` re-derived every claim below through independent code, and reached its
verdicts *after* this file was written. Where the two disagree, the red team wins.

| claim | verdict |
|---|---|
| the legality probe measures something board-reading does not | **DEAD** — board argmax + the rules engine beats it in 11/11 spaces (raw 0.9838 vs 0.9382), pearson 0.9981, and the residual orders nothing |
| the untrained nulls sink the between-space story | **WEAKENED both ways** — among the 6 ordinary SAEs macro-F1 is perfectly monotone in L₀ (ρ = +1.000, n=6), but no information model orders the full roster (SpaDE R² 0.973 at F1 0.429, against `random_topk` R² 0.579 at 0.774). At matched k = 80, training wins by +0.151 |
| `random_proj` scores 0.9036 | **does not replicate** — seed sd 0.0094 over 6 seeds (0.900–0.926, mean 0.916); the shipped seed sits near its own floor and its ±0.0036 CI is 2.6× too narrow |
| SpaDE selectively destroys R3 (turn) | **WEAKENED, misattributed** — it is a colour/frame failure. `legal` (14.8%, not rare) dies with R3; rarity is perfectly confounded with frame sensitivity, and the same pattern appears in the 3-class board probe, so it is not a sparsemax artifact |
| r < 0 is a clean paired control | **SURVIVES bitwise** in all 11 spaces at code *and* prediction level; **WEAKENED as a control** — the `reference` role drifts and is never subtracted (49% of CosineSpaDE's agreement drop), and raw's *legal* control drops more than the illegal branches (0.1513 vs 0.1253), the same inversion `violation_divergence` found |
| bootstrap CIs | **SURVIVES** — the unit is genuinely the game; ±0.002 reproduced to ≤0.0004 |

One defect the red team found in the shipped CSV: `p_pred_correct_rule` is exact at
r = −1 (11527/11527), but at r ≥ 0 its target is right for only 77% of illegal branches
and **0.45%** of `opponent_only` — once the move is played the square is occupied, so
R3's truth becomes R1. **The r ≥ 0 columns of that metric are not recall.** Any read of
that metric must be taken at r = −1.

`RED_TEAM.md` also carries a "Not tested" section listing nine gaps, including fit-set
disjointness (taken on faith from the manifest) and whether the SpaDE colour collapse is
checkpoint-specific.

## Pipeline

```
python -m spade.legality_probe.fit_set      # 200 games x positions [5, 55) -> the fit set
python -m spade.legality_probe.labels       # taxonomy -> 4-way labels, per position
python -m spade.legality_probe.extract_acts # layer-5 resid for the fit set
python -m spade.legality_probe.frame_check  # -> results/frame_check.json
python -m spade.legality_probe.spaces       # encode both activation sets through 11 spaces
python -m spade.legality_probe.readout      # 4 probe arms + branch window
                                            #   -> results/{probe_fit,readout,space_quality}.csv
python analysis/legality_probe/make_figures.py            # the one figure
python analysis/legality_probe/make_figures.py --camera-ready
```

The two GPU stages are `scripts/slurm/studies/legality_probe/submit_legality_spaces.sbatch`
and `submit_legality_readout.sbatch`.

Output schemas, tidy long. `make_figures.py` reads `probe_fit.csv` only; `readout.csv` and
`space_quality.csv` are quoted throughout RED_TEAM.md but no figure reads them:

```
probe_fit.csv  space, d_space, target, n_classes, filler_dim, metric, class, value, lo, hi, n
               target in {legality, board, legality_wrong_player, legality_shuffled}
               metric in {macro_f1, accuracy, per_class_f1, per_class_recall}
               class "R1_no_start" is R1 over the 60 non-starting squares
readout.csv    space, category, rule, role, r, metric, mean, lo, hi, n
               category = a 7-way taxonomy name, "empty_pooled", "illegal_pooled",
                          or "" for a rule aggregate (rule set instead)
               role in {reference, control_legal, control_legal_2, illegal}
               per-r metrics: macro_f1, macro_f1_minus_raw, f1_{legal,R1,R2,R3,R1_no_start},
                              agreement_with_raw, p_pred_legal_at_played_square,
                              p_pred_correct_rule
               r = 0 scalars: drop_macro_f1, excess_macro_f1, drop_agreement_with_raw
space_quality.csv  space, d_space, l0_window, prefix_bitwise, n_window_positions,
                   probe_fingerprint, l0, variance_explained, reconstruction_f1
```

**Read every CSV with `keep_default_na=False`.** `category` and `rule` are empty strings on
the aggregate rows; parsed as NaN, every aggregate silently disappears.

`macro_f1_minus_raw` is the per-game **paired** difference from raw at the same offset;
its CI is not derivable from the two marginal CIs, which is why it is a column of the
CSV rather than arithmetic in the figure. Rule aggregates carry `category = ""` with
`rule` set, so a rule mean is never averaged from its members' marginal CIs.

Probe fit set: `analysis/linear_probe/activations/` (gitignored; a worktree points at the
main checkout's copy through `SPADE_LINEAR_PROBE_ACTIVATIONS`). Read-out set: the
violation-divergence branches under `$SPADE_DIVERGENCE_DIR`, window r ∈ [−8, 24] with
*t0* round-robin over [10, 34], so **every game contributes to every offset** and n is
constant across the whole x-axis.

## The nine spaces

`spade.spaces.representative_spaces()` filtered to `name == "raw" or entry.seed == 42`:
raw plus the eight seed-42 dictionaries. This is **not** `{"raw", *BASELINE_CONFIGS}`,
which silently drops the default SpaDE (`SpaDE-lam1e-03-gam1e-04`) and substitutes the
degenerate γ=1e-2 checkpoint excluded at `src/spade/spaces.py:31`;
`src/spade/bracket_flip/sae_arm.py:45` has exactly that bug, which is why
`analysis/bracket_flip/results/sae_arm.csv` carries `SpaDETrainer-othello` and no default
SpaDE.

`random_proj` and `random_topk` are untrained controls, and they are **not the same kind of
control**. `random_topk` has L0 80, exactly TopK's k, and is the sparsity-matched null the
k=80 dictionaries are argued against. `random_proj` is a ReLU of a dense 512 → 8192
projection at L0 4101, matched to nothing; it preserves nearly all of the input, so it
bounds what a probe read-out in an 8192-d space can demonstrate. Drawing them as one gray
dashed pair tells the reader something false, so `paper_style` gives them `NULL` and
`CEILING` styling respectively, and the figure labels each with what it is matched to.

## What this study cannot claim, stated before the results

**A macro-F1 level is not a result here.** Legality is a deterministic function of the
board, and the board already decodes at ~0.98, so a high legality macro-F1 in any space
is mostly downstream of board-reading rather than evidence that the model represents
rules. Three controls pin this down:

- Legality computed for the **wrong player** still reads at macro-F1 **0.9072**
  [0.9033, 0.9112] against 0.9379 [0.9343, 0.9416] for the right one (bilinear probe),
  reproduced independently at **0.9018** against 0.9388 by a plain linear probe
  (`results/frame_check.json`). A result that survives two probe families is worth more
  than either alone.
- A shuffled-label fit gives accuracy **0.5554** and macro-F1 **0.2856** — R1 alone is
  53.9% of all cells, so plain accuracy near 0.9 means nothing and macro-F1 is the only
  primary metric this study reports.
- An **untrained** 512 → 8192 random projection reads legality at **0.9036**
  [0.8988, 0.9084], which is within 0.004 of the wrong-player control and above three of
  the eight trained dictionaries. Whatever a high legality macro-F1 measures, a wide
  random map has it.

Everything below therefore rests on **differences**: between spaces at a fixed offset,
and between r < 0 and r = 0 within a space. Every figure is built so that a level cannot
be read off it as a finding.

**The class balance drifts with absolute ply**, which is why any read across r must be a
paired difference from raw at the same offset.
R1 is exactly (ply + 5)/64 by construction and R2 collapses late:

| ply | legal | R1 | R2 | R3 |
|---|---|---|---|---|
| 5 | 0.100 | 0.156 | 0.646 | 0.098 |
| 21 | 0.182 | 0.406 | 0.281 | 0.130 |
| 37 | 0.169 | 0.656 | 0.082 | 0.093 |
| 53 | 0.066 | 0.906 | 0.004 | 0.024 |

*t0* ranges over [10, 34] and r over [−8, 24], so absolute ply varies both across rows at
a fixed r and systematically with r. The drift is common to every space, so a per-game
paired difference against raw at the same r cancels it exactly. **The slope of an
absolute macro-F1 curve against r is not degradation.**

## Where the turn-dependence lives

Fraction of 4-way labels that change when legality is computed for the wrong player,
by correct-frame class: `legal` **0.703**, `R1` **0.000**, `R2` **0.000**, `R3` **1.000**;
0.1997 of all squares overall. The zeros are structural rather than lucky —
`occupied_own` ↔ `occupied_opp` and `empty_own_only` ↔ `empty_no_bracket` each swap
*within* their rule group, so R1 and R2 are frame-invariant by construction. All of the
taxonomy's turn-dependence sits on the `legal` ↔ `R3` boundary, which is the cleanest
available statement of what R3 is, and it explains why the wrong-player control's whole
deficit lands in `legal`.

R3 is **not** rare: 9.6% pooled, third-largest of the four classes, and a plain linear
probe reads it at 0.845 F1.

## Hyperparameters

From `spade.bracket_flip.tpr_probe`: ROLE_DIM 52, LR 1e-2, betas (0.9, 0.99),
weight decay 2.0, 120 epochs, batch 1024, seed 1111, 8000 training positions. Upstream's
wd=1e-2 was tuned on ~1.1M positions and overfits at this scale.

`FILLER_DIM = 3`, from a sweep on raw layer 5:

| d_f | macro-F1 |
|---|---|
| 2 | 0.8891 |
| **3** | **0.9374** |
| 4 | 0.9378 |

d_f=2 costs 4.8 points, concentrated in R3 and R2. d_f=4 buys 0.0004 and makes the
filler geometry uninformative, since four classes in four dimensions can be exactly
orthogonal. Three is the pin.

`train_tpr_probe.BOARD_LABEL_OPTIONS` must be set to 4 in that module's namespace before
the probe is constructed: it is imported by value from
`external/tpr_othello/src/train_probe.py:34`, and it sizes `filler_embeddings` and the
bias *and* scales the loss. Left at 3 you get three fillers and no error.

## The result in one paragraph

The between-space ranking is **not** explained by information loss, and it is not evidence
about rules either. It is a claim about code geometry. An untrained 512 → 8192 random
projection supports the legality probe at **0.9036**, above three of the eight seed-42
dictionaries, and takes the smallest hit of any space on the study's headline disruption
metric (`drop_agreement_with_raw` = 0.0308, best of eleven) — so a wide random map has
whatever a high legality macro-F1 measures. But the deflationary reading that would follow
("the probe just measures how much of the residual survives") is false: the red team's
`linear_r2`, how well raw is linearly recoverable from each space, is **0.973 for SpaDE**
against a legality macro-F1 of **0.430**, while `random_topk` has `linear_r2` **0.579** and
macro-F1 **0.774**. Legality information is linearly present in SpaDE's code; the bilinear
role-filler probe cannot read it out of the basis SpaDE writes it in. Two further results
stand. **First**, every trained dictionary loses an extra **0.027–0.062** macro-F1 at the
injected move itself, over and above its own r < 0 lossiness, while the wide random
projection loses **−0.005** there and the sparsity-matched null **+0.014** — the trained
dictionaries are the ones whose read-out degrades where the activation leaves the training
distribution. **Second**, the two SpaDE variants fail *selectively*: they hold R1 and R2 at
the sparsity-matched null's level and drop R3 (turn) from 0.81 to **0.006 / 0.048**, the
one class carrying all of the taxonomy's turn-dependence, while their board probe loses
mine-vs-theirs (F1 0.47 / 0.59) and keeps empty-vs-occupied (0.80 / 0.82).

## What holds

**1. Training buys 0.15 of legality read-out at matched sparsity, and nothing at all
against an unmatched wide random map.** Held-out legality macro-F1, 95% bootstrap CI over
n = 40 held-out source games:

| space | legality | board | space | legality | board |
|---|---|---|---|---|---|
| raw (reference) | 0.9379 [0.9343, 0.9416] | 0.9820 | GatedAnneal | 0.8854 [0.8813, 0.8898] | 0.9430 |
| Standard | 0.9466 [0.9435, 0.9498] | 0.9777 | SpaDE | 0.4297 [0.4258, 0.4335] | 0.6345 |
| PAnneal | 0.9288 [0.9246, 0.9326] | 0.9791 | CosineSpaDE | 0.4896 [0.4833, 0.4960] | 0.6994 |
| GatedSAE | 0.9119 [0.9086, 0.9154] | 0.9565 | **random TopK** ($L_0$ 80) | **0.7742** [0.7697, 0.7785] | 0.8500 |
| TopK | 0.9235 [0.9199, 0.9272] | 0.9714 | **random proj.** ($L_0$ 4101) | **0.9036** [0.8988, 0.9084] | 0.9735 |
| BatchTopK | 0.9259 [0.9220, 0.9297] | 0.9719 | | | |

TopK − random TopK = **+0.149** at identical L0 = 80, and that is the only comparison in
the table with a matched control. `random_proj` is matched to nothing: it is a nearly
information-preserving expansion, so a probe reads it, and the fact that it beats
GatedAnneal, SpaDE and CosineSpaDE is a statement about those dictionaries' information
content and not about rules. The figure draws the two controls in different styles for this
reason.

**2. The trained dictionaries, and only the trained dictionaries, lose extra macro-F1 at
the violation itself.** `excess_macro_f1` on pooled empty-square violations — a space's own
r < 0 → r = 0 drop minus raw's drop on the same bootstrap draws, n = 2000 source games:

| space | excess | space | excess |
|---|---|---|---|
| Standard | +0.0294 [+0.0277, +0.0313] | BatchTopK | +0.0384 [+0.0366, +0.0404] |
| PAnneal | +0.0268 [+0.0252, +0.0286] | GatedAnneal | +0.0617 [+0.0590, +0.0643] |
| GatedSAE | +0.0271 [+0.0251, +0.0291] | **random TopK** | **+0.0137** [+0.0109, +0.0164] |
| TopK | +0.0378 [+0.0358, +0.0397] | **random proj.** | **−0.0051** [−0.0068, −0.0035] |

The paired curves in `readout.csv` show the same thing as a step: the delta from raw is flat through
r = 0 for both untrained controls and dips by 0.023–0.057 for every trained dictionary,
recovering by r = 1. `random_proj`'s r < 0 baseline (0.898) sits inside the trained
dictionaries' range (0.887–0.943), so the floor objection below does not apply to *this*
comparison. The reading is that a dictionary trained on the model's on-distribution
activations reconstructs the off-distribution activation an illegal move produces worse —
which is a statement about SAEs and distribution shift, not about rules.

**3. What a space supports and what the violation costs it answer to different metrics.**
Spearman ρ over the eight trained dictionaries (descriptive at n = 8):

| x | legality macro-F1 at r = −1 | drop in agreement with raw |
|---|---|---|
| $L_0$ over the window | +0.10 | −0.74 |
| variance explained | +0.60 | −0.95 |
| board probe, same space | **+0.98** | −0.69 |

L0 ranges 50× across these dictionaries and sees nothing of the level; the board probe sees
almost all of it. The disruption at the violation runs the other way and is tracked by
reconstruction fidelity. Neither column contains a rule-specific quantity, and neither is
the same thing as information preservation — see `linear_r2` under *what does not hold*.

**4. Both SpaDE variants lose R3 and keep R1 and R2, read on the shared pre-move board.**
P(the probe names the violated rule at the square the branch is about to play), at r = −1,
n = 1997–2000 source games:

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw | 0.979 | 0.911 | 0.814 [0.797, 0.832] |
| the six ordinary dictionaries | 0.948–0.982 | 0.866–0.935 | 0.708–0.828 |
| random projection | 0.962 | 0.878 | 0.743 [0.723, 0.762] |
| **random TopK** | **0.892** | **0.785** | **0.473** [0.450, 0.496] |
| **SpaDE** | **0.871** | **0.835** | **0.006** [0.003, 0.010] |
| **CosineSpaDE** | **0.873** | **0.810** | **0.048** [0.038, 0.058] |

SpaDE and the sparsity-matched null agree on R1 to 0.021 and on R2 to 0.050, and differ on
R3 by 0.467. That contrast needs no extrapolation, which is why F3a draws all three rules
on one row per space. Within-rule ÷ between-rule spread over the member categories is
**0.44** (mean over the 11 spaces), so the three-rule grouping is a grouping and not a
label. The same signature is in the held-out fit: SpaDE's per-class F1 is R1 0.826,
R2 0.763, R3 **0.011**, legal **0.119**, and its board probe reads `empty` at 0.802 but
`mine` at 0.467 and `theirs` at 0.634. All of the taxonomy's turn-dependence sits on the
legal ↔ R3 boundary (see *Where the turn-dependence lives*), and that boundary needs
mover-relative colour, which is exactly the board channel SpaDE drops.

**What claim 4 does not establish.** R3 is the hardest and rarest class (9.6%), so a
uniformly degraded space loses it first, and R3 detection needs several squares of
mover-relative colour at once, so a per-square colour channel at F1 ≈ 0.5 can multiply down
to near zero without anything rule-specific happening. F3b plots R3 against the same space's
overall level so a reader can see where SpaDE falls relative to the other nine; these data
do not settle whether the shortfall exceeds a multiplicative account.

## What does not hold, and why

**The legality probe adds nothing over board-reading.** A `board_oracle_legality` arm —
decode the board with the 3-way probe, then apply the rules to the decoded board — scores
**0.984** on raw against the legality probe's **0.938**, and beats the legality probe in
every space. There is no residue for the legality probe to be reading that board-reading
does not already supply. Every claim below is therefore about how a *space* is read, not
about the model having a rule representation over and above its board representation.

**"The ranking just measures information preservation" is false, and so is the opposite.**
An untrained wide random map beats three trained dictionaries and ρ = +0.98 between board
macro-F1 and legality macro-F1 — which invites the deflation. The red team's `linear_r2`
rules it out: SpaDE's code linearly recovers raw at **0.973** while reading legality at
**0.430**, and `random_topk` recovers raw at **0.579** while reading legality at **0.774**.
The ordering these figures show is a property of the **basis** each dictionary writes in
and of what a bilinear role-filler probe can extract from it — not of how much information
survived. No sentence of the form "dictionary X preserves the rules better than dictionary
Y" is supported; "a rank-constrained role-filler probe reads X more easily than Y" is.

**`excess_macro_f1` is not scale-free and must not be read as a ranking.** SpaDE
(−0.0763) and CosineSpaDE (−0.0575) are negative only because their r < 0 baselines are
already 0.42 and 0.49, so there is less left to lose at r = 0. The metric is usable exactly
where the baselines are comparable, which is claim 2's trained-vs-`random_proj` contrast,
and nowhere else. No figure plots it. `drop_agreement_with_raw`, the headline the study
committed to, has a milder version of the same problem — SpaDE agrees with raw on only 0.66
of squares before anything happens — which is why F4's bottom row is read against its top
row rather than alone. **No normalisation has been introduced to repair either; if one is
wanted it needs to be chosen deliberately, not by the plotting code.**

**F3 at r = 0 is degenerate and was moved.** At the offset the illegal move is played the
square is occupied, every space reads it as R1, and R3 correct-rule recall is 0.000–0.007
in all eleven spaces including raw. That is a property of the target, not of the spaces.
The panel now reads r = −1, the bitwise-shared pre-move board — so F3 is a read-out property
of the space and not an effect of the violation. The mover-relative frame also alternates
each ply, so R3 is near zero at every even offset by construction (raw's R3 recall over
r = −8…−1 is 0.13, 0.35, 0.13, 0.45, 0.09, 0.59, 0.01, 0.81); r = −1 is the phase in which
the branch's move is about to be played.

**The contract's claim that plain accuracy ≈ 0.9 for a probe that has learned nothing.**
The shuffled-label floor is accuracy 0.5554, macro-F1 0.2856.

**The contract's claim that R3 is rare.** It is 9.577% pooled and the third-largest of the
four classes (legal 14.79%, R1 53.91%, R2 21.73%).

**The contract's branch count of 13531.** The branch set is n = 17527, from 2000 source
games.

**`frac_recovered` in `results/acceptance/all_results.csv` is all NaN.** F4 no longer reads
that file at all: `results/space_quality.csv` carries L0, variance explained and
reconstruction F1 measured on this study's own read-out window, for all eleven spaces
including the two untrained controls, which is what lets F4 place `random_topk` at exactly
TopK's L0.

## Figures

`figures/f0_which_space` — the only figure. PNG for reading here, PDF for the paper;
`--camera-ready` writes the PDF and moves the title into the caption.

One bar per space, ordered best to worst, spanning raw, the eight dictionaries and the two
untrained controls. It answers the question anyone actually asks of this study, and it
carries the two reasons that answer is not what it looks like. The black tick on each bar
is the same space's 3-class board probe and is always higher: legality is a deterministic
function of the board, board macro-F1 tracks legality at pearson 0.998, and board argmax
plus the rules engine beats the legality probe in all 11 spaces. The two bottom rows are
untrained: a dense random projection at L0 4101 reaches 0.904 and outranks a trained
dictionary, so a high bar mostly means the map preserved information. The comparison that
survives is sparsity-matched, TopK 0.923 against random TopK 0.774 at L0 80.

Four earlier figures (per-space dumbbells, the paired r-window, per-rule at r = −1, and
metric-vs-rules) were cut. Their numbers are in `results/readout.csv` and are quoted
throughout RED_TEAM.md; `make_figures.py` no longer reads that file. The r-window figure
carried the study's one surviving positive claim — the six ordinary dictionaries each lose
an extra 0.02–0.06 macro-F1 at the injected move while neither untrained control loses
anything there — so that claim now lives only in prose and in the CSV.

## Known gaps

- Seed 42 only, layer 5 only, one probe architecture for the read-out.
- The eight dictionaries are not matched on information content, so between-family
  comparisons in F1 and F4 are confounded by L0 and reconstruction quality. Only TopK vs
  `random_topk` is a matched comparison, and it is a trained-vs-untrained one.
- ρ in F4 is over n = 8 dictionaries, which is descriptive and carries no interval.
- `analysis/bracket_flip/results/sae_arm.csv`'s `acc_rel_L5` covers eight of the nine
  spaces but was computed over the wrong SpaDE checkpoint (see above), so it is a
  cross-check on the board probe for seven spaces, not a substitute for it.
- The probe is fitted on legal positions, so its read-out on the off-distribution
  activations after r = 0 is evidence, not ground truth — the same caveat the
  violation-divergence probe read-out carries. Claim 2 is precisely a claim about that
  off-distribution behaviour, so it inherits the caveat rather than escaping it.
