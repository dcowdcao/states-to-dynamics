# Legality probe: is a rule violation still readable after an SAE?

A 4-way bilinear (TPR) probe — role = 64 squares, filler = {legal, R1, R2, R3} — is fit
independently in each of nine representation spaces at OthelloGPT layer 5, then read out
over the violation-divergence branch set, which runs a game 0 → *t0* and injects one
illegal move at *t0*. The four classes collapse the seven-way taxonomy of
`src/spade/othello/taxonomy.py` onto the three constraint rules R1–R3 via `RULE_MEMBERS`.

The question is not whether legality is decodable. It is whether a sparse dictionary
that keeps the *board* also keeps the *rules*.

## Pipeline

```
python -m spade.legality_probe.labels      # taxonomy -> 4-way labels, per position
python -m spade.legality_probe.fit         # 9 spaces x {legality, board, wrong-player}
                                           #   -> results/probe_fit.csv
python -m spade.legality_probe.readout     # branches x window -> results/readout.csv
python analysis/legality_probe/make_figures.py            # F1-F4
python analysis/legality_probe/make_figures.py --camera-ready
```

Output schemas, tidy long, both consumed by `make_figures.py` and by nothing else:

```
probe_fit.csv  space, d_space, target, n_classes, filler_dim, metric, class, value, lo, hi, n
               target in {legality, board, legality_wrong_player}
               metric in {macro_f1, accuracy, per_class_f1, per_class_recall}
readout.csv    space, category, rule, role, r, metric, mean, lo, hi, n
               category = a 7-way taxonomy name, "empty_pooled", or "" for a rule aggregate
               metric in {macro_f1, macro_f1_minus_raw, p_pred_legal_at_played_square,
                          p_pred_correct_rule, agreement_with_raw}
```

`macro_f1_minus_raw` is the per-game **paired** difference from raw at the same offset;
its CI is not derivable from the two marginal CIs, which is why it is a column of the
CSV rather than arithmetic in the figure. Rule aggregates carry `category = ""` with
`rule` set, and F3 asserts they exist rather than averaging the members itself.

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
SpaDE. `random_proj` and `random_topk` are carried as nulls only — gray, dashed, last.

## What this study cannot claim, stated before the results

**A macro-F1 level is not a result here.** Legality is a deterministic function of the
board, and the board already decodes at ~0.98, so a high legality macro-F1 in any space
is mostly downstream of board-reading rather than evidence that the model represents
rules. Two controls pin this down:

- Legality computed for the **wrong player** still reads at macro-F1 **0.9101** against
  0.9374 for the right one (bilinear probe), reproduced independently at **0.9018**
  against 0.9388 by a plain linear probe. A result that survives two probe families is
  worth more than either alone.
- A shuffled-label fit gives accuracy **0.5610** and macro-F1 **0.2926** — R1 alone is
  54% of all cells, so plain accuracy near 0.9 means nothing and macro-F1 is the only
  primary metric this study reports.

Everything below therefore rests on **differences**: between spaces at a fixed offset,
and between r < 0 and r = 0 within a space. Every figure is built so that a level cannot
be read off it as a finding.

**The class balance drifts with absolute ply**, which is why F2 is a paired difference.
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

*(stub — fill from `results/probe_fit.csv` and `results/readout.csv` once they exist.
It must be a statement about **differences**: which spaces lose legality by more than
they lose the board, and how much further the gap opens at r = 0 than at r < 0. A
sentence of the form "space X decodes legality at 0.9" is not a result of this study and
must not appear here.)*

## What holds

*(stub — one numbered claim per finding, each with its estimator, its CI, and n, in the
register of `analysis/violation_divergence/README.md`. Candidate slots:*

*1. Whether the board→legality gap (F1a) separates the dictionary families, and whether
that separation survives the board probe as a covariate (F4c).*

*2. Whether the paired Δ from raw at r = 0 (F2b–e) is distinguishable from the same Δ at
r < 0 — i.e. whether an SAE loses more legality **at the violation** than it loses
anywhere.*

*3. Which of R1 / R2 / R3 each space stops naming at the played square (F3), with the
member categories drawn behind each rule mean and the within-group ÷ between-group
spread reported.)*

## What does not hold, and why

*(stub — the register `analysis/violation_divergence/README.md` uses for its own
falsified premises. Anything the figures rule out goes here rather than being dropped.
Known entries already:*

- *The contract's claim that plain accuracy ≈ 0.9 for a probe that has learned nothing.
  The shuffled-label floor is accuracy 0.5610, macro-F1 0.2926.*
- *The contract's claim that R3 is rare. It is 9.6% pooled and the third-largest class.*
- *The contract's branch count of 13531. The branch set is n = 17527.*
- *`frac_recovered` in `results/acceptance/all_results.csv` is all NaN, so F4 uses L0 and
  variance explained only.)*

## Figures

`figures/f1_board_vs_legality` (F1) · `figures/f2_readout_window` (F2, headline) ·
`figures/f3_per_rule` (F3) · `figures/f4_metric_vs_rules` (F4).

**F1** is a dumbbell per space: the open marker is the 3-way board probe
{empty, theirs, mine}, the filled one the 4-way legality probe, both fitted in the same
space with the same machinery. Two adjacent bars would put the eye on the two levels;
the dumbbell puts it on the segment, which is the quantity. Panel (b) is the legality
probe's per-class F1, which is where the loss localizes. Raw's wrong-player control is
drawn as a vertical rule so no bar is read against zero.

**F2** is the headline. Panel (a) is raw and the two nulls in absolute macro-F1, carried
so the ply-driven drift is visible rather than hidden; panels (b)–(e) are the eight
dictionaries as per-game paired differences from raw at the same r, one panel per
dictionary family. Eleven curves on one axis is past FIGURE_GUIDELINES §4.1's five-series
limit, and faceting by family puts exactly two dictionaries per panel against a zero line
that *is* raw.

**F3** is the per-rule read-out at the played square at r = 0, three panels, each
labelled with how many taxonomy categories its rule averages (2 / 3 / 1) and drawing
those members behind the mean.

**F4** asks which standard metric predicts rule preservation: L0 on a log axis (a 50×
range across these dictionaries), variance explained (trapped in 0.971–0.999, which is
itself the finding), and the board probe in the same space. The three panels share a y
axis so the comparison is the first thing read.

## Known gaps

- The results sections are stubs; only the controls and the label statistics above are
  measured.
- Seed 42 only, layer 5 only.
- `results/acceptance/all_results.csv` is the source of L0 and variance explained; it
  carries nine seed-42 rows, and `frac_recovered` is empty in all of them.
- `analysis/bracket_flip/results/sae_arm.csv`'s `acc_rel_L5` covers eight of the nine
  spaces but was computed over the wrong SpaDE checkpoint (see above), so it is a
  cross-check on the board probe for seven spaces, not a substitute for it.
- The probe is fitted on legal positions, so its read-out on the off-distribution
  activations after r = 0 is evidence, not ground truth — the same caveat the
  violation-divergence probe read-out carries.

## Test-only scaffolding

`smoke/synthetic_results.py` writes fake `probe_fit.csv` / `readout.csv` conforming to
the frozen schemas, so the figures could be written and reviewed before the real data
landed. It has no other purpose and should be deleted once the real CSVs exist.
