# Move-legality probe: can a linear read-out say *that* a move was illegal, and *where*?

A plain L2 logistic regression is fit on OthelloGPT's layer-5 residual at the position of a
played move, and asked one question per position: was the move just played legal? Two
heads — 2-class (legal vs illegal) and 4-class (legal, R1, R2, R3) — are fit independently
in each of 11 representation spaces, then run over every position of 34,546 game sequences
that are legal everywhere except at one known index.

The point is not that legality is decodable. It is whether the probe's output, read across
a whole game, **puts its maximum on the move that is actually illegal**, and whether a
sparse dictionary keeps that.

This is not `analysis/legality_probe`, which asks a bilinear probe about the legality of
every *candidate* square before a move is made. Here there is one label per *played* token,
which is what makes a per-position curve over a game meaningful.

## What the probe reads

The residual at position *t* has seen `tokens[:t+1]`, so it is the first place in the
network where the move at *t* exists. The label is
`spade.othello.taxonomy.classify(board(tokens[:t]), tokens[t])`, collapsed onto the three
constraint rules. Reading one position earlier would be a different study: at *t−1* the
network has not been shown the move yet and the question is unanswerable.

## Balance is structural, not a reweighting

A fit-set **item** is one (source game, move index) pair. It carries the move the game
really played — legal by construction — and one substituted illegal move per constraint
rule that has a candidate there. All four rows share `tokens[:t]` and differ in exactly one
token.

`probe.balanced` keeps whole items, so every split holds the same number of rows per class
*and* the classes have identical game and move-index marginals. Nothing is subsampled to
force balance, and a probe that scored well by reading move index would score at chance.
The 2-class set draws one illegal row per item at seed 42 — never a mean over the three,
which would give the illegal class 1/√3 of its real spread.

Splits are by source game (60/20/20), so no position in one split shares a prefix with
another.

| | train | val | test |
|---|---|---|---|
| source games | 480 | 160 | 160 |
| items (2-class): rows *per class* | 27,743 | 9,258 | 9,255 |
| items with all three rules (4-class): rows *per class* | 23,641 | 7,699 | 7,774 |

`results/fit_set_balance.csv` also carries the unbalanced availability behind those
numbers. R1 is available at every position — an occupied square always exists — so it ties
`legal` exactly; R2 is the scarcest (24,163 of 27,743 train items) because late boards run
out of empty squares, and it is R2's availability that sets the 4-class item count.

## The read-out corpus

`spade.violation_divergence`'s spliced branch tree: 3,923 source games, each forked at one
position *t0* ∈ [10, 34], each fork differing from the game as played in exactly one token.
Every game contributes a **reference** (the game verbatim), two **legal controls** (a
different legal move at *t0*), and one branch per illegal category. 34,546 sequences.

Its 3,923 source games are disjoint from the probe's 800 by exact token match — they are
literally the first 3,923 games of the same seeded stream, all of which the fit set skips.

## Ground truth stops at *t0*, and the figures say so

A branch keeps the source game's `tokens[t0+1:]`. That suffix is a legal continuation of
the *reference's* board and of no other; after an illegal move no board exists for it to be
legal on, and the engine refuses to build one (`OthelloBoardState.umpire` asserts the
square is empty and that something flips). So:

- the probe is written out at **all** 59 positions, and the curve past *t0* is drawn,
  hatched, and labelled as carrying no ground truth;
- every localization statistic is taken over *t* ∈ [1, *t0*];
- the **legal control branch of the same game at the same *t0*** is the false-alarm rate of
  exactly that statistic — same prefix, same position, same estimator, legal move.

Position 0 is the attention sink and is dropped at both ends — not fit, not scored. This is
not cosmetic: left in, the probe emits a per-space *constant* there (1.0 in BatchTopK, 0.0
in raw), and one saturated position took the argmax away from *t0* in every single game.

## The control the probe has to beat

`spade.move_legality.model_baseline` scores each position by `1 − P_model(the token played
there)`, off the next-move logits alone. No probe, no fit set, no labels.
`analysis/generation_legality` measures that OthelloGPT puts 0.9982 of its next-move mass
on legal squares, so a played illegal move is *already* improbable under the model, and a
probe that only recovers that has read the output distribution rather than the
representation. It is the `model_baseline` row of `results/localization_by_space.csv`.

## Results

### The probe works, and it is not the thing that matters

In raw layer 5 the 2-class probe reaches held-out **macro-F1 0.889 / AUROC 0.942** on fit-set
positions, and over the branch tree its argmax over *t* ∈ [1, *t0*] lands on the spliced move
in **87.4%** [86.4, 88.4] of games against a legal-control false-alarm rate of **5.0%** and a
pick-a-position chance of 4.8% (`chance_hit_rate`, the mean 1/(*t0*+1) over the arm's games).
Paired against the same game's legal control at *t0*, AUROC is **0.960** [0.956, 0.965].
The control's own null — scored against the *second* legal
control every game carries — is **0.502** [0.491, 0.512] in raw and lands inside [0.492,
0.505] in all eleven spaces, which is the check that the statistic has no thumb on it.

**And the model's own next-move distribution does it perfectly, with no probe at all.**
`1 − P_model(the move played)` localizes the spliced move in **3923/3923** games and reaches
**AUROC 1.0000** against the legal control. The margin is not marginal: the illegal move's
score is above every other position in its own sequence by a median of 0.072, in 100.0% of
games, and its minimum over 3,923 games is 0.9992 where the legal arms sit at 0.902.

So *"a probe can find the illegal move"* is **not, on its own, evidence about the
representation** — the output head already gives it away, and every probe here is strictly
worse than reading the logits. What is left for the probe is what the logits do not
factor: which rule was broken, and how the answer degrades under a dictionary.

### R1 is the hole, and it is specifically an occupied *opponent* square

Per category, raw layer 5, AUROC at *t0* against the paired legal control:

| category | rule | n | probe AUROC | probe hit rate | model logits AUROC |
|---|---|---|---|---|---|
| `occupied_own` | R1 | 3923 | 0.930 | 0.759 | 1.000 |
| `occupied_opp` | R1 | 3923 | **0.819** | **0.502** | 1.000 |
| `empty_detached` | R2 | 3874 | 0.9999 | 0.999 | 1.000 |
| `empty_own_only` | R2 | 3299 | 0.999 | 0.999 | 1.000 |
| `empty_no_bracket` | R2 | 3838 | 0.998 | 0.991 | 1.000 |
| `opponent_only` | R3 | 3920 | 0.999 | 0.993 | 1.000 |

Every violation on an *empty* square is read off at ≥0.998. Playing on a square that already
holds a disc is read off at 0.82–0.93, and playing on an **opponent's** disc is the worst
cell in the study — the argmax finds it barely more often than it misses. The model's own
logits are at 1.000 on all six, so this is a property of the linear read-out of the layer-5
residual, not of what the model knows.

That **inverts** `othello/taxonomy.py`'s ordering, which sorts the categories by how much
board computation the violation takes to detect and puts occupancy first as the shallowest
check. `results/rule_posterior_raw.csv` — the mean 4-class posterior at *t0*, one row per
arm — shows the mechanism: the probe puts 0.247 of its mass on R1 for moves that are
*legal*, and only 0.546 on R1 for moves that really are R1. The legal and R1 classes are the
pair it cannot separate.

It also lines up with `analysis/generation_legality`, which finds the model's own residual
illegal mass skewed **7.1×** toward R3 and **3.0×** toward R2 while R1 is suppressed to
**0.10×**. The violations the representation marks most sharply are the ones the model
actually makes; the one it marks worst is the one it essentially never makes.

### The rise after *t0* is the splice, not the illegality

Raw, 2-class, mean over games: the pre-event level is 0.123 for every arm (identical by
construction — the branches share `tokens[:t0]` bitwise). At *r* = 0 the illegal arm hits
0.866. After it, the illegal arm averages 0.422 — but the **legal** control averages 0.362
against the reference's 0.156. Roughly six sevenths of the post-event elevation is present
in a branch whose *t0* move is perfectly legal. That is the hatched region earning its
label: what the probe reads after *t0* is mostly "this suffix does not belong to this
board", and it is why nothing there is scored.

### Across spaces: width buys more than training does

Held-out test macro-F1 and pooled localization, best and worst:

| space | 2-class F1 | 4-class F1 | localization | AUROC at *t0* |
|---|---|---|---|---|
| Standard (L1 0.059) | **0.899** | **0.820** | **0.902** | **0.966** |
| BatchTopK (k=80) | 0.893 | 0.790 | 0.889 | 0.964 |
| TopK (k=80) | 0.894 | 0.789 | 0.884 | 0.962 |
| SpaDE (λ=1e-3, γ=1e-4) | 0.893 | 0.788 | 0.881 | 0.959 |
| **raw residual** | 0.889 | 0.773 | 0.874 | 0.960 |
| PAnneal | 0.888 | 0.782 | 0.758 | 0.959 |
| CosineSpaDE (γ=1.0) | 0.865 | 0.720 | 0.742 | 0.924 |
| `random_proj` (ceiling) | 0.894 | 0.790 | 0.888 | 0.960 |
| `random_topk` (null) | 0.853 | 0.704 | 0.813 | 0.928 |

**The untrained ceiling is the row to read first.** A ReLU of a *random* 512→8192 projection
reaches 0.894 / 0.790 / 0.888 — above raw on all three, and above every trained dictionary
except Standard. Almost all of the apparent SAE gain over the raw residual is the 16×
expansion, not what the dictionary learned; Standard clears `random_proj` by 0.006 (2-class
F1) and 0.030 (4-class), and nothing else clears it at all. The sparsity-matched
`random_topk` at L₀ 80 is the honest null for the k=80 pair, and TopK/BatchTopK beat it by
0.04 / 0.09, so training does buy something at matched sparsity — just less than width does.

**Two dictionaries lose the localization without losing the fit.** PAnneal scores 0.888 on
held-out fit-set positions, essentially raw's 0.889, and then localizes at 0.758 against
raw's 0.874. Fit-set accuracy and localization on the branch tree are not the same
measurement — the second is a harder, shifted one — and a table of the first would have
called PAnneal a faithful dictionary.

## Caveats

1. **The no-probe baseline may be flattered by how the branches were drawn.** A branch token
   is drawn uniformly from its taxonomy category, not from the model's own illegal mass. The
   generation study finds the model's real illegal moves concentrate on near-misses
   (`empty_no_bracket`, `opponent_only`), which are by construction the ones it finds
   likely — so a branch set sampled from the model's own illegal conditional would shrink the
   `1 − P(played)` margin and might not be perfectly separable. **That is the experiment that
   would decide whether the probe is redundant**, and it has not been run.
2. **The baseline is a detector, not a classifier.** One scalar cannot name which rule was
   broken, so `rule_posterior_raw.csv` has no baseline row. This is not proof that the rule
   is unavailable without a probe — a cleverer oracle-free statistic might recover it.
3. Ground truth exists only for *t* ≤ *t0*; see above. Nothing past *t0* is scored, and the
   figures hatch it.
4. `stream_games` reads a ~1% window of the corpus, not a sample of it
   (`docs/game_corpus.md`). The probe games and the read-out games are disjoint but adjacent
   in one stream order, and a different seed is not a replication.
5. Single seed (42) throughout: for the fit-set draw, the splits, the balanced draw and the
   bootstrap. There are no across-seed error bars on any probe number.
6. Only `raw` is rendered. Any other space is one `localization.py --space <name>` away, and
   `space_comparison.py` scores all eleven into `localization_by_space.csv` without drawing
   anything.

## Figures

| file | what it answers |
|---|---|
| `raw_2class.png` | does the residual stream localize an illegal move at all? |
| `raw_4class.png` | the same question per rule, from the 4-class head |

Both are event-locked to the spliced move at *t0*: `P(illegal)` and `1 − P(legal)` on a full
0–1 axis, the reference game and the same game's matched legal control drawn beside the
illegal arms, the shared prefix shaded and the no-ground-truth region hatched.

Every band is a 95% percentile bootstrap CI resampling **source games** as clusters (1,000
resamples). A game's branches share `tokens[:t0]` and are not independent draws; resampling
rows would give intervals several times too tight. Every arm draws one branch per game at
seed 42 rather than averaging a game's categories, which would give it 1/√k of its spread.

Everything else the study measures is a table: `localization_<space>.csv` and
`localization_by_space.csv` (localization rate, rank of *t0*, paired AUROC, pick-a-position
chance) and `rule_posterior_raw.csv` (mean 4-class posterior at *t0* per arm).

## Reproducing

```bash
sbatch scripts/slurm/studies/move_legality/submit_move_legality_fit_set.sbatch --n_games 800
sbatch --dependency=afterok:<id> scripts/slurm/studies/move_legality/submit_move_legality_readout.sbatch
python -m spade.move_legality.model_baseline
python analysis/move_legality/localization.py --space raw
python analysis/move_legality/space_comparison.py
```
