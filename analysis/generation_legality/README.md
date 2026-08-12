# Generation legality: does OthelloGPT play legal moves, and how does it fail?

Teacher-forced on 1000 real legal games, the model's next-move distribution at every ply is
split by the 7-way taxonomy of `src/spade/othello/taxonomy.py`: mass on legal squares, mass
per constraint rule R1–R3, mass on the pass token. The same positions record what it would
actually *play* — the argmax move and one temperature-1.0 draw — and a separate arm samples
whole games from scratch and asks where the first illegal move lands.

Every other study in this repo *injects* an illegal move and measures what happens
downstream. This one asks the prior question those studies never answer: does the model
produce illegal moves on its own, and are they of a particular kind?

## Pipeline

```
python -m spade.generation_legality.positions   # 1000 games x t in [0, 56), + taxonomy
python -m spade.generation_legality.scan        # one teacher-forced pass -> per-position mass
python -m spade.generation_legality.rollout     # 1000 free-running games at T=1.0
python -m spade.generation_legality.aggregate   # -> results/*.csv
python analysis/generation_legality/make_figures.py [--camera-ready]
```

The three GPU steps are `scripts/slurm/studies/generation_legality/submit_generation_legality.sbatch`
(one H100, ~5 min). Heavy intermediates live under `study_dir("generation")`; the
per-position table is `$SPADE_GENERATION_DIR/scan/positions.csv`, 56,000 rows.

## What is measured, exactly

`logits[:, t]` predicts the move at ply *t*+1, and `classify_prefix(game, t+1)` gives the
category of all 60 candidate squares for that ply. **t runs over [0, 56)** — every move of
the game except the last two. At *t* = 56 and 57 the mover forfeits in 13.5% and 4.0% of
games, and `classify() == LEGAL` matches the engine only on non-forfeit positions. A game is
taken whole or not at all, so n is constant across the move axis and the bootstrap unit is
the game; whole-or-nothing acceptance over [0, 56) drops 41 of 1041 games (3.9%), over
[0, 58) it would drop 18%.

**Disjointness.** The 1000 games are streamed at seed 42 and excluded by exact token match
from the violation-divergence read-out set, which is 3,923 games — so this study reads
stream rows ~3,924 onward. They are *not* disjoint from anything else: the legality-probe
fit set applies the same exclusion to the same stream and its 200 games are this study's
first 200. Nothing here fits a model, so that overlap costs nothing.

**These games are not held out from OthelloGPT.** The corpus is the model's training
distribution, and `stream_games` reads a fixed leading window of it rather than sampling
(`docs/game_corpus.md`) — the same ~1.6% slice every study walks, with one opening line
truncated out of it. "Held out" here means held out from a sibling study, not from the model.
Read every number below as *in-distribution* behaviour.

**The pass token is never folded in.** Token 0 does not occur in this corpus; the model puts
4.3e-7 of its mass there and it is reported as its own column, not added to `p_legal`.

## The numbers

95% percentile bootstrap over n = 1000 source games, 2000 resamples, seed 0.

| quantity | value |
|---|---|
| mean mass on legal squares | **0.998248** [0.998019, 0.998453] |
| mean mass on illegal squares | **1.752e-3** [1.546e-3, 1.981e-3] |
| median per-position illegal mass | 2.489e-4, IQR [1.515e-4, 3.910e-4] |
| mass on the pass token | 4.26e-7 [4.08e-7, 4.50e-7] |
| greedy move illegal | **18 / 56,000** = 3.21e-4 [1.61e-4, 5.18e-4] |
| sampled move illegal (T = 1.0, one draw/position) | **93 / 56,000** = 1.66e-3 [1.29e-3, 2.09e-3] |
| a uniform mover over the 60 candidates | 0.850 — 485× the model's rate |
| free-running 59-ply game contains an illegal move | **11.5%** [9.5%, 13.6%] |

**It gets worse with move index, by two orders of magnitude.** Mean illegal mass runs
3.2e-5 at *t* = 0 → 1.7e-4 at *t* = 10 → 1.5e-3 at *t* = 30 → 5.1e-3 at *t* = 55. The greedy
arm plays nothing illegal before *t* = 24 (0 of 24,000 positions) and 8 of 8,000 in
*t* ∈ [48, 56).

**The sampled arm reproduces the mass arm**, 1.66e-3 against 1.75e-3, which is the check
that the mass columns mean what they say: a single draw per position is an unbiased estimate
of the same quantity, and it lands inside the mass CI.

### The skew, against a per-position matched null

The null is the model's own illegal mass at each position, redistributed uniformly over that
position's illegal squares. It has to be per-position: R1 owns 5 of 60 squares at ply 1 and
~55 of 60 by ply 56, so a pooled square-count null would mostly draw that drift.

| rule | mass | sampled play (n = 93) | greedy play (n = 18) |
|---|---|---|---|
| R1 placement (2 cat.) | **0.103** [0.091, 0.116] | 0.040 [0.000, 0.089] | **0.000** [0.000, 0.000] |
| R2 bracketing (3 cat.) | **2.985** [2.616, 3.361] | 2.987 [1.893, 4.387] | 1.758 [0.000, 7.015] |
| R3 turn (1 cat.) | **7.090** [6.616, 7.593] | 7.269 [5.909, 8.870] | 11.030 [9.301, 13.856] |

The behaviour matches the mass: 66% of illegal mass is R3, and 71% of the 93 sampled illegal
moves and 16 of the 18 greedy ones are R3.

**The rule grouping hides half the story, and the figures say so.** Within-rule ÷
between-rule spread of the log₁₀ skew is **0.64** over the six categories, and of the log₁₀
mass share **0.95** — at ≈1 the three rules are a label rather than a finding on the mass
axis. Both numbers are one rule's doing: R2's three members span 1.73 dex where R1's two span
0.11 and R3 has a single member.

| category | skew | | category | skew |
|---|---|---|---|---|
| `occupied_own` (R1) | 0.098 | | `empty_own_only` (R2) | 0.326 |
| `occupied_opp` (R1) | 0.107 | | `empty_no_bracket` (R2) | **6.658** |
| `empty_detached` (R2) | 0.449 | | `opponent_only` (R3) | **7.090** |

The two over-weighted categories are the two **near-miss** ones: a square that touches
opponent discs but closes no line, and a square that would flip for the *opponent*. Both look
like legal moves under any rule short of the full one. Everything the model suppresses —
occupied squares, squares touching nothing, squares touching only its own discs — is
detectable from the target square and its neighbourhood alone. So the one-sentence version is
not "the model breaks R3" but *the model's residual errors are near-misses: it has the
geometry and loses the last bracketing/turn check.*

**The three shares are compositional** — they sum to 1 by construction — so R2 and R3 sitting
above 1 is partly forced by R1 sitting below it. The claim that survives that objection is
the ordering and the per-category magnitudes, not the absolute ratios.

**The skew is a late-game phenomenon.** Every rule sits at ratio ≈ 1 for *t* < 20 and fans
out after: by *t* = 55 R2 and R3 are ~30× and R1 is ~0.02. The pooled numbers above average
over that; `by_move.csv`'s `skew_*` rows carry it per ply. F2 shows the same drift on the raw
mass axis, where R3 crosses R2 in the midgame and R1 turns over around *t* = 40.

## Figures

Two figures, one panel each. **Their y axes are not the same quantity**: F1 is cumulative
per game and only rises, F2 is per move. Reading either as the other is the one misreading
the pair invites, so each states which it is on its own y axis.

`figures/generation_legality.png` — **OthelloGPT puts 0.9982 of its next-move mass on legal
squares, and 11.5% of the 59-ply games it generates for itself still go illegal.**
P(at least one illegal move by move *t*) against move number, two arms: `free_running`,
1000 whole games sampled at T = 1.0 from a uniform-random legal opening and judged by the
engine — this is the measured, averaged-over-games curve — and `teacher_forced`, the same
quantity composed from the per-position mass of a model that never sees its own output
(∏(1 − p) over a game's plies, then averaged over games). The two are the same statistic
under two sequence distributions, so they share a hue and split on line style. The gap at
the end, 11.5% against 8.3%, is the cost of self-conditioning and is the reason the composed
arm is not a substitute for the measured one. Bands are 95% percentile bootstrap CIs over
games (2000 resamples).

`figures/generation_legality_by_rule.png` — **The residual illegal mass is not spread over
the illegal squares the way the board offers them: R3 (turn confusion) dominates from the
midgame on and R1 (placement) is suppressed.** P(illegal move at move *t*) against move
number, split into the three constraint rules, on a log axis, with `any illegal` as the
reference. The six taxonomy categories are drawn faint behind the rule that sums them, and
the rule series are **sums**, not means, so the legend says `sum of 3`
(FIGURE_GUIDELINES 1.7). The axis floor comes from the rule curves, so a faint member —
`empty_detached` decays four decades below R2 by the endgame — leaves the axis rather than
setting it for the whole panel. The within-rule ÷ between-rule spread of 0.95 is printed
under the panel with the reason: on this axis R2 is a label and R1 is a class.

There is no third figure, and no skew, density or played-rate panel. Those measurements were
not deleted with their panels — `rule_skew.csv`, `played_rate.csv` and `overall.csv` still
carry every number this README quotes — but the ratio-to-matched-null view now lives in the
CSVs and the prose above, not in a plot.

Both figures are PNG for reading here; `--camera-ready` writes vector PDFs and moves the
titles into these captions.

## What this cannot claim

- **In-distribution only.** Every teacher-forced position sits on a real legal prefix from
  the model's own training distribution. The free-running arm is the only one that leaves it,
  and it is 22% worse than the composed teacher-forced curve at 57 plies (10.1% vs 8.3%) —
  that gap is error compounding and it is measured, not modelled.
- **The composed curve assumes independence across plies within a game.** It is a composition
  of per-position marginals, not a statement about a joint distribution. It is drawn beside
  the measured arm precisely so a reader can see how much that assumption costs.
- **One model, one checkpoint, one seed, one corpus window.** Seed 42 for the games, 42 for
  the position sampler and the rollouts, 0 for the bootstrap. The corpus is synthetic
  random-legal play with a truncated `e6` opening, so per-square asymmetries are not safe to
  read off this (`docs/game_corpus.md` §2).
- **18 greedy illegal moves is 18.** The greedy arm's per-rule CIs are wide and its R2 bar
  spans [0, 7.0]. A NaN CI in `rule_skew.csv` means the matched null was empty in some
  bootstrap resamples — for a category with no available square at the plies where that arm
  failed, which is a property of the board and not of the estimator.
- **Forfeit positions are excluded**, so this says nothing about the plies where a player has
  no legal move. Those are also exactly the plies where the taxonomy and the engine disagree.

## Output schemas

All in `results/`, tidy long. No column is ever an empty string, so no `keep_default_na`
trap.

```
by_move.csv         t, metric, mean, lo, hi, n_games
                    metric in p_{legal,illegal,pass,R1..R3,<category>},
                              uniform_illegal, skew_{R1..R3,<category>}
played_rate.csv     arm, t_lo, t_hi, rate, lo, hi, n_games, n_illegal
rule_skew.csv       arm, level, name, rule, share_observed, share_expected, ratio,
                    lo, hi, n_games, n_events, n_positions
                    arm in {mass, greedy, sampled}; level in {rule, category}
survival.csv        arm, plies, value, lo, hi, n_games
                    arm in {teacher_forced, free_running}
overall.csv         metric, value, lo, hi, n_games, n
                    includes rule_{skew,mass}_within_over_between (GUIDELINES 1.7)
```
