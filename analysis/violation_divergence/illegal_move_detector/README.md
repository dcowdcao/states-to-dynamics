# The legal/illegal move detector, drawn

Four panels of one claim: a reference-free score on the model's own trajectory reacts to an
illegal move, at a stated operating point, and the reaction is graded rather than clean. A
fifth asks which dictionary reads that reaction best once every dictionary is held at the same
sparsity.

These are figure panels, not a new measurement. The statistic, the window and the threshold
rule all come from `../changepoint_localization`; this study re-runs them on **every** eligible
game of the spliced tree so the panels share one corpus, and draws them.

**Spliced only.** Every row here is a branch of `study_dir("divergence")/branches/branches.pt`:
`tokens[:t0]` shared with the source game, one token replaced at *t0*, and the suffix kept
**verbatim**. Nothing is regenerated after *t0*. The straightness corpus appears in exactly one
place — the per-move-index legal profile that "normal" is measured against — and never as a
scored game.

## The score, and what a position means

Per move index *t*, fit the mean and SD of raw layer-5 step length over legal games, and score
`g_t = −(d_t − μ(t)) / σ(t)`: how much **shorter** than usual the model's state update is.
Then contrast `g` across each candidate — the mean over the *M* = 4 steps after it minus the
mean over the 4 before — and scan. Higher means more illegal. Raw layer 5 throughout: these
panels are about what the detector does, not which dictionary does it best, which is
`changepoint_localization`'s question.

`τ` is the cut that maximizes balanced accuracy, found by exact breakpoint sweep
(`changepoint.threshold`), **refit on this population**. It is 0.619 here against the 0.869 in
`changepoint_localization/results/threshold_localization.csv`, which was fitted on the 575
games `curvature_discriminant` selected before `extend_branches` grew the tree, and on a
profile that included the games it scored.

**The objective is nearly flat across that whole range, which is the thing to remember about
τ.** Balanced accuracy is 0.6703 at τ = 0.619 and 0.6644 at 0.869: a cut moved by 0.25 buys
0.006 accuracy. What τ selects is the trade — 0.658 TPR / 0.683 TNR here, 0.576 / 0.753 there.
Quote τ with that caveat or not at all.

## Corpus

| | |
|---|---|
| eligible games | 2,415 (branch-complete, *t0* ∈ [10, 34], odd-parity source) |
| branch rows | 21,735 — reference, two legal controls, six illegal categories per game |
| scanned band | moves 6–55, 50 candidates — every position an *M* = 4 window reaches |
| profile fit | 1,222 straightness legal games, excluding every game scored here |

Eligibility is `curvature_discriminant/branch_geometry.py`'s own rule, imported rather than
restated. The odd-parity filter and the `held_out_legal` exclusion are what keep a trajectory
out of the baseline it is scored against.

## Figures

| file | what it shows |
|---|---|
| `detector_overview.png` | the four panels below on one canvas: (a) the boards, (b) the trace, (c) AUROC by rule, (d) the distributions |
| `board_states.png` | the board at *t0*−2 … *t0*+2 of game 3889: the legal move played at each of the four surrounding plies, and the illegal move the drawn branch plays at *t0* |
| `detector_trace.png` | the detector score for that same game over moves *t0*−5 … *t0*+5, three rule arms against the legal control, with τ |
| `auroc_by_rule.png` | AUROC at *t0*, illegal branch against the same game's legal control, pooled and per rule, n = 500 games |
| `score_histograms.png` | the distributions that AUROC summarizes: the score at *t0* for legal, illegal, R1, R2, R3, the same 500 games, with τ |
| `score_histograms_GatedAnnealTrainer-othello.png` | the same five distributions read through the GatedAnneal dictionary, same 500 games — panel (b) of `changepoint_localization/figures/threshold_detection.png`, whose panel (a) ranks that space first |
| `auroc_by_space_l0_matched.png` | the same AUROC, pooled illegal against the matched legal control, split by **representation space** instead of by rule: raw layer 5 and the eight L0-matched dictionaries, same 500 games |
| `sae_detector_overview.png` | `detector_overview`'s SAE counterpart: (a) `auroc_by_space_l0_matched`'s bars, (b) the score distributions in whichever space (a) ranks first. **(a) is the at-*t0* paired statistic**, so raw layer 5 reads 0.681 here and in `detector_overview` (c); `changepoint_localization`'s `threshold_detection` panel (a) asks the per-position question instead and reads 0.716 |

`results/eval_games.csv` is that evaluation set written down — the 500 game ids and their *t0*.
`move_legality.detection_auroc` reads it for the paired bars of
`figures/probe_directions_and_detection`, so those bars and these two panels score one set of
games rather than two built by the same recipe.

Band semantics: every interval is a 95% percentile bootstrap over **source games** as
clusters, 1,000 resamples, seed 42 — games and never positions, because one game's candidates
overlap in *M*−1 steps.

### Figure notes

`board_states` — game 3889 at *t0* = 20. The four non-*t0* panels mark the move actually
played, with the bracket lines that make it legal. They do not name their square: the *t0*
panel is the only one whose square is a claim. **A position has one move played in
it, so the *t0* panel marks one**: the branch the trace's peak belongs to, which for this game
is the **R2 bracketing** branch at h4 (`empty_no_bracket` — a line of opponent discs out of the
square that nothing of the mover's closes, drawn to the empty square that breaks it). The label
is that rule's code, in that rule's colour, so the board says which of R1, R2 and R3 the event
is; the trace draws all three branch arms as curves beside it. That panel is framed so the event is findable. Every marked square carries
the disc the mover would place and a verdict — a green tick on a legal move, a red cross on the
illegal one.

**Under each panel is what the detector says about that ply, and whether it is right**, green
for right and red for wrong. Off *t0* the row scored is the reference game itself, whose move
there is the legal one it played, so anything above τ is a false alarm; at *t0* it is the drawn
branch. On this game the detector is right at four of the five and false-alarms at *t0*−1.
Read that as one game, not as a rate — 37% of legal positions sit above τ. It
is a property of this illustrative game, not a measurement: the population numbers are
`auroc_by_rule` and `score_histograms`.

**Each rule is drawn from the member category that is legible as that rule**, chosen by
`select_game.py`'s `PREFERRED` table rather than by whichever branch came first: at this *t0*,
R1 is `occupied_opp` (f5), R2 `empty_no_bracket` (h4) and R3 `opponent_only` (f1). This matters
— R2's shallowest member, `empty_detached`, touches no opponent disc at all, so it draws an
empty square with nothing to see, which is exactly why `figures/othello_rules` avoids it too.
The pooled illegal arm's own draw for this game *is* that shallow member, which is why the
board draws `PREFERRED`'s branch and not the pooled one.

`auroc_by_rule` — what the bars compare, since the panel no longer prints it: each is the
illegal branch against **the same source game's legal control at the same *t0***, ranked on the
score at that game's own *t0*, one value per branch. `legal / illegal` is the pooled arm (one
illegal branch per game); R1/R2/R3 are that same arm split by rule.

**The evaluation set is flat over *t0*.** `detector.stratified` takes an equal quota of games
at every *t0* in the scored range — **n = 500 over *t0* = 10…34, 20 games at every one of the
25 positions** — because the tree
is not balanced on move index: `extend_branches.py` forked 1,665 of its 2,278 eligible games at
*t0* = 20 and about 30 at each other value, so any prefix of game ids is ~46% one position, and
move index is the strongest nuisance variable this tree has. `move_legality.detection_auroc`
deals its groups the same way.

**The range is [10, 34]** — the tree's whole fork window, and the same range
`move_legality.detection_auroc` scans. It used to stop at 16, then 14, because `detector.band`
passed the ablation's *M*max = 12 to `candidates`: a margin that buys one chance level across
every window size that study compares, and costs 16 positions at each end. This study never
varies the window — it scores *M* = 4 and nothing else, which reaches moves 6–55. The price of
scoring what *M* reaches is that this candidate set is no longer the ablation's, so neither τ
nor a peak position is comparable with that study position for position.

**n is `detector.N_EVAL` = 500** — 20 games at each of the 25 positions. The corpus ceiling
under an equal quota is 550, since the thinnest position (*t0* = 32, 33) holds 22; 500 spends
all but that last 9% and states as a round number. n = 1000 is not reachable under an equal
quota at all — it would need 40 games at every position.

`score_histograms` — same corpus, same eligibility, same statistic and the **same 500
games** as `auroc_by_rule`, on five arms held aligned game by game (asserted on `game_id` and
on *t0*). So the panel is literally the distribution those bars summarize, which it was not
while the two drew different prefixes of the tree.

`auroc_by_space_l0_matched` — the one panel here that is not raw layer 5. Its bars are the
eight `results/l0_operating_point/picks.csv` checkpoints, one per architecture, each the
nearest-L0 member of its own grid to a shared target of L0 219.16 (184-260 in practice), so a
gap between two bars is the architecture and not its sparsity — which is not true of the
seed-42 roster, whose L0s span 32.5 to 1,808. The statistic, the arms and the games are
`auroc_by_rule`'s: pooled illegal branch against the same source game's legal control, ranked
on the score at that game's own *t0*, over the 500 games of `results/eval_games.csv`, which
this script **reads** rather than re-deals. Whiskers are 95% percentile bootstrap CIs over
source games, 1,000 resamples, seed 42 — one branch per game per arm, so a game is a cluster of
one here. The gray band is the empirical chance level, each space's own legal control scored
against the second legal control the same game carries; it sits at 0.486-0.507. Bars sorted by
height, raw layer 5 in reference styling wherever it lands, hue = architecture family and hatch
= its second dictionary, drawn by `changepoint_localization.make_figures.panel` so it cannot
drift from that study's by-space bars. Numbers in `results/auroc_by_space_l0_matched.csv`; the
paired differences in `results/auroc_by_space_l0_matched_paired.csv`, on the bars' own draws.

**Four dictionaries clear raw's 0.681 [0.658, 0.704] on the paired difference**: GatedSAE
0.723 [0.700, 0.745] by +0.042 [+0.031, +0.054], BatchTopK 0.710 by +0.029 [+0.015, +0.044],
TopK 0.708 by +0.028 [+0.015, +0.040] and GatedAnneal 0.703 by +0.022 [+0.011, +0.034]. PAnneal
0.677 ties raw (-0.004 [-0.019, +0.013]), and SpaDE 0.634, Standard 0.612 and CosineSpaDE 0.498
sit below. **GatedSAE has the highest point estimate but is not separable from BatchTopK**
(+0.013 [-0.000, +0.027]); it does clear TopK (+0.015 [+0.003, +0.027]) and GatedAnneal. On the same corpus at
*unmatched* sparsity `changepoint_localization` calls GatedAnneal the best space, so that
ranking was reading L0 as much as architecture — see that study's "At matched L0" section.

`score_histograms` — the five series are Gaussian KDEs (Scott's rule), not bars. The y axis is
still a count: each curve is its density × n × one 22nd of the pooled range, so its height is
"games per slice of that width" and reads on the scale the binned version drew. One shared
width for all five, as the shared bin edges were. The bootstrap band the KDE helper returns is
not drawn — five overlapping curves cannot each carry one and stay readable.

`detector_trace` — no uncertainty band by construction: four branches of one prefix, so there
is nothing to average over. The shaded region is `t <= t0 - M`, not `t < t0`: the MOSUM window
reads *M* = 4 steps forward, so the score is branch-identical only until its window touches the
splice.

### What the panels do and do not say

**Game 3889 is illustrative, not typical.** `select_game.py` chose it *because* all three rule
arms peak at *t0*, the legal control does not, and the detector is right about four of the five
plies `board_states` draws. That last criterion is the strongest selection of the three, and
the panel says so by keeping the one false alarm: windows with no false alarm at all exist, and
they read as a cleaner detector than the population is. The population is the other two panels;
the trace is there to show the shape, and its docstring says so.

**The trace's right half is off-support.** After *t0* the branch replays the source game's
suffix onto a board that no longer exists — which is why `changepoint_localization` truncates
its curves at *r* = 0. Moves *t0*+1 … *t0*+5 read as "the score decays", never as a legality
judgement of those moves.

**τ flags the legal control too, in this game.** At *t0* the control scores 1.654 against
τ = 0.619. The three rule arms clear it by more, but the separation is in magnitude, not in
whether the cut fires — and `score_histograms` shows the same thing at population scale: 37% of
legal controls sit above τ.

**R2 ≫ R3 ≫ R1**, at 0.757 / 0.662 / 0.595 AUROC. Playing on an already-occupied square is
the hardest violation for this score to see, barely clearing chance. That is the same ordering
the supervised probe in `analysis/move_legality` finds by an unrelated route, where
`occupied_opp` is likewise its worst cell.

## Reproducing

```bash
PY=analysis/violation_divergence/illegal_move_detector
PYTHONPATH=src python $PY/extract_geometry.py     # GPU optional; raw needs no SAE encode
PYTHONPATH=src python $PY/select_game.py          # -> figure_game.json
PYTHONPATH=src python $PY/eval_games.py           # -> results/eval_games.csv, which every figure asserts
PYTHONPATH=src python $PY/figure_boards.py     [--camera-ready]
PYTHONPATH=src python $PY/figure_trace.py      [--camera-ready]
PYTHONPATH=src python $PY/figure_auroc.py      [--camera-ready]
PYTHONPATH=src python $PY/figure_histograms.py [--camera-ready]
PYTHONPATH=src python $PY/extract_geometry.py --l0-matched   # eight dictionary encodes
PYTHONPATH=src python analysis/straightness/l0_matched_profile.py   # their legal profile, once
PYTHONPATH=src python $PY/figure_auroc_l0_matched.py [--camera-ready]
```

`extract_geometry.py` writes `study_dir("divergence")/illegal_move_detector/{d,meta}.npz` and
leaves `curvature_discriminant`'s artifacts untouched — every number in
`changepoint_localization` is verified byte-identical against those, and this study must not
disturb them. Numbers behind the bars and the bins are in `results/`.
