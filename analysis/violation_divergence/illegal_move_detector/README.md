# The legal/illegal move detector, drawn

Four panels of one claim: a reference-free score on the model's own trajectory reacts to an
illegal move, at a stated operating point, and the reaction is graded rather than clean.

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
(`changepoint.threshold`), **refit on this population**. It is 0.629 here against the 0.869 in
`changepoint_localization/results/threshold_localization.csv`, and the difference is not a
disagreement: that table was fitted on the 575 games `curvature_discriminant` selected before
`extend_branches` grew the tree, and restricting this corpus to those 575 games reproduces
0.8691 exactly. Two things move it — excluding the scored games from the profile fit (+0.110)
and the larger population (−0.240).

**The objective is nearly flat across that whole range.** Balanced accuracy is 0.6721 at
τ = 0.629 and 0.6662 at τ = 0.869: a 0.24 move in the cut buys 0.006 accuracy. What τ actually
selects is the trade — 0.664 TPR / 0.680 TNR here against 0.584 / 0.748 there. Quote τ with
that caveat or not at all.

## Corpus

| | |
|---|---|
| eligible games | 2,207 (branch-complete, *t0* ∈ [16, 34], odd-parity source) |
| branch rows | 19,863 — reference, two legal controls, six illegal categories per game |
| scanned band | moves 14–47, 34 candidates |
| profile fit | 1,427 straightness legal games, excluding every game scored here |

Eligibility is `curvature_discriminant/branch_geometry.py`'s own rule, imported rather than
restated. The odd-parity filter and the `held_out_legal` exclusion are what keep a trajectory
out of the baseline it is scored against.

## Figures

| file | what it shows |
|---|---|
| `board_states.png` | the board at *t0*−2 … *t0*+2 of game 3889: the legal move played at each of the four surrounding plies, and the illegal move the drawn branch plays at *t0* |
| `detector_trace.png` | the detector score for that same game over moves *t0*−5 … *t0*+5, three rule arms against the matched legal control, with τ |
| `auroc_by_rule.png` | AUROC at *t0*, illegal branch against the same game's legal control, pooled and per rule, n = 418 games |
| `score_histograms.png` | the distributions that AUROC summarizes: the score at *t0* for legal, illegal, R1, R2, R3, the same 418 games, with τ |

Band semantics: every interval is a 95% percentile bootstrap over **source games** as
clusters, 1,000 resamples, seed 42 — games and never positions, because one game's candidates
overlap in *M*−1 steps.

### Figure notes

`board_states` — game 3889 at *t0* = 20. The four non-*t0* panels mark the move actually
played, named, with the bracket lines that make it legal. **A position has one move played in
it, so the *t0* panel marks one**: the branch the trace's peak belongs to, which for this game
is the **R2 bracketing** branch at h4 (`empty_no_bracket` — a line of opponent discs out of the
square that nothing of the mover's closes, drawn to the empty square that breaks it). It is
labelled `illegal` rather than by its rule, because "illegal" is what the position knows about
it; which rule it breaks is this file's to say, and the trace's, where all three branch arms
are drawn as curves. That panel is framed so the event is findable. Every marked square carries
the disc the mover would place and a verdict — a green tick on a legal move, a red cross on the
illegal one.

**Under each panel is what the detector says about that ply, and whether it is right**, green
for right and red for wrong. Off *t0* the row scored is the reference game itself, whose move
there is the legal one it played, so anything above τ is a false alarm; at *t0* it is the drawn
branch. On this game the detector is right at four of the five and false-alarms at *t0*−1. It
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
at every *t0* in the scored range — **22 games at each of 16…34, n = 418** — because the tree
is not balanced on move index: `extend_branches.py` forked 1,666 of its 2,207 eligible games at
*t0* = 20 and about 30 at each other value, so any prefix of game ids is ~46% one position, and
move index is the strongest nuisance variable this tree has. `move_legality.detection_auroc`
deals its groups the same way. The quota is the smallest stratum, which is what caps n at 418:
outside *t0* = 20 the whole tree holds only 541 eligible games. n grows on its own if the tree
is extended at the sparse positions.

`score_histograms` — same corpus, same eligibility, same statistic and now the **same 418
games** as `auroc_by_rule`, on five arms held aligned game by game (asserted on `game_id` and
on *t0*). So the panel is literally the distribution those bars summarize, which it was not
while the two drew different prefixes of the tree.

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
τ = 0.629. The three rule arms clear it by more, but the separation is in magnitude, not in
whether the cut fires — and `score_histograms` shows the same thing at population scale: 34% of
legal controls sit above τ.

**R2 ≫ R3 ≫ R1**, at 0.764 / 0.661 / 0.599 AUROC. Playing on an already-occupied square is
the hardest violation for this score to see, barely clearing chance. That is the same ordering
the supervised probe in `analysis/move_legality` finds by an unrelated route, where
`occupied_opp` is likewise its worst cell.

## Reproducing

```bash
PY=analysis/violation_divergence/illegal_move_detector
PYTHONPATH=src python $PY/extract_geometry.py     # GPU optional; raw needs no SAE encode
PYTHONPATH=src python $PY/select_game.py          # -> figure_game.json
PYTHONPATH=src python $PY/figure_boards.py     [--camera-ready]
PYTHONPATH=src python $PY/figure_trace.py      [--camera-ready]
PYTHONPATH=src python $PY/figure_auroc.py      [--camera-ready]
PYTHONPATH=src python $PY/figure_histograms.py [--camera-ready]
```

`extract_geometry.py` writes `study_dir("divergence")/illegal_move_detector/{d,meta}.npz` and
leaves `curvature_discriminant`'s artifacts untouched — every number in
`changepoint_localization` is verified byte-identical against those, and this study must not
disturb them. Numbers behind the bars and the bins are in `results/`.
