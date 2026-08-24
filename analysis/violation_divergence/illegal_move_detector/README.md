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
| `board_states.png` | the board at *t0*−2 … *t0*+2 of game 2498: the legal move played at each of the four surrounding plies, and one illegal candidate per rule at *t0* |
| `detector_trace.png` | the detector score for that same game over moves *t0*−5 … *t0*+5, three rule arms against the matched legal control, with τ |
| `auroc_by_rule.png` | AUROC at *t0*, illegal branch against the same game's legal control, pooled and per rule, n = 1,000 games |
| `score_histograms.png` | the distributions that AUROC summarizes: the score at *t0* for legal, illegal, R1, R2, R3, n = 200 games each, with τ |

Band semantics: every interval is a 95% percentile bootstrap over **source games** as
clusters, 1,000 resamples, seed 42 — games and never positions, because one game's candidates
overlap in *M*−1 steps.

### What the panels do and do not say

**Game 2498 is illustrative, not typical.** `select_game.py` chose it *because* all three rule
arms peak at *t0* and the legal control does not. The population is the other two panels; the
trace is there to show the shape, and its docstring says so.

**The trace's right half is off-support.** After *t0* the branch replays the source game's
suffix onto a board that no longer exists — which is why `changepoint_localization` truncates
its curves at *r* = 0. Moves *t0*+1 … *t0*+5 read as "the score decays", never as a legality
judgement of those moves.

**τ flags the legal control too, in this game.** At *t0* the control scores 1.219 against
τ = 0.629. The three rule arms clear it by more, but the separation is in magnitude, not in
whether the cut fires — and the histogram shows the same thing at population scale: 34% of
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
