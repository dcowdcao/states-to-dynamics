# Curvature at the fork, from two tensor trees (branch `fig/curvature-figure`)

A four-panel figure putting one game's rule-violating branches on the straightness
study's curvature axis, rendered from the spliced and the resampled branch trees
through one code path.

- `src/spade/violation_divergence/rule_curvature.py` — the measurement.
- `analysis/violation_divergence/rule_curvature/make_figures.py` — draws only.
- `analysis/violation_divergence/rule_curvature/README.md` — the captions, the
  θ-comparability check, and the spliced-vs-resampled table.
- 20 figures under `.../rule_curvature/figures/`: 9 spaces × 2 trees, plus one
  nine-space combined figure per tree.

## What was built

`--tree {branches,rollout}` selects the tensor tree; everything downstream is
written once. The abstraction that makes that possible is `Fork.theta` shaped
`[n_branch, K, n_turn]`, with K = 1 on the spliced tree and K = 4 on the
resampled one. Δ against the reference is then a broadcast that pairs rollout *k*
with reference rollout *k*, which is the pairing `rollout_branches`' common random
numbers were built for.

Panels: (a) the straightness study's three legal-by-construction classes over the
board, n = 2000 each; (b) game 1235's nine branches on (a)'s shared x and y,
n = 1; (c) the violation-locked Δ per rule with both legal controls as nulls;
(d) the board at *t0* with every branch move on its square. The combined figure
is (c) in all nine spaces over the same board — the `sae_profiles` precedent,
where per-family figures carry the profile and the cross-family figure carries
only the measurement.

## The θ-comparability verdict: comparable, but only per ply

Verified before drawing, and the check is in the study README:

- 1998 of 2000 reference branches are token-identical to a straightness legal
  game (same dataset, same seed), and on those the two pipelines' θ agree to
  **1.5 × 10⁻⁵ deg** against a 142 deg level. Class-mean profiles differ by at
  most 0.012 deg against a 0.094 deg SEM.
- Δθ against the reference is exactly 0.0 up to the turn at *t0* − 2 and first
  nonzero at *t0* − 1 — the straightness study's own alignment, re-derived.

The sibling audit's **NOT COMPARABLE** verdict is about `fork_plane`'s
`STRIDE = 2` window, which this figure never uses. Same branches, same game:
**107.9 deg** at stride 2 against **142.9 deg** per ply, reproducing the audit's
107.71. The audit's recommendation — recompute the branches per ply rather than
re-derive the class baselines at stride 2 — is what `rule_curvature` does, and
the module docstring says why it is not a free choice. The R²¹ trailing window is
not used either: at W = 8 a point plotted at *t* has vertex centroid *t* − 7,
which an event-locked panel cannot carry.

## Things that turned out to matter

**`CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are the same blue.**
The first draft overlaid the classes and the fork on one axes and gave one colour
two meanings — the exact failure FIGURE_GUIDELINES §5 exists to prevent. They are
now two axes sharing x and y, which also un-crushes the class curves: a 2000-game
mean spans 19 deg where one game swings 35.

**R3 has one member, and it is handled rather than silently NaN.** `Rule.sem` is
`None` at one sequence, `draw_rule` then draws no band, and the panel title says
"none where there is one". On the resampled tree R3 *does* get a band (4
rollouts) and the legend says `mean of 1 × 4 rollouts`, so the two are not
confusable.

**φ per member, then average.** `locked()` writes the nested mean out rather than
pooling sequences; pooling agrees only while every member carries the same K, and
the order is what φ's nonlinearity constrains.

**Both legal controls are drawn**, against `fork_and_board`'s one. At n = 1 a
legal alternative's Δ is the same order as the illegal branches' (at *t* − *t0* =
0: controls −5.14 / −9.25 against −13.60 .. −3.46), and a figure showing only the
illegal branches would imply a separation that is not visible at n = 1.

## What the two trees say

The response at *t0* − 1 is **bit-identical** across trees in every space (its
window is positions *t0*−2 .. *t0*, which both trees fix) — asserted, not
eyeballed. The post-*t0* tail keeps its sign and grows slightly under resampling.
What does not survive is the fine structure: the two trees' mean curves correlate
only **+0.20 (R1), +0.38 (R2), +0.60 (R3)** over *t* ≥ *t0*, and the spliced
tree's bands are ~2× too narrow because it has no continuation variance at all.
R1's localized trough (−7.51 at *t0*+1) does not survive; it becomes −4.53 at
*t0*+12, i.e. no localized trough. Full table in the study README.

Not supported: that the shared suffix is what makes the six illegal branches
co-vary after *t0*. Mean pairwise correlation is +0.536 spliced and +0.580
resampled — that is the game, not the splice.

## Shared-module changes, and their regression checks

Three edits outside the new files, each because one rule was about to be written
twice:

- `straightness_panels.align_to_event` extracted from `violation_delta` and
  `value_delta`, which now call it. Verified **array-identical to the
  pre-refactor body** on the real geometry npz for θ, ‖Δz‖ and ‖z‖; the θ trough
  reproduces the straightness README's −4.3 deg at *t0*+1.
- `branch_style.ordered_legend` accepts several axes, since this figure's series
  live on three panels, and de-duplicates by label.
- `branch_style.legend_proxy` now passes two positional args. Three empty lists
  are `zs` on a 3-D axes but a second series on a 2-D one, so the proxy was
  producing a duplicate legend entry on 2-D axes.
- `branch_style._rank` puts a `(baseline)` label first, ahead of `reference`
  (FIGURE_GUIDELINES §3). No existing figure draws a `(baseline)` label through
  this legend, so relative order is unchanged.

Regression: both `fork_and_board` figures re-render **byte-identical** (sha256)
after the shared-module edits, and reproduce their spread ratios (0.885
displacement, 0.902 R²¹). `test_curvature_plane` 3/3 and `test_divergence` 11/11.
`sae_profiles/make_figures.py` renders all 27 figures with its own
`prefix delta == 0` assert passing in every family (output deleted afterwards —
those figures are untracked and were not mine to add).

## Not done, and why

- **`analysis/straightness/make_figures.py` could not be run.**
  `results/straightness/auroc.csv` does not exist in this tree or in
  `spade_sae`; `classify.py` has not been re-run since the port. That is a
  pre-existing data gap, not a regression from the refactor — which was instead
  verified array-identical, a stronger check than a render.
- **PNG only, no camera-ready PDFs shipped.** `--camera-ready` is verified to
  render, but these are 9.6-inch README figures; a PDF is a flag away and
  20 extra tracked binaries are not worth it until one is chosen for a paper.
- **The SAE weights are symlinked into this worktree.** `checkpoints/**/ae.pt` is
  gitignored and lives only in `spade_sae`, so 11 symlinks were created to make
  the registry load. Nothing enters the commit; a fresh worktree needs them again.
- **`--game` other than 1235 is untested.** `FIGURE_GAME` is held fixed across
  both trees on purpose — the two figures must share a game or the
  spliced-vs-resampled difference cannot be attributed.
- **Grayscale.** (a) separates on line weight plus marker. In (c) the three
  Okabe–Ito rule hues are near-isoluminant, so separation rides on the markers
  (square / triangle / diamond), which is the repo-wide `branch_style` contract;
  giving the rule means distinct line styles here would have desynchronised them
  from `fork_and_board`.

PROCESS.md was deliberately not edited.
