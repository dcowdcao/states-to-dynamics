# Finish notes — dropping the random-token class from the straightness study

The straightness study built five game classes and drew all five. The random-token class
is now display-only-removed: it stays in `games.pt` and `classify.py` still scores it,
but nothing draws it, reads its AUROC rows, or fits on it. Branch `fig/drop-random-tokens`.

Every figure below was re-rendered with `/n/home10/dcao/.conda/envs/spade_env/bin/python`
and inspected. **Set `PYTHONPATH=$PWD/src` when running them here** — `spade` is installed
editable from `/n/home10/dcao/spade_sae/src`, so without it the scripts silently import the
main worktree's `paper_style` and draw five classes again.

## The seam, and why there

`CLASS_STYLE` in `src/spade/figures/paper_style.py`. Deleting the `random_tokens` entry is
the entire removal; everything else derives from it:

- `spade.figures.straightness_panels.DRAWN_CLASSES` = the `CLASSES` order filtered to
  styled classes, and `class_blocks(n_total)` turns that into the `games.pt` block slices
  every figure iterates.
- `analysis/straightness/make_figures.py`'s `OOD_PAIRS` and `DEPTH_SERIES` are built from
  `DRAWN_CLASSES`, so no figure asks `auroc.csv` for a `legal_vs_random_tokens` row.
- The four panel helpers took a `classes` argument next to `blocks`; they now iterate
  `blocks` alone, so there is no longer a way to pass a class the study does not draw.

**`CLASSES` in `spade.straightness.build_games` is untouched, deliberately.** It fixes the
block layout of `games.pt`, and every geometry npz indexes into it by position. Removing
the class there would re-slice every array — championship games would be read as random
tokens — and nothing would raise. The list is a data contract; the style dict is a display
decision; they are now separated.

Two consumers were fitting on the class rather than merely drawing it, and both had to
change or the removal would have been cosmetic:

- `density_panel`'s KDE grid spanned all of `per_game`, so the random-token block set the
  x-range of a panel it drew no curve in.
- `feature_pca`'s PCA fit its axes, variance ratios and sign pinning on all 10 000 games.

## Figures touched — 5 classes before, 4 after, all of them

| figure | before → after |
|---|---|
| `figures/theta_dist_layer5.png` | 5 → 4 curves; x-range 137-146° instead of 132-146° |
| `figures/auroc_vs_depth.png` | 4 → 3 series + null; legend 5 → 4 |
| `figures/curvature_vs_t.png` | 5 → 4 profiles (panel b unchanged, violation-locked) |
| `figures/step_vs_t.png` | 5 → 4 profiles |
| `figures/mean_vs_var_theta.png` | 4 → 3 facets |
| `figures/sae_families_auroc.png` | 4 → 3 panels |
| `figures/norm_diagnostics.png` | 5 → 4 profiles; 4 → 3 depth series |
| `sae_profiles/curvature_vs_t_*.png`, `step_vs_t_*.png`, `norm_vs_t_*.png` (24) | 5 → 4 profiles in panel (a) |
| `sae_profiles/theta_dist_comparison.png` | 5 → 4 curves; panel titles now Δmedian viol/champ |
| `sae_profiles/violation_spike_comparison.png`, `quantity_comparison.png` | unchanged — violation-locked throughout |
| `feature_pca/restrav_pca.png` | 5 → 4 clouds, **and a refit plane** |

No empty legend entry, no facet gap and no stale n anywhere; every render exited 0 (nothing
was OOM-killed). `sae_profiles/make_figures.py` takes ~5 min for its 27 figures, which is
past a two-minute agent-shell timeout: background it.

## What the retirement cost

Three findings die. **None was refuted** — the study stopped measuring the class. All three
are listed in `docs/claims_status.md` § *Retired — do not restate*.

1. **`theta_dist_comparison`'s claim, "Every dictionary keeps the random-token shift — and
   most surface a human-game shift that raw layer 5 hides."** The figure stays; its subject
   changes to the class that is still drawn. Every dictionary keeps the *violation* shift
   too, and unlike the random-token version the claim needs checking rather than eyeballing:
   the eight SAE mean-θ AUROCs run 0.39-0.46 with every bootstrap CI upper bound below 0.5
   (largest, GatedSAE, 0.481). The second half of the claim is unaffected and is what the
   figure's README section was already about.
2. **`sae_profiles/README.md`'s norm/L0 regime table** (pass-through / suppressed /
   inverted). Every cell was a random/legal ratio. The three-regime split it named is
   readable off the `quantity_comparison` table that stays — the same four spaces whose
   norm response to a violation is positive — so the cross-reference to it was rewritten
   rather than deleted. `per_family_step` and `per_family_norm` no longer return the ratios.
3. **`feature_pca/README.md`'s "PC1 is the random-token axis, everywhere."** This one did
   not merely lose its subject; refitting the plane changed every number in that README.
   PC1 carries 16-24% of the variance rather than 43-55%. Alignment with raw L5 falls from
   0.76-0.99 to 0.13-0.95 — the old near-unanimity was every dictionary reproducing one
   gross OOD direction, not the descriptor. Championship is now the only class that moves
   in the plane (0.3-2.1 pooled SD along a PC; rollout and violation never exceed 0.27),
   which is the figure's new suptitle. The CosineSpaDE sub-cluster survives, re-derived in
   the new plane: 302 of 8000 drawn games (violation 113, legal 112, rollout 77,
   championship 0), same delayed-code-recruitment mechanism.

## One thing corrected that the removal only exposed

`analysis/straightness/README.md` claimed violations "read marginally *curvier* than legal
in layers 0-3 (0.56 at L0)". They do not. The legal-vs-legal null is not centred on 0.5
early — it reads 0.538 at W_E and 0.556 at L0 — and violations score 0.520/0.556/0.560/
0.533/0.507 across W_E-L3, inside the null's CI at every one of them. The claim was the
noise floor. Rewritten to what the figure shows: indistinguishable through layer 3, falling
away at layer 4 (0.454, CIs still touching), clear of the band from layer 5
(0.397 [0.372, 0.421] against a null of 0.504 [0.469, 0.541]).

This was invisible while random tokens were on the plot at 1.00.

## Committed numbers now unreferenced

**None, in this repo** — the straightness results CSVs are not committed here.
`results/straightness/` holds only a `.gitkeep`; the `auroc.csv` in the working tree is
untracked (copied from the frozen repo so the figures could be re-rendered) and was left
alone, as were the `legal_vs_random_tokens` rows `classify.py` still writes into it. Those
rows and the whole `random_tokens` block of `summary_stats.csv` are now written and never
read. In the frozen source repo, `results/straightness/{auroc.csv,summary_stats.csv}` are
committed and carry the same unreferenced rows; that repo is read-only and untouched.

Prose numbers that were deleted rather than orphaned, for the record: the depth table's
`legal vs random_tokens` row (0.81 / 1.00 / 0.01 / 0.02 / 1.00), the norm/L0 regime table's
16 ratios, and `feature_pca`'s old alignment table.

One stale cross-reference is knowingly left: `docs/port_notes/straightness.md` § *Claims
hygiene* records that at port time the surviving prose said "every family preserves the
separation (0.01-0.21)". That is a dated record of the port, not a live claim; the live one
in `docs/claims_status.md` now reads 0.39-0.46.

## Judged wrong in the partial work inherited

The seam, the panel-signature change, the KDE-grid fix and the PCA refit were already right
and were kept, as were the recomputed medians, bandwidths, norm means and step-length
crossings — every one of them checks out against the npz. Four things did not:

- The early-layer "curvier" claim above, which the previous pass rewrote into the README
  without checking it against the null band the figure draws.
- `feature_pca/README.md` was not touched at all, so its Results section still said PC1 was
  the random-token axis while the script it documents had already been refit.
- `sae_profiles/README.md` kept the norm/L0 regime table and two cross-references to it,
  including a random-game norm figure (+14%) nothing computes now.
- `restrav_pca`'s suptitle asserted championship separation "in most dictionaries" with
  nothing behind it. `class_shift` now computes the standardized displacement and
  `make_figures.py` prints it beside the alignment table, so the README's numbers come out
  of the script that draws the figure.

## Known deviation, not fixed

`feature_pca`'s `pca_panel` draws one scatter per class in a loop, so championship — the
class the figure is now about — is painted last and on top. FIGURE_GUIDELINES §4.6 wants a
single shuffled scatter with class identity carried by the centroid and contour, which the
panel already has. Pre-existing, and a rewrite rather than a fix; flagged here instead of
being folded into this change.
