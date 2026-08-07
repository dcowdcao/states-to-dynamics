# Finish note — dropping the shaded uncertainty ribbons from `violation_divergence`

Branch `fig/bands-divergence`. Every per-series shaded CI/SD/IQR band in
`analysis/violation_divergence/` is gone. Null envelopes and gray annotation spans stayed.
Five of the seven figures now carry no uncertainty on the figure at all; that is a live
FIGURE_GUIDELINES §1.2 violation and the decision below is yours, not mine.

## What was removed

| file | what it was | what replaced it |
|---|---|---|
| `make_figures.py` `series()` → F1 `divergence_layer5` | 7 × 95% CI of the mean over source games | **nothing** |
| `make_figures.py` `series()` → F3 `depth_sweep` | same, 9 categorical ticks | **error bars** (§4.1: few layers → markers + error bars, and the caption already said "error bars") |
| `theta_comparability` `level_panel` | 2 × ±1.96 SEM over 2000 games | **nothing** — the band was ≤ 0.34° wide against the 34.4° gap the panel annotates, i.e. invisible |
| `game_trajectory` `marginal_panel` | 9 × ±1 SD over K = 4 rollouts (rollout variant only) | **nothing** |
| `behavioral` `curve()` → F6 | 8 × 95% percentile bootstrap over source games | **nothing** |
| `probe_readout` `curve()` → F7 | 6–8 × 95% bootstrap over source games | **nothing** |
| `curvature_discriminant` `band_panel` → panels (j), (k) | IQR over the 8 SAE dictionaries around their median | **nothing** — the median line stays; each dictionary is a row of `evaluation.csv` |

Every interval removed is still computed and still written to its CSV
(`divergence_curves*.csv`, `behavioral_curves*.csv`, `probe_*_illegal_square.csv`,
`evaluation.csv`). Nothing was deleted from the measurement side, and no numeric output
changed: the CSVs re-render byte-identically.

## What was kept, and why

- **`fork_expanding_curvature/make_figures.py` ~191** — `fill_between([j−0.36, j+0.36],
  null.min(), null.max(), color=GRID)`. This is a **null envelope**, not a series' spread:
  it is the range of the time-shuffled null across branches, drawn as a gray slab in one
  categorical slot with the observed points scattered on top of it. Kept, as instructed.
  That figure re-renders byte-identical.
- **`r21_shared_basis/make_figures.py` ~399** — two 5–95% permutation-null envelopes, not
  per-series bands. The `facecolor=face` / `alpha=0.18 if face != "none" else 1.0` branch
  is drawing **two different things**: for `"activation refit"` the facecolor is `"none"`
  and the alpha is 1.0, so it is a hatched (`///`) outline in `NULL["color"]` with no fill
  at all; only the `"shared"` arm is an actual gray fill. Both are the null, both are gray,
  both are labelled `null, 5–95% (...)`. Kept. That study is untouched — no re-render.
- **`fork_curvature_space`'s jackknife ellipse** — PROCESS.md (2026-08-05) records that its
  semantics were deliberately moved into a legend entry. It is `Ellipse(..., fill=False)`
  in `embedding_panels.class_ellipse`: an **unfilled outline**, so it is not shaded
  contouring and the objection does not reach it. Kept, untouched.
- All `axvspan` / `axhspan`: the pre-*t0* prefix spans, the `r ≥ 8` saturation span in F1,
  `game_selection`'s population IQR span, `curvature_discriminant`'s within-game
  label-permutation null. Structure and nulls, not per-line shading.

## Panels that now show a mean with no uncertainty — needs your call

1. **F1 `divergence_layer5` (a) and (b).** (b) is the claim-supporting panel: the paired
   illegal-minus-control excess, and the study's headline `ratio_illegal_control_L5` is
   quoted with a CI. Six ribbons around zero were the worst clutter in the study, so I do
   not think they should come back as-is, but this is the panel that most needs something.
2. **F6 `behavioral` (a) and (b).**
3. **F7 `probe_readout` (a) and (b).**
4. **`game_trajectory` (b), rollout only.** n = 4 within one game; the figure was already
   labelled shape-only, so the loss here is smallest.
5. **`curvature_discriminant` (j), (k).** The median over 8 dictionaries with no spread.
   §1.7 would want the 8 members drawn faint instead — I did not, because these are
   third-of-a-row panels and 8 extra lines each is worse than the band was. That is the
   cheapest fix if you want one.

`theta_comparability` (a) is *not* on that list: its interval was smaller than the line
width, and the caption now says so with the number.

## Captions, titles and prose reconciled

- `make_figures.py` — `F1_NOTE` both variants ("bands are 95% CI…" → "each curve is a mean
  over source games / branches, 95% CIs in `divergence_curves*.csv`"). `F3_NOTE` unchanged;
  it already said "error bars" and now that is true. `series()` gained a docstring saying
  why.
- `theta_comparability/make_figures.py` — figure note "(a) mean ± 95% CI over games" →
  "mean over games; its 95% CI half-width is ≤ 0.34° at every *t*, well inside the gap the
  panel annotates, and is not drawn". `theta_comparability/README.md` §6 likewise.
- `game_trajectory/make_figures.py` — `variant_note`'s "Band = ±1 SD (n = K)" → "No spread
  is drawn over those K". `delta_hat` returns the mean only (the SD had no other caller).
  `game_trajectory/README.md` — the "**The band in (b) is ±1 SD…**" paragraph and the
  "Reading them honestly" bullet both rewritten.
- `behavioral/make_figures.py` — F6's legend title, both variants ("bands: 95% CI…" →
  "curves are means over source games/branches … their 95% CIs are in
  `behavioral_curves*.csv`"). `named()` no longer renames `lo`/`hi`.
- `probe_readout/make_figures.py` — `BAND` → `ESTIMATOR` ("mean over source games; 95%
  bootstrap CIs in the CSV"), `band_legend()` → `curve_legend()`; it is the legend title in
  three panels and in the rollout figure's two under-figure legends.
  `probe_readout/RESULTS.md` §F7 rewritten.
- `curvature_discriminant/make_figures.py` — `band_panel()` → `auroc_panel()`, series label
  "8 SAE dictionaries (median, IQR)" → "(median)", figure caption's "band = IQR over the 8
  dictionaries" → names `evaluation.csv` instead. The percentile call is now a plain
  `np.median`.
- `RESULTS.md` (study level) — "Bands are 95% CI over source games" → "Error bars are…";
  the FIGURE_GUIDELINES compliance paragraph no longer claims every curve carries
  uncertainty and states the F1 exception explicitly; the integrator note's "a
  category-styled curve with a CI band" → "a category-styled mean curve".
- **`rule_curvature/README.md`** — a leftover from commit 951485e, which dropped that
  study's SEM ribbons in the code and the panel title but not in its README. Four places
  still described the band ("The band is ±1.96 SEM over the sequences a rule averages…",
  "R3 has a band here", the "(no band)" table cell, "the spliced tree's bands understate
  the uncertainty by 2×"). Reconciled; the ±/SEM *numbers* in the comparison table are
  untouched, since those are printed statistics rather than a drawn band.

Left alone deliberately: `README.md`'s "0.44–0.54 band", `RED_TEAM.md`'s "1–6% band" and
`sae_spaces`'s "0.91–1.03 band" are numeric ranges in prose, not figure elements.
`fork_and_board` and `rule_curvature`'s "band of white beside the board" is a layout term
for the legend strip.

## Shared-helper change I am recommending but did not make

**`src/spade/figures/straightness_panels.py` still draws ±1.96 SEM fills**, at lines 73
(`kde_ci`), 137 (`value_profile_panel`), 153 (`profile_panel`) and 186 (`violation_panel`).
That file is a sibling's. It reaches into this study through
**`rule_curvature/make_figures.py:127`**, which calls `profile_panel` for its panel (a) —
so `curvature_rules_*` still ships three shaded class bands in its top-left panel while its
own panels (b) and (c) have had theirs removed. That is the one visible inconsistency left
in this directory and I could not fix it from here. `theta_comparability` imports only
`DRAWN_CLASSES` from that module, so nothing else of mine is affected.
*Renumbered since — the panels named here are the four-panel layout: (b), one game's
fork on (a)'s scale, was removed, so the (c) above is now (b) and the board (d) is now
(c). See `docs/finish_notes/panelb-docs.md`.*

If `profile_panel` loses its band, `rule_curvature/README.md:85` ("each a mean ± 95% CI
over n = 2000 games") and `rule_curvature/make_figures.py:228` (panel (a)'s title, "mean
± 95% CI, n = 2000 games per class") both need the same edit, and both are outside this
branch.

## Rendering

`PYTHONPATH=$PWD/src SOURCE_DATE_EPOCH=1`, `/n/home10/dcao/.conda/envs/spade_env/bin/python`,
login node. Every figure below was rendered before the change (reproducing its committed
bytes exactly, which is what makes the after-diff readable), then after, then looked at.

Rendered and inspected — spliced and rollout where the script has `--variant`:
`divergence_layer5`, `depth_sweep`, `theta_sampling_mismatch`, `game_1235_fork_pca`,
`f7_probe_readout`, `f6_behavioral_divergence`, `space_fork_grid_discriminant`,
`r21_expanding_magnitude` + `space_fork_grid_r21_expanding`.

`game_1235_fork_pca.png` (spliced) and both `fork_expanding_curvature` figures re-render
**byte-identical**, which is the check that the spliced path and the kept null envelope
were not touched.

Nothing failed and nothing OOMed. Two notes for whoever renders next:

- `r21_shared_basis` needs the SAE weights, which are gitignored and absent from a fresh
  worktree. `checkpoints/*/trainer_*/ae.pt` are symlinked here to
  `/n/home10/dcao/spade_sae/checkpoints/`; the symlinks are untracked and cost nothing.
  With them the whole study renders on the login node in ~6 min.
- `space_fork_grid_r21_isomap{,_rollout}.png` do not byte-reproduce (`Isomap(n_jobs=-1)`'s
  eigensolver, ~1e-5 of pixels by 1 grey level — PROCESS.md already records this). I
  reverted those two to their committed bytes rather than ship the noise.

## Unrelated rot found on the way, not fixed

- **`RESULTS.md`'s figure list describes F1 and F3 as four panels each.** Both have been two
  panels since the cosine panels were dropped (`make_figures.py` says so in a comment) and
  the F3 bullet enumerates a panel (d) that no longer exists. I edited only the sentences
  about bands; correcting the panel inventory is a separate change and I did not want to
  guess at it inside this one.
- ~~**`probe_readout/make_figures.py:127 parity_panel()` is dead.**~~ Deleted with
  `PARITY_BRANCHES` on `gray/gray-parent`; `probe_parity.csv` still holds the data.
