# Shaded uncertainty bands out of the straightness figures (branch `fig/bands-straightness`)

Every per-series `fill_between` in the straightness study is gone. Where the
panel had markers to hang an interval on, the band became thin 95% CI bars at
those markers; where it did not, the curve now carries no interval and this note
says so. Gray annotation spans and the gray legal-vs-legal null envelope stay —
they mark structure and a null range, not one series' uncertainty.

- `src/spade/figures/straightness_panels.py` — the shared panels.
- `analysis/straightness/make_figures.py`, `.../sae_profiles/make_figures.py` — the figures.
- `analysis/straightness/README.md`, `.../sae_profiles/README.md` — the captions.

## What replaced the bands

`straightness_panels.ci_bars` is the one styling rule: `errorbar(fmt="none",
elinewidth=0.9, capsize=1.8)` at the positions `markevery` already draws. Four
classes stagger their markers, so the bars land on different x and stay
separable — the thing overlapping ribbons were not.

| panel | before | now |
|---|---|---|
| `profile_panel`, `value_profile_panel` (curvature / step-length / norm vs. board position) | ±1.96 SEM ribbon over 58 positions | CI bars at the ~7 marker positions per class |
| `violation_panel` (event-locked Δ) | ±1.96 SEM ribbon over the 21-position window | CI bars at all 21 markers |
| `f2_auroc_vs_depth`, `f6` panel (b) | per-class bootstrap CI ribbon across 9 depths | CI bars at the 9 depth markers, asymmetric `[y−lo, hi−y]` from the same bootstrap columns |
| `sae_profiles.combined_comparison` | ±1.96 SEM ribbon per quantity | CI bars at each quantity's marker positions |

The AUROC-vs-depth panels are the case the guidelines already name (§4.1, "few
layers → markers + error bars"), so those got better, not just quieter.

## Figures that now carry no uncertainty display

The KDE density panels. `density_panel` drew a percentile bootstrap band per
class; four densities whose medians sit ~0.3° apart overlap too much for any
per-curve interval — bars on a density curve read as bin noise — so the band was
deleted with nothing in its place. This is a live conflict with
FIGURE_GUIDELINES §1.2 / §4.4 and the owner should decide it:

- `analysis/straightness/figures/theta_dist_layer5.png` (F1, both panels)
- `analysis/straightness/sae_profiles/figures/theta_dist_comparison.png` (9 panels)

Mitigation, not a fix: both captions now say the curves carry no interval and
point at the figure where the same separation is scored with a bootstrap CI
(`auroc_vs_depth.png`, `sae_families_auroc.png`). `kde_ci` still returns its
band — `analysis/bracket_flip` draws it, and that study is not mine.

Two panels also keep bars that are invisible at print size, because n = 2000
makes the SEM far smaller than a marker: `step_vs_t.png` (a) and
`norm_diagnostics.png` (a). The bars are drawn and correct; the honest reading
is that the between-class gap dwarfs the within-class error, which is what those
two panels claim. Worth a look if the owner wants the point made louder.

## Verified by eye

Rendered with `PYTHONPATH=$PWD/src` against the netscratch geometry npz:

- `analysis/straightness/figures/`: `theta_dist_layer5`, `curvature_vs_t`,
  `step_vs_t`, `mean_vs_var_theta`.
- `analysis/straightness/sae_profiles/figures/`: all 27 (8 families × 3, plus
  `violation_spike_comparison`, `theta_dist_comparison`, `quantity_comparison`).

**Not verified with real numbers:** `auroc_vs_depth.png`,
`sae_families_auroc.png`, `norm_diagnostics.png`. `results/straightness/auroc.csv`
does not exist — `classify.py` has not been re-run since the port — so
`make_figures.main()` cannot complete. Their drawing code was checked against a
synthetic `auroc.csv` (scratchpad only, never written into `results/`), which
confirms the categorical x + asymmetric `yerr` lays out correctly; the numbers in
those three renders were fake. Re-run `python -m spade.straightness.classify`
before trusting them.

Captions that moved: the KDE note in F1 and `theta_dist_comparison`; the
`mean ± 95% CI` line in `curvature_vs_t`, `step_vs_t`, `norm_diagnostics`,
`quantity_comparison`, and `sae_profiles.CI_NOTE` (which covers all 24
per-family figures); the `theta_dist_layer5`, `auroc_vs_depth`, `curvature_vs_t`
and `theta_dist_comparison` rows of the two READMEs.
