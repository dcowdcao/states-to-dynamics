# Finish note — dropping the shaded bands from bracket_flip, rules_causal, legality_probe

Three `fill_between` ribbons removed. Each one annotated a curve that has no other
uncertainty display, so none of them was a straight delete: the interval moved onto the
markers the curve already carries, drawn thin with a small cap. Nothing else changed about
what is plotted.

| figure | was | is |
|---|---|---|
| `bracket_flip` fig2 (a) | 95% bootstrap fill under three KDE curves | bars at the 1–2 marked points per curve |
| `rules_causal` `pair_patch`, `forfeit` (b) | 95% CI fill under each layer sweep | bars at every layer marker |
| `legality_probe` F2 (a)–(e) | 95% bootstrap fill under 11 curves × 5 panels | bars at every fifth offset, staggered per series as the markers already were |

**No figure lost its uncertainty display.** Where a panel had one, it still has one.

## The guidelines conflict this leaves

`FIGURE_GUIDELINES.md` names the band as the *form* the uncertainty takes, in four places
that the tree now contradicts:

- §4.1 "Many layers → line + shaded band"
- §4.2 "Mean + band; say what the band is on the figure"
- §4.4 "Carry a bootstrap band over items" (the KDE rule, which is exactly `bracket_flip`
  fig2 (a))
- §7 checklist, "Uncertainty drawn; band semantics + n stated"

The substance of every one of those — uncertainty is drawn, its semantics and n are on the
figure — still holds. Only the shape is wrong. `FIGURE_GUIDELINES.md` is shared with the
sibling branches, so it is not edited here; the owner should decide the wording once, for
all of them.

## Shaded regions deliberately kept

These are not per-series uncertainty and were left alone:

- `legality_probe` F2, `axvspan` over `r < 0` — the bitwise-shared prefix window.
- `legality_probe` F2 (e), `axhspan` — the y range of panels (b)–(d), so the change of
  scale is on the page.
- `rules_causal` `unflip_edit`, `axvspan` over the `random` population — marks the null's
  slot on a categorical axis.
- `bracket_flip` fig2 (b), gray `axhspan` — the damage control (log-mass on the whole legal
  set), which §3 asks for as a gray band.

**One judgment call to check.** `legality_probe` F4 panels (b) and (e) draw the two
untrained controls as a horizontal level with `axhspan(lo, hi)` in the series colour: that
*is* a shaded uncertainty band, per series. It stayed, for two reasons — those controls have
no x position, so there is no marker to hang a bar on and any substitute would be invented;
and the interval is ~0.001 wide under a dashed line, i.e. not the thing that was
unreadable. One line to delete if you disagree, plus the caption clause below.

## Captions, titles and prose reconciled

- `bracket_flip/make_figures.py` fig2 (a) legend title: "band: 95% percentile bootstrap" →
  "bars at the markers: 95% percentile bootstrap", rewrapped so it clears the share inset's
  title.
- `legality_probe/make_figures.py`: the shared `BAND` constant is now `BARS`
  ("bars: 95% bootstrap CI over source games"). It feeds F2, F3 and F4 — **F3 and F4 never
  had a band**, they have always drawn `errorbar`, so this also fixes two captions that
  were wrong before this change.
- F2 caption: dropped "CI half-widths … are drawn but mostly narrower than the lines" for
  "Half-widths are ≤ 0.006, mostly shorter than the marker", and said the bars are at every
  fifth offset. Kept to the original length so the tight bbox does not grow and shrink the
  panels.
- F4 caption: "and the same interval shades each control level", covering the `axhspan`
  above.
- `rules_causal/make_figures.py`: both legend titles "band: 95% CI over {pairs,events}" →
  "bars: …".
- `rules_causal/README.md`: "Error bars and bands are 95% CIs" → "Error bars are 95% CIs".

Nothing else in the three directories describes a band. The reports under
`bracket_flip/reports/` that mention shaded bands (`RESULTS_A3.md` F5, `RESULTS_A4.md`
F3/F4) describe figures the tree no longer builds; they are records of what was drawn then
and were left as they are.

## Rendering

Both PNG and `--camera-ready` PDF re-rendered for every figure touched, on the login node,
no OOM. `analysis/bracket_flip/sae_pca_figure.py` was not touched and needs no rerun, so
none of the gitignored `ae.pt` / `linear_probe.pt` inputs were needed.

`fig1_flip_and_basis` and `f1_probes_per_space` re-render byte-identically as PNG; their
PDFs differ only in the embedded creation date, so those two are reverted rather than
committed.

`analysis/rules_causal/make_figures.py` writes its three figures next to itself and none of
them is tracked. They were rendered and inspected, then deleted, so the study still ships
no figure files — regenerate with `PYTHONPATH=src python analysis/rules_causal/make_figures.py`.
