> **Correction, read first.** A3's wave-2 record, not the study's current position
> (`README.md`, `RED_TEAM.md`, `docs/claims_status.md`). Three things below were
> retired afterwards. `delta_peak_L5`, `delta_peak_r_L5` and the Headline's "peak
> offset migrates 0.00 → 5.68" are max-of-noise — a within-row permutation null
> reproduces the peak to 0.4% at every layer and 79–83% of the depth drift, and
> those panels were deleted as a correctness fix. `peak_layer` = 7 is vacuous as
> defined: the metric normalizes within each space, so an excess over the input
> floor has no cross-space meaning. And "`occupied_opp` behaves as a no-op" was
> answered the other way by the probe read-out — the square decodes the mover's
> colour in 78.6% of events, a recolor (`probe_readout/RESULTS.md`, `RED_TEAM.md`).
> The plateau, the ratio and the never-halves fraction are means and stand.

# A3 raw activation space: M1, M2, M3 across `token_embed` and `resid_post_0..7`

Everything below is read from `$SPADE_DIVERGENCE_DIR/divergence/*.npz` (2000
games; the nine raw spaces were all present when I ran, as were A4's eight SAE
ones) by `analysis/violation_divergence/make_figures.py`,
which writes `results/violation_divergence/{depth_sweep,divergence_curves,contract_keys}.csv`
and figures F1 and F3. Numbers here are copied from those CSVs, not rounded from
memory; `contract_keys.csv` is the machine-readable form of the table below.

Scope was cut mid-task: M5 (behavioural) went to A7 and M6 (probe) to A6. I had
read `unflip_edit.probe_directions` and `rules.causal.boards` but computed no M5
or M6 number, so there is nothing for them to cross-check against. One note that
may save A6 a step: the probe's `MINE` is the player to move at position *s*+1,
fixed by `extract_sae_training_data.py`'s `focus_states * (-1 + 2*(arange(L)%2))`
— even move index gets multiplier −1 — and `unflip_edit.build_rows` scores
against `regular_moves(state, -mover)`, consistent with that.

## Cross-validation contract

`n_games` **2000**, `n_branches` **15531** (13531 non-reference rows in the npz
plus the 2000 reference rows that are dropped as zero-by-definition).

Per-category n, which differs and is never pooled silently below:
`occupied_own` 2000, `occupied_opp` 2000, `opponent_only` 1998, `empty_detached`
1956, `empty_no_bracket` 1941, `empty_own_only` **1636**, `control_legal` 2000.

| category | n | `delta_peak_L5` | `delta_peak_r_L5` | `delta_plateau_L5` | `delta_peak_WE` |
|---|---|---|---|---|---|
| `occupied_own` | 2000 | 0.791890 ± 0.004952 | 5.9205 ± 0.3108 | 0.444340 ± 0.005487 | 1.43746 |
| `occupied_opp` | 2000 | 0.803008 ± 0.004937 | 5.0395 ± 0.2865 | 0.448818 ± 0.005534 | 1.43118 |
| `empty_detached` | 1956 | 0.882829 ± 0.004338 | 2.9269 ± 0.2479 | 0.490414 ± 0.005266 | 1.38523 |
| `empty_own_only` | 1636 | 0.896604 ± 0.004934 | 2.8203 ± 0.2641 | 0.496315 ± 0.006027 | 1.40422 |
| `empty_no_bracket` | 1941 | 0.842089 ± 0.004410 | 4.4678 ± 0.2848 | 0.488644 ± 0.005587 | 1.40102 |
| `opponent_only` | 1998 | 0.869390 ± 0.004462 | 3.6917 ± 0.2639 | 0.483470 ± 0.005612 | 1.41755 |
| illegal pooled | 11531 | 0.845980 ± 0.002039 | 4.1893 ± 0.1155 | 0.474544 ± 0.002308 | 1.41321 |
| `control_legal` | 2000 | 0.845555 ± 0.004543 | 5.2605 ± 0.2809 | 0.484598 ± 0.005676 | 1.41587 |

`ratio_illegal_control_L5` = **0.979253**, 95% CI [0.972786, 0.985832]
(1000-resample bootstrap over source games, not over rows — a game contributes up
to six illegal rows that share its prefix).

`peak_layer` = **7**. Definition used: the layer maximising
`max_r [δ̂_layer(r) − δ̂_WE(r)]` on category-mean curves, pooled illegal — the
`excess_over_we` column of `depth_sweep.csv`, so A5 can check the argmax
without re-reading the npz. Since
`δ̂_WE` is a delta function, the maximiser is always `r = 1` and the excess is
just `δ̂(1)`: 0.4096, 0.4252, 0.4598, 0.4965, 0.5485, 0.6586, 0.7138, **0.7284**
for L0…L7. It is monotone in depth, so `peak_layer` is the last layer rather
than an interior optimum — worth stating, because "the peak layer is 7" and
"there is no peak" are the same fact here.

`± ` is a 95% CI of the mean over source games (one row per game within a
category, so the game is the resampling unit and no clustering correction is
needed). All values from `results/violation_divergence/depth_sweep.csv`.

### The one place A5 will disagree with me if we are not careful

`delta_peak_L5` / `delta_peak_r_L5` above are **`persistence()`'s per-branch
statistics**, averaged over rows — each branch's own argmax over r ≥ 0. That is
the interface's definition and reproduces METRICS_REPORT's "peaks ~0.85 at r≈5".

The **category-mean curve is maximal at r = 0 in all 9 spaces and all 7
categories**, and decays monotonically from there (modulo the parity ripple
below). At layer 5, pooled illegal: max of the mean curve 0.792544 at r = 0,
versus mean of per-branch peaks 0.845980 at mean offset 4.19. The per-branch
offset has **median 0** (51.9% of branches peak exactly at r = 0; q75 = 7), so
the mean 4.19 is a right tail, and taking an argmax over 25 offsets of a noisy
row biases the value up by 6.7%. Both conventions are in `depth_sweep.csv` as
`delta_peak_row` / `peak_r_row` and `delta_peak_curve` / `peak_r_curve`; F3's
panel (c) carries the caveat in-panel. Neither number is wrong, they answer
different questions, and a mechanical comparison that mixes them will look like
a contradiction when it is not.

## Headline

At layer 5 an illegal move moves the residual stream **no further from the real
game than a different legal move does**: pooled illegal plateau 0.474544 against
the legal control's 0.484598, ratio 0.9793 [0.9729, 0.9858] — significantly
*below* 1. The separation is entirely within the taxonomy, and it tracks board
change rather than legality: paired per source game, the two zero-board-change
R1 categories sit 8% below the control (`occupied_own` −0.04026 [−0.04368,
−0.03684], `occupied_opp` −0.03578 [−0.03955, −0.03201]) while the R2 empty
categories sit at or slightly above it (`empty_own_only` +0.01232 [+0.00766,
+0.01698], `empty_detached` +0.00497 [+0.00097, +0.00897], `empty_no_bracket`
+0.00379 [−0.00069, +0.00826]) and R3 `opponent_only` is indistinguishable from
it (−0.00107 [−0.00528, +0.00314]). What does separate every illegal branch from
the control is **direction**: `cos(Δ_cat, Δ_control)` at r = 0 is 0.44–0.54
depending on category, rising to 0.78–0.86 by r = 24, so the branches leave along
partly different directions and re-align as the shared suffix reasserts itself.
Down the stack the picture is the study's "computed, not inherited" claim in four
numbers: the token embedding's divergence is a delta function — δ̂(0) = 1.41321
≈ √2, and **exactly 0.000000 at every r ≥ 1** — while the residual stream's
plateau climbs monotonically 0.2026 → 0.5680 from L0 to L7, its per-branch peak
offset migrates 0.00 → 5.68, and the fraction of branches whose divergence never
halves within 24 plies goes 0.0% → 52.9%. Every layer converts more of a
one-token input difference into an offset the model still carries at the end of
the game, and none of it can be inherited from the input, because the input
carries nothing past r = 0.

## Two things the design doc asked for as results, not assumptions

**`occupied_opp` behaves as a no-op, not a recolor.** Paired per source game at
layer 5, its plateau exceeds `occupied_own`'s by only +0.00448 [+0.00218,
+0.00677], against a no-op-to-board-change gap (control minus `occupied_own`) of
+0.04026. So `occupied_opp` sits **11.1%** of the way from "the board did not
change" to "a legal move changed the board". The difference from `occupied_own`
is real (the CI excludes zero) but an order of magnitude smaller than a recolor
reading would predict.

**`occupied_own` vs `control_legal`, the study's cleanest contrast.** The
zero-board-change violation diverges *less* than the board-changing legal move,
by 0.04026 in plateau (8.3%) and 0.0537 in per-branch peak. Read plainly: raw
layer-5 divergence magnitude is dominated by how much the board changed, and a
violation that changes nothing registers as less of an event than a legal move
that flips 1.93 discs. If there is a "the model noticed something impossible"
signal in raw space, it is not in ‖Δ‖.

## M2, and a 2-ply parity structure nobody has flagged

`cos(Δ(r), Δ(0))` at layer 5 does not decay smoothly. It **alternates with ply
parity**: mean 0.1080 at odd r (0.0625 at r = 1) against 0.2439 at even r (0.4610
at r = 2), the ripple damping but persisting to r = 24. δ̂ carries the same
period-2 ripple with the opposite sign — odd offsets are +2.02% above the local
level at layer 5, +6.93% at layer 0, +0.50% at layer 7. The natural reading is
the mover-relative board code: every ply the {mine, theirs} frame flips, so a
fixed board difference projects onto alternating directions. It is largest at
layer 0 and smallest at layer 7, i.e. the network partially cancels it. F1's
panel (c) is where it is visible; I did not chase it further, and it is a real
candidate finding for whoever writes the discussion.

## Figures

Both under `analysis/violation_divergence/figures/`, PNG by default and vector
PDF under `--camera-ready` (which also drops the in-figure title). Colours and
markers are `paper_style.CATEGORY_STYLE`; `control_legal` takes `NULL` styling
and is last in every legend; `rule_key` prints R1–R3 under both.

- **F1 `divergence_layer5`** — event-locked at layer 5, four panels: (a) δ̂(r)
  per category with the pre-t0 region shaded and annotated zero-by-construction,
  (b) the paired excess over `control_legal`, which is where the categories
  actually separate, (c) `cos(Δ(r), Δ(0))`, (d) `cos(Δ_cat, Δ_control)`. Panels
  (c) and (d) are M2, which the design doc's figure list has no figure for; I
  attached them to F1 rather than invent an F8, since they share the window,
  the styling and the n.
- **F3 `depth_sweep`** — `token_embed` as its own leftmost tick separated by a
  rule, then L0…L7, four panels: (a) peak δ̂, (b) plateau δ̂ with the W_E anchor
  at exactly 0 called out, (c) per-branch peak offset, (d) censored fraction of
  the half-life. Bands are 95% CI over source games; panel (d) is a normal
  approximation on a proportion.

Checked against FIGURE_GUIDELINES: no colour literals, every curve carries
uncertainty with its semantics and n stated in-figure (n is in each legend
entry), the null is gray/dashed/last, the layer indexing base is in the axis
label, the event-locked panels have the dashed gray line at 0 and the shaded
pre-event window, and both figures were read in grayscale before shipping.

Two knowing deviations. §4.1's "≤5 series per panel" is exceeded: seven series
is the study's independent variable, and `CATEGORY_STYLE` is built for exactly
this seven-way display with hue = rule, lightness = depth within rule, marker =
rule. §4.2's "annotate the extremum" is done on one curve only (the largest,
`empty_own_only` at r = 0, 0.866) — seven annotations would be unreadable.

## What I could not settle

- **M2's illegal-vs-control cosine has no matched null.** There is one
  `control_legal` branch per game, so `cos(Δ_control, Δ_control) ≡ 1` and I have
  nothing to compare 0.44–0.54 against. I cannot tell whether that value means
  "illegal moves leave along a different direction" or "any two distinct moves
  leave along directions ~60° apart". The *contrast across categories* is
  interpretable and is a finding (occupied 0.52–0.54, empty 0.44–0.49); the
  absolute level is not. **A second legal control branch per game would settle
  it** and costs one more row per game in `branches.py`.
- **`empty_own_only` is the largest-divergence category and the one missing in
  18.2% of games**, spread across all t0 (DATA_REPORT). Its numbers above are
  conditioned on prefixes that offer an own-only square, which is a board-shape
  selection, not a depth effect. I cannot separate "own-only violations diverge
  most" from "prefixes that admit an own-only square diverge most". A
  within-game comparison restricted to the 1636 games where the category exists
  would help; I did not run it.
- **Half-life at layers 6 and 7 is past its censoring point** for several
  categories: at L7 the censored fraction runs 0.470–0.576 and the median is
  therefore `inf` for `control_legal`, `occupied_own`, all three R2 categories
  and the pooled illegal row. Those medians are lower bounds, not estimates.
  `depth_sweep.csv` carries `half_life_censored` next to every median for
  exactly this reason; never quote the median alone at L6–L7.
- I did not test the parity ripple against anything. It is described, not
  explained.

## Requests for the integrator

1. **No `paper_style` change needed.** `CATEGORY_STYLE` covered all six
   categories and `NULL` covered the control; I added no colour and needed no
   new entry. The one thing I built locally that a second agent may want is
   `series()` / `legend_order()` in `make_figures.py` (a category-styled curve
   with a CI band, and taxonomy-order-then-null legend sorting). If A4's SAE
   figures grew the same helpers, they are a candidate for
   `src/spade/figures/violation_panels.py` at integration — I deliberately did not
   create that file.
2. **`make_figures.py` writes the result CSVs as well as the figures.** My file
   list allowed `make_figures.py` and `results/violation_divergence/*.csv` but
   no third module, so the aggregation lives in the figure entry point. If you
   would rather split it, the seam is clean: `depth_table`, `curve_table` and
   `contract_table` have no matplotlib in them.
3. **Decide which peak convention the paper quotes**, per the section above.
   My recommendation is the per-branch statistic for the contract (it is the
   interface's) and the mean-curve statement in the prose, because "the average
   trajectory is furthest from the reference at the moment of the violation and
   never comes back" is the true sentence and "it peaks five plies later" is
   not.
