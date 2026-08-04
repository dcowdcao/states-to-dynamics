# The fork in R^21: what the curvature descriptor can and cannot carry

Feasibility study for projecting `space_fork_grid_pca`
(`analysis/violation_divergence/game_trajectory`) through the ReStraV descriptor
R^21 — 7 raw step lengths, 6 raw turning angles, 4 moments of *d*, 4 moments of
*θ* — instead of through a PCA of the raw activation displacement.

```
python analysis/violation_divergence/fork_expanding_curvature/cache_features.py   # once
python analysis/violation_divergence/fork_expanding_curvature/feasibility.py      # the numbers
python analysis/violation_divergence/fork_expanding_curvature/make_figures.py [--camera-ready]
```

## Verdict

**R^21 cannot carry a move-index curve on this window.** The descriptor needs 8
sampled positions (14 plies at STRIDE=2) to be R^21 at all; the figure has 4
sampled positions before *t0* (8 plies). Every window long enough to fill R^21 is
more than twice as long as the pre-branch stretch the figure exists to show, so
the shared prefix at the origin and the spray at *t0* — the whole content of the
figure — fall outside the descriptor's domain.

The well-posed use of R^21 here is **one 21-D point per branch** over the whole
post-*t0* stretch. That is a §4.6 embedding scatter, not a fork.

The expanding window anchored at *t0* is the least-wrong way to get a curve and is
rendered here, but it starts 14 plies late and 13 of its 21 coordinates are frozen
along it. Treat it as a diagnostic.

## 1. The truncation trap

`restrav_features` slices `d[:, :7]` and `theta[:, :6]`. A short window returns a
**shorter vector, with no error**:

| W (sampled positions) | plies | n_d | n_θ | returned width | contents |
|---|---|---|---|---|---|
| 2 | 2 | 1 | 0 | — | `IndexError` from `moment4`: `amin` over an empty θ |
| 3 | 4 | 2 | 1 | **11** | d1–d2, θ1, + 8 moments |
| 4 | 6 | 3 | 2 | **13** | d1–d3, θ1–θ2, + 8 moments |
| 5 | 8 | 4 | 3 | **15** | d1–d4, θ1–θ3, + 8 moments |
| 6 | 10 | 5 | 4 | **17** | d1–d5, θ1–θ4, + 8 moments |
| 7 | 12 | 6 | 5 | **19** | d1–d6, θ1–θ5, + 8 moments |
| ≥ 8 | ≥ 14 | ≥ 7 | ≥ 6 | **21** | the full descriptor |

The width is wrong *and* the column meanings shift: at W=5, column 4 is *θ1*, not
*d5*. Zero-padding to 21 does not repair that — the coordinates would have to be
scattered into their correct slots and the padding is fabricated, so a curve drawn
across a region where W grows through 3,4,…,7 moves partly because a coordinate
switched on. `window.r21` asserts `shape[1] == 21` and every helper here goes
through it, so a short window is a crash.

## 2. Curve points, and what happens to *t0*

Game 0, *t0*=10, 17 sampled positions, r ∈ [−8, 24]. Measured from the
activations: branches are identical before *t0* and differ from *t0* on
(asserted); a window is contaminated iff it contains a post-*t0* position, which
at W=8 matches the descriptor's own contamination exactly (asserted).

**Trailing (causal)** — drawn at the window's last position:

| W | dim | curve pts | first drawn r | pts free of the branch | first contaminated r | smearing |
|---|---|---|---|---|---|---|
| 3 | 11 | 15 | −4 | 2 | 0 | 0 |
| 4 | 13 | 14 | −2 | 1 | 0 | 0 |
| 5 | 15 | 13 | 0 | **0** | 0 | 0 |
| 6 | 17 | 12 | +2 | **0** | +2 | 0 |
| 7 | 19 | 11 | +4 | **0** | +4 | 0 |
| **8** | **21** | **10** | **+6** | **0** | +6 | 0 |
| 9 | 21 | 9 | +8 | 0 | +8 | 0 |
| 13 | 21 | 5 | +16 | 0 | +16 | 0 |
| 17 | 21 | 1 | +24 | 0 | +24 | 0 |

**Centered** — smears instead of truncating:

| W | dim | curve pts | first drawn r | pts free of the branch | first contaminated r | smearing |
|---|---|---|---|---|---|---|
| 3 | 11 | 15 | −6 | 2 | −2 | 2 plies |
| 4 | 13 | 14 | −6 | 1 | −4 | 4 plies |
| 5 | 15 | 13 | −4 | 0 | −4 | 4 plies |
| 6 | 17 | 12 | −4 | 0 | −4 | 4 plies |
| 7 | 19 | 11 | −2 | 0 | −2 | 2 plies |
| **8** | **21** | **10** | **−2** | **0** | −2 | **2 plies** |
| 9 | 21 | 9 | 0 | 0 | 0 | 0 |
| 13 | 21 | 5 | +4 | 0 | +4 | 0 |

The deciding row: a curve point free of the branch needs the whole window before
*t0*, i.e. **W ≤ 4**, which returns R^13 at most. No W is both full-width and
leaves a prefix. This is the window design (8 pre-branch plies) against the
descriptor design (14 plies minimum), not a tuning knob.

STRIDE=1 would halve the requirement to 7 plies, but the board code is
mover-relative and the frame inverts every ply, so turning angles across a frame
flip are a property of the code rather than the trajectory — it would make 6 of
the 21 coordinates a frame artifact. Not recommended.

## 3. What moves along the expanding window

6 curve points, r = 14…24 (W = 8…13):

| k | W | r | ‖ΔΔ21‖ since the previous point: raw block (d1–7, θ1–6) / moment block |
|---|---|---|---|
| 11 | 8 | 14 | first drawn point |
| 12 | 9 | 16 | **0.000e+00** / 1.297e+01 |
| 13 | 10 | 18 | **0.000e+00** / 8.303e+00 |
| 14 | 11 | 20 | **0.000e+00** / 2.393e+01 |
| 15 | 12 | 22 | **0.000e+00** / 1.228e+01 |
| 16 | 13 | 24 | **0.000e+00** / 4.975e+00 |

Exactly zero, not small: an expanding window's first 7 steps are its first 7 steps
forever. **13/21 of the descriptor is a per-branch constant and 8/21 is the
curve**, and those 8 are cumulative moment estimators whose sample size *is* the
window length — *min d* and *max d* are monotone in W by construction.

Is the motion the trajectory or the estimator? Null: recompute the same 6
descriptors over a random permutation of the post-*t0* positions (20 shuffles), so
temporal order is destroyed and window growth is the only thing left. Statistic:
path length after the first point / ‖Δ21‖ there, median over the 8 non-reference
branches.

| space | observed | time-shuffled null |
|---|---|---|
| raw layer 5 | 1.24 | 1.00 |
| PAnneal (pen 0.042) | 1.21 | 1.03 |
| Standard (L1 0.059) | 1.26 | 1.18 |
| GatedSAE (L1 1.37) | 1.47 | 0.91 |
| TopK (k=80) | 1.78 | 0.85 |
| GatedAnneal (pen 0.955) | 1.52 | 0.88 |
| BatchTopK (k=80) | 1.47 | 0.89 |
| CosineSpaDE (γ=1.0) | 1.86 | 1.22 |
| SpaDE (λ=1e-3, γ=1e-4) | 1.62 | 1.11 |

Observed beats the null everywhere, by 1.2–2.1×, with heavily overlapping
per-branch spreads. Most of the curve's length is estimator drift; one game cannot
resolve the rest.

## 4. Separation, and why the large number is not a win

One effect size throughout: illegal-vs-control centroid gap on the fitted plane
divided by the pooled within-group per-axis SD (√(mean squared distance to own
centroid / 2)), 6 illegal branches vs 2 controls, reference excluded. PC1+PC2
variance in parentheses. *p* is a label-permutation null over all 28 relabellings
of the 8 branches (floor 1/28 = 0.036), defined only where there is one point per
branch.

| space | raw disp., per position | raw disp., per branch | R^21 expanding | R^21 one point |
|---|---|---|---|---|
| raw layer 5 | 0.33 (0.53) | 1.54 (0.88) *p*=0.32 | 1.52 (0.58) | 1.54 (0.63) *p*=0.18 |
| PAnneal | 0.29 (0.50) | 1.65 (0.88) *p*=0.32 | 1.26 (0.60) | 1.41 (0.64) *p*=0.32 |
| Standard | 0.38 (0.48) | 1.76 (0.88) *p*=0.25 | 1.31 (0.59) | 1.30 (0.65) *p*=0.39 |
| GatedSAE | 0.09 (0.44) | 0.89 (0.85) *p*=0.68 | 0.12 (0.58) | 0.50 (0.68) *p*=0.89 |
| TopK | 0.17 (0.50) | 1.79 (0.86) *p*=0.21 | 1.55 (0.56) | 1.35 (0.60) *p*=0.46 |
| GatedAnneal | 0.22 (0.45) | 1.23 (0.86) *p*=0.39 | 0.51 (0.65) | 0.42 (0.73) *p*=0.93 |
| BatchTopK | 0.09 (0.50) | 0.98 (0.86) *p*=0.61 | 0.58 (0.67) | 0.31 (0.70) *p*=1.00 |
| CosineSpaDE | 0.20 (0.34) | 1.30 (0.83) *p*=0.39 | 0.91 (0.62) | 1.03 (0.72) *p*=0.61 |
| SpaDE | 0.45 (0.43) | 1.87 (0.89) *p*=0.25 | 0.72 (0.53) | 1.18 (0.56) *p*=0.57 |

Column 1 is the existing figure's plane and reproduces the study's calibration
(0.09–0.45 here vs a quoted 0.06–0.33; the offset is the pooled-SD convention, the
range and ordering match). Columns 3–4 look 5× better. They are not.

Column 2 is the control that explains it: the *same raw activation displacement as
column 1*, averaged over the post-*t0* positions to one point per branch. It
reaches 0.89–1.87 — as large as or larger than either R^21 column, in every space.
The gain is the collapse of the position axis, not the descriptor: column 1's
denominator is the spread over 13 positions × 8 branches, columns 2–4's is the
spread over 8 branches. Column 2 also holds the most variance in two PCs
(0.83–0.89, vs 0.53–0.73 for R^21 and 0.34–0.53 for the current figure).

The permutation nulls close it: with 6 illegal and 2 control branches from one
game, **no space in any one-point column reaches even the 0.036 floor** (best
*p*=0.18). The bar was low; R^21 does not clear it, and the apparent clearing is
an artifact of the statistic's denominator.

Two side findings:

- **An unscaled PCA of R^21 is a plane of turning angles.** On raw layer 5, PC1
  puts 1% of its energy on the step-length coordinates, loading hardest on *var θ*,
  *θ2*, *max θ* — *d* is in activation-norm units and *θ* in degrees. Every plane
  here is fit after rescaling each coordinate to unit RMS, a diagonal map, so zero
  displacement is still the origin and the figure's uncentred-PCA rationale
  survives.
- **The legal controls diverge more than most illegal branches in R^21.** In raw
  layer 5 the two controls carry the largest ‖Δ21‖ of any branch at most offsets
  (`r21_expanding_magnitude` panel a). Whatever the descriptor separates on this
  game, it is not legality.

## Figures

| file | what |
|---|---|
| `space_fork_grid_r21_expanding` | the fork across all nine spaces, expanding-window R^21 plane. Planes are refit per space, so only the *shape* is comparable across cells. The dotted riser from *t0* is the 14 plies where the descriptor does not exist |
| `r21_expanding_magnitude` | (a) ‖Δ21‖ relative to the reference's descriptor norm, raw layer 5, with the undefined region shaded; (b) how far the curve travels after its first point, per space, against the time-shuffled null |

Both are n=1, shape only, as in the parent fork study. Uncertainty in (b) is the
per-branch scatter (n=8 branches, one game) and the null's range over branches.

## Recommendations

1. Do not replace the fork figure's plane with an R^21 curve — the figure's claim
   is the branch point, and R^21 cannot see the branch point.
2. If R^21 belongs in this study, it belongs as **one point per branch** over the
   whole post-*t0* stretch (W=13, fully populated, no window to defend), drawn as
   a §4.6 embedding scatter over the full 13,531-branch population. At n=1 the
   permutation floor is 0.036 and nothing reaches it.
3. Keep the raw-displacement plane for the fork: it is the only candidate that
   draws the prefix at the origin and the spray at *t0*.
4. `restrav_features` should assert its output width rather than letting
   `d[:, :7]` truncate silently. Flagged only — `src/` is out of scope here.
5. If this variant is adopted, `branch_style` / `legend_order` / `fork_panel` are
   now duplicated between this study and `game_trajectory` and should be lifted
   into `spade.figures`.
