# Divergence by space: how far the fork panels' branches actually run

The fork grids in `../fork_and_board` and `../game_trajectory` draw the branches
separating in nine representation spaces, and a 3-D projection can only ever show the
*shape* of that separation. The question it poses and cannot answer is how much: at
r = +6, is the illegal branch twice as far from the reference as a second legal move, or
a tenth as far? This is that number, on the study's event-locked axis, over the whole
population.

It is the companion to those figures, not a standalone result. Same nine cells in the
same order with raw layer 5 first, same rule grouping, same `branch_style` colours and
markers, same control-as-null. `../make_figures.py`'s F1 is the thing it replaces: F1
draws δ̂ = ‖Δ‖ / ‖z_ref‖ at layer 5 and nowhere else, on a raw activation axis that no
fork panel is drawn in.

```
python analysis/violation_divergence/divergence_by_space/measure.py       [--plane r21] [--variant rollout] [--n_games 100]
python analysis/violation_divergence/divergence_by_space/make_figures.py  [--plane r21] [--variant rollout] [--n_games 100] [--raw] [--camera-ready]
```

`measure.py` writes `$SPADE_DIVERGENCE_DIR[/rollout]/divergence_by_space/curves.npz`
(~35 min spliced, ~45 min rollout on one core, I/O-bound); `make_figures.py` reads it and
writes the figure plus `results/violation_divergence/divergence_by_space[_rollout].csv`,
one row per (space, series, measured_in, r) with the mean and its 95% percentile bootstrap
CI (1000 resamples of the source games, seed 42).

`--n_games N` draws the study's **shared N-game population** instead of the whole tree:
the first N branch-complete games forking at t₀ = 20 (`branches.games_at_t0`, 1696 exist),
which is the same set every other N-game sweep here runs over. It is a separate artifact
at every step — `curves_nN.npz`, `divergence_by_space_nN.png`, `..._nN.csv` — so it never
overwrites the whole-tree figure. n = 100 takes ~2 min.

## The figures

`figures/divergence_by_space.{png,pdf}` · `figures/divergence_by_space_rollout.{png,pdf}`
· `figures/divergence_by_space_n100.{png,pdf}` ·
`figures/divergence_by_space_r21_n100.{png,pdf}` ·
`figures/divergence_by_space_rollout_n100.{png,pdf}` ·
`figures/divergence_by_space_r21_rollout_n100.{png,pdf}` ·
`figures/divergence_by_space_n100_l0_matched.{png,pdf}`

**In the plane the fork panels draw, R2 and R3 run 6–24% further from the reference than
a legal alternative does — and roughly half of that is the plane, not the branch.** One
panel per representation space, raw layer 5 first so the eye calibrates on the reference
space (§4.6). x is r = s − t₀, event-locked to the branch move: the dashed vertical rule
is the branch position, the shaded band left of it is the shared prefix, where every
branch is bitwise identical to the reference and the distance is 0 by construction. y is
each branch's distance from the reference branch, **in the same 2-D PCA plane its fork
panel is drawn in** (solid, filled markers), with the same distance in that cell's full
space underneath (dotted, open markers); both divided by the same cell's own legal
control, so the null sits at 1 on average over r ≥ 0. One curve per constraint rule, the
mean over its member categories (n = 2, 3, 1), with the count in the legend entry;
`control_legal` is the null, in gray. Bands are 95% percentile bootstrap CIs of the mean
(1000 resamples of the source games, seed 42) over **n = 1553
branch-complete source games** — one game is one item, so no clustering correction — and
are about ±0.015 wide, hairline at this n. t₀ round-robins over [10, 34] and the window
is r ∈ [−8, +24] sampled every 2 plies at t₀'s parity, so every game contributes to every
offset and n is constant across the x-axis. Panels whose plane curve leaves the y-cap
carry the value they reach.

`../fork_and_board`'s panel (c) is **this same quantity collapsed over r ≥ 0** — the bar
is the in-plane curve's mean, the open marker is the full-space curve's mean, and the
divisor is identical. Read the two together: (c) gives one number per rule per space for
one illustrative game, this gives the whole trajectory of that number over 1553.

## The n = 100 twin

`figures/divergence_by_space_n100.png` is the same figure over the shared 100-game
population, so it can be read beside the other n = 100 figures in this study — same games,
same t₀ = 20, same event-locked axis. Mean over r ≥ 0 in the plane, against the
whole-tree figure above:

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw layer 5 | 0.989 (0.993) | 1.157 (1.142) | 1.130 (1.122) |
| CosineSpaDE | 1.009 (1.017) | 1.387 (1.236) | 1.305 (1.206) |
| SpaDE | 1.008 (1.019) | 1.224 (1.213) | 1.174 (1.176) |
| TopK | 0.983 (0.973) | 1.086 (1.064) | 1.097 (1.069) |

Whole-tree value in brackets. Every cell reproduces to within 0.02 except CosineSpaDE,
whose R2/R3 run 0.10–0.15 higher — that cell's r = 0 spike is the one place the population
matters: it reaches 4.4× at n = 100 against 3.1× on the whole tree, and its CI half-width
there is 0.59 against 0.11. Away from r = 0 the CI is 0.05–0.09, so the plateau where the
claim lives carries the same reading at n = 100 as at n = 1553.

The rule means hide the same amount of within-rule structure in both: the within- ÷
between-rule spread of the displacements is a median 0.924 [IQR 0.842–0.983] over these
100 games, above 1 in 21% of them, against 0.945 [0.872–1.020] over the whole tree. The twin's sweep computes that
statistic per game rather than quoting `rule_groups`' whole-tree number, and
`make_figures.py` prints it for the population drawn.

## The L0-matched twin

The roster swap runs over all four arms of the study, so the L0-matched grid is the
canonical grid cell for cell: {displacement, R²¹} × {spliced, rollout} × {ratio, raw}.

`figures/divergence_by_space_n100_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_n100_raw_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_r21_n100_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_r21_n100_raw_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_rollout_n100_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_rollout_n100_raw_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_r21_rollout_n100_l0_matched.{png,pdf}` ·
`figures/divergence_by_space_r21_rollout_n100_raw_l0_matched.{png,pdf}`

Every one of the eight: **n = 100** source games, the study's shared population at t₀ = 20
(`branches.games_at_t0`); the **roster** is `results/l0_operating_point/picks.csv`, one
checkpoint per SAE type at the shared **L0 = 219.2**, with raw layer 5 leading as the
reference cell; the **corpus** is the branch tree the arm names — `spliced` replays the
source game's own suffix after t₀, `rollout` samples every branch's continuation from
OthelloGPT (K = 4); the **estimator** is the mean over games of the branch's distance from
the reference, in the refit 2-D plane and in the full space, each divided by that cell and
game's own legal control (the raw twins skip the divisor), with a 95% CI over the 100 games.

```
python analysis/violation_divergence/divergence_by_space/measure.py --n_games 100 --l0-matched                            # ~4 min
python analysis/violation_divergence/divergence_by_space/measure.py --n_games 100 --l0-matched --plane r21                # ~20 min
python analysis/violation_divergence/divergence_by_space/measure.py --n_games 100 --l0-matched --variant rollout          # ~5 min
python analysis/violation_divergence/divergence_by_space/measure.py --n_games 100 --l0-matched --plane r21 --variant rollout
python analysis/violation_divergence/divergence_by_space/make_figures.py --n_games 100 --l0-matched \
    [--plane r21] [--variant rollout] [--raw] [--camera-ready]
```

**The whole L0-matched grid is measured on CPU, and that is a choice worth keeping.** CPU
and GPU float arithmetic differ in the last digits, so the same sweep re-run on the other
device moves a figure slightly. The two spliced-displacement cells were measured on CPU
before the other six existed and must not be regenerated to check them, so measuring the
six new cells on CPU too is what gives all eight one device; measuring them on GPU would
have left a six-plus-two grid whose cross-arm differences carry a device term nobody can
separate from the dictionary. `measure.py` is CPU-bound and I/O-bound, not GPU-bound, so
this costs nothing — run it under slurm (`scripts/slurm/submit_gpu.sbatch` with
`env CUDA_VISIBLE_DEVICES=`) for the dedicated cores, not for the accelerator.

The eight SAE cells above sit wherever each type's own default hyperparameters put it on
the sparsity axis — TopK and BatchTopK at k = 80, Standard at L1 0.059, and so on — so a
difference between two of those panels is the architecture *and* its L0, and there is no
way to say which. `--l0-matched` swaps them for `results/l0_operating_point/picks.csv`:
one checkpoint per type, each the nearest-L0 member of its own grid to the shared target
**L0 = 219.2**. Panel order, colours, markers, axes, window and population are the
canonical figure's, and each cell's title carries its own L0, so the two are read on top of
each other. Raw layer 5 is not an SAE and is the same cell in both — its 136 CSV rows are
identical to the canonical file's, which is the check that only the dictionaries moved.

**Both qualitative claims survive.** The plane inflates the rule/control ratio in **27 of
27 cells** here too, median **1.104** (range 1.070–1.325) against the canonical n = 100
figure's 1.108 (1.084–1.309). And no rule ends up further from the reference than a legal
alternative does by more than a few percent in the full space: R2 runs 0.987–1.073 and R3
0.999–1.046, against 0.977–1.059 and 0.994–1.041 canonically, while **R1 sits 8–9% below
the control in all nine spaces** (0.906–0.920) exactly as before.

**What moves is the ranking of the dictionaries, and the two SpaDE codes lose their
distinction.** Mean over r ≥ 0 in the plane, canonical n = 100 → L0-matched:

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw layer 5 | 0.989 → 0.989 | 1.157 → 1.157 | 1.130 → 1.130 |
| PAnneal | 0.984 → 0.998 | 1.144 → 1.146 | 1.124 → 1.130 |
| Standard | 1.011 → 1.000 | 1.223 → **1.140** | 1.198 → **1.128** |
| GatedSAE | 1.000 → 1.002 | 1.088 → 1.089 | 1.096 → 1.097 |
| TopK | 0.983 → 0.970 | 1.086 → 1.106 | 1.097 → 1.101 |
| GatedAnneal | 0.998 → 0.987 | 1.097 → 1.134 | 1.104 → 1.107 |
| BatchTopK | 1.003 → 0.995 | 1.110 → 1.149 | 1.115 → 1.134 |
| CosineSpaDE | 1.009 → 0.995 | 1.387 → 1.421 | 1.305 → 1.311 |
| SpaDE | 1.008 → 0.998 | 1.224 → **1.127** | 1.174 → **1.109** |

By R2 the canonical order is CosineSpaDE > SpaDE > Standard > raw > PAnneal > BatchTopK >
GatedAnneal > GatedSAE > TopK; at matched L0 it is CosineSpaDE > raw > BatchTopK > PAnneal
> Standard > GatedAnneal > SpaDE > TopK > GatedSAE. **"The two SpaDE codes inflate most" is
half an L0 effect**: SpaDE's R2 falls from 1.224 to 1.127 once it is seated at L0 232 rather
than at its own default, which lands it below raw layer 5, and Standard falls the same way.
What is left is CosineSpaDE, which is also the one pick *below* the target (L0 184) and is
the loudest cell in every figure in this study for its own reasons.

**Read the raw twin's magnitudes against the canonical one only inside a cell.** The five
Karvonen picks are fed the raw residual stream (`Matched.standardize`) where the canonical
seed-42 entries are all fed scale·(x − mean), so their latent scale changes for a reason
that is not the L0: the legal control's full-space distance goes 6.38 → 12.5 in PAnneal,
3.53 → 8.05 in Standard, 6.23 → 11.98 in GatedSAE, 6.62 → 13.04 in TopK and 7.19 → 11.78 in
GatedAnneal, while our three (BatchTopK 6.83 → 6.98, CosineSpaDE 0.0820 → 0.0831, SpaDE
0.0226 → 0.0688) move only as much as the checkpoint did. The span across the grid
therefore narrows from 586× to **193×** in the full space, and that number is a change of
input convention as much as of dictionary. The ratio figure divides all of it out, which is
why it is the one a cross-panel claim is read off.

### The other three arms

Mean over r ≥ 0 in the plane, canonical n = 100 → L0-matched, for the three arms the
spliced-displacement table above does not cover. Raw layer 5 is not an SAE and is the same
cell in both columns of every arm.

**R²¹, spliced.** Plane inflation stays 27 of 27 with the same median, 1.134, and a wider
range (1.051–1.309 → 1.032–1.363).

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw layer 5 | 1.040 → 1.040 | 1.228 → 1.228 | 1.204 → 1.204 |
| PAnneal | 1.063 → 1.035 | 1.284 → 1.248 | 1.246 → 1.166 |
| Standard | 1.053 → 1.030 | 1.309 → **1.122** | 1.294 → **1.114** |
| GatedSAE | 1.092 → 1.036 | 1.216 → 1.166 | 1.134 → 1.171 |
| TopK | 1.032 → 1.016 | 1.128 → 1.201 | 1.076 → 1.182 |
| GatedAnneal | 1.076 → 1.012 | 1.164 → 1.165 | 1.126 → 1.143 |
| BatchTopK | 1.062 → 1.014 | 1.168 → 1.280 | 1.168 → 1.232 |
| CosineSpaDE | 1.089 → 1.003 | 1.832 → 1.977 | 1.625 → 1.541 |
| SpaDE | 1.048 → 1.057 | 1.266 → **1.148** | 1.237 → 1.177 |

**Displacement, rollout.** 27 of 27, median 1.106 → 1.115, range 1.039–1.373 → 1.040–1.386.

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw layer 5 | 1.035 → 1.035 | 1.224 → 1.224 | 1.161 → 1.161 |
| PAnneal | 1.023 → 1.029 | 1.201 → 1.192 | 1.142 → 1.137 |
| Standard | 1.031 → 1.028 | 1.276 → **1.187** | 1.200 → **1.143** |
| GatedSAE | 1.018 → 1.023 | 1.112 → 1.112 | 1.087 → 1.086 |
| TopK | 1.020 → 1.025 | 1.118 → 1.170 | 1.092 → 1.123 |
| GatedAnneal | 1.015 → 1.025 | 1.106 → 1.185 | 1.082 → 1.134 |
| BatchTopK | 1.022 → 1.032 | 1.125 → 1.196 | 1.089 → 1.142 |
| CosineSpaDE | 1.026 → 1.026 | 1.488 → 1.517 | 1.363 → 1.367 |
| SpaDE | 1.042 → 1.029 | 1.319 → **1.206** | 1.224 → **1.155** |

**R²¹, rollout.** 27 of 27, median 1.146 → 1.155, range 1.057–1.453 → 1.039–1.551.

| space | R1 | R2 | R3 |
|---|---|---|---|
| raw layer 5 | 1.150 → 1.150 | 1.425 → 1.425 | 1.262 → 1.262 |
| PAnneal | 1.118 → 1.130 | 1.454 → 1.424 | 1.251 → 1.240 |
| Standard | 1.111 → 1.173 | 1.445 → **1.318** | 1.363 → **1.198** |
| GatedSAE | 1.205 → 1.180 | 1.312 → 1.260 | 1.180 → 1.140 |
| TopK | 1.186 → 1.136 | 1.263 → 1.370 | 1.130 → 1.222 |
| GatedAnneal | 1.250 → 1.138 | 1.324 → 1.350 | 1.200 → 1.222 |
| BatchTopK | 1.177 → 1.115 | 1.210 → 1.454 | 1.102 → 1.272 |
| CosineSpaDE | 1.253 → 1.122 | 2.387 → **2.678** | 2.074 → 2.150 |
| SpaDE | 1.171 → 1.197 | 1.490 → **1.322** | 1.334 → 1.230 |

**The L0 confound reverses the same four dictionaries in all four arms.** On R2, Standard
and SpaDE fall once they are seated at the shared L0 (Standard 1.309 → 1.122, 1.276 → 1.187,
1.445 → 1.318 across the three arms here; SpaDE 1.266 → 1.148, 1.319 → 1.206, 1.490 → 1.322),
while TopK and BatchTopK rise, and CosineSpaDE stays the loudest cell in every arm. That is
the same reversal the spliced-displacement twin reports, now on three more arms and two more
estimators, so **"the two SpaDE codes inflate most" is half an L0 effect** is a statement
about the grid and not about one figure. What does *not* move is the qualitative reading:
the plane inflates in 27 of 27 cells in each of the four arms, and the median inflation
changes by at most 0.01 in any of them.

The legal control's distance, mean over r ≥ 0 in each cell's own units, n = 100 — the
divisor the ratio figures remove, for the three new raw twins:

| space | R²¹ spliced plane \| full | rollout plane \| full | R²¹ rollout plane \| full |
|---|---|---|---|
| raw layer 5 | 3.123 \| 5.405 | 7.512 \| 14.53 | 2.849 \| 4.987 |
| PAnneal | 3.172 \| 5.448 | 6.825 \| 13.27 | 2.948 \| 5.090 |
| Standard | 3.241 \| 5.506 | 3.960 \| 8.201 | 2.861 \| 5.064 |
| GatedSAE | 3.322 \| 5.549 | 6.765 \| 12.73 | 2.948 \| 5.149 |
| TopK | 3.192 \| 5.505 | 7.623 \| 14.23 | 2.933 \| 5.116 |
| GatedAnneal | 3.258 \| 5.547 | 6.041 \| 12.15 | 2.902 \| 5.082 |
| BatchTopK | 3.107 \| 5.376 | 4.001 \| 7.559 | 2.846 \| 4.978 |
| CosineSpaDE | 2.388 \| 4.371 | 0.0307 \| 0.0810 | 1.918 \| 3.722 |
| SpaDE | 3.191 \| 5.378 | 0.0285 \| 0.0648 | 2.938 \| 5.018 |

The rollout displacement span narrows from the canonical roster's 782×/664× (plane/full) to
**263×/224×**, for the input-convention reason the previous section gives: Karvonen's five
picks are fed the raw residual stream and our three are not. The R²¹ columns are column-scaled
per game and span 1.7× and 1.5×, so nothing there is comparable to an activation length.

**The raw-layer-5 invariance check.** Raw layer 5 is the same cell in the canonical and the
L0-matched sweep, so its CSV rows should be identical — the check that only the dictionaries
moved. They are, exactly, in three of the four arms (136, 136 and 120 rows). In R²¹ rollout
the drawn `ratio` column is identical too, and only `ratio_hi` differs, on 1 of 120 rows by
1e-5, which is the sixth significant digit `%.6g` prints; the underlying npz differs by
3.3e-6 on values of ~12, i.e. float32 rounding in `curvature_plane.fit_plane`'s SVD. That
path does not reproduce bitwise across thread counts the way the displacement arm's Gram
route does.

## The R^21 twin

`figures/divergence_by_space_r21_n100.{png,pdf}` is the same figure with the reference
subtracted one step later: each branch's trajectory goes through ReStraV's 21 curvature
components first, and the difference is taken there —
Δφ(t) = φ(z_b(t)) − φ(z_ref(t)), φ the W = 8 trailing window of
`curvature_plane.windowed`. Nine cells, same colours and markers, same
ratio-to-the-legal-control, same 95% CI over the same 100 games. The plane is
`curvature_plane.fit_plane`'s uncentred top-2 of Δφ, refit per game, and both curves
are measured in the column-scaled field, because R²¹ mixes degrees with
activation-scale lengths.

```
python analysis/violation_divergence/divergence_by_space/measure.py      --plane r21 --n_games 100  # ~20 min
python analysis/violation_divergence/divergence_by_space/make_figures.py --plane r21 --n_games 100 [--camera-ready]
```

**The x axis is shorter: r = −4 … +24, 15 offsets.** φ's window ends at t and spans 14
plies, so it needs seven sampled positions of context and `positions` can only supply
five at t₀ = 20 — the first two offsets have no full-width window. Two branch-free
offsets survive, and `measure.py` asserts Δφ is exactly zero on both, which is what
pins where the branch is once R²¹ smears it.

**The separation is the same event, read through a sliding window.** In the
displacement figure the rule-minus-control gap is a spike at r = 0 (+0.33 in raw layer
5) decaying to +0.03 by r = +24. In R²¹ it starts at +0.11, climbs to **+0.34 at
r = +12**, and collapses to +0.06 at r = +14. Every space peaks at r = +12 (GatedSAE's
R3 at +10), and r = +12 is exactly where the branch move fills all seven steps of the
W = 8 window, and r = +14 is the first offset whose window no longer contains the
step that crosses t₀. So R²¹ does not
find separation the displacement figure missed — it spreads one ply's worth over seven
and reads it out as a ramp. **Nothing about the peak's position is a claim about the
model.**

Integrated over the window that does buy separation. Mean over r ≥ 0 in the plane,
displacement → R²¹, raw layer 5: R1 0.989 → 1.040, R2 1.157 → 1.228, R3 1.130 → 1.204;
in the full space R1 0.909 → **0.977**, R2 1.022 → 1.047, R3 1.017 → 1.038. The one
qualitative change is R1: illegal *placement* sits 8–10% **below** the control
everywhere in the displacement figure and is within 3% of it in R²¹, in every space.
The plane inflates in 27 of 27 cells here too, median 1.134 against 1.108 (range
1.051–1.309).

The price is precision and one unreadable panel. The CI half-width over r ≥ 0 roughly
doubles — median 0.148 against 0.094 — because φ of a short window is a noisier
statistic than a displacement. And CosineSpaDE's R2/R3 leave the y-cap for seven
consecutive offsets rather than the displacement figure's one at r = 0 (peak 2.98× at
r = +12 against 4.44× at r = 0), so that cell shows its annotated peak and little else.
Reading it needs the CSV.

The rule grouping is no better in R²¹: the within- ÷ between-rule spread over these 100
games is a median 0.886 [IQR 0.752–1.022] against the displacement field's 0.924
[0.842–0.983] — the same "barely coherent" verdict, with a wider spread.

## Why the plane, and what the plane costs

The y quantity is `branch_panels.plane_displacement`'s rule, resolved in r instead of
averaged over it: the in-plane distance from the reference branch, where the plane is the
uncentred top-2 PCA of the displacement D(t) = z_b(t) − z_ref(t), refit per game and per
space exactly as `fork_plane.pca2` fits it. The reference's D is exactly zero, so it
projects to the origin and distance-from-the-reference and distance-from-the-origin are
the same number.

A plane holds only part of a displacement, and a different part of each branch, so a
branch that leaves the plane reads as a branch that stayed near the reference. That is
why the full-space distance ‖D‖ rides underneath every in-plane curve. Mean over r ≥ 0,
spliced tree, as a multiple of the cell's legal control:

| space | R1 plane \| full | R2 plane \| full | R3 plane \| full |
|---|---|---|---|
| raw layer 5 | 0.993 \| 0.905 | 1.142 \| 1.009 | 1.122 \| 1.009 |
| PAnneal | 0.992 \| 0.905 | 1.140 \| 1.009 | 1.123 \| 1.010 |
| Standard | 1.007 \| 0.909 | 1.197 \| 1.026 | 1.163 \| 1.018 |
| GatedSAE | 0.983 \| 0.900 | 1.064 \| 0.972 | 1.070 \| 0.987 |
| TopK | 0.973 \| 0.895 | 1.064 \| 0.978 | 1.069 \| 0.993 |
| GatedAnneal | 0.984 \| 0.897 | 1.071 \| 0.971 | 1.072 \| 0.983 |
| BatchTopK | 0.988 \| 0.899 | 1.086 \| 0.979 | 1.085 \| 0.993 |
| CosineSpaDE | 1.017 \| 0.918 | 1.236 \| 1.016 | 1.206 \| 1.013 |
| SpaDE | 1.019 \| 0.920 | 1.213 \| 1.019 | 1.176 \| 1.013 |

**The plane inflates the rule/control ratio in 27 of 27 cells, median 1.108×, range
1.077–1.217.** The two SpaDE codes inflate most (1.16–1.22 on R2/R3), TopK least (1.08).
The single-game figures report a much larger inflation — `../fork_and_board`'s game 1235
reaches 4.08 in the plane against 1.53 in the full space, 2.7× — because that game is an
argmax over branch separation, not a draw. Over the population the inflation is about
11%, and the honest reading of a fork panel is "the spread you see is roughly a tenth
larger than the branch actually is", not "several times".

The sharpest place to see the mechanism is r = 0 in raw layer 5. In the full space the
legal control is displaced 1.384× the window mean and R2 1.528×, a 10% gap. In the plane
the control is 1.133× and R2 1.711×, a 51% gap. The plane is fit where the *illegal*
branches travel, so it keeps 51% more of R2's displacement than of the control's, and
the ratio read off it is that difference in shares, not a difference in distance.

## What the full space says

**No rule ends up further from the reference than a legal alternative does.** Over r ≥ 0
the full-space ratios are 0.895–0.920 for R1, 0.971–1.026 for R2 and 0.983–1.013 for R3;
nothing exceeds the control by more than 2.6% and R1 sits 8–10% below it in every space.
This reproduces the parent study's `ratio_illegal_control_L5` = 0.979 in a different
estimator (in-plane-fit spaces, per-game control normalisation, stride-2 window) and
extends it from layer 5 to all eight dictionaries. **Eight SAE dictionaries do not make
an illegal move look further away than a legal one.** The one thing that changes across
the grid is how much the *plane* exaggerates.

## Cross-panel comparability

The plane is refit per game and per space, and its units are the space's own. Measured
here, the legal control's mean full-space displacement runs from **13.19 in raw layer 5 to
0.0225 in SpaDE — 585×** for the same branches in the same games. Nine raw curves side by
side would be nine unit systems, and nine panels side by side get compared whatever a
disclaimer says. Every curve is therefore divided by **that cell's own `control_legal`,
averaged over r ≥ 0**: dimensionless, a within-cell contrast by construction, the null at
1 by construction, and the same form `../fork_and_board`'s panel (c) and the parent
study's `ratio_illegal_control_L5` already use. The divisor is in the y-axis label rather
than only here, because a reader who misses it misreads every panel.

A scalar divisor, not the control's curve: dividing pointwise would pin the control to a
flat 1 and throw away its shape, and 0/0 in the shared prefix. With a scalar the control
keeps its own trajectory and averages to 1.

## The raw pair

`figures/divergence_by_space_n100_raw.{png,pdf}` ·
`figures/divergence_by_space_r21_n100_raw.{png,pdf}` ·
`figures/divergence_by_space_rollout_n100_raw.{png,pdf}` ·
`figures/divergence_by_space_r21_rollout_n100_raw.{png,pdf}`

```
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100 --variant rollout
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100 --plane r21
python analysis/violation_divergence/divergence_by_space/make_figures.py --raw --n_games 100 --plane r21 --variant rollout
```

`--raw` is a flag on the figure, not a sweep. `measure.py` already writes both distances
raw — `plane` and `full` in each npz, in each cell's own units — and the division is
`normalised()`'s, so the raw pair reads the same four n = 100 npz the ratio pair does and
needs no re-measure.

**Everything but the y-axis is the ratio figure's: same nine cells in the same order, same
`branch_style` colours and markers, same 95% CI over the same 100 games, same shaded
prefix and branch rule.** What goes is the ratio's own furniture — the dashed null at
y = 1, the "above 1" note, the 2.0 cap and the callouts that read against it — because
none of it means anything on a distance axis.

**Read one figure for the size of a distance and the other for the comparison between
panels. A number does not carry across.** The raw figure is the only place a cell's
magnitude appears: the ratio pins every panel's control at 1, so nine panels that differ
700-fold look identical. The ratio figure is the only form in which the nine may be set
side by side at all, and it is the only one whose rule-vs-control read is paired per game.
Neither substitutes for the other, and the styling being identical is exactly why this
paragraph is here.

The legal control's distance from the reference, mean over r ≥ 0 in each cell's own units,
n = 100 — the divisor the ratio figure removes:

| space | spliced plane \| full | rollout plane \| full |
|---|---|---|
| raw layer 5 | 8.486 \| 13.25 | 7.512 \| 14.53 |
| PAnneal | 3.964 \| 6.382 | 3.467 \| 6.763 |
| Standard | 2.057 \| 3.532 | 1.715 \| 3.631 |
| GatedSAE | 3.814 \| 6.229 | 3.349 \| 6.480 |
| TopK | 4.317 \| 6.616 | 3.988 \| 7.320 |
| GatedAnneal | 4.583 \| 7.191 | 4.049 \| 7.752 |
| BatchTopK | 4.458 \| 6.829 | 4.050 \| 7.478 |
| CosineSpaDE | 0.0370 \| 0.0820 | 0.0313 \| 0.0815 |
| SpaDE | 0.0121 \| 0.0226 | 0.00961 \| 0.0219 |

**700× between the top and the bottom of that column** (raw layer 5 against SpaDE, in
the plane, spliced; 586× in the full space, which is the whole tree's 585× above, and
782×/664× on the rollout tree). That is the number "Cross-panel comparability" asserts,
now on an axis.

The same table for the R²¹ pair, where the span is 1.4–1.5×:

| space | spliced plane \| full | rollout plane \| full |
|---|---|---|
| raw layer 5 | 3.123 \| 5.405 | 2.849 \| 4.987 |
| PAnneal | 3.104 \| 5.384 | 2.894 \| 5.018 |
| Standard | 3.000 \| 5.302 | 2.841 \| 4.991 |
| GatedSAE | 3.354 \| 5.495 | 2.880 \| 5.055 |
| TopK | 3.388 \| 5.663 | 2.928 \| 5.186 |
| GatedAnneal | 3.350 \| 5.553 | 2.833 \| 5.057 |
| BatchTopK | 3.313 \| 5.595 | 2.953 \| 5.170 |
| CosineSpaDE | 2.444 \| 4.453 | 1.976 \| 3.843 |
| SpaDE | 3.022 \| 5.192 | 2.791 \| 4.824 |

**These are not physical units and the R²¹ pair is not a raw figure in the sense the other
pair is.** `curvature_plane.fit_plane` divides each of the 21 descriptor columns by its SD
over that game's Δφ rows, `Plane.project` re-applies the same scale, and `measure.py` takes
the full-space norm of the scaled field too — so both R²¹ curves are lengths in **per-game
descriptor-SD units**, which is why nine dictionaries land within 1.4× of each other rather
than 700×. Nothing on that axis is degrees, activation units, or comparable between games.
The column scaling is not cosmetic: R²¹ mixes turning angles with activation-scale step
lengths whose variances differ ~450× across dictionaries, and a raw norm of it would report
whichever component carries the biggest units.

What the raw view adds, on the displacement pair:

- **The excess is 100× smaller where the ratio is largest.** CosineSpaDE's R2 is the grid's
  biggest ratio (1.387) and its biggest ratio spike (4.44× at r = 0); in that space's units
  the same curve sits 0.011 above its control, against raw layer 5's 1.04. The ratio ranks
  cells by excess *relative to a divisor that itself varies 700×*, so a tall bar there is a
  small divisor as readily as a long branch.
- **The distances shrink after the branch.** The control's in-plane displacement peaks at
  r = 0 and falls to 0.68–0.76× of it by r = +24 in the seven dense cells (0.46× in
  CosineSpaDE). The R²¹ pair runs the other way — 1.25 → 2.81 in raw layer 5, 2.24× — since
  φ's window keeps absorbing post-branch steps. Both directions survive the division, but
  only the raw axis says how far anything moved.

What the ratio view does that the raw view cannot, beyond the shared axis:

- **The rule-vs-control contrast is paired per game there and unpaired here.** `ratios()`
  divides each game by *that game's* control before averaging, so it estimates E[rule /
  control]; a raw figure invites the reader to compare two population means, E[rule] against
  E[control]. Those differ, and in R²¹ they differ enough to change the verdict in **14 of
  27 spliced cells**: R1 reads 1.03–1.09 in the ratio figure and 0.92–0.98 of its control in
  population means, and GatedSAE's, TopK's, GatedAnneal's and BatchTopK's R3 cross from
  above the control to below it. On the displacement pair five cells flip, all R1, all
  within 0.04 of the null. **A raw panel is not the ratio panel rescaled** — no constant
  divisor relates them, because the divisor is per game.
- **The null has a position.** The ratio figure's y = 1 is where "no further than a legal
  alternative" lives, and an eye can check a curve against a rule. In the raw figure the
  control is one more curve among four, and every ordering has to be read off two lines that
  nearly coincide.

**r < 0 is exactly zero, not nearly zero.** Every branch is bitwise identical to the
reference over the shared prefix, so the full-space distance there is 0 and `measure.py`
asserts it; the in-plane distance is an eigenvector component of a matrix whose row is zero
and comes out below 1e-18 in absolute value. On a raw axis that region is the true
origin rather than a rescaled one, and the shaded band plus the in-panel `≡ 0` label carry
it exactly as they do in the ratio figure.

**The displacement pair is the one figure in this study with a per-panel y axis, and it
has to be.** What is drawn spans 1360× spliced (SpaDE 0.0097 against raw layer 5's 13.16)
and 1671× rollout — a wider span than the control column above, which is why the titles
quote the drawn one. On a single shared linear axis the two SpaDE panels stop existing:
SpaDE's entire rule-vs-control separation is 0.0087, 0.06% of a 15.1-unit axis, 0.07 pt at
`CELL_H` against the 1.7 pt the figure draws its lines with; CosineSpaDE's is 0.69 pt. Both
render as one flat line on zero and Standard gets 23% of the axis, so `sharey=False` is
forced by the data and not chosen. The cost is real: **panel heights are not comparable,
only within-panel shapes are**, the same disclaimer §4.6 puts on a per-panel PCA refit, and
the per-panel tick labels are what carry it. Log y was built and rejected — 3.6 decades of
axis spent on a within-panel signal of ~0.1 decade leaves the gray legal control invisible
in all nine panels, which fails §3 outright.

`--shared-y` draws the displacement pair on one autoscaled linear axis anyway. It is not a
deliverable; it exists so the collapse above can be re-checked rather than taken on trust.

**The R²¹ pair keeps one shared linear axis** and needs no deviation at all: every panel
peaks between 4.63 and 5.78 (within 1.3×, 1.5× on the rollout tree) and uses at least 58%
of it. The 11.7× between the smallest and largest value drawn is inside one panel,
CosineSpaDE, not between panels.

The autoscaled raw limits keep the ratio figure's own proportions rather than inventing new
ones: bottom at −0.045 × top, which is the −0.09/2.0 the ratio panels use, so zero sits
4.31% up the axis in every panel of every figure and the shared-prefix region reads as
visibly zero instead of resting on the spine.

CSVs are `results/violation_divergence/divergence_by_space[_r21][_rollout]_n100_raw[_l0_matched].csv`,
the same shape as the ratio ones with `ratio*` renamed `distance*`. Ground truth computed
independently from the npz is in `verification/`.

## The rollout variant

`--variant rollout` reads the tree where every branch, the reference included, samples
its own continuation from OthelloGPT instead of replaying the source game's. A branch is
K = 4 rows there, and the reduction is `../fork_and_board`'s, not F1's: each rollout is
differenced against **the reference drawn from its own uniform stream** (rows are
K-contiguous and the sampler is keyed on (game, k)) and the K displacement *vectors* are
averaged before the plane is fit, so the plane here is the plane those fork panels draw.
F1's `reduce_k` averages the scalar δ̂ instead. The two differ: averaging vectors cancels
whatever is incoherent across a branch's four samples, so magnitudes here are
systematically smaller than a scalar mean over K would give. Dividing by the control
removes most of that and not all of it, and **the size of the residual is not measured** —
it would take a second pass with the other reduction.

That tree saturates. Past r = +8 the illegal and control δ̂ agree to within 0.5% and cross
by r = +13, so both classes are climbing toward the divergence of unrelated activations
whatever legality does (`../README.md`). The figure draws the whole window and **shades
r ≥ 8 as not-evidence** rather than stopping there, because the event-locked shape before
that point is the content and a truncated axis hides where the saturation starts.

**The branch move stops setting the distance, in every space.** Over the whole window,
pooled across the nine cells, the full-space ratios are R1 0.969, R2 1.019, R3 1.016
against the control's 1.000 — nothing more than 2% off the null, where the spliced tree
put R1 8–10% below it. In the shaded region the ordering is gone entirely (r ≥ 10: R1
0.974, R2 0.997, R3 0.992). Inside the evidence window the separation survives but is
small: r ∈ [0, 8] gives R1 0.961, R2 1.054, R3 1.056 in the full space.

The plane still inflates, and by more here: 27 of 27 cells above 1, median **1.105**,
range 1.053–1.268, and inside the evidence window the pooled inflation reaches **1.29 on
R2**. So a fork panel drawn on the rollout tree shows R2 at 1.36× the control when the
branch is really at 1.05×. That is the figure's own warning about reading a projection,
at its largest.

r = 0 is the same measurement in both trees for δ̂ (the rollout copies the prefix
verbatim), but **not for this quantity**, because the plane is refit on the rollout's
K-averaged displacements: raw layer 5, R2 at r = 0 is 1.711 in the spliced plane and
1.802 in the rollout plane. The full-space numbers move too (1.528 → 1.363), which is the
vector-vs-scalar K reduction above, not a difference in the model's behaviour at t₀.

## Choices and limits

- **Stride 2, not every ply.** `fork_plane.positions` samples at t₀'s parity because the
  board code is mover-relative and alternates each ply; the plane is the fork panels'
  plane only if it is fit on the positions they draw. F1's every-ply axis is the raw-δ̂
  figure's; this one inherits the fork rule. Cost: 17 offsets instead of 33.
- **`control_legal_2` is dropped**, from the plane fit and from the panels, exactly as
  `../fork_and_board` drops it — a second legal move is the null *for* the control and
  belongs where the control is measured. Its rows are in the npz for anyone who wants the
  legal-vs-legal null in this metric; nothing here draws it.
- **n = 1553 is `complete_games`**, the games carrying the reference, both controls and
  all six illegal categories. That fixes each rule mean's composition at 2/3/1 members in
  every game, so a rule's curve is not partly a change in which categories it averages.
  The weaker `rule_complete_games` population (2000) would let that vary.
- **The rule means hide a weak grouping.** `rule_groups` measures within- ÷ between-rule
  spread of the displacements at 0.945 [IQR 0.872–1.020] on this same population and
  field, i.e. a rule's members travel barely closer to each other than to another rule's.
  Three confident lines should not be read as three coherent classes. That number lives
  here and not on the figure, which carries no footer text: a deliberate departure from
  FIGURE_GUIDELINES 1.7, which asks for it in-panel.
- **`--plane r21` is n = 100 in both trees.** The whole tree would need t₀ ≥ 18 for a
  branch-free window to exist at all (`curvature_plane.usable`); the shared t₀ = 20
  population sidesteps that. On the rollout tree the K = 4 rows are averaged after φ,
  never before, and `measure.py` asserts Δφ collapses on both branch-free offsets.
- **The raw pair departs from FIGURE_GUIDELINES 4.2**, which says a quantity whose scale
  differs across panels goes on the axis as a paired ratio and never raw. 4.2 also asks
  for the reference's raw scale per panel; the ratio figure is the compliant one and the
  raw pair is that scale published as a figure rather than only as the table above. It is
  a companion, never the figure a cross-panel claim is read off.
- **Seed 42 only**, like the rest of the study.
- **The rollout K reduction is the fork figures' and not F1's**, and the difference is
  unmeasured. See "The rollout variant" above.
- **The y-cap is 2.0** and a few panels' r = 0 spike runs past it (two spliced, three
  rollout); each prints the value it reaches. One shared y-axis is the point of the
  normalisation, and
  the alternative — a top set by one cell's spike — leaves the plateau, where the claim
  is, in the bottom eighth of every panel.
- **Not measured here**: whether the inflation is the plane's *variance share* per branch
  (the `../fork_and_board` identity ratio_plane = ratio_full × share_illegal/share_control
  predicts it exactly, and the shares are not written out by `measure.py`). Adding the
  per-branch share would settle it in one more column.
- `fork_plane.space_features` rebuilds the SAE encoder on every call, so it costs a
  checkpoint load per game. `measure.py` builds the nine encoders once instead; the fix
  belongs in `src` and is not made here.
