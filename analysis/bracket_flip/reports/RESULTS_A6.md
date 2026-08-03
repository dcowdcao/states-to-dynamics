# A6 — M6, the causal cross-check

`src/rules/causal/unflip_edit.py` edits layer-5 `resid_post` along **linear-probe**
directions so a flipped square reads its pre-flip colour, and reports **+1.07**
log-mass onto the counterfactual board's legal set at β=1 against **+0.10** for a
magnitude-matched random direction. M6 redoes that edit with the paper's **TPR**
intervention — `ΔB = r_s (f_mine − f_theirs)ᵀ` mapped back to the residual through
`pinv(M)` — on the mined single-flip events, scored by the same code.

**The two methods agree.** At matched edit magnitude the TPR edit moves *more*
mass onto the counterfactual legal set than the linear edit (+1.50 vs +1.05, both
against a +0.12 random null) and changes *fewer* other squares' decoded class
(0.19 vs 0.22 of 63). The effect is not bought with collateral at the scale that
matters: at TPR scale 4, the smallest that reliably flips the read-out (97.8% of
events), the flanked square's edit disturbs **fewer** other squares than a
magnitude-matched random direction does (0.250 vs 0.281). Collateral only becomes
the story above scale 8, where the effect is large but half the events take
board-wide damage.

Job **35961417**, `kempner_h100`, **3 m 02 s** wall (11 forward passes over 3,673
edits, plus figures). Reproduce: `sbatch shell/slurm/submit_bracket_intervene.sh`.

## 1. What was run

- Events: all 2,000 mined single-flip events (`$SPADE_BRACKET_DIR/events/events.csv`),
  replayed with `rules/causal/boards.py`. Sequence is `prefix + m`; the edit lands
  at layer 5, position *t*, the injected move's own position; the score reads the
  next-move logits at that position.
- Counterfactual board: the real post-*m* board with the edited square set back to
  the opponent's colour. Diagnostic squares are legal on exactly one of the two
  boards; the score is Δ log-mass on `cf_only` (should rise) and on `real_only`
  (should fall) against an unedited run. `set_logmass` is **imported from**
  `unflip_edit.py`, not reimplemented.
- Populations, all edited theirs→mine at layer 5:

  | population | square | n rows |
  |---|---|---|
  | `flipped` | *s*, the disc *m* just flipped | 1,370 |
  | `settled` | a mover-coloured disc **not** flipped, age ≥ 10 plies (`unflip_edit`'s definition) | 933 |
  | `random` | *s*, random direction, magnitude matched per event | 1,370 |

- Edit families: `tpr` (scale ∈ {1,2,3,4,6,8,12,16} on the unnormalized least-norm
  `edit_direction`), `linear` (unflip_edit's protocol, β ∈ {1,2}), and
  `tpr_matched` (TPR direction rescaled per event to the linear edit's own length —
  the head-to-head at exactly matched magnitude).

Event accounting: 2,000 events → 1,370 usable `flipped` rows. 630 events (31.5%)
were dropped because un-flipping *s* leaves the opponent's legal set unchanged or
empty, so there is no diagnostic square; 411 events have no settled disc.
Log-mass on `cf_only` is undefined when that set is empty, so the `d_lp_cf` means
below are over n=986 (`flipped`/`random`) and n=532 (`settled`); `d_lp_real` over
n=891 and n=619. Every n is in `analysis/bracket_flip/results/intervention_summary.csv`.

Sanity: the TPR probe reads the edited square as `theirs` before the edit in
**99.64%** of `flipped` rows and 99.04% of `settled` rows; the linear probe agrees
(99.56% / 99.14%). Δ log-mass on the *full* legal set is −0.0002 at scale 4, so the
edit is not simply inflating or deflating legality everywhere.

## 2. Head to head

All on the same 2,000 events, same scoring. Uncertainty is a 95% percentile
bootstrap of the mean over events (1,000 resamples). Collateral = other squares
(of 63) whose decoded class changed, read with the edit's own probe; "clean" =
fraction of events with zero collateral.

| edit | ‖Δh‖ | flip rate | Δ log-mass cf-only | Δ log-mass real-only | collateral | clean |
|---|---|---|---|---|---|---|
| **linear, β=1** (reference) | 2.66 | 1.000 | **+1.051** [+1.011, +1.095] | −0.098 [−0.116, −0.081] | 0.217 | 0.823 |
| **TPR, matched to β=1** | 2.66 | 0.976 | **+1.496** [+1.447, +1.550] | −0.095 [−0.111, −0.079] | 0.194 | 0.836 |
| random null at that size | 2.66 | 0.004 | +0.121 [+0.101, +0.143] | −0.000 | 0.159 | 0.867 |
| **linear, β=2** | 5.32 | 1.000 | **+2.971** [+2.872, +3.074] | −1.323 [−1.427, −1.212] | 0.650 | 0.606 |
| **TPR, matched to β=2** | 5.32 | 1.000 | **+4.105** [+3.987, +4.235] | −1.450 [−1.558, −1.348] | 0.458 | 0.690 |
| random null at that size | 5.32 | 0.003 | +0.447 [+0.404, +0.486] | +0.000 | 0.528 | 0.652 |
| **TPR, scale 4** (smallest reliable) | 3.08 | 0.978 | **+1.792** [+1.736, +1.841] | −0.148 [−0.169, −0.129] | 0.250 | 0.799 |
| random null at that size | 3.08 | 0.003 | +0.159 [+0.136, +0.185] | −0.000 | 0.281 | 0.779 |

Against the published linear-probe numbers (`analysis/rules_causal/README.md`,
job 34239305, a different item set — any legal move, `|flip_set|` up to 5+):

| quantity | published (linear probe) | this study, linear on my events | this study, TPR |
|---|---|---|---|
| `flipped`, β=1 / matched | +1.070 [+1.028, +1.121] | +1.051 [+1.011, +1.095] | +1.496 [+1.447, +1.550] |
| `random`, same magnitude | +0.103 [+0.084, +0.124] | +0.121 [+0.101, +0.143] | +0.121 [+0.100, +0.141] |
| effect ÷ null | 10.4× | 8.7× | 12.4× |
| `settled`, β=1 / matched | +1.235 [+1.159, +1.309] | +1.296 [+1.225, +1.366] | +1.755 [+1.667, +1.838] |
| `flipped`, β=2 / matched | +3.040 [+2.928, +3.149] | +2.971 [+2.872, +3.074] | +4.105 [+3.987, +4.235] |

**Verdict: the two methods agree.** Same sign, same order of magnitude, same ~10×
separation from the random null, and the same asymmetry (real-only mass falls while
counterfactual-only mass rises). Rerunning the *linear* edit on my single-flip event
set reproduces the published number to within 2% (+1.051 vs +1.070), which says the
event set and the reimplemented scoring are not doing the work. The TPR edit is
42% *stronger* at identical ‖Δh‖ and 39% stronger at β=2 — a difference in
efficiency, not in conclusion.

Two secondary agreements worth recording. First, `settled ≈ flipped` under the TPR
edit exactly as under the linear one (+1.76 vs +1.50 at matched magnitude; +2.01 vs
+1.79 at scale 4): a just-flipped disc is no more fragile than one that has sat for
ten plies, so what the model uses is the board state, not the recency of the flip.
Second, the two probes agree on *when* an edit lands: at TPR scale 4 the flip rate
is 0.978 read by the TPR probe and 0.983 read by the linear probe. The two curves
never differ by more than 0.10, and that worst case is at scale 2 where the edit is
half-landed; the linear probe is the more permissive of the two below scale 4
(0.40 vs 0.30 at scale 2, 0.83 vs 0.77 at scale 3).

## 3. Scale sweep and collateral

`flipped`, TPR edit. `‖Δh‖/‖h‖` is against the mean layer-5 residual norm 23.3.

| scale | ‖Δh‖ | ‖Δh‖/‖h‖ | flip rate | Δ cf-only | Δ real-only | collateral | clean | random-null collateral |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.77 | 0.033 | 0.044 | +0.278 | −0.001 | 0.058 | 0.944 | 0.054 |
| 2 | 1.54 | 0.066 | 0.302 | +0.663 | −0.005 | 0.104 | 0.903 | 0.131 |
| 3 | 2.31 | 0.099 | 0.771 | +1.170 | −0.035 | 0.172 | 0.852 | 0.195 |
| **4** | **3.08** | **0.132** | **0.978** | **+1.792** | **−0.148** | **0.250** | **0.799** | 0.281 |
| 6 | 4.62 | 0.199 | 1.000 | +3.270 | −0.865 | 0.386 | 0.720 | 0.447 |
| 8 | 6.17 | 0.265 | 1.000 | +4.831 | −1.968 | 0.575 | 0.632 | 0.645 |
| 12 | 9.25 | 0.397 | 1.000 | +7.149 | −3.605 | 1.040 | 0.496 | 1.183 |
| 16 | 12.33 | 0.529 | 1.000 | +8.077 | −4.587 | 1.744 | 0.344 | 1.984 |

**Smallest scale that reliably flips the decode: 4** (97.8% of events; 100% from
scale 6). This is the value PROBE_REPORT §5 predicted from median λ* ≈ 1.84 plus
the 96% flip rate it measured at scale 4 on held-out activations, and it lands the
same way here on real event activations.

Reading the collateral column against the last one is the point. From scale 2
upward the TPR edit's collateral is **below** what a magnitude-matched random
direction produces (0.250 vs 0.281 at scale 4; 1.744 vs 1.984 at scale 16); only at
scale 1, where the edit almost never lands, is it marginally above (0.058 vs 0.054).
So at the scale that reliably flips, the intervention is clean by the only
comparison that controls for the size of the perturbation: 80% of events see no
other square move at all, and the 0.25 squares that do move are fewer than a random
push of the same length moves. **The effect at scale 4 is not bought with
board-wide damage.**

Above scale 8 that stops being true in absolute terms — at scale 16 only 34% of
events are collateral-free and the mean is 1.7 squares — but the effect there is
also no longer specific: the random null itself reaches +1.94 log-mass at that
size, a quarter of the +8.08 the real edit gets. Large edits move mass because they
are large. The honest operating point is scale 3–4.

The random direction's collateral is not zero and grows with magnitude, which is
the correct behaviour for a null: at ‖Δh‖ = 12.3 a random push is half the
residual's own length and moves the read-out wherever it lands. What separates the
edit from the null is not that the null is inert but that the null is *undirected*:
its flip rate at the target square stays at 0.003–0.014 across the entire sweep
while the TPR edit goes 0.044 → 1.000.

## 4. Both controls

1. **Magnitude-matched random direction** at the same square, same per-event
   ‖Δh‖, generated from a fixed seed so the same vectors are used in every pass.
   Flip rate 0.003–0.014 across the sweep (vs 0.978 for the edit at scale 4).
   Δ cf-only +0.159 at scale 4 against +1.792 — **11.3×**. Δ real-only is
   statistically zero at every scale ([−0.001, +0.001] at scale 4), so the random
   direction produces no asymmetry at all: it inflates both sides slightly, while
   the real edit moves mass *from* one set *to* the other.
2. **The same edit on squares that were not flipped** (`settled`, mover-coloured,
   age ≥ 10). +2.006 [+1.920, +2.092] at scale 4, against +1.792 for `flipped` —
   a shade *larger*, with a shade more collateral (0.310 vs 0.250) and a slightly
   lower flip rate (0.966 vs 0.978). This reproduces `unflip_edit`'s
   `settled ≈ flipped` finding, and it is the right result to report as a control
   rather than to explain away: the edit is not specific to just-flipped squares,
   because nothing about the model's use of a disc depends on when it arrived. A
   control that *should* separate — and does — is the random direction.

## 5. What failed, and what to be careful of

- **31.5% of events yield no diagnostic square.** Un-flipping *s* often leaves the
  opponent's legal set identical, which makes the score undefined. This is the same
  filter `unflip_edit.build_rows` applies, so the comparison is unaffected, but the
  effective n is 986, not 2,000, and any claim about "all mined events" is wrong.
- **The scale knob and β are not the same quantity**, and putting both on a "scale"
  axis would be meaningless. β=1 is defined as "move the projection onto the
  probe direction to the target class's mean"; the TPR scale is a plain multiplier
  on a least-norm residual step. That is why the head-to-head above is run at
  matched ‖Δh‖ (`tpr_matched`) and why the figure's x axis is ‖Δh‖.
- **`linear` flip rate is 1.000 by construction** when read with the linear probe:
  the edit is *defined* as moving the projection to the target class's mean, so the
  argmax follows. Read with the TPR probe the same edit flips 0.958 of events. The
  numbers in the flip-rate column are therefore not a fair contest between methods,
  only a statement of when each edit has landed in its own terms. The Δ log-mass
  columns are the comparable ones.
- **Collateral is instrument-dependent.** Each edit is judged by its own probe in
  the table above; the CSV also carries the cross-decoder columns
  (`collateral_tpr` / `collateral_lin` for every condition). They agree to within
  0.05 squares up to scale 8 and diverge above it (TPR edit at scale 16: 1.74 by
  the TPR probe, 2.20 by the linear probe) — one more reason not to read the
  large-scale end of the sweep as a measurement.
- **`analysis/linear_probe/activations/` is gitignored**, so a worktree cannot see
  it. `tpr_intervene.py` resolves it through `SPADE_LINEAR_PROBE_ACTIVATIONS`, the
  same override `tpr_probe.py` and `act_geometry.py` use, and
  `submit_bracket_intervene.sh` sets a default. Without it the run fails at load.
- **The published +1.07 is measured on a different item set** — every legal move,
  including moves that flip up to five discs at once — while these events are all
  `|flip_set| == 1`. That is why the linear edit was re-run here rather than quoted:
  +1.051 on the single-flip subset against +1.070 on the mixed set says flip-set
  size does not move this number much, but it had to be checked, not assumed.
- Nothing crashed on the cluster. The one bug found in development was a bool
  tensor subtraction in the collateral count, fixed before the production run.

## 6. Files

```
src/bracket_flip/tpr_intervene.py                        the edit, the sweep, the scoring
shell/slurm/submit_bracket_intervene.sh                  job 35961417
analysis/bracket_flip/intervention_figures.py            F6
analysis/bracket_flip/figures/f6_tpr_intervention.{png,pdf}
analysis/bracket_flip/results/intervention_per_event.csv   one row per (event, population, method, scale)
analysis/bracket_flip/results/intervention_summary.csv     the tables above, with every CI and n
analysis/bracket_flip/results/intervention_manifest.json   probe paths, skips, seeds
```

F6 panels: (a) Δ log-mass onto counterfactual-only squares against edit size, (b)
the same off real-only squares, (c) flip rate of the edited square's decode, (d)
collateral against edit size, (e) effect against collateral — the panel that
decides whether the intervention is clean, (f) the per-event distribution at
scale 4, which the means in (a) hide. The linear-probe edit is drawn as the
reference (black), the random direction as the null (gray, dashed, last).
