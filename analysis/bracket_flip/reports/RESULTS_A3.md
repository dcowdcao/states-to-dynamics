> **Correction.** This report's headline — *the flanked square is the one square whose
> class does not change* — was retired by red-teaming; see `docs/claims_status.md` for
> what survives and what may not be restated.

# A3 — probe-space read-out of a bracketing flip

M1 (decode), M2 (displacement field) and M5 (depth) with figures F1, F2, F5.
Everything below is layer-5 `resid_post` unless a layer is named, decoded with a
TPR (`d_r`=52, `d_f`=2, seed 1111) fit on this study's own 20k probe set.

Reproduce (the login node is capped at 1 core; 36 probe fits need a compute node):

```
sbatch --account=ba_lab --partition=sapphire --cpus-per-task=16 --mem=64G --time=2:00:00 \
  --wrap="SPADE_BRACKET_DIR=... python src/bracket_flip/probe_space.py --threads 16"
python analysis/bracket_flip/make_figures.py [--camera-ready]
```

Job 35957638, 16 cores, ~13 min. Probes cache to `$SPADE_BRACKET_DIR/probe_space/`,
CSVs to `analysis/bracket_flip/results/`, figures to `analysis/bracket_flip/figures/`.

## Cross-validation contract

| key | value | 95% CI | as computed here |
|---|---|---|---|
| `n_events` | **2000** | — | every row of `events/events.csv`; nothing dropped |
| `p_s_unchanged_rel` | **0.9880** | [0.9830, 0.9925] | `argmax` class at *s* identical before and after |
| `p_unflipped_swap_rel` | **0.9668** | [0.9652, 0.9684] | codes 3+4, `theirs→mine` or `mine→theirs`; n = 58,434 discs |
| `p_s_absolute_correct` | **0.9880** | [0.9830, 0.9925] | *s* reads the opponent's colour before **and** the mover's after |
| `acc_rel_L5` | **0.9829** | — | held-out board state, 12,000 positions × 64 squares |
| `acc_abs_L5` | **0.7349** | — | same probe architecture and seed, absolute-colour labels |
| `peak_layer` | **4** | — | `argmax` over blocks 0–7 of `p_s_absolute_correct` (0.9920) |

Intervals are percentile bootstrap over **events** (B = 2000), not over squares:
one event per game, so events are independent and the 64 squares inside one are
not. All raw numbers in `analysis/bracket_flip/results/`.

**Three warnings about these keys before anyone compares them.**

1. `p_s_absolute_correct` is **not independent evidence**. Composing the relative
   label with whose turn it is maps "opponent's colour before" to `theirs` and
   "mover's colour after" to `theirs` as well, so the metric is identically
   `P(theirs, theirs)`. Measured: the two agree to every printed digit at all
   nine layers. Relabeling is exact, as the design says; it is not a second
   measurement, and F1(b) is a deterministic reflection of F1(a), not a new panel.
2. `p_s_unchanged_rel` is weak on its own — a probe that answered `empty`
   everywhere would score 1.0. The strict form `P(theirs, theirs)` = 0.9880 is
   the number to quote, and here it happens to equal the loose one. *Both forms
   were retired afterwards: a board oracle scores 1.0 on either. Quote neither —
   see the correction above.*
3. `peak_layer = 4` reports the probe as much as the model. See M5.

## M1 — decode

Decoded class at *s*, all 2000 events, layer 5:

| condition | `empty` | `theirs` | `mine` |
|---|---|---|---|
| before *m* | 0.0045 | **0.9930** | 0.0025 |
| after *m* | 0.0010 | **0.9950** | 0.0040 |
| after the control *m'* | 0.0025 | 0.0065 | **0.9910** |

The control row is the measurement that makes the headline mean something.
Under *m'* the perspective flips exactly as it does under *m*, *s* is not
flipped, and *s* swaps: `p_s_unchanged_ctrl` = **0.0075** [0.0040, 0.0115].
So *s* holds still under *m* (0.9880) and moves under *m'* (0.9925) — a
double dissociation, not a claim that *s* is inert.

Unflipped discs, by colour: mover's **0.9656** [0.9632, 0.9677] (n = 28,393),
opponent's **0.9680** [0.9659, 0.9700] (n = 30,041).

Agreement with A2's alignment check, which used the *committed linear probe* on
the same activations (`DATA_REPORT.md`): *s* reads `theirs` before 0.9950 vs my
0.9930, and *s*-unchanged 0.9900 vs my 0.9880. A2's 0.9757 for unflipped discs
is "the class changed"; mine is the strict `theirs↔mine` swap, so those two are
not the same quantity and should not be diffed.

### The basis test: the model does not carry absolute colour

Same architecture, same seed, same 8000/12000 split; only the labels change.

| probe | mover-relative | absolute colour |
|---|---|---|
| TPR (`d_r`=52, `d_f`=2) | 0.9829 | **0.7349** |
| linear (512×8×8×3) | 0.9909 | **0.7374** |

24.8 points. Decomposing it settles what the number means:

| | occupancy (empty vs occupied) | colour, among truly occupied |
|---|---|---|
| TPR, relative | 0.9933 | 0.9720 |
| TPR, absolute | 0.9929 | **0.5107** |
| linear, relative | 0.9981 | 0.9840 |
| linear, absolute | 0.9977 | **0.5145** |

Always answering "black" scores 0.5083 on occupied squares. The absolute probes
beat that by **0.24 and 0.62 points**. The occupancy-only ceiling — perfect
empty/occupied, then one colour always — is 0.4609 + 0.2740 = **0.7349**, which
is `acc_abs_L5` to four decimals. The absolute-colour probe carries essentially
no absolute colour.

Two things this rules out. It is not the TPR's low-rank factorization: the
unfactorized linear probe collapses by the same 24.8 points and lands 0.62 above
the ceiling. And it is not a regularization artifact: both bases were fit at the
same weight decay, epochs and split, as `PROBE_REPORT.md` §7 asks.

So mover-relative is the model's basis and absolute colour is something we
compute from it — Li et al.'s result for linear probes, reproduced here for a
tensor-product probe as well. It also means the "white becomes black" framing is
ours, not the model's; the model's own statement of the event is "the square
that did not swap".

### Robustness (slices of `m1_per_event.csv`)

| slice | n | `p_s_unchanged_rel` | `p_s_absolute_correct` | `p_s_unchanged_ctrl` |
|---|---|---|---|---|
| mover = black (the literal "white before, black after") | 1004 | 0.9871 | 0.9871 | 0.0060 |
| mover = white | 996 | 0.9890 | 0.9890 | 0.0090 |
| strict control | 1596 | 0.9881 | — | 0.0056 |
| loose control | 404 | 0.9876 | — | 0.0149 |
| both positions inside the probe's [5, 55) range | 1906 | 0.9879 | 0.9879 | — |
| outside it (t = 5, or t ≥ 55) | 94 | 0.9894 | 0.9894 | — |

No slice moves the headline by more than 0.2 points. The one visible gradient is
board density: by *t* quartile the read-out is 0.9980 (t ≤ 13), 0.9864, 0.9862,
0.9806 (t > 38) — 1.7 points lost across the game, consistent with late boards
being harder for the probe, not with the flip being read differently.

The loose controls' `p_s_unchanged_ctrl` is 2.7× the strict ones' (0.0149 vs
0.0056), which is the expected direction: a loose *m'* may put a disc on one of
*s*'s rays. Both are far below 1.

## M2 — displacement field (F1)

Drawn in gauge-invariant logit differences, per the design: `x = ℓ_mine −
ℓ_theirs`, `y = ½(ℓ_mine + ℓ_theirs) − ℓ_empty`. Filler coordinates are not used
anywhere in this report — `PROBE_REPORT.md` §4 measures the theirs–mine angle
moving 131.3° → 113.3° → 104.5° across seeds at 0.0006 of accuracy, so a panel
drawn there would not survive a reseed.

Mean position per role (logits), layer 5, n as in the table:

| role | n squares | before | after | control *m'* |
|---|---|---|---|---|
| flipped *s* | 2,000 | (−8.15, 4.19) | (−7.49, 5.84) | (**+7.07**, 4.02) |
| unflipped, mover's | 28,393 | (+6.61, 4.46) | (−7.03, 5.16) | (−7.05, 5.12) |
| unflipped, opponent's | 30,041 | (−7.14, 5.08) | (+6.65, 4.52) | (+5.05, 4.71) |
| placed *m* | 2,000 | (+0.54, −8.95) | (−8.14, +3.33) | (+1.08, −7.16) |
| empty | 65,566 | (+0.58, −8.56) | (+0.55, −8.54) | (+0.25, −8.23) |

95% CI half-widths on these means are 0.03–0.14 logits, smaller than the plotted
markers, so F1 also carries the *dispersion* (50% Gaussian contour of the point
cloud) to keep the reader from mistaking a mean shift for class separation.

Per-event median displacement `‖Δ‖` (median over events, IQR in brackets):

| role | native basis | absolute colour |
|---|---|---|
| flipped *s* | **3.8** [2.3, 5.6] | **16.0** [13.9, 17.9] |
| unflipped, mover's | 14.8 [13.2, 16.6] | 3.0 [2.5, 3.4] |
| unflipped, opponent's | 14.9 [13.3, 16.5] | 2.9 [2.5, 3.3] |
| placed *m* | 15.4 [13.3, 17.5] | 14.7 [13.0, 16.4] |
| empty | 4.4 [3.8, 5.0] | 3.3 [3.0, 3.7] |

**In the probe's own basis the flanked disc moves less than an empty square
does** (3.8 vs 4.4) while every other disc moves ~15. Relabel to absolute colour
and the table inverts exactly. That inversion is the design's "a flip is the
absence of a swap", measured.

## M5 — depth (F5)

Layers are `resid_post` of transformer blocks 0–7. The leftmost tick is the
input space, `W_E[token] + W_pos[pos]`, verified equal to `blocks.0.hook_resid_pre`
to 0.0 max absolute difference — so it needs no forward pass and is exact.

| layer | `p_s_unchanged_rel` | `p_s_absolute_correct` | `p_unflipped_swap_rel` | `p_s_unchanged_ctrl` | TPR rel | linear rel | gap (pts) |
|---|---|---|---|---|---|---|---|
| embed | 0.5580 | 0.4060 | 0.3300 | 0.5555 | 0.6252 | 0.6255 | 0.03 |
| 0 | 0.8120 | 0.8115 | 0.7904 | 0.1925 | 0.8841 | 0.8910 | 0.69 |
| 1 | 0.9145 | 0.9145 | 0.8840 | 0.0810 | 0.9324 | 0.9362 | 0.38 |
| 2 | 0.9635 | 0.9630 | 0.9380 | 0.0295 | 0.9615 | 0.9646 | 0.31 |
| 3 | 0.9820 | 0.9820 | 0.9632 | 0.0185 | 0.9763 | 0.9785 | 0.22 |
| **4** | 0.9920 | **0.9920** | **0.9715** | 0.0130 | **0.9833** | 0.9877 | 0.44 |
| 5 | 0.9880 | 0.9880 | 0.9668 | **0.0075** | 0.9829 | **0.9909** | 0.80 |
| 6 | 0.9720 | 0.9705 | 0.9392 | 0.0100 | 0.9725 | 0.9882 | 1.57 |
| 7 | 0.9670 | 0.9630 | 0.9282 | 0.0170 | 0.9685 | 0.9867 | 1.82 |

**The input anchor does the work the design asked it to.** At the embedding the
read-out is 0.4060 and unflipped discs swap 0.3300 of the time — the flip
structure is computed by the stack, not inherited from the token and position.

**`peak_layer = 4`, and it is a probe artifact as much as a model fact.** The
TPR's own accuracy peaks at block 4; the linear probe's peaks at block 5, where
it is 0.9909 against the TPR's 0.9829. Reading the gap instead of the curve:
the TPR is handicapped by 0.44 points at layer 4 and 0.80 at layer 5, a
difference of 0.36 points — the same size as the 0.40-point drop in
`p_s_absolute_correct` between those layers. The layer-4 peak is within the
capacity confound and should not be quoted as "the flip is written by block 4".
The honest statement is that the read-out saturates over blocks 3–5 and the
decline after that tracks the TPR's own late-layer droop (1.57 and 1.82 points),
where the linear probe stays flat.

**A5: if `peak_layer` disagrees, check the estimator before the data.** The
argmax over blocks of `p_s_unchanged_rel` is also 4, and of `p_unflipped_swap_rel`
also 4, so the three read-out statistics agree with each other; the disagreement
that matters is with the linear probe's 5.

The control curve is the cleanest depth signal in the figure: `p_s_unchanged_ctrl`
falls monotonically from 0.5555 at the embedding to 0.0075 at block 5 and turns
back up at 6–7 with the probe's accuracy.

## F2 — the worked example

`analysis/bracket_flip/figures/f2_worked_event.png`, from `results/f2_event.csv`.

Selection rule, mechanical and stated so it cannot be a cherry-pick: among the
**34** events at the median *t* (25) that carry a *strict* control, take the one
whose flipped-square displacement is closest to the median over all 2000 events.
Target 3.848, chosen event 3.860 — row 1882, *t* = 25, mover white, *s* = e4,
*m* = f3.

The rule targets typicality of the *measurement*, and one thing about the drawn
event is nonetheless better than average: all 128 square-decodes (64 squares × two
boards) are correct. That happens for **825 / 2000 = 41.2%** of events; the median
event gets 1 square wrong and the mean 1.67, at a per-square accuracy of 0.9874
(before) and 0.9865 (after). So the panel is cleaner than a coin flip would give,
by luck of the displacement criterion rather than by selection on decode quality —
worth knowing before reading the two decode panels as representative.

The panels: true board before/after (absolute discs, *s* and *m* outlined),
the TPR decode of the same two boards (the probe's classes), the three logits at
*s* across before/after/control, and this one event's 64 squares in F1(a)'s
coordinates. *s* reads `theirs` before (ℓ = 13.7 over `empty`) and after (15.7),
and `mine` under the control (13.7 vs `theirs` 10.0).

## Figures: what every band is, and n

| figure | element | semantics | n |
|---|---|---|---|
| F1(a,b) | error bars on each arrow head | 95% percentile CI on the mean, cluster bootstrap over events, B=2000 | 2,000 events |
| F1(a,b) | ellipses | 50% dispersion contour of a Gaussian fit to the point cloud — **not** a CI; dotted = before, solid = after | 2,000–65,566 squares, per role in the legend |
| F1(c,d) | shaded band | 95% percentile CI, bootstrap over events at fixed full-sample bandwidth | 2,000 events |
| F1(c,d) | curve | Gaussian KDE, Scott's rule; h = 0.21–0.72 (native), 0.13–0.68 (absolute) logits | 2,000 |
| F1(c,d) | ▼ | median; IQR in the legend entry | 2,000 |
| F2 | none | one event; no aggregation, so no uncertainty | 1 |
| F5(a,b) | shaded band | 95% percentile CI, cluster bootstrap over events, B=2000 | 2,000 events; 58,434 unflipped discs for the swap curves |
| F5(c) | shaded region | the TPR−linear capacity gap, one fit per point (no resampling) | 12,000 held-out positions × 64 squares |

F5(c) has no interval: each point is a single probe fit, and a seed-to-seed
spread is 0.0006 (`PROBE_REPORT.md` §3), an order of magnitude below the gaps
being read. Saying so is more honest than drawing a band that means something else.

## What failed, and what is soft

1. **The login node has one core.** `nproc` = 1 under the cgroup and load average
   ~28, so the first attempt was heading for two hours. The 36 probe fits belong
   on a compute node; the reproduce command above is the one that works.
2. **`peak_layer` is confounded and I could not de-confound it.** See M5. The
   number is reported because the contract asks for it, with the caveat attached.
3. **`p_s_absolute_correct` adds no information** over `p_s_theirs_both`. This is
   arithmetic, not a bug, but a reader who treats the contract's relative and
   absolute rows as two checks will be double-counting one.
4. **F1(b) is a reflection, not a measurement.** Orienting the absolute panel so
   the mover is black is exactly `x → −x` on the after points. Drawn because it
   makes the claim legible; it should never be cited as independent.
5. **The absolute panel needs an orientation at all** because half the events
   have a white mover; an un-oriented `ℓ_black − ℓ_white` field averages the two
   halves to nothing. Both halves are in the robustness table instead.
6. **94 events (4.7%) read a position outside the probe's training range** [5, 55):
   `before` sits at *t*−1, so *t* = 5 is one short at the bottom and *t* ≥ 55 runs
   past the top. Measured, no effect (0.9894 vs 0.9879).
7. **The linear capacity reference is my own fit, not the committed one.**
   `analysis/linear_probe/train_linear_probe.py` imports `transformer_lens` and
   the ARENA utilities at module scope, so it cannot be imported as a library,
   and its committed accuracies are on a different 200-game ARENA corpus. I
   re-implemented its sweep recipe (Adam, lr 1e-2, 200 epochs, full batch, same
   8000-row split) against this study's activations so the two probes differ only
   in architecture. That reproduces 0.9909 at layer 5 against the committed
   0.9896 — a different corpus, so the two are close but not the same number.
8. **`tpr_probe._fit` is imported across module privacy.** See the requests below.
9. **Not attempted:** the TPR intervention (M6, deferred by design), any
   activation-space geometry (M3/M4, A4's), and per-square-role breakdowns of the
   basis test.

## Requests for shared files (not edited by me)

1. **`src/figures/paper_style.py`: add a probe-family and label-basis pair.**
   F5(c) compares two probe families across two label conventions, a dimension no
   existing dict covers. `analysis/bracket_flip/make_figures.py` carries a local
   `PROBE_STYLE` / `BASIS_STYLE` whose colours are indexed out of the imported
   `OKABE_ITO` (no literals), which should move into `paper_style.py` before a
   second figure needs it and picks a different hue.
2. **`src/figures/paper_style.py`: move `ROLE_TEXT` there, next to `RULE_TEXT`,
   with a `role_key()` beside `rule_key()`.** It currently lives in
   `probe_space.py`, so the plotting script imports torch and the vendored TPR to
   print five lines of legend key.
3. **`src/bracket_flip/tpr_probe.py`: the cache path does not encode the training
   set.** `train()` writes `$SPADE_BRACKET_DIR/tpr_probes/tpr_r52_f2_layer{L}.pt`
   whether it was fit on the committed 10k set or this study's 20k set, so two
   agents fitting different sets silently overwrite each other — and A4 loads that
   path live for the edit direction. A3 therefore caches under
   `probe_space/` and calls the module-private `_fit`. Folding the training-set
   identity into `_path()` would make both the private import and the second
   cache unnecessary.
