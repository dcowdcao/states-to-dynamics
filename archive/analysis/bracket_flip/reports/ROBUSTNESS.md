> **Correction.** The headline this report stabilizes — `p_s_unchanged_rel`, "the
> flanked square is the one square whose class does not change" — was retired: it is
> the probe's accuracy at *s* squared, and §3's own result that it tracks accuracy
> across ranks at r ≈ 0.999 is the evidence that killed it, read here as support.
> A stable number is not an informative one. `peak_layer` = 4 is the TPR's answer;
> an independent linear probe gives 5–6. Kept as the record of the sweep, not as a
> statement of results — `docs/claims_status.md`.

# Robustness — A3's headline numbers under reseeding, retuning and re-ranking

A3 reports one TPR probe at one seed with a weight decay that A1 retuned. This
refits it 68 times over three axes and recomputes the five headline numbers at
every setting.

Reproduce (68 fits, one job; the login node is capped at 1 core):

```
cd shell/slurm && sbatch submit_bracket_robustness.sh
```

All 68 fits ran in job **35962124**, one H100, **2 min 21 s** — 22 s for the first
(CUDA warm-up), 1 s each after. Probes cache to `$SPADE_BRACKET_DIR/robustness/`,
never to `tpr_probes/` or `probe_space/`, so the study's committed fits are
untouched; the CSVs in `analysis/bracket_flip/results/robustness_*.csv` are from
job 35962816, which recomputed every metric against that cache in **40 s**.

Throughout: 2,000 events, 58,434 unflipped discs, 12,000 held-out positions ×
64 squares. Only gauge-invariant quantities appear — accuracies, decoded
classes, proportions. Filler coordinates are not comparable across fits
(`PROBE_REPORT.md` §4), so reseeding them would measure nothing.

**The refit reproduces A3 exactly.** Seed 1111 / wd 2.0 / r=52 on the H100 gives
`acc_rel_L5` = 0.9828867316246033 and `p_unflipped_swap_rel` = 0.9668172639216894,
bit-identical to the CPU probe committed in `probe_space/`. The fitting loop and
the device are therefore not free choices; the three below are.

## Verdicts

| number | A3 | seed (n=6) | weight decay | role dim | verdict |
|---|---|---|---|---|---|
| `acc_rel_L5` | 0.9829 | 0.9829–0.9835 | 0.9764–0.9829 | 0.8308–0.9916 | **ROBUST** |
| `acc_abs_L5` | 0.7349 | 0.7349–0.7366 | 0.7306–0.7352 | 0.6687–0.7376 | **ROBUST** |
| `p_s_unchanged_rel` | 0.9880 | 0.9880–0.9895 | 0.9740–0.9880 | 0.6920–0.9985 | **SENSITIVE**, one-sided |
| `p_unflipped_swap_rel` | 0.9668 | 0.9667–0.9675 | 0.9508–0.9668 | 0.7799–0.9846 | **SENSITIVE**, one-sided |
| `peak_layer` | 4 | 4 at 6/6 seeds | — | — | **ROBUST** |

"One-sided" is the finding that runs through §2 and §3: across the 15 rows of the
three sweeps the two read-out proportions track layer-5 relative accuracy at
Pearson **0.9988** and **0.9944**, slope 1.96 and 1.25. Nothing in the sweep moves them independently
of probe quality, and every configuration with a better probe reports a *higher*
proportion. 0.9880 is a floor, not an estimate: the best probe tested (r=64,
`acc_rel` 0.9916) reads 0.9985, i.e. 3 events of 2,000 wrong.

**Seed 1111 is the least favourable of the six seeds** on `acc_rel_L5` and
`acc_abs_L5`, tied-lowest on `p_s_unchanged_rel`, second-lowest on
`p_unflipped_swap_rel`.
A3's committed numbers are the conservative end of their own seed distribution.

## 1. Seeds — wd 2.0, r=52, layer 5

| seed | `acc_rel_L5` | `acc_abs_L5` | `p_s_unchanged_rel` | `p_unflipped_swap_rel` | `peak_layer` |
|---|---|---|---|---|---|
| 1111 (A3's) | 0.9829 | 0.7349 | 0.9880 | 0.9668 | 4 |
| 7 | 0.9835 | 0.7357 | 0.9880 | 0.9675 | 4 |
| 2026 | 0.9833 | 0.7366 | 0.9895 | 0.9670 | 4 |
| 4242 | 0.9831 | 0.7361 | 0.9895 | 0.9669 | 4 |
| 31337 | 0.9830 | 0.7356 | 0.9885 | 0.9672 | 4 |
| 90210 | 0.9833 | 0.7354 | 0.9895 | 0.9667 | 4 |
| **mean ± sd** | 0.98320 ± 0.00023 | 0.73572 ± 0.00059 | 0.98883 ± 0.00075 | 0.96701 ± 0.00029 | — |
| **range (points)** | 0.065 | 0.171 | 0.150 | 0.079 | 0 |

The right yardstick is A3's own event bootstrap, which resamples the 2,000
events. Seed noise is far below it: 0.150 points of seed range against a
0.95-point-wide CI for `p_s_unchanged_rel`, and 0.079 against 0.32 points for
`p_unflipped_swap_rel`. The full seed range of `p_s_unchanged_rel` is **3
events**, since one event is 0.05 points. Reseeding does not threaten any of the
five numbers.

## 2. Weight decay — seed 1111, r=52, layer 5

A1 retuned upstream's 1e-2 to 2.0 on an inner split, selecting on accuracy. The
question is whether the read-out proportions came along for the ride.

| wd | `acc_rel_L5` | `p_s_unchanged_rel` | `p_unflipped_swap_rel` | `acc_abs_L5` |
|---|---|---|---|---|
| 0.01 (upstream) | 0.9765 | 0.9740 | 0.9508 | 0.7306 |
| 0.5 | 0.9779 | 0.9800 | 0.9533 | 0.7349 |
| 1.0 | 0.9809 | 0.9840 | 0.9613 | 0.7352 |
| **2.0 (A3's)** | **0.9829** | **0.9880** | **0.9668** | **0.7349** |
| 4.0 | 0.9764 | 0.9860 | 0.9524 | 0.7307 |

**They came along for the ride, and this must be stated plainly.**
`p_s_unchanged_rel` moves 1.4 points across the grid and rises monotonically
with wd over [0.01, 2.0] — the same interval on which wd was selected for
accuracy. That is 9× the seed spread and 1.5× the width of A3's event bootstrap.
`p_unflipped_swap_rel` moves 1.6 points the same way. A3's 0.9880 and 0.9668 are
therefore reported at the setting that maximizes them within the tuned range.

Two things limit the damage. The retune was selected on an inner split of the
*training* 8000 with the eval 2000 untouched (`PROBE_REPORT.md` §6), so this is
not leakage. And the direction of the A3 claim is untouched at every wd: even at
upstream's 1e-2, *s* holds still 0.9740 of the time while unflipped discs swap
0.9508 of the time — the double dissociation survives the worst setting in the
grid. What does not survive is the fourth decimal.

`acc_abs_L5` stays in 0.7349–0.7352 over wd ∈ [0.5, 2.0] and falls to 0.7306 at
both ends, tracking `acc_rel_L5`. The ceiling result of §4 holds at every wd.

## 3. Role dimension — seed 1111, wd 2.0, d_f=2, layer 5

| `d_r` | params | `acc_rel_L5` | `p_s_unchanged_rel` | `p_unflipped_swap_rel` | `acc_abs_L5` |
|---|---|---|---|---|---|
| 16 (upstream's) | 17,414 | 0.8308 | 0.6920 | 0.7799 | 0.6687 |
| 32 | 34,822 | 0.9355 | 0.8885 | 0.8921 | 0.7124 |
| **52 (A3's)** | 56,582 | **0.9829** | **0.9880** | **0.9668** | **0.7349** |
| 64 | 69,638 | 0.9916 | 0.9985 | 0.9846 | 0.7376 |

Role overlap and what it costs, same probes:

| `d_r` | off-diag \|cos\| mean | max | flip rate @ scale 8 | collateral mean | zero-collateral | worst |
|---|---|---|---|---|---|---|
| 16 | 0.1833 | 0.8697 | 0.795 | 3.90 | 0.03 | 14 |
| 32 | 0.1022 | 0.6878 | 0.985 | 2.82 | 0.15 | 11 |
| **52** | 0.0546 | 0.3405 | 1.000 | 0.80 | 0.55 | 4 |
| 64 | 0.0290 | 0.2133 | 1.000 | 0.24 | 0.81 | 3 |

Collateral is the count of the other 63 squares whose decoded class changes under
a `theirs→mine` edit at *s*, one square per activation, n=200, at the fixed scale
8 that flips *s* every time at r≥52 (`PROBE_REPORT.md` §5 measures 0.87–1.75 mean
for the six transitions on A1's committed-set probe; 0.80 here for `theirs→mine`
on the study's own probe set).

**The read-out result does depend on having 52 role dimensions — in magnitude,
not in sign.** Below 52 it degrades in lockstep with accuracy, and at r=16 it
collapses (0.6920): upstream's default cannot see the flip at all at our data
budget. Above 52 it improves. So 52 is not a value the conclusion is balanced on;
it is a point on a monotone curve, and A3 sits on the conservative side of it.
The 64-role probe is the useful bound: with 69,638 parameters, still 71% of the
linear probe's, it reads *s* as unchanged in 1,997 of 2,000 events.

Role overlap is the mechanism, exactly as A1 predicted: from r=64 to r=16 mean
|cos| rises 6.3× and collateral 17×. Note that 64 roles in 64 dimensions
*could* be orthogonal and the fit does not make them so (max |cos| 0.213), so
collateral never reaches zero.

## 4. The absolute-colour ceiling

The occupancy-only ceiling is a property of the labels, not of any probe:
classify empty/occupied perfectly, then always answer the majority colour. On the
12,000 held-out positions the eval label shares are empty 0.4609, black 0.2740,
white 0.2651, so

> ceiling = 0.4609 + 0.2740 = **0.73494**

The measured `acc_abs_L5` is 0.73490 — 0.004 points *below* a strategy that never
looks at colour. **A3's decomposition is confirmed independently, from the labels
alone.** The same construction applied to the relative labels
(empty 0.4609, theirs 0.2909, mine 0.2482) gives a ceiling of 0.75181, and that
is the contrast that makes the number mean something:

| probe (seed 1111) | accuracy | its own occupancy ceiling | over ceiling | colour on occupied | majority colour | over majority |
|---|---|---|---|---|---|---|
| relative | 0.9829 | 0.7518 | **+23.11 pts** | 0.9720 | 0.5396 | **+43.24 pts** |
| absolute | 0.7349 | 0.7349 | **−0.004 pts** | 0.5107 | 0.5083 | **+0.24 pts** |

Same architecture, same activations, same split, same wd; only the labels differ.
The relative probe clears its ceiling by 23 points and the absolute probe by
zero.

Across the six seeds, `acc_abs_L5` − ceiling is **−0.004 to +0.167 points**
(mean +0.078), and colour-on-occupied beats always-black by **0.24 to 0.55
points** (mean 0.38). A3 quotes 0.24, the low end. The finding is not
seed-dependent: no seed produces an absolute probe that reads colour.

Across ranks, capacity buys relative colour and does not buy absolute colour:

| `d_r` | colour on occupied, relative | absolute | absolute over always-black (pts) |
|---|---|---|---|
| 16 | 0.7631 | 0.4496 | −5.87 |
| 32 | 0.9099 | 0.4913 | −1.70 |
| 52 | 0.9720 | 0.5107 | +0.24 |
| 64 | 0.9849 | 0.5138 | **+0.55** |

Quadrupling the role dimension moves relative colour accuracy 22.2 points and
absolute colour 0.55 points past the always-black baseline. Across weight decays
the absolute probe never exceeds +0.29 points over always-black and at wd=4 it
falls *below* it (−0.41). At every one of the 15 settings, the absolute-colour
probe is within a fraction of a point of a strategy that never looks at colour.

This is the finding that answers the study's original question, and it survives
every axis resampled here: **the model's basis is mover-relative, and absolute
colour is something we compute from it.**

## 5. Peak layer

`peak_layer` is the argmax over blocks 0–7 of `p_s_absolute_correct`, one probe
fit per (seed, layer) — 48 fits.

| layer | 1111 | 7 | 2026 | 4242 | 31337 | 90210 |
|---|---|---|---|---|---|---|
| 3 | 0.9820 | 0.9795 | 0.9815 | 0.9815 | 0.9820 | 0.9815 |
| **4** | **0.9920** | **0.9910** | **0.9910** | **0.9935** | **0.9935** | **0.9925** |
| 5 | 0.9880 | 0.9880 | 0.9895 | 0.9890 | 0.9885 | 0.9890 |
| 6 | 0.9705 | 0.9675 | 0.9700 | 0.9665 | 0.9700 | 0.9675 |

**`peak_layer` = 4 at 6/6 seeds**, with a margin over layer 5 of 0.15–0.50 points
(mean 0.36). The argmax of `p_s_unchanged_rel`, of `p_unflipped_swap_rel` and of
`acc_rel` is also 4 at all six seeds. The number does not move between 4 and 5,
so the study can quote a single peak layer for this probe.

That is all it establishes. A3's caveat stands untouched: the linear probe peaks
at block 5, and this sweep contains no linear reference, so a stable TPR peak at
4 is a stable fact about the TPR. Reseeding cannot separate the probe from the
model — it shows only that the confound is not seed noise.

## What this sweep does not cover

1. **One activation set.** Every fit reuses the same 20,000-position probe set and
   the same 2,000 events, so nothing here resamples the *data*. A3's event
   bootstrap does that, and it remains the wider interval for the two read-out
   proportions.
2. **`d_f` = 2 throughout.** The filler dimension is the fourth free choice and is
   not swept; at `d_f`=2 there is no dimensionality reduction in filler space,
   which is why A3 can draw F1 directly, so changing it changes more than a
   hyperparameter.
3. **No linear reference**, so §5's confound is reported, not resolved.
4. **The wd grid is coarse at the top.** wd=4.0 is past the flat region A1 mapped
   (1.5–2.5) and loses 0.65 points of accuracy; the interesting comparison —
   whether some wd between 2 and 3 reads *s* higher still — is not run, and would
   be selecting on the eval set if it were.
5. **Collateral is measured for one transition** (`theirs→mine`) at one scale, the
   one relevant to a bracketing flip. A1 covers all six.
