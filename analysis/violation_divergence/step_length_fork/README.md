# Step length at the fork

The fork family draws **where** a branch goes. This draws **how far it moves per
step** while it gets there. For each branch *b* of a source game, at the
study's sampled positions,

```
s_b(t) = ‖ z_b(t) − z_b(t − 2) ‖
```

and the panels draw the paired ratio **s_b(t) ÷ s_ref(t)** — the branch's step
length against the reference branch's at the same position of the same game.
Nine panels, one per representation space, event-locked to the branch at *t₀*.

Nothing else in the study draws this. `../rule_curvature` draws the turning
angle θ and only θ; the R²¹ figures (`../fork_curvature_space`,
`../r21_shared_basis`) carry step length as 7 of 21 components inside a
descriptor and never on its own axis.

Two figures, one quantity and one panel grid: **one selected game**, which is an
illustration of the construction, and its **100-game population twin**, which is
the one that carries an interval. The reduction and the interval are the whole
difference between them, so they are one script.

```
python analysis/violation_divergence/step_length_fork/measure.py [--n_games 100] \
    [--variant spliced|rollout]
python analysis/violation_divergence/step_length_fork/make_figures.py \
    [--game 1235 | --n_games 100] [--variant spliced|rollout] [--camera-ready]
```

| file | what it is |
|---|---|
| `figures/game_1235_step_length_fork.{png,pdf}` | the nine-space grid, one game, spliced branches, layer 5 |
| `figures/n100_step_length_fork.{png,pdf}` | the same grid over 100 source games, with a bootstrap CI over games |
| `$SPADE_DIVERGENCE_DIR/step_length_fork/n100.npz` | `measure.py`'s output: per space, game, branch and position, the absolute step length and the ratio |
| `results/violation_divergence/step_length_fork.csv` | one row per (space, series, position): the **absolute** step length and the ratio, for the reference, the three rules, all six member categories and the control |
| `results/violation_divergence/step_length_fork_n100.csv` | the same rows over the population: the mean with its bootstrap CI, the median, and the paired contrast against the legal control |
| `figures/n100_step_length_fork_rollout.{png,pdf}` | the population grid over the **rollout** branch set, same 100 games (below) |
| `$SPADE_DIVERGENCE_DIR/rollout/step_length_fork/n100.npz` | its sweep output, `n_rollouts` = 4; it never renders over the spliced npz |
| `results/violation_divergence/step_length_fork_n100_rollout.csv` | its table, columns identical to the spliced one |
| `figures/n100_step_length_fork_l0_matched.{png,pdf}` | the population grid with the eight SAE cells seated at one shared $L_0$ (below) |
| `$SPADE_DIVERGENCE_DIR/step_length_fork/n100_l0_matched.npz` | its sweep output; it never renders over the canonical npz |
| `results/violation_divergence/step_length_fork_n100_l0_matched.csv` | its table, columns identical to the canonical one |

The default is the spliced set (`$SPADE_DIVERGENCE_DIR/branches`): every branch
replays the source game's `tokens[t0+1:]`, so a difference at *r* > 0 descends
from one token. The rollout set forks the suffix too — a different quantity, and
therefore its own figure and its own files, never mixed into these
(**The rollout twin**, below).

## Caption

**At the branch the model's next step changes length — longer for a turn
violation, shorter for an occupied-square one — for two or three steps, and a
second legal move at the same position does most of it too.** One panel per representation
space, raw layer 5 first so the eye calibrates on it; layer-5 `resid_post`,
game 1235, *t₀* = 20. y is each branch's step length divided by the **reference
branch's step length at the same position in the same game**, so the reference
is exactly the black line at 1 and the panels of nine spaces whose absolute step
lengths differ ~580× can share one axis. x is move index *t*, sampled every 2
plies at *t₀*'s parity, so one step spans 4 plies; the dashed vertical rule is
the branch position *t₀* and the shaded band left of it is the shared prefix,
where every branch is token-identical and the ratio is **exactly 1 by
construction** — asserted in all nine spaces on every render, not assumed, and
drawn rather than cut away. One curve per constraint rule, the mean over its
member categories (n = 2, 3, 1) with the members drawn faint behind it;
`control_legal` — a second legal move at *t₀* — is the empirical null, in grey.
**n = 1 game: no interval over games, and none is drawn.** The within ÷
between-rule spread of the drawn ratio over *t* ≥ *t₀* is 0.757 in raw layer 5
(0.70–0.90 across the nine spaces), so the three curves are three labels more
than three findings and the faint member traces carry the evidence. Each panel
prints its largest rule-mean excursion with its offset.

**Game 1235 is selected, not typical.** It is the argmax, over all 1550
branch-complete games, of the mutual separation of its six illegal displacement
trajectories, scored in raw layer 5 at full width (`../game_selection`).
Population median 0.705; this game 1.080. This is an illustration of the
construction, not a claim-supporting figure, and no number here has a CI. The
study's claim-supporting numbers are in `../README.md`, over 13,531 branches.

## The sampling convention, which is load-bearing

Positions come from `fork_plane.positions` at `STRIDE = 2` — the fork family's
convention — so **one step spans 4 plies** inside one mover's frame. The board
code at layer 5 is mover-relative, so the frame inverts every ply and a per-ply
path is a near-reversal sawtooth; sampling every second ply removes it.

`../rule_curvature` samples **every ply** instead. The two are different
quantities, not two estimates of one. On the same layer-5 corpus the legal class
steps **29.2 per ply against 18.9 at stride 2**, and its turning angle reads
142.0° against 107.9° — and every class contrast changes sign between the two
(`../theta_comparability`). **The two conventions must never share an axis.**
The figure carries this sentence; anything read off it is a stride-2 quantity.

A step is attributed to the position it *ends* on, so a step is branch-free
exactly when that position precedes *t₀*. At *t₀* = 20 the drawn offsets are
*r* = −6, −4, …, +24 and the first three are branch-free.

## Why the ratio, and where the absolute lengths went

The nine spaces are nine unit systems. The reference branch's own mean step
length over *t* ≥ *t₀* runs from **18.6 in raw layer 5 to 0.032 in SpaDE**, a
factor of ~580, and inside one space it swings ~2× over the window with no
monotone trend (raw layer 5: 13.1 to 24.7). A grid of raw step lengths would be
nine incommensurable panels, and a reader would compare them anyway.

Dividing by the reference branch's step length **at the same position in the
same game** fixes both. It is a paired, within-game contrast: it cancels the
per-position swing the branches share, the reference is 1 by construction, and
the legal control's own ratio is the null the illegal ones are read against.
This is the same form `../fork_and_board`'s panel (c) and
`../divergence_by_space` argue in.

Nothing is lost to the division: `step_length_fork.csv` carries the absolute
`step_length` beside every `ratio`, and the reference's per-space scale is the
first column of the table below. The absolute grid was drawn from that CSV
before this one was committed. It shows one thing the ratio removes on purpose:
**every branch, in every space, tracks the same per-position swing** — a trough
near *t* = 24 and a peak near *t* = 38, ~2× deep in raw layer 5 — which is a
property of the game and the position, not of the branch. The fork's effect
rides on top of it and is smaller than it. Nine panels of that cannot share an
axis, and on separate axes the swing is what a reader sees.

## The numbers, on the one game

Game 1235, layer 5, spliced branches, printed by `make_figures.py` on every
render. `s_ref` is the reference branch's own absolute step length over the
drawn window (the divisor). Rule and control cells are the **mean ratio over
the 13 offsets *r* = 0, 2, …, 24**, with **the value at *r* = 0** in brackets.

| space | s_ref range | within ÷ between | R1 placement (2) | R2 bracketing (3) | R3 turn (1) | control (null) |
|---|---|---|---|---|---|---|
| raw layer 5 | 13.1–24.7 | 0.757 | 1.012 (0.86) | 1.060 (1.18) | 1.124 (1.53) | 1.001 (1.23) |
| PAnneal | 6.33–11.8 | 0.710 | 1.014 (0.84) | 1.066 (1.18) | 1.142 (1.57) | 0.995 (1.22) |
| Standard | 3.69–6.85 | 0.704 | 1.000 (0.86) | 1.065 (1.15) | 1.144 (1.55) | 1.000 (1.20) |
| GatedSAE | 5.73–11.0 | 0.813 | 1.011 (0.80) | 1.038 (1.00) | 1.112 (1.25) | 1.004 (1.15) |
| TopK | 6.23–11.5 | 0.778 | 1.016 (0.84) | 1.041 (1.04) | 1.104 (1.32) | 0.996 (1.17) |
| GatedAnneal | 7.03–13.6 | 0.769 | 1.009 (0.82) | 1.030 (1.04) | 1.096 (1.36) | 1.005 (1.31) |
| BatchTopK | 6.45–12.0 | 0.770 | 1.008 (0.83) | 1.023 (1.07) | 1.087 (1.40) | 0.989 (1.18) |
| CosineSpaDE | 0.107–0.178 | 0.899 | 0.948 (0.83) | 1.031 (1.32) | 1.124 (1.74) | 0.976 (1.11) |
| SpaDE | 0.0253–0.0398 | 0.758 | 1.009 (0.91) | 1.050 (1.21) | 1.086 (1.51) | 1.005 (1.19) |

Four things it says, in the order they matter:

1. **The step at the fork is different, and the direction splits the rules.** At
   *r* = 0 the turn violation steps **1.25–1.74×** the reference's, and the two
   occupied-square violations step **0.80–0.91×** — *shorter*. R1 is below 1 at
   *r* = 0 in all nine spaces and R3 above 1 in all nine. R2's three members
   straddle: `empty_own_only` 1.39 and `empty_detached` 1.21 against
   `empty_no_bracket` 0.95 in raw layer 5, which is most of why R2's mean sits
   between them.
2. **A second legal move does it too.** The control steps **1.11–1.31×** at
   *r* = 0. Playing a different token lengthens the step whether or not the
   token is legal; at *r* = 0 R3 is the only rule above the control, by
   **1.04–1.57×** (only 1.04 in GatedAnneal, 1.09 in GatedSAE, 1.24 in raw
   layer 5), R1 is below it in all nine spaces and R2 in seven of the nine.
3. **It decays fast.** In raw layer 5 R3 runs 1.53, 1.33, 1.35 at *r* = 0, 2, 4;
   from *r* = +6 on its largest excursion is 1.16, at *r* = +8. Averaged over
   the whole window every rule lands within 15% of 1 and R1 within 6%.
4. **Past *r* = +4 there is nothing above the null's own noise.** The control's
   largest excursion from 1 over *r* ≥ 0 is ±0.18 to ±0.31 depending on the
   space, and past *r* = +4 no rule mean leaves that band in any panel — except
   *r* = +8 in Standard and CosineSpaDE, one offset in two of nine. Inside it,
   **R3 leaves the band in all nine spaces**, R2 in six and R1 in three. Read
   the first three offsets; treat the rest as the null's wobble drawn twice.

## The population twin

The first **100** of the **1696** branch-complete games forking at *t₀* = 20,
ascending by game id (`branches.games_at_t0`) — the population every *n*-game
sweep in this study draws, so this figure and `../fork_population` are over the
same games. No game-level criterion enters it. Game 1235 is in it as one ordinary
member, 42nd by id.

Same quantity, same nine panels, same axis. What is different:

- Each curve is the **mean over games of that game's own ratio**, the rule mean
  taken over its member categories *inside* the game. The band is a **95%
  percentile bootstrap CI over source games**, 2000 resamples, seed 0, resampling
  **games and not branches** — a game's nine branches share a prefix and the
  reference they are divided by.
- **No faint member traces.** A member at this *n* is itself a mean over 100
  games, so drawn faint it would read as game-to-game spread, which is exactly
  what the band already carries.
- **Both legal controls are drawn.** In this figure the control is not a divisor
  (the reference is), so `control_legal` and `control_legal_2` are two draws of
  one null — a different legal token at *t₀*, measured exactly as the illegal
  branches are. Drawing both is the honest empirical null: the gap between the
  two grey curves is the size of a difference that means nothing, and it is
  never resolved (+0.008 to +0.023 at *r* = 0, every interval straddling 0).
  Averaging them into one null curve was rejected — a null averaged over *k*
  candidates has ~1/√*k* the spread of the signal it is read against
  (FIGURE_GUIDELINES §3).
- Each panel prints the **paired contrast** of its largest rule against
  `control_legal`, at the fixed offset *r* = 0, on the same games. Two point
  estimates side by side is what this study already had to correct once
  (`../fork_population`).

Raw layer 5, *r* = 0 and the window mean over *r* ≥ 0, with the paired contrast
against the legal control:

| series | at *r* = 0 | median | mean over *r* ≥ 0 | (series − control) at *r* = 0 |
|---|---|---|---|---|
| R1 placement | **0.936** [0.910, 0.961] | 0.916 | 0.986 [0.981, 0.992] | **−0.060** [−0.081, −0.038] |
| R2 bracketing | **1.099** [1.068, 1.131] | 1.067 | 1.010 [1.004, 1.015] | **+0.104** [+0.077, +0.132] |
| R3 turn | **1.076** [1.044, 1.108] | 1.035 | 1.008 [1.001, 1.015] | **+0.080** [+0.051, +0.111] |
| `control_legal` (null) | 0.995 [0.967, 1.024] | 0.987 | 0.991 [0.984, 0.997] | — (the divisor) |
| `control_legal_2` (null) | 1.016 [0.986, 1.046] | 0.991 | 0.990 [0.984, 0.997] | +0.021 [−0.004, +0.047] |

### What survives, and what does not

**1. A second legal move does *not* lengthen the step.** This is the largest
correction. On game 1235 the control steps **1.11–1.31×** the reference at
*r* = 0; over 100 games it steps **0.989–1.007×**, and its interval straddles 1
in all nine spaces. The one-game figure's headline reading — *any* different
token lengthens the step — was that game.

**2. R1 shortens the step, and that is what replicates.** At *r* = 0, R1 is
**0.890–0.957** with every interval clear of 1 (upper bounds ≤ 0.984) in all
nine spaces, and paired against the control it is resolved negative in all nine,
**−0.032 to −0.117**. The direction the one game gave (0.80–0.91) survives; the
magnitude roughly halves. It is also the same rule, and the same direction, that
`../fork_population` found to be the only resolved contrast on the displacement.

**3. R3's 1.53× does not survive, and neither does R3 > R2 > R1.** R3 at *r* = 0
is **1.076** in raw layer 5 against 1.526 on game 1235, and it sits *below* 1 in
the four gated/TopK spaces (0.971–0.996, every interval straddling 1). Paired
against the control it is resolved positive in five spaces (+0.072 to +0.304 in
raw, PAnneal, Standard, SpaDE, CosineSpaDE), unresolved in three, and resolved
*negative* in GatedAnneal (−0.030 [−0.057, −0.000]); R2 is resolved negative in
GatedSAE (−0.034 [−0.063, −0.003]). The ordering fails differently: **R2 − R3**
at *r* = 0 is **+0.023** [+0.001, +0.047] in raw layer 5 — R2, not R3, is the
larger excursion — resolved in three of nine spaces and unresolved in the rest.
What *is* resolved in all nine is R2, R3 > R1: R3 − R1 = +0.072 to +0.345,
R2 − R1 = +0.057 to +0.414. So the population orders the rules as
**{R2, R3} > control > R1**, with R2 and R3 indistinguishable from each other,
and not as R3 > R2 > R1.

**4. It decays as fast, off a much lower peak, and the window means are within
5% of 1.** raw layer 5 R3 runs **1.076, 1.057, 1.012** at *r* = 0, 2, 4 (n = 1:
1.53, 1.33, 1.35). Window means: R1 0.962–0.999, R2 0.972–1.052, R3 0.976–1.045,
control 0.970–0.999 — every rule within 5% of 1 in every space, and paired
against the control the window-mean gaps are ≤ +0.023 (+0.046 in CosineSpaDE).
**Do not read the one game's "past *r* = +4 nothing leaves the control's band"
against this figure**: the control's own excursion band is now ±0.016–0.075
instead of ±0.18–0.31, because it is a mean over 100 games rather than one
game's noise, and a band that narrows ten-fold is not the same band. Against the
narrow one, 2–3 of the 3 × 10 rule-mean values past *r* = +4 leave it in raw
layer 5, PAnneal, Standard and SpaDE, and none in the other five.

**Game 1235 is not what drives the disagreement.** Its own *r* = 0 values in raw
layer 5 are R1 0.862, R2 1.184, R3 1.526, control 1.232, all far outside the
population's; dropping it from the 100 moves every raw-layer-5 mean by **≤
0.005**. Being one game of 100 is the whole difference.

**The rule grouping is weak here too, and by the same margin.** Within ÷
between-rule spread of the drawn ratio over *t* ≥ *t₀*, per game: raw layer 5
median **0.866** [IQR 0.728–1.021], above 1 in 29% of games; across the nine
spaces the medians run 0.778 (CosineSpaDE) to 0.969 (GatedSAE). The one game's
0.757 was near the good end of that distribution.

**CosineSpaDE is again the loudest cell, and again for its own reason.** R2
1.366 [1.318, 1.417] and R3 1.298 [1.249, 1.346] against a null of 0.994, the
largest paired contrasts anywhere (+0.372 and +0.304). It is also the one cell
where the effect is nearly per-game universal — R2 above 1 in **98%** of the 100
games and R3 in 91%, against 72% and 62% in raw layer 5. Its absolute steps are
~0.13 with a narrow spread, so a small absolute change is a large ratio; read it
against its own control, as with every cell.

## Caption for the population figure

**A move that plays onto an occupied square takes a measurably *shorter* next
step than a different legal move does; a bracketing or turn violation takes a
slightly longer one in five of nine spaces; and a legal alternative itself does
nothing.** Nine panels, one per representation space, raw layer 5 first; layer-5
`resid_post`, 100 source games all forking at *t₀* = 20. y is each branch's step
length divided by **that game's own reference branch's step length at the same
position**, so the reference is the black line at 1 and the nine spaces share one
axis. x is move index *t*, sampled every 2 plies at *t₀*'s parity, so one step
spans 4 plies; the dashed rule is *t₀* and the shaded band left of it is the
shared prefix, where the ratio is 1 by construction (asserted in all nine spaces,
per game, on every sweep). Each curve is the mean over the 100 games of that
game's own ratio, the rule mean taken over its 2, 3 or 1 member categories inside
the game; **bands are 95% percentile bootstrap CIs of that mean over source
games** (2000 resamples, seed 0, resampling games and not branches). The two grey
curves are the two legal controls — two draws of one empirical null, since
neither is a divisor here. Each panel prints the paired contrast of its largest
rule against `control_legal` at *r* = 0, taken inside each game. Within ÷
between-rule spread over *t* ≥ *t₀*, per game, raw layer 5: median 0.866 [IQR
0.728–1.021]. The population is the first 100 of the 1696 branch-complete games
forking at *t₀* = 20, ascending by game id, with no game-level criterion.

## The L0-matched twin

`figures/n100_step_length_fork_l0_matched.{png,pdf}` ·
`results/violation_divergence/step_length_fork_n100_l0_matched.csv`

```
python analysis/violation_divergence/step_length_fork/measure.py      --n_games 100 --l0-matched  # ~3 min
python analysis/violation_divergence/step_length_fork/make_figures.py --n_games 100 --l0-matched
```

The eight SAE cells above are each type at its own default sparsity — TopK and BatchTopK
at k = 80, Standard at L1 0.059 — so "R2/R3 do nothing in the gated/TopK spaces" is a
statement about four dictionaries that are also four times sparser than the rest, and the
architecture cannot be told from the L0. `--l0-matched` seats them at
`results/l0_operating_point/picks.csv` instead: one checkpoint per type, the nearest-L0
member of its own grid to the shared target **L0 = 219.2**. Same 100 games, same axis,
same panel order, same styling; each cell's title carries its own L0. Raw layer 5 is not an
SAE and is the same cell in both — its 192 CSV rows are identical to the canonical file's.

**R1 shortens the step, and that is what survives.** At *r* = 0 R1 is 0.900–0.936 with
every interval clear of 1 in all nine spaces, and paired against `control_legal` it is
resolved negative in all nine, **−0.047 to −0.093** (canonically −0.032 to −0.117). Same
rule, same sign, same order of magnitude. The control still does nothing: 0.979–1.000 at
*r* = 0, every interval straddling 1.

**The result that does not survive is "R2 and R3 are absent in the gated/TopK spaces", and
it reverses.** Paired against the legal control at *r* = 0, canonical → L0-matched:

| space | R2 − control | R3 − control |
|---|---|---|
| raw layer 5 | +0.104 → +0.104 | +0.080 → +0.080 |
| PAnneal | +0.094 → +0.065 | +0.072 → +0.047 |
| Standard | +0.103 → **+0.016 [−0.008, +0.042]** | +0.083 → **+0.016 [−0.009, +0.041]** |
| GatedSAE | −0.034 [−0.063, −0.003] → −0.017 [−0.044, +0.012] | −0.019 → −0.010 |
| TopK | −0.027 [−0.055, +0.002] → **+0.053 [+0.027, +0.079]** | −0.012 → **+0.051 [+0.023, +0.079]** |
| GatedAnneal | −0.027 [−0.056, +0.002] → **+0.045 [+0.021, +0.069]** | **−0.030 [−0.057, −0.000]** → **+0.040 [+0.015, +0.065]** |
| BatchTopK | −0.017 [−0.046, +0.012] → **+0.105 [+0.076, +0.136]** | −0.005 → **+0.076 [+0.044, +0.109]** |
| CosineSpaDE | +0.372 → +0.430 | +0.304 → +0.305 |
| SpaDE | +0.124 → +0.053 | +0.097 → +0.036 |

TopK, GatedAnneal and BatchTopK go from unresolved (GatedAnneal's R3 from resolved
**negative**) to resolved positive on both rules, at 0.7–1.0× raw layer 5's gap. Only
GatedSAE stays without a resolved excess, and its point estimate halves toward zero. In the
other direction Standard's +0.103/+0.083 collapse to an unresolved +0.016 and SpaDE's halve.
**At their own default sparsity the four gated/TopK dictionaries were too sparse to show the
fork's step-length change; at L0 ≈ 220 they show it about as well as the raw residual
stream does.** The canonical figure's "resolved positive in five spaces, negative in
GatedAnneal" is an L0 ordering, not an architecture ordering.

CosineSpaDE is again the loudest cell (+0.430 on R2) and it is again the pick furthest from
the target on the low side (L0 184); read it against its own control, as always.

**The rule grouping stays weak.** Within ÷ between-rule spread of the drawn ratio over
*t* ≥ *t₀*, per game, raw layer 5: median 0.866 [IQR 0.728–1.021] — the same cell, so the
same number — and 0.848 (CosineSpaDE) to 0.965 (GatedSAE) across the eight dictionaries,
against the canonical 0.778–0.969.

**One caveat the figure carries in its note block.** Karvonen's five picks were trained on
the raw residual stream and ours on scale·(x − mean); each is fed the activations it was
trained on, so the two families' latents are not on one input scale. The ratio is a
within-game, within-cell contrast, so it is unaffected — but the absolute `step_length`
column in the CSV is not comparable between the two files for those five cells.

## The rollout twin

Same 100 games, same nine panels, same axis, one thing changed: the branch set.
In `$SPADE_DIVERGENCE_DIR/rollout` every branch — **the reference included** —
keeps `tokens[:t₀]` and its own branch token and then samples the rest of the
game from OthelloGPT at temperature 1, K = 4 times per branch
(`spade.violation_divergence.rollout_branches`). Nothing replays the source
game's suffix, so the reference is no longer the one in-distribution sequence in
the set. The price is single-token attribution: two branches now differ at every
position after *t₀*, not at *t₀* alone, so a difference at *r* > 0 is the branch
move **plus everything the model then chose to do about it**.

`branches.games_at_t0` returns the same 100 ids on both trees (10, 35, 85, 110,
…), so the two figures are over the same games and are read side by side.

### The reduction, and the order it is taken in

A branch is 4 rows. Every row is encoded and turned into its own step length, and
the mean over the four is taken **last**: the SAE encode is nonlinear, so a mean
taken before it is the code of an average activation and not the average of four
codes.

The ratio is formed **within a rollout index** — s_b^k(t) ÷ s_ref^k(t), then the
mean over k — because a branch's k-th rollout and the reference's k-th are driven
by the same uniform stream, so the pairing holds the sampled continuation fixed
and lets only the branch token vary. Common random numbers, the same argument
that makes this figure bootstrap games rather than branches. Ratio-of-means
(average the four lengths, divide once) was rejected on three counts: E[s_b/s_ref]
≠ E[s_b]/E[s_ref], so it is a different quantity, not a rounder version of this
one; it throws away the pairing the seeds were built to provide; and it cannot
hold the shared prefix at exactly 1 branch by branch, which is the check the
sweep asserts in all nine spaces on every run.

### *r* ≤ 0 is the spliced figure exactly, by construction

The step drawn at *r* = 0 is ‖z(t₀) − z(t₀−2)‖, and both positions are causally
upstream of the resampling — every rollout of every branch shares
`tokens[:t₀+1]`. The two npz files confirm it: `step` and `ratio` at every drawn
*r* ≤ 0 are **bitwise equal** across the two trees, in all nine spaces, for all
100 games. The fork-position numbers below are therefore not a replication of the
spliced result, they are the same measurement; everything the rollout set can add
is at *r* ≥ +2.

### Raw layer 5

| series | at *r* = 0 | mean over *r* ≥ 0 | (series − control) at *r* = 0 | (series − control) over *r* ≥ 0 |
|---|---|---|---|---|
| R1 placement | **0.936** [0.910, 0.961] | 1.008 [1.002, 1.014] | **−0.060** [−0.081, −0.038] | −0.004 [−0.010, +0.002] |
| R2 bracketing | **1.099** [1.068, 1.131] | 1.036 [1.031, 1.042] | **+0.104** [+0.077, +0.132] | **+0.024** [+0.019, +0.030] |
| R3 turn | **1.076** [1.044, 1.108] | 1.029 [1.021, 1.036] | **+0.080** [+0.051, +0.111] | **+0.017** [+0.010, +0.024] |
| `control_legal` (null) | 0.995 [0.967, 1.024] | 1.012 [1.006, 1.018] | — (the divisor) | — |
| `control_legal_2` (null) | 1.016 [0.986, 1.046] | 1.016 [1.010, 1.023] | +0.021 [−0.004, +0.047] | +0.004 [−0.002, +0.011] |

1. **R1 still steps shorter than a legal alternative, and at *r* = 0 that is
   arithmetic rather than evidence.** 0.890–0.957 across the nine spaces with
   every interval clear of 1, paired against the control **−0.032 to −0.117** and
   resolved negative in all nine — the spliced numbers to the last digit, because
   the measurement is the same one. Past the fork the story changes: over
   *r* ≥ 0 the paired R1 − control gap is **−0.004** [−0.010, +0.002] in raw
   layer 5, unresolved there and in PAnneal, Standard, CosineSpaDE and SpaDE, and
   resolved negative only in the four gated/TopK spaces (−0.011 to −0.013). The
   shortened step is a fork-position effect, and resampling the suffix does not
   extend it.
2. **Once the suffixes diverge every curve lifts ~1%, both nulls included, and
   that is the denominator.** `control_legal` over *r* ≥ 0 goes from 0.991
   [0.984, 0.997] spliced to **1.012** [1.006, 1.018] here, above 1 with the
   interval clear of it in all nine spaces (1.008–1.018). It is not a branch
   effect: at *r* = +12 in raw layer 5 the across-game correlation between the
   control's step length and the reference's falls from **0.73** spliced to
   **0.09** here, and the mean of the ratio (1.009) parts company with the ratio
   of the means (0.997) — with the denominator no longer moving with the
   numerator, the convexity of 1/s pushes E[s_b/s_ref] above E[s_b]/E[s_ref].
   Read the paired contrast, which subtracts a lift both nulls carry; the level
   of a curve above 1 past the fork is not a finding on this tree.
3. **R2's edge over the null survives further out than it does spliced.**
   Window-mean paired gaps in raw layer 5: R2 **+0.024** [+0.019, +0.030] against
   +0.019 [+0.013, +0.024] spliced, R3 +0.017 [+0.010, +0.024] against +0.018
   [+0.011, +0.024]. Per offset, R2 − control is **+0.023** [+0.008, +0.036] at
   *r* = +8, where the spliced set gives an unresolved +0.003 [−0.012, +0.017],
   +0.015 [+0.001, +0.027] at +10 and +0.020 [+0.006, +0.033] at +12, and is
   unresolved from *r* = +16 on. Against the control's own excursion band —
   ±0.027 here against ±0.023 spliced, so a comparable band — **8** of the
   3 × 10 rule-mean values past *r* = +4 leave it in raw layer 5 (spliced: 2), 11
   in PAnneal, 13 in CosineSpaDE and 0–3 in the gated/TopK spaces.
4. **The rule grouping is as weak as it is on the spliced set.** Within ÷
   between-rule spread of the drawn ratio over *t* ≥ *t₀*, per game, raw layer 5:
   median **0.896** [IQR 0.829–1.014], above 1 in 30% of games (spliced 0.866
   [0.728–1.021], 29%); the nine-space medians run 0.842 (CosineSpaDE) to 0.976
   (GatedSAE).

### The window is not truncated at *r* = +8

`../fork_and_board` draws its rollout figures only to `ROLLOUT_RMAX = 8` because
past that its illegal and control δ̂ agree to within 0.5%. That rule does not
transfer here and the window is the spliced twin's, *r* = −6 … +24:

- **δ̂ is cumulative and this is not.** Distance from the reference grows and then
  saturates against a ceiling set by how far two independently sampled
  continuations drift apart, so its late offsets draw a separation that has
  already closed. A step length is per-position: it returns towards 1 and stays
  there, which is a reading rather than an artifact, and the two nulls are drawn
  through the same offsets so a reader can see what "nothing" looks like at each.
- **There is resolved signal past +8** (point 3): cutting at +8 would drop
  R2 − control at *r* = +10 and +12, both clear of 0.
- **Comparability is the point of the figure.** The spliced twin is the thing
  this is read against; a shorter x-axis would make part of the visible
  difference an axis difference.

### Caption

**Resampling every branch's continuation leaves the fork-position result exactly
where it was — it is the same measurement — and adds two ply of resolved
separation after it, on top of a ~1% lift that both nulls carry.** Nine panels,
one per representation space, raw layer 5 first; layer-5 `resid_post`, 100 source
games all forking at *t₀* = 20, the rollout branch set (K = 4 sampled
continuations per branch, temperature 1). y is each branch's step length divided
by that game's own reference branch's at the same position, formed within a
rollout index and averaged over the four afterwards; x is move index *t*, sampled
every 2 plies at *t₀*'s parity. The dashed rule is *t₀* and the shaded band left
of it is the shared prefix, where the ratio is 1 by construction — asserted in
all nine spaces, per game, on every sweep. Each curve is the mean over the 100
games of that game's own ratio, the rule mean taken over its 2, 3 or 1 member
categories inside the game; **bands are 95% percentile bootstrap CIs of that mean
over source games** (2000 resamples, seed 0, resampling games and not branches).
The two grey curves are the two legal controls, two draws of one empirical null.
Each panel prints the paired contrast of its largest rule against `control_legal`
at *r* = 0, taken inside each game. Everything at *r* ≤ 0 is bitwise identical to
the spliced twin, since the step ending at *t₀* precedes the resampling.

## Reading it honestly

- **One game, no interval — and the population disagrees with it.** Every number
  in "The numbers, on the one game" is a single measurement with no CI, on a
  selected game. Nothing there licenses "an illegal move lengthens the step" as a
  claim about OthelloGPT — it licenses "on this game it does, at the fork, for
  R3", and over 100 games it does not: R3's 1.53 becomes 1.08, the control's
  1.23 becomes 1.00, and R2 overtakes R3. Read the population twin for anything
  claim-shaped; read the one game for what the construction is.
- **The control is the scale, not the line at 1.** The reference is 1 by
  construction, so a curve above 1 means nothing on its own. On the one game the
  question is whether a rule leaves the *control's* own excursion band, and only
  R3 does so in all nine spaces, at *r* ≤ +4. On the population that band test is
  replaced by the paired contrast against the control on the same games, and the
  rule resolved in all nine spaces is **R1**, downward.
- **R1 is the occupied-square rule, so "shorter step" may be about the branch's
  board delta and not about legality.** Its two members are the only categories that
  place no new disc — `occupied_own` changes nothing, `occupied_opp` recolors (78.6%,
  `../README.md`) — while every R2 and R3 category plays onto an empty square. That
  categorical split is perfectly confounded with R1-vs-rest and nothing in this
  taxonomy separates it. The *graded* part is testable inside R1, and it is there:
  paired per game at *r* = 0, `occupied_own` − `occupied_opp` is **−0.061** [−0.088,
  −0.033] in raw layer 5 and resolved negative in eight of nine spaces (−0.038 to
  −0.077; CosineSpaDE's −0.018 [−0.056, +0.022] is the exception). The category that
  leaves the board untouched steps shortest, which is what the board-delta account
  predicts, so on this quantity that account survives. Separating the categorical half
  needs a branch this taxonomy lacks — an illegal move onto an empty square breaking
  nothing else, or a legal one that recolors. Until then R1's shortened step is a
  direction with an unresolved alternative explanation, and a figure that upgrades it
  to a claim has to rule that alternative out first.
- **The delta grades this quantity and not the other two, and that is a reading, not
  a result.** The same paired within-R1 contrast run in the other two fields over the
  same 100 games gives +0.003 [−0.026, +0.033] on the displacement (`../fork_population`,
  resolved in none of nine) and +0.042 [−0.013, +0.098] in full R²¹
  (`../r21_run_population/board_delta_contrast.py`, resolved in one of nine, and in the
  *opposite* sign). The three do not disagree so much as differ in what they can say.
  The displacement contrast is on a comparable scale with a comparable interval
  (half-width 0.030 against this one's 0.027), so it has the power to see a −0.06
  gradient and does not: on that quantity the account is excluded, not merely
  unsupported. The R²¹ contrast is not — an effect near 0.03 against a half-width near
  0.05 cannot be resolved either way, and closing it wants roughly 3x the games, which
  the 1696-game *t*₀ = 20 pool would afford. What is left is that a step length at
  *r* = 0 is close to a direct read of how much the residual stream had to change
  across the branch token, which is where a board-delta account should bite; the two
  shape-flavoured fields are measuring something the delta does not set. That is a
  story the numbers permit, not one they establish.
- **The rule grouping is weak.** Within ÷ between-rule spread is 0.70–0.90 on
  this quantity on game 1235 — better than the 0.885 the displacement field gives
  on the same game, which is one game and not evidence that step length groups
  the rules better. Over the population the per-game medians run 0.778–0.969 and
  the statistic is above 1 in 19–43% of games, so the grouping is weak there too.
  R2's members disagree in sign at the fork on the one game; read the faint
  traces.
- **R3 is one category.** `R3 turn (mean of 1)` is `opponent_only` — one branch
  of one game on the single-game figure, one branch per game on the population
  one. Either way it is the least averaged of the three, and on the one game it
  is also the largest curve in every panel, which the population does not
  reproduce.
- **The nine panels share an axis because the quantity is a ratio, and for no
  other reason.** A tall curve in a cell says that dictionary's code moved
  further relative to its own reference, not that the dictionary separates
  violations better. Nothing here ranks the nine.
- **The stride is not a detail.** Every value is a stride-2 quantity. The
  per-ply convention gives a different level (29.2 vs 18.9 on the legal corpus)
  and flips class contrasts; a number from here and a number from
  `../rule_curvature` cannot be compared, subtracted or plotted together.
- **CosineSpaDE is the loudest panel and the least trustworthy.** Its absolute
  steps are ~0.13 with a narrow spread, so a small absolute change is a large
  ratio; on the one game its R3 reaches 1.74 against a control of 1.11, and over
  the population R2 reaches 1.37 against a control of 0.99. It is also the one
  cell whose effect is nearly per-game universal (R2 above 1 in 98% of the 100
  games). Read it against its own control, as with every cell.
