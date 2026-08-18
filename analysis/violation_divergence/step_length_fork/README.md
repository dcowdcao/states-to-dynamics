# Step length at the fork

The fork family draws **where** a branch goes. This draws **how far it moves per
step** while it gets there. For each branch *b* of one source game, at the
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

```
python analysis/violation_divergence/step_length_fork/make_figures.py \
    [--game 1235] [--camera-ready]
```

| file | what it is |
|---|---|
| `figures/game_1235_step_length_fork.{png,pdf}` | the nine-space grid, spliced branches, layer 5 |
| `results/violation_divergence/step_length_fork.csv` | one row per (space, series, position): the **absolute** step length and the ratio, for the reference, the three rules, all six member categories and the control |

Spliced branches only (`$SPADE_DIVERGENCE_DIR/branches`): every branch replays
the source game's `tokens[t0+1:]`, so a difference at *r* > 0 descends from one
token. The rollout set forks the suffix too, which is a different quantity and a
different figure.

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

## The numbers

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

## Reading it honestly

- **One game, no interval.** n = 1, and a selected game. Every number above is
  a single measurement with no CI; the population twin is a separate change.
  Nothing here licenses "an illegal move lengthens the step" as a claim about
  OthelloGPT — it licenses "on this game it does, at the fork, for R3".
- **The control is the scale, not the line at 1.** The reference is 1 by
  construction, so a curve above 1 means nothing on its own. The question the
  panel answers is whether a rule leaves the *control's* own excursion band,
  and the only rule that does so in all nine spaces is R3, at *r* ≤ +4.
- **The rule grouping is weak.** Within ÷ between-rule spread is 0.70–0.90 on
  this quantity — better than the 0.885 the displacement field gives on the same
  game, which is one game and not evidence that step length groups the rules
  better. R2's members disagree in sign at the fork. Read the faint traces.
- **R3 is one branch.** `R3 turn (mean of 1)` is `opponent_only`, a single
  branch of a single game. It is the largest curve in every panel and it is also
  the least averaged one.
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
  ratio; its R3 reaches 1.74 and its control only 1.11. Read it against its own
  control, as with every cell.
