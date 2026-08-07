# The fork and the board it forks on

One figure for the whole construction. Panel **(b)** is what the branches *are* —
the Othello position at the end of the shared prefix, with each branch move on
the square it plays. Panel **(a)** is where those same branches *go* — one 3-D
fork per representation space, raw layer 5 first. Before this, the two halves
lived in separate figures and a reader had no way to connect a coloured
trajectory to an actual move.

Eight branches are drawn — the played move, one legal control, and one move per
illegal category — and the six illegal categories are grouped into the three
constraint rules they violate, so the figure carries five series.

Panel **(c)** is the number (a) asks the reader to eyeball and does not answer:
how far each rule runs from the reference, once per cell, in the plane that cell
draws and in that cell's full space, as a multiple of the same cell's legal
control. *[(c): the fork as a number](#c-the-fork-as-a-number-and-why-it-is-a-ratio)*
below is what a reader has to know before reading a bar off it.

```
python analysis/violation_divergence/fork_and_board/make_figures.py \
    [--game 1235] [--plane displacement|r21] [--embedding pca|isomap] \
    [--variant spliced|rollout] [--camera-ready]
```

| file | plane (a) is drawn in | branch set |
|---|---|---|
| `game_1235_fork_and_board_displacement_pca` | PCA of `Δ(t) = z_b(t) − z_ref(t)` — the default | spliced |
| `game_1235_fork_and_board_r21_pca` | PCA of `Δφ(t) = φ(z_b(t)) − φ(z_ref(t))`, inside the 21-D ReStraV curvature descriptor | spliced |
| `game_1235_fork_and_board_displacement_pca_rollout` | same displacement plane | rollout |
| `game_1235_fork_and_board_r21_pca_rollout` | same R²¹ plane | rollout |

Both panels draw the same game, which is what makes the pairing mean anything.
The default is **game 1235** (*t0* = 20); `--game` asserts the game it is given
carries all nine branch types rather than silently drawing fewer. Both branch
sets carry it, and both give 1550 branch-complete games.

**`--variant spliced`** (the default) is the study's own branch set: every branch
replays the source game's `tokens[t0+1:]`. **`--variant rollout`** reads
`$SPADE_DIVERGENCE_DIR/rollout/`, where every branch — the reference included —
samples its own continuation from OthelloGPT (K = 4 per branch, temperature 1.0,
seed 42; 17,527 branches, 70,108 rows over the same 2000 games). It fixes a real
defect in the spliced design and buys a new one; the section *[The rollout
variant](#the-rollout-variant-and-why-its-window-stops-at-r--8)* below is what a
reader has to know before looking at either rollout figure.

## Captions

The figures carry no prose. What a reader needs in order not to over-read them is
here, per figure, and this is the caption the figure is published with
(FIGURE_GUIDELINES §2, §5). What stays inside the figure is the apparatus a
reader cannot look up while looking at it: panel titles and letters, per-panel
space names, axis labels, the legend, the board's reading key, and the R1/R2/R3
key naming each rule's member categories.

### `game_1235_fork_and_board_displacement_pca`

**One prefix, eight futures: where each branch of game 1235 goes in nine
representation spaces, and what each branch is on the board.** Panel **(a)**: one
3-D fork per space, raw layer 5 first. The plane embeds the displacement
`Δ(t) = z_b(t) − z_ref(t)` by uncentred PCA and move index *t* is the third axis,
so the reference is exactly the vertical line at the origin and the shared prefix
collapses onto it; positions are sampled every 2 plies at *t0*'s parity; view
fixed at `elev = 18, azim = −58`. One path per rule is the mean of its member
categories' displacements, **taken in the plane, after projection** — PCA here is
an uncentred linear map, so that is exactly the projection of the mean
displacement (with `--embedding isomap` it is not linear, and the path is a mean
of embedded points rather than of displacements). Each rule's members are drawn
faint behind their mean, and every legend entry carries n, including
`R3 turn (mean of 1)`, which is one branch and not an average. **The plane is
refit per space**, so the axes are arb. units and the three ticks per plane axis
are a within-panel scale bar only: **cells are not comparable across the grid**,
in position or in magnitude, only in the *shape* of the fork. Panel **(b)**: the
position at the end of the shared prefix, discs in absolute colour. 10 moves are
legal at *t0*; the played move and one of the other 9 are branched. An outlined
cell is a branch square, its category marker in the corner so a disc underneath
stays visible; the small dashed translucent disc in the opposite corner is **the
disc that branch would place — never a disc on the board**, which matters most on
the two `R1 occupied` squares, where the real disc's colour is the only evidence
for which of the two categories the branch is; the dashed ring marks a disc the
played move flips. Panel **(c)**: per cell and per rule, the mean over the rule's
member categories (n = 2, 3, 1, one dot per member) of that branch's mean
distance from the reference over the 13 drawn offsets *r* = 0, 2, …, 24 —
**as a bar in the plane the cell above draws and as an open marker in that
cell's full space, both divided by the same cell's legal control**. The
reference is 0 and the control 1, both by construction. The division is what
makes nine refit planes readable side by side at all; the raw distances differ
by ~800× across the grid and are never drawn. **Every bar in this figure stands
above its marker, by a factor that is a property of the plane and not of the
branch** — see the section below, which is the section to read before quoting a
number off (c).

**Game 1235 is selected, not typical.** It is the argmax, over all 1550
branch-complete games, of the mutual separation of its six illegal displacement
trajectories, scored in raw layer 5 in the full 512-dimensional space and never
in an SAE space or a 2-D projection (`../game_selection`). Population median
0.705; this game 1.080. It is the clearest available illustration of the
construction, not a typical one — see the section below for what that costs.

**n = 1, and no error bars over games.** This is an illustration of the design,
not a claim-supporting figure. Every number — the 0.978 ratio, the 0.4985
legal-vs-legal null, the plateau — belongs to `../README.md`, where it carries its
CI over 13,531 branches.

**The rule grouping is weak.** Within-rule ÷ between-rule mean pairwise distance,
on the full 512-d layer-5 displacement trajectories over *t* ≥ *t0*, is **0.885**
on this game against a population **median 0.945 [IQR 0.872–1.020] over all 1550
branch-complete games, above 1 in 30% of them**
(`python -m spade.violation_divergence.rule_groups`). A rule's members travel
barely closer to each other than to another rule's, so the faint member lines and
not the three means carry the evidence.

**One legal control is drawn, as the null.** Its own null — control 2, scored
against control 1 exactly as the illegal branches are — is not drawn here. It is
in `../game_trajectory` and in `../README.md`, which is where the legal-vs-legal
scale lives.

### `game_1235_fork_and_board_r21_pca`

**The same eight branches and the same board, with panel (a)'s plane taken inside
the 21-D ReStraV curvature descriptor instead of the activation space.** Panel
(a) embeds `Δφ(t) = φ(z_b(t)) − φ(z_ref(t))`, where φ is the 7 step lengths, 6
turning angles and their 8 moments over the **W = 8 sampled positions (14 plies)
ending at *t***; the plane is the top-2 uncentred PCs of the column-scaled Δφ,
refit per space. **15 of the study window's 17 offsets carry a descriptor** — the
two earliest windows run off the board — **and 2 of those are branch-free**, where
Δφ is exactly 0, so the prefix still *is* the origin (asserted at render time in
all nine spaces). Panel **(c)** measures this plane and this space: the bar is a
distance in the R²¹ plane the cell draws and the marker is a distance in the
**column-scaled** Δφ, which is the vector `fit_plane` actually projects, so the
two are the same quantity at two dimensionalities and neither is in raw
descriptor units. Everything else — panel (b), the sampling, the view, n = 1,
the selection of game 1235, one control, per-space refits and the
not-comparable-across-cells disclaimer — reads exactly as for the displacement
figure above.

**(c) reads differently here, and the difference is real.** R3 sits 1.69–2.47×
the control in the full R²¹ space against 1.34–1.61× in the displacement space:
a turn violation moves the curvature descriptor further, relative to a legal
alternative, than it moves the activation. R1 is the other way — 0.88–1.32×,
straddling 1, so on this game an occupied-square violation bends the trajectory
no more than a second legal move does. **Do not read that as R²¹ being the
better space.** It is one game, the two fields are scaled differently by
construction (Δφ is column-scaled, Δ is not), and the population sweep that
would license the comparison has not been run for this statistic in either
field.

**φ is nonlinear, so it is applied to each member category on its own and the
rule mean is taken afterwards, in the plane.** Averaging the six trajectories
first and taking step lengths and angles of the average would measure the
curvature of a path no branch takes. The projection *is* linear, so
mean-then-project and project-then-mean agree exactly, and the mean is taken in
the plane exactly as the displacement figure takes it.

**The rule grouping is no better here.** The same within ÷ between statistic on
the column-scaled Δφ, raw layer 5, all 21 components, *t* ≥ *t0*, is **0.902** —
against the 0.885 the displacement figure reports on the same game and window.
**Do not read that pair as R^21 being worse.** Over a population the ordering
reverses and the two are indistinguishable: R^21 median **0.936 [IQR 0.827–0.998],
above 1 in 25%** over 60 games with *t0* ≥ 18
(`python -m spade.violation_divergence.rule_groups --field r21`), against the
displacement median 0.945. The two sweeps are not the same games — the R^21 one
needs *t0* ≥ 18 so that a drawn window is branch-free — so the medians are not a
paired comparison. What replicates is that the grouping is weak in both fields.

### `game_1235_fork_and_board_displacement_pca_rollout`

**The same eight branches of the same game, drawn out of the rollout branch set:
every branch, the reference included, samples its own continuation from
OthelloGPT rather than replaying the source game's.** Panel (a) is the same
uncentred PCA of `Δ(t) = z_b(t) − z_ref(t)`, refit per space, and panel (b) is
unchanged — the board does not know what continued from it. Three things are
different and all three are on the figure.

**Each path is a mean over the branch's K = 4 rollouts, paired on the sample
index.** A rollout's reference is the one drawn from the same uniform stream
(`rollout_branches.py` keys the sampler on `(game_id, k)`), so the difference is
taken within a sample and averaged after — common random numbers, not two
independent draws. The plane projection is linear, so mean-then-project and
project-then-mean agree exactly, and the rule mean over member categories is
taken in the plane as before.

**The window stops at *r* = +8, not +24.** See the section below: past *r* ≈ +8
the illegal and control δ̂ agree to within 0.5% over the whole 2000-game set, so
the extra offsets would draw a fork whose separation is not there. The drawn
window is *r* ∈ [−8, +8], 9 sampled positions.

**The rule grouping is unchanged.** Within ÷ between-rule mean pairwise distance
on the K-averaged full-512-d layer-5 displacement, *t* ≥ *t0*, is **0.887**
against the spliced figure's 0.885 on the same game — one game, so read it as
"the rollout did not change the grouping", not as a measurement.

**(c) averages 5 offsets, not 13** — *r* = 0, 2, …, 8, the window `ROLLOUT_RMAX`
leaves, and the panel says so. It is where the saturation shows up as a number:
in the full space R1 runs **0.89–0.99× the control across all nine cells**, i.e.
an occupied-square branch ends up no further from the reference than a second
legal move, and R3's excess falls from 34–61% on the spliced set to 11–44%. The
bars keep more separation than the markers do (R3 1.21–2.96× in the plane)
because the plane is still fit on the branches it draws. R2 is the clearest
case: its bar stands above 1 in all nine cells while its marker sits at or below
1 in four of them (GatedSAE 1.01, TopK 1.00, BatchTopK 0.99, GatedAnneal 0.96).

Everything else — n = 1, the selection of game 1235, one control, per-space
refits, the not-comparable-across-cells disclaimer, the fixed camera — reads
exactly as for the spliced displacement figure above, and the same caveat about
over-reading a selected game applies with more force here, since the rollout set
has not been swept for a best game at all.

### `game_1235_fork_and_board_r21_pca_rollout`

**The rollout branches in the R²¹ curvature plane.** Panel (a) embeds
`Δφ(t) = φ(z_b(t)) − φ(z_ref(t))` exactly as the spliced R²¹ figure does, and the
K = 4 rollouts are averaged **after φ**, never before: φ is nonlinear, so the
descriptor of an averaged trajectory is the curvature of a path no rollout takes.
The order is φ per row → difference against the same rollout's reference → mean
over K → project.

**The prefix still collapses.** With the window cut to *r* ≤ +8 the drawn offsets
are *r* = −4 … +8, of which **2 are branch-free**, and `project()` asserts
`max |Δφ| = 0` there in all nine spaces on every render — it passes. That is not
a formality in this variant: it is the check that a rollout's activations before
*t0* really are bit-identical to every other branch's, which they are (max
|z − z_row 0| = 0 over all 36 rows of the game at every position < *t0*, raw
layer 5).

**Within ÷ between is 0.714 here**, against 0.902 on the spliced set for the same
game and field. One game and one draw of four continuations: it is not evidence
that the rules cohere better under rollout, and no population sweep of the R²¹
ratio has been run on this branch set.

**(c) has one cell that sets the whole y scale, and it is not an artifact.** In
CosineSpaDE the control's Δφ is the smallest of the nine cells (mean 1.72
against 2.4–4.6 elsewhere) while its illegal branches are among the largest, so
its ratios reach **12.3 in the plane and 5.1 in the space** and the other eight
cells are compressed against them. The axis is not capped and no bar is broken:
one linear axis from zero is what a bar chart owes the reader, and the
compression is the honest consequence of putting nine cells on it. It is also
the sharpest illustration of the panel's own rule — these are *within-cell*
numbers, and a cell whose control barely moves will report large ones. Read
CosineSpaDE against its own control, not against raw layer 5's bars. The rest of
(c) here reads as elsewhere: R1 below 1 in eight of nine cells, R3 above 1 in
all nine, and the bar below its marker for R1 in eight of nine — the plane
understating, where the spliced figures have it overstating.

## (c): the fork as a number, and why it is a ratio

A fork panel asks the reader to judge how far each branch runs from the
reference, and until (c) nothing on the figure answered it. (c) is that
judgement as a number: for each branch, its **mean distance from the reference
over the drawn offsets *r* ≥ 0** (`spade.figures.branch_panels.plane_displacement`),
averaged over a rule's member categories, with the members drawn as dots so the
spread the mean hides is on the panel. The drawn window is what it averages
over — 13 offsets *r* = 0, 2, …, 24 on the spliced figures, **5 offsets
*r* = 0, 2, …, 8 on the rollout ones**, where `ROLLOUT_RMAX` cuts the fork — and
the panel prints which.

### The comparability problem, and what was done about it

The plane is refit per space and its axes are that space's own units. In the
displacement plane, `control_legal` sits a mean 2.80 from the reference in raw
layer 5 and 0.0036 in SpaDE: the same branch, the same game, a factor of ~800.
Nine such numbers side by side would be nine unit systems in one bar chart, and
a reader would compare them, because that is what a bar chart is for.

**So (c) never draws a distance. It draws each branch's distance ÷ the same
cell's legal control's**, which is dimensionless, is a within-cell contrast by
construction, and is the form the rest of this study already argues in (illegal
δ̂ ÷ control δ̂ in `../README.md`). The reference is 0 and the control is 1 in
every cell, both by construction, and the dashed line at 1 is drawn.

That makes side-by-side reading defensible. It does **not** make the nine bars
nine measurements of the same thing, for a reason the panel also draws:

### It measures the plane, not the space — so the marker is there too

A plane holds part of a displacement, and it holds a *different* part of each
branch. In raw layer 5, the displacement plane carries **30% of the control's
displacement and 81% of `opponent_only`'s** — the plane is the top two PCs of
exactly these branches, so it is fit where the largest of them travel. The ratio
(c) would report from the plane alone is therefore inflated by the ratio of
those shares, exactly:

```
plane ratio = full-space ratio × (branch's share of its own ‖Δ‖ ÷ control's share)
      4.08  =           1.53   ×            (0.810 ÷ 0.303)
```

The error runs the other way too: a branch that leaves the plane reads as a
branch that stayed near the reference. Neither is visible on a fork panel.

So every bar carries an **open marker: the same ratio measured in that cell's
full space** — the SAE's own 8192-d code, or the column-scaled Δφ in R²¹, the
space the plane was fit in. The gap between a bar and its marker *is* the part
the plane does not hold. Where the two disagree, the marker is the answer to
"how far", and the bar is the answer to "how far *in this picture*".

This is also what makes the across-cell reading honest. Displacement, spliced,
R3: the **bars** run 3.81–6.18 across the nine cells and the **markers** run
1.34–1.61. The spread the bars show across the grid is ~9× the markers' and is
mostly the plane fit moving, not the spaces differing. Read differences between
cells off the markers.

### The numbers

Displacement plane, spliced branches, game 1235, layer 5, mean over
*r* = 0, 2, …, 24, as a multiple of each cell's own `control_legal`, printed by
`make_figures.py` on every render — **plane | full space**:

| cell | R1 placement (2) | R2 bracketing (3) | R3 turn (1) |
|---|---|---|---|
| raw layer 5 | 1.22 \| 1.05 | 2.43 \| 1.20 | 4.08 \| 1.53 |
| PAnneal | 1.19 \| 1.05 | 2.37 \| 1.19 | 4.29 \| 1.56 |
| Standard | 1.22 \| 1.04 | 3.17 \| 1.22 | 6.18 \| 1.60 |
| GatedSAE | 1.74 \| 1.18 | 2.02 \| 1.13 | 3.81 \| 1.40 |
| TopK | 1.69 \| 1.11 | 2.62 \| 1.16 | 4.69 \| 1.45 |
| GatedAnneal | 1.44 \| 1.07 | 2.11 \| 1.12 | 3.84 \| 1.34 |
| BatchTopK | 1.59 \| 1.10 | 2.63 \| 1.16 | 5.01 \| 1.53 |
| CosineSpaDE | 1.34 \| 1.14 | 3.26 \| 1.30 | 4.65 \| 1.61 |
| SpaDE | 1.23 \| 1.02 | 2.86 \| 1.19 | 5.00 \| 1.49 |

What the full-space column says, in one line: on this game a rule's branches sit
**2–30% further from the reference than a second legal move does**, and the
ordering R3 > R2 > R1 holds in all nine spaces. The plane column says 1.2×–6.2×
of the same thing.

The other three figures, as ranges over the nine cells (plane | full space):

| figure | R1 | R2 | R3 |
|---|---|---|---|
| displacement, rollout | 0.79–0.95 \| 0.89–0.99 | 1.09–1.88 \| 0.96–1.12 | 1.21–2.96 \| 1.11–1.44 |
| R²¹, spliced | 0.57–1.37 \| 0.88–1.32 | 1.25–2.22 \| 1.09–1.58 | 2.24–3.94 \| 1.69–2.47 |
| R²¹, rollout | 0.21–2.75 \| 0.62–1.76 | 0.55–7.18 \| 0.56–3.30 | 1.16–12.33 \| 1.06–5.06 |

The rollout displacement figure is the saturation of the section below as one
number per rule: **R1 falls under 1 in every cell** — those branches end up
*closer* to the reference than a second legal move does — and R3's full-space
excess shrinks from 34–61% to 11–44%. The R²¹ rollout figure is the one to
distrust: its ranges are wide because a single cell (CosineSpaDE) carries the
top of every one of them, and it is one game, one draw of four continuations,
and a descriptor whose window still reaches 14 plies back at *r* = +8. It is
also where the plane *understates* rather than overstates — the bar sits below
its marker for R1 in eight of the nine cells — which is the same distortion
running the other way.

### What (c) does not carry

- **No interval over games.** n = 1, like the rest of the figure. The dots are
  the within-rule spread over member categories (n = 2, 3, 1) and nothing else;
  `R3 turn` has one member, so its bar has one dot sitting on it and is not an
  average. The within ÷ between-rule spread ratio of **0.885** on this game says
  that spread is nearly as wide as the spread between rules, so the three bars
  are three labels more than three findings.
- **No claim about a space.** A cell whose bars are tall has a plane that
  happens to hold more of this game's displacement, not a dictionary that
  separates violations better. Nothing in this figure licenses ranking the nine.
- **Not the study's ratio.** `../README.md`'s 0.978 is mean illegal δ̂ ÷ mean
  control δ̂ over 13,531 branches with a CI. (c)'s numbers are one game, no CI,
  and a different estimator (a mean of per-branch distances, not a ratio of
  population means). They are not comparable and neither replicates the other.
- In grayscale the three rule hues are told apart by the marker shape on each
  bar (`RULE_STYLE`: square, triangle, diamond) and by position in the group,
  since a bar cannot carry a marker of its own.

## The rollout variant, and why its window stops at *r* = +8

Every spliced branch replays `tokens[t0+1:]` from the source game, and that
suffix is a legal continuation of **the reference's** board alone — the reference
keeps a 36-move legal continuation, a legal control's survives a median of 2
moves, and after an illegal move there is no board for it to be legal on. The
rollout set removes that asymmetry by regenerating every suffix, the reference's
included. What it costs is single-token attribution: two rollout branches differ
at *t0* **and** at every position after it, so a displacement at *r* > 0 is the
branch move plus everything the model chose to do about it.

That shows up as saturation. Layer 5, full set, per-branch mean over K then mean
over branches (11,527 illegal and 2,000 control branches over 2000 games), read
off `$SPADE_DIVERGENCE_DIR/[rollout/]divergence/resid_post_5.npz`:

| *r* | spliced δ̂ illegal | spliced ratio | rollout δ̂ illegal | rollout ratio |
|---|---|---|---|---|
| 0 | 0.792 | 1.0316 [1.0266, 1.0366] | 0.792 | **identical by construction** |
| 1 | 0.658 | 0.947 | 0.811 | 0.970 |
| 4 | 0.590 | 0.9305 [0.9229, 0.9382] | 0.985 | 0.9855 [0.9809, 0.9904] |
| 8 | 0.554 | 0.9314 [0.9231, 0.9396] | 1.118 | 0.9957 [0.9926, 0.9990] |
| 13 | 0.527 | 0.954 | 1.200 | 1.0010 — crosses over |
| 24 | 0.461 | 0.9797 [0.9715, 0.9873] | 1.222 | 1.0034 [1.0017, 1.0053] |

Ratio = mean illegal δ̂ ÷ mean control δ̂; brackets are a 95% percentile bootstrap
over the 2000 games, 2000 resamples. Three things to take from it:

- ***r* = 0 is the same number in both variants**, to four decimals, because the
  activation at *t0* depends only on `tokens[:t0+1]` and the two sets share those
  bits. The rollout adds nothing at *r* = 0 and everything after it. (This is
  also the cheapest available check that the rollout activations are right.)
- **Rollout δ̂ climbs where spliced δ̂ decays**, from 0.79 to **1.236** by *r* = 20
  — 96% of the **1.293** two-unrelated-games-at-the-same-position level that
  `../RED_TEAM.md` measures in this space, against the spliced plateau's 0.46–0.50
  (~36%). The spliced study's "δ̂ has not saturated" finding does **not** carry
  over; under rollout it very nearly has.
- **The illegal/control separation dies with it.** Spliced holds ~0.93 across the
  whole window. Rollout starts at 0.970, is inside 1.5% of 1 by *r* = 4, inside
  0.5% by *r* = 8, and crosses **above** 1 by *r* = 13. The tiny intervals mean
  those late values are resolvably different from 1 in the *wrong* direction, not
  that a signal survives.

So a rollout fork figure is informative at short horizons and misleading at long
ones, and drawing it to *r* = 24 would have been drawing 16 offsets of nothing.
`ROLLOUT_RMAX = 8` in `make_figures.py` is that cut, the figure states it, and it
is the one number to revisit if the branch set changes.

**This does not retract the spliced study.** The two sets answer different
questions — spliced isolates one token against a shared future, rollout lets the
model write the future and measures the pair afterwards — and `branches.pt`
remains the tensor the single-token claims rest on. Neither figure is the other's
replication.

## Game 1235 is selected, and the figure says so

Game 1235 is the **argmax**, over all 1550 branch-complete games, of the mutual
separation of its six illegal displacement trajectories — scored in raw layer 5
in the full 512-dimensional space, never in an SAE space or a 2-D projection
(`../game_selection`). Population median 0.705; this game 1.080.

That is load-bearing, not boilerplate. On this game the reference and both
controls separate from all six illegal branches **in all nine spaces**, which
looks like a result. Over 40 games that ordering holds in only 0.07–0.17 of
them, against a 0.036 chance floor. The selected game displays structure that
does not generalise, so a reader who does not know the game was chosen will
over-read the panels — which is why the disclosure leads the caption above and
why a `--game` other than the default must carry its own.

## Layout

The board is square and the fork grid is three cells wide, so a naive stack
would leave a band of white beside the board. That band carries the figure's
apparatus instead: the **one shared legend** — both panels take colour, marker,
line style and label from `spade.figures.branch_style`, so a second legend would
say the same thing twice — and the board's reading key. The board's cell is as
wide as it is tall, so the square board fills it rather than leaving a strip, and
the legend block is centred in the band rather than pinned to its top.

The R1/R2/R3 key goes under the whole figure, which is the only place with the
width its three lines need (FIGURE_GUIDELINES §5: shorthand in a legend earns a
key inside the figure). It names each rule, how many categories it averages, and
what each of them is, driven by `RULE_MEMBERS` and `CATEGORY_GLOSS` so the counts
cannot disagree with the legend's `(mean of n)`.

Panel **(c)** is a third full-width row under the board rather than an inset in
each of the nine cells: nine cells × five series is 45 numbers, which no set of
in-panel labels survives, and the whole point of the panel is that the nine are
read against one axis. The canvas grows by (c)'s height plus one more row gap,
so the fork grid and the board keep the size they had before it — the cells are
within 0.4% of their previous height. The bars sit next to nothing else, which
is what a nine-group grouped bar chart needs at 9 pt.

Panel letters sit top-left of their region; the working PNG carries an in-figure
title and `--camera-ready` drops it and emits vector PDF (§2, §6).

## What is plotted

Panel (a) is `spade.violation_divergence.fork_plane`'s embedding, drawn by
`spade.figures.branch_panels.rule_fork_panel`. Move index is the third axis, so the
plane is a 2-D embedding of the **displacement** `Δ(t) = z_branch(t) − z_ref(t)`
and never of raw activations: move index dominates activation geometry, so a raw
PCA would put it on PC1 and encode time twice. With Δ, the reference is exactly
the vertical line at the origin and the spray at *t0* is the whole content.
Positions are sampled every 2 plies at *t0*'s parity — the board code is
mover-relative, so at every ply the path is a sawtooth that hides everything
else.

Panel (b) is `branch_panels.board_panel`. Markers sit in the top-left corner of
their square, never at the centre: `occupied_own` and `occupied_opp` play onto a
square that already holds a disc, and covering that disc would destroy the
evidence for the category. The outlined cell, in the branch's own colour and
line style, is what identifies the square. Under the rule grouping the two R1
squares carry the same colour and marker, so a reader tells them apart by the
square and by the disc under it — the board note says so.

Every branch square also carries **the disc the move would place**, in the
mover's colour, pinned to the corner opposite the category marker. It has to be
unmistakably hypothetical, because a reader who miscounts the position misreads
the whole figure: it is half a real disc's radius, translucent, and dashed, where
a real disc is solid, full size and solid-edged — a difference that survives
grayscale. The corner, rather than the centre, is exactly the `occupied_*`
problem again: those two squares already hold a real disc, and its colour is the
reader's only evidence for which of the two categories the branch is, so nothing
is ever drawn over it. It is drawn on **all** eight branch squares, the reference
and the control included, not only the illegal ones: every branch is a move that
places a disc, and a marking that appeared only on illegal squares would imply
the placement is what makes them illegal, which is false for four of the six
categories.

## The second plane: R^21

`--plane r21` replaces the displacement embedding with a plane inside the 21-D
ReStraV curvature descriptor (`spade.violation_divergence.curvature_plane`, which
`../fork_curvature_space` measures in detail). φ is the 7 step lengths, 6 turning
angles and their 8 moments over the **W = 8 sampled positions ending at *t***, and
panel (a) embeds `Δφ(t) = φ(z_b(t)) − φ(z_ref(t))`. Panel (b) is unchanged: the
board does not know what plane the trajectories were drawn in.

Three things the R^21 version has to get right, all of them checked at render
time or in `spade.violation_divergence.test_curvature_plane`:

- **φ goes on each member category separately, and the rule mean is taken
  afterwards.** φ is nonlinear, so `mean φ(z_b) ≠ φ(mean z_b)`: averaging the six
  trajectories first and taking step lengths and angles of the average measures
  the curvature of a path no branch takes. The plane projection *is* linear, so
  mean-then-project and project-then-mean agree, and the figure takes the mean in
  the plane exactly as the displacement version does. The caption says where the
  mean is taken.
- **The difference is taken inside R^21, never φ of the displacement.** `D` is
  identically zero over the shared prefix, so its steps have zero length, its
  angles are undefined, and the reference branch comes out 53.8% NaN
  (`../fork_curvature_space/README.md`). Δφ keeps both invariants: the reference
  is the vertical line at the origin, and the prefix collapses onto it exactly.
  `project()` asserts `max|Δφ| = 0` over the branch-free offsets, in every one of
  the nine spaces, on every render.
- **The window costs offsets.** A full-width window spans 14 plies, so it needs
  prefix context the study window `r ≥ −8` does not carry. The positions are
  extended back to the start of the game and the first `W − 1` are dropped: at
  *t0* = 20, **15 of the study window's 17 offsets carry a descriptor and 2 of
  those are branch-free**. The figure states this and the *z* axis starts where
  the descriptor does, not where the study window does.

The `--embedding` choice belongs to the displacement plane; `--plane r21` fits
its own uncentred PCA of the column-scaled Δφ and asserts that `--embedding` is
left at `pca` rather than silently ignoring it.

## One trajectory per rule, and what that costs

The six illegal categories are averaged into the three rules they violate
(`spade.othello.taxonomy.RULE_MEMBERS`): **R1 placement** = `occupied_own`,
`occupied_opp`; **R2 bracketing** = `empty_detached`, `empty_own_only`,
`empty_no_bracket`; **R3 turn** = `opponent_only`. Both panels take
`RULE_STYLE`, so a colour and a marker mean a rule in either half, the R1/R2/R3
key under the legend now names the series exactly, and two branch squares of the
same colour on the board are two members of one rule.

The groups are sizes 2, 3 and 1, so every legend entry carries n — including
**`R3 turn (mean of 1)`**, which is one branch and not an average of many.

The mean is taken in the plane, after projection. PCA here is an uncentred
linear map (`fork_plane.pca2`), so that is exactly the projection of the mean
displacement; Isomap is not linear, and there it is a mean of embedded points —
which is why the mean is taken after projection rather than before, one rule for
both embeddings.

**The grouping is weak, and the figure says so.** Within-rule ÷ between-rule mean
pairwise distance, on the full 512-d layer-5 displacement trajectories over
*t* ≥ *t0*, is **0.885** on game 1235 and **median 0.945 [IQR 0.872–1.020]** over
all 1550 branch-complete games — above 1 in 30% of them
(`python -m spade.violation_divergence.rule_groups`). The R^21 figure recomputes
it in its own space — the same statistic on the column-scaled Δφ, all 21
components, raw layer 5, *t* ≥ *t0* — and gets **0.902** on the same game, with a
population of **median 0.936 [IQR 0.827–0.998], above 1 in 25%** over 60 games
with *t0* ≥ 18 (`--field r21`). The rules cohere no better in curvature space than
in displacement space; the single-game pair (0.902 vs 0.885) orders them the other
way from the medians, which is why neither ordering is worth quoting. Each caption
says which field its number comes from. A rule's members travel
barely closer to each other than to another rule's. Three confident lines would
be a lie about that, so each rule's members are drawn **faint behind their mean**
and the ratio rides in the caption: the members, not the means, are the
evidence. Game 1235 sits near the first quartile, so its grouping is *better*
than typical — which is itself selection, since the game is the argmax of the
mutual separation of the same six trajectories.

## One control, and where the null went

`control_legal_2` is **not drawn**. One legal control remains, in the null
styling — grey, dashed, last in the legend — so the reader sees "a legal
alternative, for scale" without a second grey mark on an adjacent square.

That has a cost, and it is not cosmetic. `branches.py` builds two legal controls
because *one gives no null*: control 2 is scored against control 1 exactly as the
illegal branches are, and the study's first pass shipped with one control and
could not interpret its own direction result (`../RED_TEAM.md` F1). Dropping it
here removes the reader's visual answer to "is that separation big compared to
two legal moves differing?". Nothing measured changes — `branches.pt` still
carries both rows, and every number in `../README.md` is still scored against
`control_legal` with `control_legal_2` as its null — but the figure now has to
point at where that null lives, and its caption does: `../game_trajectory`,
which draws all nine branches including control 2 and pairs them with the δ̂
marginal, and `../README.md`, which carries the legal-vs-legal numbers with CIs.

Dropping the row also drops it from the PCA fit, since the plane is fit on what
is drawn. Cells shift slightly against the nine-branch version; nothing in them
is comparable across cells anyway.

## Axes

- Each fork cell: `PC1 (x% var)` / `PC2 (y% var)` per §4.6 — **per component,
  never summed**, and per panel, since the plane is refit per space, and *nothing
  else*. The R^21 label used to append the descriptor component the PC loads
  hardest on (`PC1 (44% var, Δ mean θ)`); it no longer does, because that named a
  quantity the axis is not. PC1 in raw layer 5 loads 0.360 on `mean θ`, which is
  13% of the direction, over a near-tied same-signed `mean d` 0.328 and `max θ`
  0.323, and 16 of 21 coordinates are needed to reach 90% of it. Which coordinate
  "wins" flips between those near-ties with the plane: the `whole` variant of
  `../fork_curvature_space`, same game and space, leads on `min d` instead.
  Isomap cells read `arb. units` and carry no ratio. The two plane axes carry
  **three sparse ticks, symmetric about zero** so the origin — where the reference
  sits in a displacement plane — always gets one. They are a within-panel scale
  bar and nothing else: the plane is refit per space, so the caption's "not
  comparable across the grid" disclaimer is what keeps them honest (§4.6, §4.8).
  The ordered axis keeps its ticks and reads `move index t`, once per row.
- Board: `file` / `rank (1 at the top)`.
- Panel (c): `distance from the reference, ÷ the cell's own legal control`. The
  divisor is in the axis label rather than the caption because it is what makes
  the nine groups one axis at all — a reader who misses it reads nine
  incommensurable planes as one bar chart. Zero-based and linear (§1.6), one
  panel for one scale, and the tick labels are 9 pt like the rest of (c).

## Reading it honestly

- **Planes are refit per space.** Cells are comparable in the *shape* of the
  fork and never in position or magnitude; SAE latent spaces live on different
  scales. The caption says so.
- **(c) is a ratio because a distance would be a lie.** Nine refit planes carry
  nine unit systems — the same control branch sits 2.80 from the reference in
  raw layer 5 and 0.0036 in SpaDE — so (c) divides every distance by its own
  cell's legal control and never draws the distance. Even then, compare cells
  through the open markers: the bars carry the plane's per-cell distortion on
  top of the quantity.
- **A bar in (c) measures the picture, not the space.** The plane is fit on the
  branches it draws, so it holds 81% of `opponent_only`'s displacement and 30%
  of the control's in raw layer 5, and the bar inflates by exactly that ratio.
  The marker beside it is the same contrast in the full space. Quote the marker.
- **One game is n = 1, and a selected one.** See the section above: the panels
  look cleaner than a typical fork does, by construction. This is an
  illustration of the design, not a claim-supporting figure. Every number — the
  0.978 ratio, the 0.4985 legal-vs-legal null, the plateau — belongs to
  `../README.md`, where it carries its CI over 13,531 branches.
- **Three lines are six categories.** The rule means are a labelling
  convenience, not a demonstration that the rules are the natural unit; the
  spread ratio near 1 says the members barely cohere. Read the faint member
  lines. If a claim ever rests on a rule, it has to be scored per category and
  aggregated with a CI, in `../README.md`, not read off these means.
- **Only one control is drawn.** The legal-vs-legal scale is in
  `../game_trajectory` and `../README.md`; the caption names them rather than
  the figure showing one.
- **The two variants are not two measurements of one thing.** A spliced fork and
  a rollout fork of the same game do not "agree" or "disagree"; they are one
  token against a shared future and one token against a future the model wrote.
  Compare them only through the section above, which says what each can carry.
- **A rollout path is short on purpose.** It stops at *r* = +8 because the
  separation it draws is gone past there, not because the data runs out. Anyone
  extending `ROLLOUT_RMAX` owes the reader the ratio table again.
- The view angle is fixed at `elev=18, azim=−58` and stated in the caption, so a
  favourable camera cannot pass as a result.
