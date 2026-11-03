# M6 / F7 — what the layer-5 board probe decodes off an illegal branch

Agent A6. Code: `decode.py` (measurement → `results/violation_divergence/probe_*.csv`)
and `make_figures.py` (F7). 3923 source games, 34546 branch rows, layer-5
`resid_post`, and the project's one board read-out `spade.probes.load.board(5)`
(`[512, 64, 3]` flat, classes `EMPTY, THEIRS, MINE`, **0.9937** test accuracy over
20,000 `adamkarvonen/othello_45MB_games` games split by game).

## The frame convention — read this before any number below

The probe is **mover-relative**: at position *s* it predicts the board after move
*s* with `mine` = the player to move at *s+1*. After an illegal move whose turn it
is is exactly what is in dispute, so "mine" is not a well-defined label for an
illegal branch. Every number here therefore uses **one target for every branch**:

> **The reference game's board, labelled in the reference game's frame.**
> At offset *r* the target for square *q* is `EMPTY` if the reference board is
> empty at *q* after move *t0 + r*; `MINE` if it holds the colour that moves at
> position *t0 + r + 1* in the reference game; `THEIRS` otherwise. The mover comes
> from replaying the reference game through `OthelloBoardState.umpire`, never from
> `s % 2`.

Two consequences worth stating plainly:

- **"the illegal branch decodes a disc" means** `argmax` over the probe's three
  classes at that square is not `EMPTY`. It is a claim about the branch's decoded
  board, scored on the reference board's coordinates — not a claim that the branch
  "believes" anything about its own frame.
- At *r* = 0 the player who moved at *t0* is `THEIRS` in this frame (the reference
  game has already handed the turn over). So a freshly placed disc — legal or
  illegal — should read `THEIRS`, and that is what `p_mover_colour` measures: the
  probability that the square decodes **the colour of whoever moved at t0**, with
  the frame's alternation divided out at every offset.

**Does the engine-replay mover actually differ from parity here?** Yes, but almost
nowhere in this window. Across the whole analysis grid the two conventions disagree
on **185 of 129,459 (game, offset) cells, 0.14%** — this corpus forfeits at the very
end of a game, past *r* = 24 for nearly every source. On exactly those cells the
probe is near chance under either convention (engine frame **0.585**, parity frame
**0.414**, n = 185 cells), so the choice moves no headline number. It does now lean
the right way, which it did not under the retired ARENA probe (0.554 / 0.441).
`probe_sanity.csv` carries both frames at every offset; the `frame_diagnostic` rows
are that comparison.

## Sanity check: the probe on the reference branch

Reference-branch accuracy against the true board, engine frame, mean over 3923 games:

| *r* | −8 | 0 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| all 64 squares | 0.9999 | 0.9996 | 0.9986 | 0.9961 | 0.9867 |

At *r* = 0, by square class: empty 0.9999, untouched (theirs) 0.9993, untouched
(mine) 0.9992, the reference's own move square 1.0000, discs it flipped 0.9997.
This sits **above** the probe's 0.9937 test accuracy because that figure is over
every position of a game while this window is not: *t0* ∈ [10, 34] with 2000 of the
3923 games at *t0* = 20, so *r* ∈ [−8, 24] lands almost entirely in the mid-game,
where the probe is strongest. The slow decline with *r* is board fill running toward
the end of the game.

## The contract keys

`p_illegal_disc_decoded[cat]` — events whose illegal branch decodes a disc
(≠ `EMPTY`) on the illegally played square at *r* = 0, with the 95% bootstrap CI
over source games and, beside it, the same square read off the reference and
control branches:

| category | n | **p_illegal_disc_decoded** | 95% CI | reference branch | control branch |
|---|---|---|---|---|---|
| `occupied_own` | 3923 | **1.0000** | [1.0000, 1.0000] | 0.9995 | 0.9997 |
| `occupied_opp` | 3923 | **0.9997** | [0.9992, 1.0000] | 0.9995 | 0.9995 |
| `empty_detached` | 3874 | **0.9951** | [0.9930, 0.9972] | 0.0000 | 0.0000 |
| `empty_own_only` | 3299 | **1.0000** | [1.0000, 1.0000] | 0.0003 | 0.0003 |
| `empty_no_bracket` | 3838 | **0.9997** | [0.9992, 1.0000] | 0.0000 | 0.0000 |
| `opponent_only` | 3920 | **1.0000** | [1.0000, 1.0000] | 0.0003 | 0.0000 |

The two occupied categories are ~1 for every branch by construction — the square
already holds a disc — so their row is a floor check, not a result; their result is
the colour, below. For the four empty-square categories the answer is unambiguous:
**the illegal branch writes the illegal disc onto its decoded board in 99.5–100% of
events, and neither anchor writes anything there (≤ 0.0003).** The ceiling anchor
— the reference branch read at its own played square, a disc that really was placed
— is 1.0000 [1.0000, 1.0000]. The illegal disc is decoded as reliably as a real one.

`parity_agree_r0[cat]` — events whose illegal branch decodes the same mover as the
reference at *r* = 0. Per-event the decoded mover is a majority vote over the
squares no branch of that game plays or flips: a frame-flipped branch swaps
`THEIRS` and `MINE` on all of them at once, so the vote is the read-out.

| category | n | **parity_agree_r0** |
|---|---|---|
| `occupied_own` | 3923 | 0.9992 |
| `occupied_opp` | 3923 | 1.0000 |
| `empty_detached` | 3874 | 0.9894 |
| `empty_own_only` | 3299 | 0.9815 |
| `empty_no_bracket` | 3838 | 0.9995 |
| `opponent_only` | 3920 | **0.9916** |
| reference branch | 3923 | 1.0000 |
| `control_legal` | 7846 | 1.0000 |

**The R3 prediction is falsified.** `opponent_only` is the one category whose
violation *is* turn confusion, and it does not confuse the decoded turn: 0.9916,
mid-pack, above `empty_detached` and `empty_own_only`. No category drops below
0.981 at any offset in the window. The model does not read an illegal
opponent-flipping move as "the opponent moved"; it reads it as the mover placing a
disc — and **100.0%** of the time the disc it writes there carries the **mover's**
colour, not the opponent's (`p_mover_colour_r0`, `opponent_only`).

## Persistence: how many plies the illegal disc survives

Per event, the length of the unbroken run of offsets *r* = 0, 1, 2, … at which the
illegal branch's decode at the illegally played square **disagrees with the
reference board's truth there**. 25 = the whole post-*t0* window.

| category | n | median | mean | ≥ 1 ply | ≥ 8 | ≥ 16 | = 25 |
|---|---|---|---|---|---|---|---|
| `empty_detached` | 3874 | **24** | 19.00 | 0.995 | 0.878 | 0.715 | 0.471 |
| `empty_own_only` | 3299 | **21** | 18.32 | 1.000 | 0.884 | 0.672 | 0.386 |
| `empty_no_bracket` | 3838 | **19** | 17.43 | 1.000 | 0.862 | 0.617 | 0.343 |
| `opponent_only` | 3920 | **14** | 14.38 | 1.000 | 0.715 | 0.463 | 0.238 |
| `occupied_opp` | 3923 | **3** | 4.66 | 0.764 | 0.226 | 0.061 | 0.010 |
| `occupied_own` | 3923 | **0** | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |

**The answer: about 14–24 plies, median.** Between 34% and 47% of empty-square
events carry the illegal disc unbroken to the end of the 24-ply window. This is not
a transient at the injection position — the model keeps an impossible disc on its
board for most of the rest of the game.

## `occupied_opp`: recolor, not no-op

At *r* = 0, P(the illegally played square decodes the colour of whoever moved at
*t0*), with the same square read off the reference branch as the anchor:

| category | illegal branch | reference branch | control branch | excess |
|---|---|---|---|---|
| `occupied_own` | 1.000 | 0.999 | 0.999 | **+0.001** |
| `occupied_opp` | **0.896** | 0.144 | 0.143 | **+0.752** |

`occupied_own` sits on its own anchor: the square already holds the mover's disc,
the illegal branch agrees, nothing happened. That is a **no-op**, and its median
persistence of **0 plies** says the same thing — this violation leaves no trace at
all on the decoded board.

`occupied_opp` does not. The square holds the **opponent's** disc on the reference
board, and the illegal branch reads the **mover's** colour there in 89.6% of events
against a 14.4% anchor. (The anchor is not 0 because the reference move sometimes
flips that very square — which is exactly why the anchor has to be measured rather
than assumed.) **This is the recolor reading, not the no-op reading, by a 0.75
margin.** Placing on an opponent's square is represented as *capturing* it.

Two checks on that verdict:

- **It is not parity confusion.** A frame-flipped branch would produce the same
  observation. Restricted to the events whose decoded parity is confirmed aligned
  at *r* = 0, the number is **0.8960** — unchanged from 0.8960.
- **It washes out fast.** The excess over the anchor falls +0.752 (*r*=0) → +0.263
  (*r*=8) → +0.110 (*r*=16) → +0.053 (*r*=24), and the median disagreement run is
  3 plies. So `occupied_opp` recolors, but unlike the four empty-square categories
  the recolor does not persist: the model repairs a wrong colour far faster than it
  removes a disc that should not exist.

## By square class, at *r* = 0

Accuracy of each branch's decoded board against the reference board, per class
(reference-branch column is the same measurement on the same squares, so the
comparison is within one forward pass):

| square class | `occupied_own` | `occupied_opp` | `empty_detached` | `empty_own_only` | `empty_no_bracket` | `opponent_only` | reference branch |
|---|---|---|---|---|---|---|---|
| the reference move's square | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | 1.000 |
| discs the reference move flipped | 0.043 | 0.193 | 0.019 | 0.009 | 0.057 | 0.025 | 1.000 |
| untouched, theirs | 0.997 | 0.999 | 0.969 | 0.946 | 0.995 | 0.977 | 0.999 |
| untouched, mine | 0.933 | 0.899 | 0.960 | 0.961 | 0.927 | 0.963 | 0.999 |
| empty | 1.000 | 1.000 | 0.995 | 0.994 | 0.998 | 0.997 | 1.000 |

Two things fall out of the control class:

- The mirror image of the headline holds: the illegal branch does **not** place the
  reference's disc (0.000–0.001 vs 1.000) and does **not** flip what the reference
  move flipped (0.01–0.19 vs 1.000). Its board is the branch's own move, cleanly.
- The untouched discs — the control that rides along in the same forward pass —
  are **not** left alone. They fall from 0.999 to as low as 0.899, and which colour
  takes the loss tracks the violation: the categories that place a disc into a
  populated neighbourhood lose most on `untouched, theirs` (`empty_own_only` 0.946,
  `empty_detached` 0.969), while the two occupied categories lose most on
  `untouched, mine` (`occupied_opp` 0.899, `occupied_own` 0.933) despite changing
  little or nothing about the board. Reading either as the model computing flip
  consequences of the illegal disc is the natural story, but it is an inference
  from an accuracy drop, not a measurement of which squares flipped — A4/A5 can
  settle it.

## F7

`figures/f7_probe_readout.png` (`--camera-ready` → vector PDF, titles dropped for
the caption). Two event-locked panels on the *r* ∈ [−8, +24] window, mean over
source games with per-curve n in the legend, pre-*t0* region shaded, dashed grey
line at *r* = 0. **No interval is drawn.** Eight overlapping shaded CIs hid the
anchor curves the panels are read against, so they were dropped; the 95% bootstrap
CI (1000 resamples over source games) stays in `probe_illegal_square.csv` as
`lo`/`hi`, and the legend title says so. That leaves the panels with no on-figure
uncertainty, against FIGURE_GUIDELINES 1.2.

- **(a)** decoded occupancy of the illegally played square, four empty-square
  categories, with **both anchors labelled**: the reference branch at the same
  square (floor — nothing was played there) and the reference branch at its own
  played square (ceiling — a disc that really was placed). `control_legal` takes
  `NULL` styling and coincides with the floor anchor to within 0.0001; its legend
  entry says so, since two curves on top of each other otherwise read as one.
- **(b)** the occupied pair: P(decodes the mover's colour) for the illegal branch
  (solid) and for the reference branch at the same square (dotted, same hue).

The parity read-out has no panel: `parity_agree_r0` never leaves [0.981, 1.000] in
this window, so the curves sit flat and carry no result. `probe_parity.csv` keeps
the measurement.

Caption, first sentence: *An illegal move puts its disc on OthelloGPT's decoded
board and leaves it there for a median of 14–24 plies; a move onto an opponent's
square is decoded as capturing it, not as a no-op.*

### Figure notes

Prose that used to sit on the figures in light gray, kept here because it defines
the shaded band, the n, and the contrast the panels are read against.

`f7_probe_readout.{png,pdf}` and `f7_probe_readout_rollout.{png,pdf}`:

- Shaded band, *r* ∈ [−8, 0]: "shared prefix — branches coincide by construction."
- Panel (a): "nothing played here — both anchors rise only as the shared suffix
  fills the square."
- Panel (b): "recolor: **+0.75** at *r* = 0 against the reference branch on the same
  square" (drawn as a double-headed arrow between the `occupied_opp` solid and
  dotted curves at *r* = 0), and "no-op: occupied (own) tracks its own reference
  anchor."
- Rule key: **R1 placement** — a move puts one disc of the mover's color on an
  empty square. **R2 bracketing** — the disc must close a gapless line of opponent
  discs with one of the mover's own. **R3 turn** — colors alternate; legality is
  judged for the mover, not the opponent.

`f7_probe_readout_rollout.{png,pdf}` only:

- "the sampled reference's board ends at its first illegal move: **1.0%** of window
  positions have no board and are dropped, so n falls with *r*." The figure used to
  print 1.1%, hard-coded against the 2000-game tree.

## The rollout variant

`probe_rollout_*.csv` and `f7_probe_readout_rollout.*`, from the tree where every
branch — the reference included — samples its own continuation, K = 4 per branch.
Everything at *r* = 0 is **identical** to the spliced tree by construction: tokens
up to and including *t0* are copied from the source branch, so the probe reads the
same activation there. The variant only moves what depends on *r* > 0:

| | spliced | rollout |
|---|---|---|
| median persistence, `empty_detached` | 24 | 25 |
| median persistence, `empty_own_only` | 21 | 25 |
| median persistence, `opponent_only` | 14 | 19 |
| `occupied_opp` excess at *r* = 8 | +0.263 | +0.196 |
| reference decode accuracy at *r* = 24 | 0.9867 | 0.9879 |

The illegal disc persists *longer* when the continuation is sampled, which is the
expected direction: a sampled suffix never plays the move that would legitimately
fill or flip the square, so nothing forces the decoded board back onto the
reference's. 1.0% of window positions have no reference board there — the sampled
reference stops at its first illegal move — and are dropped rather than guessed.

## For the integrator

1. **I added `decode.py` next to `make_figures.py`.** My assignment listed
   `make_figures.py`, `figures/`, `RESULTS.md` and `results/violation_divergence/probe_*.csv`.
   Folding a probe pass over 2 GB of activations into the plotting script would make
   every figure tweak re-run it, so the measurement is its own entry point, matching
   `src/spade/bracket_flip/probe_space.py` → `analysis/bracket_flip/make_figures.py`.
   Say the word and I will merge them.
2. **`paper_style` request: a branch-role style.** F7 uses *hue* for the category
   and *line style* for which branch is being read at that square (solid = the
   illegal branch, dotted = the reference branch at the same square). That is the
   same shape as the existing `BASIS_STYLE`, and A3's F1/F6 will want it too. I
   worked around it with literal dash patterns local to `make_figures.py`; the
   entry belongs in `paper_style.py`, which only you extend. Suggested:
   `BRANCH_ROLE_STYLE = {"illegal": {"ls": "-"}, "reference_at_square": {"ls": (0, (1, 1.4))}}`.
3. **The probe takes no flag.** `decode.py` calls `spade.probes.load.board(5)`,
   which reads `$SPADE_PROBES_DIR`, so a worktree needs no path argument. The old
   `--probe` flag existed only because `analysis/linear_probe/activations/linear_probe.pt`
   was gitignored and lived in one checkout; it is gone.
4. **`--smoke` writes `probe_smoke_*.csv`**, so a smoke run cannot overwrite the
   committed full-set CSVs. Delete those before merging if any are lying around.
5. Nothing outside my assigned paths was touched.
