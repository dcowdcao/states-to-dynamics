# M6 / F7 — what the layer-5 linear probe decodes off an illegal branch

Agent A6. Code: `decode.py` (measurement → `results/violation_divergence/probe_*.csv`)
and `make_figures.py` (F7). 2000 source games, 15531 branch rows, layer-5
`resid_post`, the committed probe `analysis/linear_probe/activations/linear_probe.pt`
(`[512, 8, 8, 3]`, classes `EMPTY, THEIRS, MINE`, published eval accuracy 0.9896),
loaded and reshaped exactly as `src/spade/rules/causal/unflip_edit.py::probe_directions`
does.

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
nowhere in this window. 317/2000 reference games desync (a forfeit makes the same
colour move twice) somewhere inside their 59 tokens, but the median first desync is
at **position 56** and only **4 games** desync at or before position 40 — the corpus
forfeits at the very end of a game. Across the whole analysis grid the two
conventions disagree on **129 of 66000 (game, offset) cells, 0.20%**. On exactly
those cells the probe is at chance under either convention (engine-frame accuracy
**0.465**, parity-frame **0.533**, n = 129 cells), so the choice moves no headline
number. `probe_sanity.csv` carries both frames at every offset; the
`frame_diagnostic` rows are that comparison.

## Sanity check: the probe on the reference branch

Reference-branch accuracy against the true board, engine frame, mean over 2000 games:

| *r* | −8 | 0 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| all 64 squares | 0.9976 | 0.9945 | 0.9906 | 0.9835 | 0.9626 |

At *r* = 0, by square class: empty 0.9991, untouched (theirs) 0.9903, untouched
(mine) 0.9875, the reference's own move square 0.9900, discs it flipped 0.9897.
This matches the probe's published 0.9896 and licenses everything below. The slow
decline with *r* is board fill plus the probe's training range (turns 5–54); at
*t0* = 34, *r* = 24 lands on position 58, outside it.

## The contract keys

`p_illegal_disc_decoded[cat]` — events whose illegal branch decodes a disc
(≠ `EMPTY`) on the illegally played square at *r* = 0, with the 95% bootstrap CI
over source games and, beside it, the same square read off the reference and
control branches:

| category | n | **p_illegal_disc_decoded** | 95% CI | reference branch | control branch |
|---|---|---|---|---|---|
| `occupied_own` | 2000 | **0.9990** | [0.9975, 1.0000] | 0.9950 | 0.9925 |
| `occupied_opp` | 2000 | **0.9995** | [0.9985, 1.0000] | 0.9945 | 0.9950 |
| `empty_detached` | 1956 | **0.9852** | [0.9796, 0.9903] | 0.0015 | 0.0000 |
| `empty_own_only` | 1636 | **0.9945** | [0.9902, 0.9976] | 0.0012 | 0.0018 |
| `empty_no_bracket` | 1941 | **0.9959** | [0.9928, 0.9985] | 0.0010 | 0.0010 |
| `opponent_only` | 1998 | **0.9995** | [0.9985, 1.0000] | 0.0000 | 0.0000 |

The two occupied categories are ~1 for every branch by construction — the square
already holds a disc — so their row is a floor check, not a result; their result is
the colour, below. For the four empty-square categories the answer is unambiguous:
**the illegal branch writes the illegal disc onto its decoded board in 98.5–99.95%
of events, and neither anchor writes anything there (≤ 0.0018).** The ceiling anchor
— the reference branch read at its own played square, a disc that really was placed
— is 0.9975 [0.9955, 0.9995]. The illegal disc is decoded as reliably as a real one.

`parity_agree_r0[cat]` — events whose illegal branch decodes the same mover as the
reference at *r* = 0. Per-event the decoded mover is a majority vote over the
squares no branch of that game plays or flips: a frame-flipped branch swaps
`THEIRS` and `MINE` on all of them at once, so the vote is the read-out.

| category | n | **parity_agree_r0** |
|---|---|---|
| `occupied_own` | 2000 | 0.9990 |
| `occupied_opp` | 2000 | 0.9995 |
| `empty_detached` | 1956 | 0.9867 |
| `empty_own_only` | 1636 | 0.9859 |
| `empty_no_bracket` | 1941 | 0.9990 |
| `opponent_only` | 1998 | **0.9930** |
| reference branch | 2000 | 1.0000 |
| `control_legal` | 2000 | 1.0000 |

**The R3 prediction is falsified.** `opponent_only` is the one category whose
violation *is* turn confusion, and it does not confuse the decoded turn: 0.9930,
mid-pack, above `empty_detached` and `empty_own_only`. No category drops below
0.986 at any offset in the window. The model does not read an illegal
opponent-flipping move as "the opponent moved"; it reads it as the mover placing a
disc — and 96.6% of the time the disc it writes there carries the **mover's**
colour, not the opponent's (`p_mover_colour_r0`, `opponent_only`).

## Persistence: how many plies the illegal disc survives

Per event, the length of the unbroken run of offsets *r* = 0, 1, 2, … at which the
illegal branch's decode at the illegally played square **disagrees with the
reference board's truth there**. 25 = the whole post-*t0* window.

| category | n | median | mean | ≥ 1 ply | ≥ 8 | ≥ 16 | = 25 |
|---|---|---|---|---|---|---|---|
| `empty_own_only` | 1636 | **18** | 16.71 | 0.994 | 0.834 | 0.581 | 0.312 |
| `empty_detached` | 1956 | **16** | 15.23 | 0.985 | 0.729 | 0.534 | 0.309 |
| `empty_no_bracket` | 1941 | **16** | 15.94 | 0.996 | 0.794 | 0.545 | 0.286 |
| `opponent_only` | 1998 | **13** | 13.92 | 0.999 | 0.713 | 0.436 | 0.205 |
| `occupied_opp` | 2000 | **2** | 3.50 | 0.670 | 0.159 | 0.036 | 0.003 |
| `occupied_own` | 2000 | **0** | 0.03 | 0.020 | 0.000 | 0.000 | 0.000 |

**The answer: about 13–18 plies, median.** Roughly 30% of empty-square events carry
the illegal disc unbroken to the end of the 24-ply window. This is not a transient
at the injection position — the model keeps an impossible disc on its board for most
of the rest of the game.

## `occupied_opp`: recolor, not no-op

At *r* = 0, P(the illegally played square decodes the colour of whoever moved at
*t0*), with the same square read off the reference branch as the anchor:

| category | illegal branch | reference branch | control branch | excess |
|---|---|---|---|---|
| `occupied_own` | 0.980 | 0.988 | 0.989 | **−0.008** |
| `occupied_opp` | **0.786** | 0.150 | 0.144 | **+0.636** |

`occupied_own` sits on its own anchor: the square already holds the mover's disc,
the illegal branch agrees, nothing happened. That is a **no-op**, and its median
persistence of **0 plies** says the same thing — this violation leaves no trace at
all on the decoded board.

`occupied_opp` does not. The square holds the **opponent's** disc on the reference
board, and the illegal branch reads the **mover's** colour there in 78.6% of events
against a 15.0% anchor. (The anchor is not 0 because the reference move sometimes
flips that very square — which is exactly why the anchor has to be measured rather
than assumed.) **This is the recolor reading, not the no-op reading, by a 0.64
margin.** Placing on an opponent's square is represented as *capturing* it.

Two checks on that verdict:

- **It is not parity confusion.** A frame-flipped branch would produce the same
  observation. Restricted to the 1999 events whose decoded parity is confirmed
  aligned at *r* = 0, the number is **0.7864** — unchanged from 0.7860.
- **It washes out fast.** The excess over the anchor falls +0.636 (*r*=0) → +0.249
  (*r*=8) → +0.112 (*r*=16) → +0.042 (*r*=24), and the median disagreement run is
  2 plies. So `occupied_opp` recolors, but unlike the four empty-square categories
  the recolor does not persist: the model repairs a wrong colour far faster than it
  removes a disc that should not exist.

## By square class, at *r* = 0

Accuracy of each branch's decoded board against the reference board, per class
(reference-branch column is the same measurement on the same squares, so the
comparison is within one forward pass):

| square class | `occupied_own` | `occupied_opp` | `empty_detached` | `empty_own_only` | `empty_no_bracket` | `opponent_only` | reference branch |
|---|---|---|---|---|---|---|---|
| the reference move's square | 0.002 | 0.001 | 0.005 | 0.005 | 0.004 | 0.002 | 0.990 |
| discs the reference move flipped | 0.052 | 0.201 | 0.073 | 0.062 | 0.078 | 0.047 | 0.990 |
| untouched, theirs | 0.979 | 0.986 | 0.862 | 0.836 | 0.947 | 0.917 | 0.990 |
| untouched, mine | 0.923 | 0.887 | 0.930 | 0.941 | 0.918 | 0.953 | 0.988 |
| empty | 0.996 | 0.998 | 0.971 | 0.968 | 0.988 | 0.985 | 0.999 |

Two things fall out of the control class:

- The mirror image of the headline holds: the illegal branch does **not** place the
  reference's disc (0.001–0.005 vs 0.990) and does **not** flip what the reference
  move flipped (0.05–0.20 vs 0.990). Its board is the branch's own move, cleanly.
- The untouched discs — the control that rides along in the same forward pass —
  are **not** left alone. They drop from 0.990/0.988 to as low as 0.836. The drop
  is larger on `untouched, theirs` for the empty-square categories that place a
  disc into a populated neighbourhood (`empty_own_only` 0.836, `empty_detached`
  0.862) than for `occupied_own`, which changes nothing (0.979). Reading that as
  the model computing flip consequences of the illegal disc is the natural story,
  but it is an inference from an accuracy drop, not a measurement of which squares
  flipped — A4/A5 can settle it.

## F7

`figures/f7_probe_readout.png` (`--camera-ready` → vector PDF, titles dropped for
the caption). Three event-locked panels on the *r* ∈ [−8, +24] window, mean with a
95% bootstrap CI over source games (1000 resamples, band semantics and per-curve n
printed in the panel), pre-*t0* region shaded and annotated as fixed by
construction, dashed grey line at *r* = 0, rule key under the panels.

- **(a)** decoded occupancy of the illegally played square, four empty-square
  categories, with **both anchors labelled**: the reference branch at the same
  square (floor — nothing was played there) and the reference branch at its own
  played square (ceiling — a disc that really was placed). `control_legal` takes
  `NULL` styling and coincides with the floor anchor to within 0.002; its legend
  entry says so, since two curves on top of each other otherwise read as one.
- **(b)** the occupied pair: P(decodes the mover's colour) for the illegal branch
  (solid) and for the reference branch at the same square (dotted, same hue). The
  *r* = 0 gap is annotated with its size.
- **(c)** decoded parity per category plus both legal branches, on a 0.955–1.004
  axis because nothing leaves it.

Caption, first sentence: *An illegal move puts its disc on OthelloGPT's decoded
board and leaves it there for a median of 13–18 plies; a move onto an opponent's
square is decoded as capturing it, not as a no-op.*

## For the integrator

1. **I added `decode.py` next to `make_figures.py`.** My assignment listed
   `make_figures.py`, `figures/`, `RESULTS.md` and `results/violation_divergence/probe_*.csv`.
   Folding a 4-minute probe pass over 938 MB of activations into the plotting script
   would make every figure tweak re-run it, so the measurement is its own entry
   point, matching `src/spade/bracket_flip/probe_space.py` → `analysis/bracket_flip/make_figures.py`.
   Say the word and I will merge them.
2. **`paper_style` request: a branch-role style.** F7 uses *hue* for the category
   and *line style* for which branch is being read at that square (solid = the
   illegal branch, dotted = the reference branch at the same square). That is the
   same shape as the existing `BASIS_STYLE`, and A3's F1/F6 will want it too. I
   worked around it with literal dash patterns local to `make_figures.py`; the
   entry belongs in `paper_style.py`, which only you extend. Suggested:
   `BRANCH_ROLE_STYLE = {"illegal": {"ls": "-"}, "reference_at_square": {"ls": (0, (1, 1.4))}}`.
3. **The probe is gitignored** and lives only in the main checkout, so `decode.py`
   defaults `--probe` to `repo_root()/analysis/linear_probe/activations/linear_probe.pt`
   and a worktree run must pass the main checkout's path. Nothing else needs a flag.
4. **`--smoke` writes `probe_smoke_*.csv`**, so a smoke run cannot overwrite the
   committed full-set CSVs. Delete those before merging if any are lying around.
5. Nothing outside my assigned paths was touched.
