# The branch board: what the nine futures are

The fork figure (`../game_trajectory`) draws nine trajectories in a projected
space. It never says what the nine branches *are*. This one does: the Othello
position at the end of the shared prefix, with every branch move drawn on the
square it plays.

**Superseded by `../fork_and_board`**, which draws this board as panel (b) under
the same nine branches in every representation space, with one legend for both.

```
python analysis/violation_divergence/branch_board/make_figures.py [--game 1235] [--camera-ready]
```

`--game` indexes `branches.pt`'s `game_id`; the default is the game the fork
figure draws, so the two figures show the same fork.

## What is plotted

The board is `replay(tokens[:t0])` — the position every branch shares, since the
branches are token-identical up to *t0* and differ only in the move played
there. Discs are absolute colour; the mover at *t0* is named under the panel,
which is what makes "own" and "opponent" readable.

Each branch square is **outlined in its own category's colour and line style**,
with the category's marker in the square's top-left corner. Colours, markers and
labels come from `spade.figures.branch_style`, the same module the fork figure
draws from, so a series means the same thing in both figures and the legends sort
identically (reference first, the two legal nulls last).

**Markers sit in the corner, never at the centre.** Two illegal categories,
`occupied_own` and `occupied_opp`, play onto a square that already holds a disc.
A centred marker would have to cover it — and covering the disc destroys the
evidence for the category, since the disc's colour *is* what makes the move
occupied-own rather than occupied-opp. One placement rule for all nine keeps the
grammar uniform; the outlined cell, not the marker's position, identifies the
square.

The discs the **played** move flips carry a dashed ink ring — drawn outside the
disc, so it is visible over black and white alike. Only the reference's flips are
shown: the two legal controls flip discs too, and drawing all three flip sets
turns the board into a knot of rings with no reader question behind it.

## Reading it

- The `R1 occupied (own)` cell holds a disc of the mover's colour and the
  `R1 occupied (opp)` cell holds one of the opponent's. That is the whole
  definition of the two categories, visible on the board.
- The `R2` cells are all empty: detached (touching nothing), own-only (touching
  only the mover's discs), no-bracket (touching opponent discs but closing no
  line). Ordered by how much of the board you must scan to rule the move out.
- `R3 opponent-only` is a square that *would* be legal if the other player were
  to move. Turn confusion, not geometry.
- The two grey dashed cells are legal moves the mover could have played. They are
  the study's nulls: whatever an illegal branch does downstream, a legal
  alternative branch is the scale it is measured against.
- Not every legal move is drawn. The legend states how many the position has;
  the branch set samples the played move and two others.

One game, n = 1. This is an illustration of the design, not a claim; the study's
numbers and their CIs live in `../README.md`.
