# The fork and the board it forks on

One figure for the whole construction. Panel **(b)** is what the nine branches
*are* — the Othello position at the end of the shared prefix, with each branch
move on the square it plays. Panel **(a)** is where those same nine branches
*go* — one 3-D fork per representation space, raw layer 5 first. Before this,
the two halves lived in separate figures and a reader had no way to connect a
coloured trajectory to an actual move.

```
python analysis/violation_divergence/fork_and_board/make_figures.py \
    [--game 1235] [--embedding pca|isomap] [--camera-ready]
```

Both panels draw the same game, which is what makes the pairing mean anything.
The default is **game 1235** (*t0* = 20); `--game` asserts the game it is given
carries all nine branch types rather than silently drawing eight.

## Game 1235 is selected, and the figure says so

Game 1235 is the **argmax**, over all 1550 branch-complete games, of the mutual
separation of its six illegal displacement trajectories — scored in raw layer 5
in the full 512-dimensional space, never in an SAE space or a 2-D projection
(`../game_selection`). Population median 0.705; this game 1.080.

That is load-bearing, not boilerplate. On this game the reference and both
controls separate from all six illegal branches **in all nine spaces**, which
looks like a result. Over 40 games that ordering holds in only 0.07–0.17 of
them, against a 0.036 chance floor. The selected game displays structure that
does not generalise, so the disclosure rides in the legend-title block on the
figure itself (`branches.selection_note`) and not only here: a reader who does
not know the game was chosen will over-read the panels. A different `--game`
swaps the note for one saying that game was passed explicitly.

## Layout

The board is square and the fork grid is three cells wide, so a naive stack
would leave a band of white beside the board. That band carries the figure's
apparatus instead: the **one shared legend** — both panels take colour, marker,
line style and label from `spade.figures.branch_style`, so a second legend would
say the same thing twice — the board's reading key, and the R1/R2/R3 rule gloss
(FIGURE_GUIDELINES §5: shorthand in a legend earns a key inside the figure).

Panel letters sit top-left of their region; the working PNG carries an in-figure
title and `--camera-ready` drops it and emits vector PDF (§2, §6).

## What is plotted

Panel (a) is `spade.violation_divergence.fork_plane`'s embedding, drawn by
`spade.figures.branch_panels.fork_panel`. Move index is the third axis, so the
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
line style, is what identifies the square.

The **two legal controls share the null styling** — grey, dashed, last in the
legend — and on this board they land on adjacent squares (h3, h4), where two
identical grey marks cannot be told apart. Control 2, the null *for* control 1,
therefore takes an open marker. Same colour, same shape, different fill, so the
pair separates without reading legend text and survives grayscale.

## Axes

- Each fork cell: `PC1 (x% var)` / `PC2 (y% var)` per §4.6 — **per component,
  never summed**, and per panel, since the plane is refit per space. Isomap cells
  read `arb. units`. The two plane axes have their ticks off (the coordinates
  mean nothing, the geometry does); the ordered axis keeps its ticks and reads
  `move index t`, once per row.
- Board: `file` / `rank (1 at the top)`.

## Reading it honestly

- **Planes are refit per space.** Cells are comparable in the *shape* of the
  fork and never in position or magnitude; SAE latent spaces live on different
  scales. The legend title says so on the figure itself.
- **One game is n = 1, and a selected one.** See the section above: the panels
  look cleaner than a typical fork does, by construction. This is an
  illustration of the design, not a claim-supporting figure. Every number — the
  0.978 ratio, the 0.4985 legal-vs-legal null, the plateau — belongs to
  `../README.md`, where it carries its CI over 13,531 branches.
- The view angle is fixed at `elev=18, azim=−58` and stated on the figure, so a
  favourable camera cannot pass as a result.
