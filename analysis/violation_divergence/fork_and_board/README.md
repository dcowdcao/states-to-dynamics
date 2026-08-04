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

```
python analysis/violation_divergence/fork_and_board/make_figures.py \
    [--game 1235] [--embedding pca|isomap] [--camera-ready]
```

Both panels draw the same game, which is what makes the pairing mean anything.
The default is **game 1235** (*t0* = 20); `--game` asserts the game it is given
carries all nine branch types rather than silently drawing fewer.

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
(`python -m spade.violation_divergence.rule_groups`). A rule's members travel
barely closer to each other than to another rule's. Three confident lines would
be a lie about that, so each rule's members are drawn **faint behind their mean**
and the ratio rides in the legend title: the members, not the means, are the
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
point at where that null lives, and its legend title does: `../game_trajectory`,
which draws all nine branches including control 2 and pairs them with the δ̂
marginal, and `../README.md`, which carries the legal-vs-legal numbers with CIs.

Dropping the row also drops it from the PCA fit, since the plane is fit on what
is drawn. Cells shift slightly against the nine-branch version; nothing in them
is comparable across cells anyway.

## Axes

- Each fork cell: `PC1 (x% var)` / `PC2 (y% var)` per §4.6 — **per component,
  never summed**, and per panel, since the plane is refit per space. Isomap cells
  read `arb. units`. The two plane axes carry **three sparse ticks, symmetric
  about zero** so the origin — where the reference sits in a displacement plane —
  always gets one. They are a within-panel scale bar and nothing else: the plane
  is refit per space, so the legend title's "NOT comparable across the grid"
  disclaimer is what keeps them honest (§4.6, §4.8). The ordered axis keeps its
  ticks and reads `move index t`, once per row.
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
- **Three lines are six categories.** The rule means are a labelling
  convenience, not a demonstration that the rules are the natural unit; the
  spread ratio near 1 says the members barely cohere. Read the faint member
  lines. If a claim ever rests on a rule, it has to be scored per category and
  aggregated with a CI, in `../README.md`, not read off these means.
- **Only one control is drawn.** The legal-vs-legal scale is in
  `../game_trajectory` and `../README.md`; this figure names them rather than
  showing one.
- The view angle is fixed at `elev=18, azim=−58` and stated on the figure, so a
  favourable camera cannot pass as a result.
