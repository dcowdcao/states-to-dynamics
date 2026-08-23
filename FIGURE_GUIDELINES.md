# FIGURE_GUIDELINES.md

House rules for every figure in this repo. Read before writing plotting code or
fixing an existing figure. Derived from NeurIPS/ICLR/ICML author instructions,
the mech-interp/SAE literature (Gemma Scope, SAEBench, Cunningham, Li et al.),
and the representation-geometry literature (Hénaff 2019/2021, Ansuini 2019,
Hosseini & Fedorenko 2023, ReStraV). Sources at the bottom.

Two audiences, one pipeline: figures start as **working artifacts** (PNG, titles
on, read in a README) and get promoted to **camera-ready** (vector PDF, titles
moved into the caption). Write the plotting code so promotion is a flag, not a
rewrite.

---

## 1. Non-negotiables

1. **Never encode by color alone.** Every series carries color **and** a marker
   shape (and line style when there are >3). ICML requires legibility in
   black/white; ~1 in 12 men has a CVD. Print the figure in grayscale before
   shipping it.
2. **Every claim-supporting curve carries uncertainty**, and the caption says
   what the band is (SD / SEM / 95% CI / IQR / bootstrap), over what source of
   variation, and n. A bare point estimate is a NeurIPS checklist Q7 violation.
   This is the field's weakest spot — SAEBench, Gemma Scope, and Gated SAEs ship
   no error bars at all — so it is cheap to be better than the literature here.
3. **Figure text is never smaller than the caption text** (≥9 pt at final printed
   size). Draw at final width; never rescale in LaTeX.
4. **Axes are labeled with units; every series is in the legend.** Say what
   "good" means when the axis direction is not obvious.
5. **No color literals in plotting scripts.** Import from
   `src/figures/paper_style.py`. One dict, one import, so a method is the same
   color in every figure. (SAEBench's own code gives `batch_topk` and `gated`
   the same purple — that is the failure this rule prevents.)
6. **Bar charts start at zero; never a dual y-axis.** Two scales → two panels.
7. **A mean over a group of series carries n and its members.** When several
   conditions collapse into one line because they share a label (six illegal
   categories into three rules), the legend entry carries n — `R2 bracketing
   (mean of 3)`, and `(mean of 1)` when the group has one member, so nobody
   reads a single condition as an average. Draw the members faint behind their
   mean, and report within-group ÷ between-group spread **on the figure**. Near
   1 the grouping is a label rather than a finding, and the figure says so
   instead of letting three confident lines imply otherwise. **When the drawn
   mean is itself over a population of items, drop the member traces**: a member
   there is another mean over the same items, so faint behind the group mean it
   reads as item-to-item spread, which is what the interval already carries. The
   spread ratio then goes on the figure as a median with IQR over items, not as
   one number. When the group
   aggregate is a **sum** of its members rather than a mean — probability mass
   over a rule's categories, a count — the legend says `sum of 3`, not
   `mean of 3`: the two have different units and only one of them is bounded by
   its members.

## 2. House style

```python
from figures.paper_style import apply_style, CLASS_STYLE, REFERENCE, NULL
apply_style()                 # working PNG
apply_style(camera_ready=True)  # vector PDF, no in-figure titles
```

`paper_style.py` owns the rcParams (Okabe–Ito cycle, `pdf.fonttype=42`, spines
off, 0.5 pt minimum line weight), the per-class color/marker dicts, and the
reference/null styling. Change it there, not per figure.

**Sizing.** Draw at the width it will be printed:

| target | width (in) |
|---|---|
| ICML single column | 3.25 |
| ICML full width (`figure*`) | 6.75 |
| NeurIPS / ICLR full width | 5.5 |
| NeurIPS / ICLR half width | 2.65 |
| repo README / working figure | 9.0 (2 panels) |

Font-to-width ratio ≈ 2.5–3 pt per inch for a two-column figure. A 12 pt label
on a 14-inch canvas renders illegibly once shrunk to 3.25 in.

**Palette (Okabe–Ito, the set ICML's accessibility guide links to):**
`#000000` black · `#E69F00` orange · `#56B4E9` sky blue · `#009E73` bluish green
· `#F0E442` yellow · `#0072B2` blue · `#D55E00` vermillion · `#CC79A7` reddish
purple. Never `jet`/rainbow. Sequential = one hue light→dark (`viridis` for
colorbars); ordered variables (width, layer, tokens) get a sequential ramp, not
categorical hues.

## 3. Baselines, nulls, controls

The rule that matters most in this repo, since nearly every study is
"condition vs. reference":

- **The reference class must look like a reference**, not like another
  condition: black or dark gray, listed first in the legend, labeled
  `... (baseline)`.
- **The null carries the signal's estimator.** One value per item in the signal
  means one value per item in the null: a per-item mean or median over k
  candidates has ~1/√k the spread and manufactures a separation the data does
  not have. Pick one candidate per item at a stated seed, and if an aggregate
  null is unavoidable, say so and compare the two spreads on the figure.
- **A null gets gray + dashed + low zorder + last in the legend**, and is *never*
  given a slot in the categorical cycle. Prefer an **empirical null** (shuffle
  labels, or score the reference against itself) over an analytic chance line;
  draw both if you have them.
- **Same measurement on null/random data → same color, dashed.** A *different
  quantity* → different color. (Ansuini: trained solid black, random-weights
  dashed black, PCA red.)
- **A ratio to a per-item reference is only readable while the two are paired.**
  Dividing by the reference's value on the *same* item keeps numerator and
  denominator correlated; when a variant breaks that (each item resampled on its
  own), the convexity of 1/x lifts every curve above 1 — the null included — and
  the level stops being a finding. Draw the null through every offset and assert
  the paired contrast, not the level, whenever the pairing is not exact.
- Chance/threshold lines are thin, dashed, gray, drawn behind data, and
  **labeled in-panel or in the legend** — never a bare unexplained line.
- An untrained/randomized-model control belongs **on the plot as a gray band**,
  not buried in a table.

## 4. Figure types

### 4.1 Depth / layer sweeps

- x = layer index, categorical. **State the indexing base in the caption**
  ("layers 0–7, where 0 is the first transformer block") — off-by-one confusion
  between "layer 0 = embedding" and "layer 0 = first block" is endemic.
- **Anchor the axis at the input.** The embedding/input space is the leftmost
  tick, separated from the network ticks by a thin rule or gap (Hénaff's
  `Pixel | LGN V1 | Perception`). This is rare in mech-interp and it is exactly
  what licenses the claim "the effect is computed, not inherited from the input."
- ≤5 series per panel; beyond that, facet or use a heatmap (32 lines is not a
  plot).
- Few layers → markers + error bars. Many layers → line + shaded band.
- When the absolute value is estimator-dominated, plot **Δ relative to the
  input** on an axis symmetric about 0, with a zero line.
- Patching / recovery sweeps: **draw and label both anchors** — the untouched
  corrupted run (0) and the ceiling (1, which a full-residual patch hits *by
  construction* when clean and corrupted share a prefix). An unlabeled line at
  1.0 reads as a target the method failed to reach rather than a sanity check.
- Comparing models of different depth → normalize to fractional depth ∈ [0,1].

### 4.2 Event-locked (peri-event) plots

For anything aligned to an intervention at position `t0`:

- x label is `position/time relative to <event>`; zero is the event.
- **Vertical line at 0: thin, dashed, gray, behind the data.** Solid black reads
  as data. If the event has duration, shade the window instead.
- **Shade the baseline/pre-event window in gray and state it numerically** in
  the caption. Show enough pre-event range (25–50% of the window) to establish
  the baseline.
- Horizontal dashed line at the baseline value.
- Mean + band; **say what the band is on the figure**, not only in the caption.
- **If a region is zero by construction, shade it and say so in-panel.** A flat
  segment with no annotation reads as a smoothing artifact or a late effect;
  annotated, it is proof the alignment and matching are correct.
- Annotate the extremum with its value and offset.
- **A quantity whose scale differs across panels goes on the axis as a paired
  ratio to the reference, never raw.** Divide by the reference item's value at
  the *same* position of the *same* item, so the division cancels the shared
  per-position trend and not merely the units — an unpaired normalization
  leaves that trend on every curve, where it is read as the response. The
  reference is then 1 by construction and the panel says so; the line at 1 is
  the reference class, styled as one and named in the legend, never a bare
  chance line; the null is the control's *own* ratio, which is what says
  whether an excursion is large. The reference's raw scale per panel goes in
  the caption and the absolute values in the CSV, or the division has thrown
  data away. Name the contrast on the axis label, not only in prose.
- **An ordering against the null is a paired contrast, not two point
  estimates.** When the signal and the null are measured on the same items, what
  the figure asserts is `signal − null` per item: the difference cancels the
  shared prefix and the common divisor, and its interval can be much tighter *or*
  much wider than either marginal one. Draw it, or print it in-panel with its
  interval and the offset it is taken at — a fixed offset, since an argmax over
  the window is a different and unstated statistic. Two error bars that happen
  not to overlap are not a measured ordering; reading one off a panel is a
  mistake this study has had to correct twice.
- **A variant of an existing figure keeps that figure's axis**, unless the axis
  is what changed. Two figures a reader is meant to set side by side — the same
  quantity over a different branch set, corpus or seed — share x range, y range,
  panel grid and note block, so what differs on the page is the data. Truncate a
  variant's window only when the extra offsets draw a separation that has already
  closed, which is a property of *cumulative* quantities (a distance from a
  reference saturates; a per-position rate does not), and say which case it is.
  The reduction the variant adds — an average over resampled continuations,
  seeds, crops — is part of the estimator and goes in the note block.
- Overlay conditions on shared axes; never split them into panels when the
  point is the difference.
- Standard layout when items are inspectable: per-item heatmap on top,
  condition-average with band below, x-axes aligned.

### 4.3 Pareto / trade-off (SAE)

- x = `$L_0$ (Sparsity)`, **log scale**; y = `Loss Recovered (Fidelity)` or
  `Δ LM Loss (nats)`.
- **The caption states which corner wins** — the single best sentence in this
  literature is Gemma Scope's *"An ideal SAE should have low delta loss and low
  L0, i.e. correspond to a point towards the bottom-left corner."* Copy the
  pattern.
- Scatter `s=100, edgecolor="black", zorder=2` over a faint L0-sorted connector
  `alpha=0.5, zorder=1`.
- 3-variable form: marker = architecture, fill color = third metric via
  `viridis` + colorbar, legend keyed by marker shape only.
- Color = method family, marker = variant. Hero treatment (larger, outlined,
  `zorder` on top) for our own method is legitimate as long as the axes are not
  rigged.

### 4.4 Distributions and angles

- **Curvature/angles in degrees**, range 0–180; Δ-curvature signed on a
  symmetric axis. No circular statistics needed for unsigned angles in [0,180).
- Skewed/bounded → **median + IQR**, not mean ± SD. Mark the median with a
  triangle and put `median = X°, IQR = Y°` in the caption.
- One condition vs. one null → **mirrored (back-to-back) histograms**, null in
  gray pointing down. Many samples, two classes → overlapping histograms.
- **Three or more classes on one axis → KDE curves, not step histograms.**
  Stacked step outlines read as bin noise and the class you care about gets lost
  in the staircase. State the kernel and the bandwidth rule (Scott/Silverman,
  with the resulting h in data units) in the caption — the bandwidth is a
  reproduction-critical hyperparameter, and an unstated one invites the "you
  smoothed away the second mode" objection. Carry a bootstrap band over items:
  resample under a **fixed full-sample bandwidth**, so the band shows sampling
  noise rather than the bandwidth chasing each resample's spread. Markers go on
  the body of the curve, never the flat tails.
- **An alignment density shows direction, never magnitude.** cos(Δ, v) positive
  on every item is compatible with v explaining a percent of ‖Δ‖². When the
  claim is "this is *the* direction", the panel carries the share of ‖Δ‖² along
  v — with its CI, its n, and the same null — as an inset bar beside the
  density, and names what the rest of the displacement is.
- Items × conditions matrix → diverging colormap centered at zero with **both
  poles named in the caption** ("blue = straightening, red = entangling"), plus
  the two marginals as error-bar plots.

### 4.5 Detection metrics (AUROC)

- **Do not flip sub-0.5 AUROCs.** In a geometry study the *direction* is the
  finding, not a nuisance. Keep the full 0→1 axis, symmetric about 0.5, and
  annotate both directions in-panel (`↑ curvier` / `↓ straighter`).
- Define the orientation once, unambiguously, in the caption: "AUROC of the mean
  turning angle for class C scored against legal; >0.5 means C is curvier."
- Chance line labeled; empirical null band preferred.
- **Bootstrap CIs over items** (percentile, ≥1000 resamples). Without them a
  reader cannot judge 0.52 vs 0.50.
- Full ROC curves belong in the appendix; AUROC numbers in tables; AUROC-vs-x
  in the main text when the claim is about how it varies.
- **A bar cannot carry a line style, so translate it to a hatch.** `SPACE_STYLE`
  separates the two dictionaries of an architecture family by dash pattern; drawn
  as bars that dimension vanishes and a family reads as one duplicated colour.
  Hatch the variant, keep the hue for the family, and print the mapping as a
  one-line in-figure key -- the same information, and it survives grayscale.
- **Sort the bars when the ordering is the claim**, rather than holding a fixed
  table order. The reference class keeps its reference styling wherever it lands,
  so the reader can count how many conditions clear it; that is the whole point of
  sorting, and pinning the reference to the first slot hides it.

### 4.6 Embedding scatter (PCA / Isomap)

Tens of thousands of points, several classes, arbitrary coordinates — the panel
that most easily becomes a color-only blob.

- **Draw the cloud once with the point order shuffled.** A per-class loop paints
  the last class on top and the reader sees a fake separation; a single shuffled
  scatter is the only honest overlap.
- Class identity then rides on a **shape-coded centroid plus a 50% probability
  contour** per class, drawn over the cloud. Those two survive grayscale, which
  a color-only cloud does not. Say what the contour is — *dispersion of a
  Gaussian fit, not a confidence interval* — in the legend title, and put n per
  class in the legend entry.
- **Axes carry the per-component variance ratio** (`PC1 (5.5% var)`), never the
  summed total, and nonlinear embeddings are labeled `arb. units`.
- **Three sparse ticks per embedding axis, symmetric about zero**
  (`MaxNLocator(3, symmetric=True)`, ≤7 pt, negative tick pad so they never
  reach the axis label). They are a within-panel scale bar and nothing more —
  the coordinates still mean nothing, the geometry does — so a panel carrying
  them **must** carry the "not comparable across cells" disclaimer too. This
  reverses a ticks-off rule: knowing how big a structure is inside one panel is
  worth more than the ambiguity the numbers cost, and the numbers make the
  disclaimer load-bearing rather than optional.
- **The dominant nuisance variable gets its own panel** on the same embedding,
  colored by a sequential ramp (here: move number t). A class layout that
  reproduces the nuisance panel is a confound, not a finding — the panel is what
  lets a reader check that, so it ships even when it is uninformative.
- Ordered sub-classes take a **light→dark ramp of the parent class's hue**;
  unordered classes take separate hues. Marker shape codes the parent class.
- Dense clouds: `rasterized=True` on the scatter artist, so the PDF keeps vector
  text and axes over a raster cloud.
- **A grid of the same embedding across spaces** (one panel per SAE / layer /
  model) refits PCA per panel, so absolute positions, axis orientation and
  variance ratios are panel-local — **state in the legend or suptitle that
  panels are not comparable point-for-point, only in within-panel arrangement.**
  Keep the class styling identical across panels (one legend, on the reference
  panel) and lead with the reference space so the eye calibrates on it. SAE
  latent spaces live on different scales — SpaDE codes on a simplex — so compare
  the *structure* of the class layout, never coordinate magnitudes.

### 4.7 Displacement fields (paired before→after arrows)

For a paired change drawn in a space, one arrow per item, tail = before,
head = after.

- **Say what the pairing is in-panel**, not only in the caption: "arrow = same
  square, before → after the flanking move". An unexplained arrow field is read
  as a flow, which it is not.
- **Arrows cannot be the only cue.** Put the class marker at the head and keep
  arrow color and marker consistent with the rest of the study. Thin shafts
  (≈0.8 pt), small heads; the head is data, the shaft is a connector.
- **Never draw every item across every event.** Aggregate (mean displacement per
  class with a spread contour) or facet, and say which you did. A raw quiver of
  10⁴ arrows is a smear.
- **The items that should not move are the control** and belong on the same
  panel, drawn in the reference style, so a near-zero arrow is visibly
  near-zero against neighbors that moved.
- **Carry the marginal.** Overlapping arrows hide magnitude, so pair the field
  with a distribution of displacement lengths per class (§4.4), which is also
  where the uncertainty and n go.
- **The interval belongs on the displacement, not on the head.** An arrow asserts
  a change, the change is paired within item, so the bar is a bootstrap of
  (after − before) over items, drawn at the head, and the legend says which
  contrast it is. An endpoint interval is not an interval on the arrow, and it is
  not the conservative choice either: where the two conditions are anticorrelated
  across items — any frame that flips with the event — the paired interval is
  *wider*. In `bracket_flip` it is 1.5–2× wider on the owner axis and 1.2–2.5×
  narrower on the occupancy axis, off the same 1,596 events.
- **Two panels, two measurements.** A second panel that is a deterministic relabel
  of the first is not independent evidence. Draw the measurement the relabel
  stands in for — a probe actually trained in the other convention — and put it on
  axes shared with the first, so a collapse shows as a collapse instead of being
  rescaled into apparent structure.
- State the space's units, and whether the axes are a learned basis. If the
  space is low-dimensional by construction (a 2-D probe filler space), say so —
  it is the difference between an honest plot and a projection.

### 4.8 Trajectories with an ordered third axis (2-D embedding × time)

A path through a learned 2-D plane, drawn against an ordered variable (move
index, layer, epoch) on a third axis. 3-D is normally a mistake — it occludes,
it cannot be read quantitatively, and it usually encodes a third *data*
dimension that a facet would carry better. It earns its place only under these
conditions:

- **The third axis is an ordered index, never a third data dimension.** If you
  are tempted to put a third principal component there, use a facet or a second
  panel instead. Label it with units like any other axis.
- **Take the ordered variable out of the plane first.** Move index dominates
  activation geometry, so a PCA of raw activations puts it on PC1 and the figure
  collapses to a slanted ribbon that encodes time twice. Embed a
  reference-subtracted or per-index-centred quantity, and say which. Doing this
  is also what discharges §4.6's nuisance-variable panel: the nuisance is
  displayed on its own axis rather than projected away.
- **Center the embedding on a meaningful origin, or state there isn't one.**
  Mean-centring a displacement moves the zero off the origin and separates the
  reference from the marks drawn there. PCA of a displacement should be
  uncentred; Isomap has no origin at all, so say so rather than implying one.
- **Ship a linear and a nonlinear embedding** when the claim is about whether
  classes separate — PCA alone cannot distinguish "not separated" from "not
  separated *linearly*". Isomap axes are `arb. units` with no variance ratio.
  Exact-duplicate rows (a reference that is zero by construction) form a
  disconnected clique in the kNN graph: fit on the non-degenerate rows and
  transform the rest, and record that in the docstring.
- **Fix and state the viewing angle** (`elev`, `azim`). An unstated camera makes
  the figure irreproducible and lets a favourable angle pass as a result.
- **Pair it with a quantitative companion.** A 3-D projection shows shape, never
  magnitude; the companion carries the number the claim rests on, with the usual
  uncertainty and n. A 2-D panel beside it is the default, and when that number is
  a ratio to a per-cell control, draw the panel on a log ratio ladder rather than
  as bars from zero (§1.6): a ratio's null is 1, and two series that differ
  several-fold — a distance in the drawn plane and the same distance in the full
  space — share one linear axis only by flattening the smaller onto the null line.
  A **table** — in the caption, or in the study README the caption points at — is
  allowed instead, and is the better answer when the panel would be a second
  estimator the reader is invited to compare with the projection by eye and cannot
  (`fork_population`: the projection's paths cancel to 20% of a game's
  displacement, the companion's numbers cancel nothing, and no shared axis makes
  the two commensurable). Two conditions if you take the table: the projection must
  say on its own face that it carries no magnitude, and the caption must name where
  the numbers are. Dropping the companion is never allowed.
- **Sparse ticks on the plane axes, full ticks on the ordered one.** Three per
  plane axis per §4.6, symmetric about zero so the origin — which in a
  displacement plane *is* the reference — always gets one. Per-panel refits are
  not comparable across panels: say so in the legend title when facetting, and
  say it whether or not the ticks are there.
- **Do not put the interval on a path that is a mean over items; put it in the
  companion.** If you draw one at all it is an *ellipse* and never a 1-D bar — the
  estimator is a two-dimensional mean and has no preferred direction, so a bar is
  a choice of axis rather than a measurement (covariance of the items' projected
  positions ÷ n at χ²₂(0.95), flat in the plane, one per path at its end, named in
  the legend as a CI of the mean, and none at all on a series that is zero by
  construction). But under a fixed 3-D camera that ellipse lies flat in the plane
  and foreshortens into an unreadable lens, and at the n where it is worth drawing
  it is about as wide as the path is long, so the ellipses stack and occlude the
  paths they belong to. That is not a tuning problem. Say in numbers, in the
  caption, that the mean direction is unresolved, and let the companion carry the
  interval. §4.6's 50% dispersion contour is a different object and the two must
  never be read for each other.
- **A mean of displacement *vectors* cancels, so print the share that survives.**
  Averaging a displacement over items (games, categories) is not averaging their
  distances: components that differ between items subtract, and the mean path is
  shorter than every item's. Unannotated, a short mean fork reads as "nothing
  moved", which is the opposite of what a mean of per-item distances reports. Put
  ‖mean Δ‖ ÷ mean ‖Δ‖ on each panel and pair the projection with the
  distance-based panel, so the two estimators are visibly two.
- Sub-sample an oscillating index rather than drawing every step. A quantity
  whose frame alternates (here: a mover-relative board code flipping each ply)
  draws as a sawtooth that hides everything else; plot one phase and say which.

### 4.9 Board panels (a discrete state, drawn where it lives)

- **One renderer owns the board.** `spade.figures.othello_board.draw_board` sets
  the grid, the algebraic axes and the discs; a panel that draws its own will
  drift out of register with every other board in the paper, and the same square
  must land in the same place everywhere. Overlays go on top of it; shading that
  belongs *under* the pieces is drawn before it.
- **Let the pieces draw themselves.** Absolute colour — a black disc is a filled
  black circle, a white one an open circle — so the board needs no legend for its
  own contents and is unchanged in grayscale.
- **Never cover the evidence.** The disc sitting on a square is often the whole
  reason the square is marked. Mark with an outline on the cell edge and a marker
  inset into a corner, never a fill or a centred marker.
- **A marker hue has to survive on a black disc, a white disc and the empty
  cell.** Mid-luminance Okabe–Ito with a white marker edge. Pure black is
  unavailable: it is already the disc colour and the disc edge.
- **Draw the rule, not just the square.** A marked square with no geometry asserts
  something the reader has to take on faith. Draw the line that makes the move
  legal, or the line that fails to close, and mark the square where it fails.
- **State the side to move, in the axes.** Any legality that depends on whose turn
  it is cannot be read without it, and a title does not survive camera-ready
  (§5), so it does not belong in one.
- **Name squares algebraically** in the label, so a claim in the prose can be
  checked against the picture square by square.
- One marked square needs no marker and no legend: the outline and its name carry
  it, and a spare marker on a board reads as another piece.
- **Two boards of the same position are one figure, not two.** The claim is "same
  board, one square different", and a reader holding a board in memory across two
  files cannot check it. Panel them side by side under one key; each panel still
  states its own side to move, because a panel gets read on its own.

### 4.10 Direction geometry (unit vectors: similarity matrices, PCA of directions)

When the objects plotted are probe *directions* rather than a sample of
activations, the failure mode is that a near-zero cosine reads as a finding when
it is what unrelated vectors in a high-dimensional space do by default.

- **Only contrasts are directions.** A k-way softmax head is unchanged by adding
  one vector to every class weight, so a raw class column is not a point in
  activation space and its cosine with anything is a property of the fit. Plot
  differences of class weights, and name the contrast in the legend.
- **Put the random-direction null on the same axes.** Unit vectors in d
  dimensions meet at E|cos| = sqrt(2/(pi d)) — 0.035 at d = 512. Prefer the
  empirical version (the same number of sampled vectors, or the probe refit on
  shuffled labels) per §3; without it every 0.05 cell invites over-reading.
- **Say what an uninformative cosine does not rule out.** Two directions can be
  near-orthogonal while one quantity is an exact nonlinear function of the
  other. The panel bounds *linear* alignment and nothing else, and the caption
  has to say so.
- **A similarity matrix is symmetric: draw one triangle**, order rows from the
  taxonomy constant rather than by clustering (so the same matrix is comparable
  across spaces), and follow §4.4 — diverging map centered at zero, symmetric
  limits, both poles named. Annotate the cells when the matrix is small enough
  to carry numbers: a heatmap is colour-only encoding, and grayscale sends +0.6
  and −0.6 to the same ink.
- **Uncertainty on a direction is a refit, not a resample of the cells.** The
  cells are deterministic given the weights, so bootstrap the probe's training
  **games** and refit. When the point estimate comes from a committed probe and
  the interval from a cheaper estimator, report the agreement between the two.
- **PCA of unit vectors is uncentred** — the zero vector is a real point (§4.8)
  and the mean direction is itself a result. Per-component variance ratios as in
  §4.6. A joint fit is dominated by whichever family has the most members (64
  squares against 3 rule contrasts), so when the question is "does B lie in A's
  subspace", fit on A and project B in.
- **Pair the scatter with the projection it is a picture of.** A 2-D loading
  plot cannot show that a direction keeps 4% of its mass in the plane; the
  companion panel is the captured-fraction curve against the same null.
- **When the directions are near-orthogonal, drop the variance panel for an
  ordinal one.** A cloud whose pairwise angles sit in a narrow band around 90°
  has no dominant plane, and a variance ratio in the axis label invites the
  reader to ask why it is 4%. Embed the *angular* distance θ = arccos(cos) —
  a metric, which 1 − cos is not — with nonmetric (ordinal) MDS, which fits only
  the rank order of the angles. Label the axes `arb. units`, report **rank
  fidelity** (Spearman between the true angles and the drawn distances, over all
  pairs) instead of a variance ratio, and say in the panel that distance is
  monotone in angle, not proportional to it.
- **An ordinal embedding ships with a structure-free null.** A plane can always
  render class *means*; that is most of what it renders. Rebuild the cloud
  keeping each family's mean direction and replacing the residuals with
  isotropic noise at the observed within-family spread, embed it the same way,
  and print its fidelity beside the real one. If the null scores at or above the
  real embedding — it does here — the panel is evidence about families and never
  about individual members, and the caption says so rather than leaving the
  reader to infer it.
- **Pin the orientation, and do not draw arrows in a space with no origin.**
  SMACOF, Isomap and UMAP fix a configuration only up to rigid motion, so a
  refit is free to rotate or mirror the panel; align the solution to a fixed
  basis (orthogonal Procrustes onto the PCA coordinates) so it cannot. For the
  same reason the committed probes are drawn as ringed markers among the cloud,
  not as arrows from a centre that means nothing.
- **Check the direction set for algebraic duplicates before embedding it, and
  score them out.** Two contrasts fit by ridge onto centred indicators over the
  same rows are exact negatives when their targets are (`1[y>0] = 1 − 1[y=0]`).
  Draw both only when the reflection is itself the point — two opposed poles of
  one axis is a real thing to show — and then say on the figure that it is an
  identity, not a measurement. Either way compute fidelity on the de-duplicated
  set: a 180° pair is the easiest thing a plane can render, so leaving it in
  inflates every score for free. Note also that antipodes cost the rest of the
  layout, since a plane cannot place many of them without distorting the radii
  around them.

## 5. Captions and panel labels

- **The caption's first sentence is the claim, not a description.** "Curvature
  reduction for natural image sequences", not "Curvature results".
- Then, per panel: what it shows, error-bar semantics, and n.
- **Reproduction-critical hyperparameters go in the caption** (λ, width, layer,
  seed count, n per class).
- Panel letters `(a)`, `(b)` bottom- or top-left of each axes, enumerated in one
  figure caption.
- **Domain shorthand in a legend earns a key — in the README, not in the
  figure.** If series are labeled with codes that live in a spec elsewhere
  (`R2 bracketing`, a category name, a condition ID), the figure carries no
  explanatory prose; the study README carries a `## Figure notes` section with
  one line per figure file, driven from the same constant the labels come from.
  This reverses an earlier rule that printed the key under the panels. Grey
  prose accumulated on every figure until it was competing with the data, and
  the same text had to be maintained in two places. The cost is real and worth
  naming: a figure lifted into a talk or a poster no longer explains its own
  codes, so whoever lifts it has to carry the README line with it.
- **Band semantics and n move with it.** §1.2 still requires them, so a removed
  block that stated a CI, an n, a bandwidth or a normalisation goes into the
  same `## Figure notes` entry verbatim — not paraphrased, not rounded.
- **A chance or threshold line keeps its label** (§3). If dropping prose would
  orphan one, move the label into the legend rather than deleting it; a bare
  unexplained line is worse than the prose was.
- **Panels that share a styling table share one legend.** When two panels of
  different kinds (a projection and a board, a curve and a matrix) draw the same
  classes from the same style dict, a second legend only repeats itself. Place
  it once, and place it where the layout already wastes room: a square panel
  beside a wide one leaves a band, and the shared legend plus the in-figure key
  are what belong there. An empty band is a layout bug, not neutral whitespace.
- **A standalone figure owns its whole key.** The rule above inverts when a grid
  is cut down to one panel: band semantics, n, seed and the meaning of any
  shading were living in the figure-level legend and die with it. Whatever the
  panel needs to be read alone belongs on its own axes, and it drops the panel
  letter along with the grid.
- Cross-reference between figures so points can be aligned across panels.
- Working figures may carry an in-figure title; **camera-ready must not** (ICML:
  "omit titles from inside the graphic file itself") — the title becomes the
  caption's first sentence.
- Every figure is referenced and interpreted in the text.

## 6. Output

| content | format |
|---|---|
| line / bar / scatter (<5k points) | **PDF** (vector) — ICML mandates it |
| heatmaps, sharp-line raster | PNG, ≥300 dpi |
| dense scatter (>10⁴ points) | PDF with `rasterized=True` on the artists |
| photos | JPEG |

`pdf.fonttype = 42` (TrueType). Matplotlib's default is Type 3, which NeurIPS's
"Type 1 or Embedded TrueType" rule rejects. Always `bbox_inches="tight"`.

## 7. Pre-commit checklist

- [ ] Colors + markers from `paper_style.py`; no literals
- [ ] Reference class styled as a reference; null gray/dashed/last
- [ ] Uncertainty drawn; band semantics + n stated
- [ ] Grouped means carry n, their members, and the within/between spread
- [ ] Axes labeled with units; direction of "good" stated if ambiguous
- [ ] Layer indexing base stated; input space anchored as its own tick
- [ ] Event-locked: dashed gray line at 0, baseline window shaded and named
- [ ] Legible at final size; grayscale-readable
- [ ] Vector PDF for camera-ready; titles moved to captions
- [ ] Figure referenced and interpreted in its README/paper section

## Sources

Conference: [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist)
(Q7 error bars) · [ICML author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
(vector mandate, label everything, ≥0.5 pt) · [ICML accessibility guide](https://icml.cc/Conferences/2024/AccessiblePapersAndTalks)
(figure font ≥ caption font; never color alone) · [ICLR formatting](https://iclr.cc/Conferences/2026/AuthorGuide).
Statistics: Henderson et al., *Deep RL that Matters* (arXiv:1709.06560);
Agarwal et al., *Statistical Precipice* (arXiv:2108.13264, IQM + stratified
bootstrap CIs). Color: [Okabe–Ito](https://jfly.uni-koeln.de/color/) ·
[Paul Tol](https://sronpersonalpages.nl/~pault/). Field conventions: Gemma Scope
(2408.05147), SAEBench (2503.09532, `graphing_utils.py`), Cunningham et al.
(2309.08600), Li et al. (2210.13382), Hénaff 2019 (Nat Neuro 22:984) / 2021
(Nat Comms 12:5982), Ansuini et al. (1905.12784), Hosseini & Fedorenko
(2311.04930), ReStraV (2507.00583). Design: Rougier et al., *Ten Simple Rules
for Better Figures* (PLOS CB 10:e1003833).
