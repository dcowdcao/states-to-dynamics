# Figure audit — `bracket_flip`

Audited against `FIGURE_GUIDELINES.md` §7 pre-commit checklist. Every PNG was
opened and looked at; every PDF was parsed for fonts, page size and raster
content. Scripts audited: `analysis/bracket_flip/make_figures.py` (F1, F2, F5)
and `analysis/bracket_flip/geometry_figures.py` (F3, F4).

**Scope corrections.** `analysis/bracket_flip/intervention_figures.py` does not
exist — F1/F2/F5 and F3/F4 are the whole set, split across the two scripts
above. There is no `f6_*`. PNG and PDF are in sync with their scripts
(mtimes match), so nothing here is a stale-render artifact.

Verdict: **F4 is the best figure in the repo and needs three small fixes. F1 and
F3 are strong with real defects. F5 has the best §4.1 compliance I have seen
here and two hard failures. F2 is the weakest and loses its own experimental
conditions in the camera-ready variant. One failure is systemic across all five
and blocks camera-ready: nothing is drawn at final printed width.**

## Figure × check

`P` pass · `F` fail · `~` partial · `–` not applicable

| # | Check (§7) | F1 | F2 | F3 | F4 | F5 |
|---|---|:--:|:--:|:--:|:--:|:--:|
| 1 | Colours + markers from `paper_style.py`, no literals | ~ | P | ~ | P | **F** |
| 2 | Never colour alone / grayscale-readable | **F** | **F** | P | P | **F** |
| 3 | Uncertainty drawn; band semantics + n on the figure | P | **F** | P | P | P |
| 4 | Axes labelled with units; direction of "good" stated | P | ~ | P | ~ | P |
| 5 | Reference styled as reference; null gray/dashed/last | P | – | P | P | ~ |
| 6 | Layer sweep §4.1 (indexing base, input anchor, anchors) | – | – | – | – | **P** |
| 7 | Distributions §4.4 (median+IQR, KDE, bandwidth, tails) | P | – | P | **P** | – |
| 8 | Displacement field §4.7 (pairing, head marker, aggregate, control, marginal, basis) | **P** | ~ | ~ | – | – |
| 9 | Legible at final printed size | **F** | **F** | **F** | **F** | **F** |
| 10 | Panel letters; no in-figure title in the PDF | ~ | ~ | ~ | ~ | **F** |
| — | PDF genuinely vector, dense scatter rasterized | P | P | P | P | P |

Bold marks where the call is decisive rather than marginal.

### Output formats and file sizes

All five PDFs are genuinely vector: real text-show operators, embedded TrueType
subsets, **zero Type 3 glyphs**. Text is selectable. No bloat anywhere.

| file | PDF | page size (in) | PNG | rasterized artists |
|---|---|---|---|---|
| f1_displacement_field | 87 KB | 10.02 × 10.02 | 930 KB | 0 (none needed) |
| f2_worked_event | 58 KB | 10.99 × 7.80 | 557 KB | 0 (none needed) |
| f3_activation_geometry | 872 KB | 13.39 × 8.44 | 1943 KB | 8 images ✓ |
| f4_specificity | 222 KB | 9.67 × 8.24 | 691 KB | 0 (26k vector path ops = KDE grids, correct) |
| f5_depth_sweep | 31 KB | 14.39 × 5.09 | 592 KB | 0 (none needed) |

F3 is the only figure with dense scatter and it correctly sets
`rasterized=True` on all three clouds — 872 KB instead of tens of MB. This is
right and nobody should "fix" it.

One quiet defect: `paper_style.apply_style` asks for Times New Roman / Nimbus
Roman, neither of which is installed on this cluster, so **every camera-ready
PDF embeds DejaVu Serif**. The figures will not match the body font of an
ICML/NeurIPS submission.

---

## Priority 1 — blocks camera-ready

### P1.1 No figure is drawn at final printed width (all five, check 9)

The hard failure. Guideline rule 3: *"Figure text is never smaller than the
caption text (≥9 pt at final printed size). Draw at final width; never rescale
in LaTeX."* Every figure is drawn at working-figure width with a 9 pt base and
7.0 pt legend titles, then shipped as a "camera-ready" PDF at that same size.
`apply_style(camera_ready=True)` (`src/figures/paper_style.py:215-220`) changes
only the font family and the save format; **`figsize` is never touched**, and
each figure function drops only the `suptitle`. "Camera-ready" is currently a
misnomer for "serif, no suptitle".

Scaled to ICML full width (6.75 in), the body text lands at:

| figure | drawn width | scale to 6.75 in | 9 pt body → | 7.0 pt legend title → |
|---|---|---|---|---|
| f5 | 14.39 in | 0.47× | **4.2 pt** | 3.3 pt |
| f3 | 13.39 in | 0.50× | **4.5 pt** | 3.5 pt |
| f2 | 10.99 in | 0.61× | **5.5 pt** | 4.3 pt |
| f1 | 10.02 in | 0.67× | **6.1 pt** | 4.7 pt |
| f4 | 9.67 in | 0.70× | **6.3 pt** | 4.9 pt |

The guidelines' own font-to-width ratio (≈2.5–3 pt per inch) says a 14.4-inch
canvas needs ~36 pt text to survive the shrink. F3 and F5 compound this with
multi-line paragraph axis labels (`geometry_figures.py:300-302`, `320-325`,
`334-335`) that become unreadable strips.

**Fix.** Make width a parameter of `camera_ready`, not a constant. Add a width
table to `paper_style.py` and have each figure function pick its target:
`make_figures.py:176` (10.0, 9.2), `:270` (11.0, 7.6), `:386` (14.4, 4.6);
`geometry_figures.py:379` (13.2, 8.6), `:442` (9.8, 8.4). F5 and F3 are 6-panel
and 3-panel layouts that will not survive a 2.1× shrink at all — they need to
be re-laid-out (2 rows) or split, not merely rescaled. This is a re-plot, not a
one-line change, and it should be done once for both scripts.

### P1.2 F5(c): probe family is encoded by colour alone, and it is a self-inflicted regression (check 1, 2)

`paper_style.PROBE_STYLE` (`src/figures/paper_style.py:149-152`) already exists
and already carries markers — `linear: marker "o"`, `tpr: marker "D"`.
`make_figures.py:46-49` **redefines it locally and drops the marker keys**:

```python
# A probe family is a dimension no paper_style dict covers yet; colours come from
# the shared Okabe-Ito palette and the entry belongs in paper_style.py
PROBE_STYLE = {
    "linear": {"color": REFERENCE["color"], "label": "linear probe"},
    "tpr": {"color": OKABE_ITO[4], "label": "TPR ($d_r$=52, $d_f$=2)"},
}
```

The comment is **factually wrong** — `paper_style` covers it, at line 149 — and
will mislead the next integrator. `make_figures.py:52-55` duplicates
`BASIS_STYLE` the same way (`paper_style.py:153-156`), and `OKABE_ITO[4]` is a
positional index into the palette: reorder the palette and the TPR silently
changes colour. That is exactly the SAEBench failure mode rule 5 exists to
prevent.

The consequence is visible in F5(c) (`make_figures.py:359-364`): marker and
line style are spent on *basis* (`▲` solid = relative, `▽` dashed = absolute)
and the *probe family* rides on colour alone. Black-solid-▲ vs orange-solid-▲
is a colour-only distinction, and the two dashed absolute-colour curves sit
coincident at y ≈ 0.738 across blocks 0–5, so in grayscale panel (c) shows two
lines where there are four.

**Fix.** Delete `make_figures.py:44-55` entirely; import `PROBE_STYLE` and
`BASIS_STYLE` from `paper_style`. Then in `capacity_panel` (`:359-364`) use
`style["marker"]` for the probe and reserve line style alone for the basis.

### P1.3 `unflipped_own` / `unflipped_opp` are separated by colour alone (F1, F2, F5 — check 2)

Root cause is in the shared file: `src/figures/paper_style.py:123-124` gives
both roles `marker: "o"`.

```python
"unflipped_own":  {"color": "#0072B2", "marker": "o", ...},
"unflipped_opp":  {"color": "#56B4E9", "marker": "o", ...},
```

§4.6's light→dark sub-class ramp justifies sharing a marker *when the sub-class
distinction is decorative*. Here it is load-bearing and the two series
physically overlap:

- **F1(a)**: the two arrows are near-horizontal, at nearly the same y, and
  **point in opposite directions**. Rendered, they read as one blue line. A
  reader cannot tell which class crosses which way — which is the panel's
  entire claim.
- **F1(c)/(d)**: medians 14.8 vs 14.9, two solid same-marker curves lying on
  top of each other.
- **F2(f)**: the dominant series in the panel, two shades of blue, same marker,
  heavily overlapping arrows.
- **F5(a)**: both rise to ≈0.93–0.97 and run together.

`geometry_figures.py:400-402` already solved this for F4 by giving
`unflipped_opp` a dash — and F4 is the only one of the five that reads in
grayscale. That decision belongs in the shared dict, not in one script.

**Fix.** Request from the integrator: add a line-style key to
`SQUARE_ROLE_STYLE` at `paper_style.py:121-127` (`unflipped_own` solid,
`unflipped_opp` `(0, (4, 1.5))`, matching `geometry_figures.py:402`), then have
F1 (`make_figures.py:87-103`, `159-162`), F2 (`:251`) and F5 (`:318-321`)
consume it. Per `docs/bracket_flip_design.md` agents must not edit
`paper_style.py` directly.

### P1.4 F5(c): legend sits on top of the data it describes, and the panel letter collides with the title (check 10)

Two defects in the same panel, both confirmed in the **PDF**, not just the PNG:

1. `make_figures.py:379-380` places the legend `loc="center right"`. The two
   absolute-colour curves run flat at y ≈ 0.738 across the full block range and
   the legend's lower entries land on them. With `legend.frameon=False`
   (`paper_style.py:208`) there is no backing box, so the legend's own short
   dashed handles are visually indistinguishable from the dashed data lines
   running beneath them. The reader cannot tell legend from data.
2. In the serif PDF the title at `make_figures.py:377-378` is wider than in the
   PNG and runs into the panel letter, rendering as **`(c)capacity: the TPR
   peaks at block 4…`** with no separating space. `panel_letter`
   (`make_figures.py:59-61`) draws at `x=0.0, y=1.02` in axes coords, which
   collides with any left-anchored title once the font changes.

**Fix.** Move the legend to `loc="lower right"` or below the axes; and give the
panel letter its own reserved strip — `geometry_figures.py:47-49` already uses
`x=-0.12`, which does not collide. Unify the two `panel_letter` definitions
(see P3.1).

---

## Priority 2 — misleading or missing information

### P2.1 F2 loses its experimental conditions in the camera-ready PDF (check 3)

`f2_worked_example` gates its `suptitle` on `camera_ready`
(`make_figures.py:282-286`), which is correct per §5 — but that suptitle is the
**only** place the figure states `t=25`, `layer 5`, that this is a *single*
event, and which square is `s`. In the PDF all of it is gone. Nothing else in
the figure says n=1.

Worse, the event's selection rule is nowhere on the figure even in the PNG.
`results/contract_a3.csv` shows it was picked properly and mechanically —
`canonical_median_displacement=3.848` vs `canonical_event_displacement=3.8596`,
i.e. the event closest to the median, chosen from a pool of 34. **That is good
practice and the figure throws it away.** An unlabelled worked example reads as
cherry-picked.

**Fix.** Move `t`, layer, `s`, "1 of 2,000 events" and "selected as the
median-displacement event of 34 candidates" into an in-panel annotation that is
*not* gated on `camera_ready` — e.g. extend the footer at
`make_figures.py:288-292`.

### P2.2 F3(c): hardcoded n in the panel title, and the direction source is never stated

`geometry_figures.py:303-304`:

```python
ax.set_title("$\\Delta$ = after $-$ before (n=2000)\n"
             "$\\Delta'$ = after $-$ control, strict (n=1596)", fontsize=8.0, color=MUTED)
```

Both counts are string literals. They happen to match
`results/variance_shares.csv` today (`n_events` = 2000 and 1596), so the figure
is not currently wrong — but the CSV carries `n_events` per row and the title
ignores it. Re-run on a different mining yield and the figure states the old n
with a straight face. This is the one place in the whole study where a number
on a figure is not derived from the data behind it.

Separately, `shares_panel` filters to one direction source at
`geometry_figures.py:282` (`d = shares[shares.direction_source == "tpr"]`) and
**never says so on the panel**. F4 shows both TPR and linear-probe directions
side by side precisely to prove the result is not probe-dependent; F3(c) shows
only TPR and looks like it covers both. Same omission applies to F3(a)/(b),
which use `m.dirs["tpr"]` via `geometry_figures.py:380`.

**Fix.** Read both counts from `shares.n_events` at `:303`; append "flip
direction from the TPR probe" to the panel.

### P2.3 F4: the zero reference is invisible and unlabelled; only one pole of the axis is named (check 4)

F4 is otherwise the strongest figure here, which makes these worth fixing.

- `geometry_figures.py:427` draws the meaningful reference — cos = 0, "no
  alignment" — as `ax.axvline(0, color=GRID, lw=1.0, zorder=1)`. `GRID` is
  `#dcdcdc` (`paper_style.py:16`), which is invisible against a white page; I
  cannot see it in any of the four panels. §3: chance/threshold lines are
  "thin, dashed, gray, drawn behind data, **and labelled in-panel or in the
  legend** — never a bare unexplained line." This one is worse than bare, it is
  absent.
- `COS_LABEL` (`geometry_figures.py:43-44`) glosses only the positive
  direction. The negative direction is where the null sits (pooled median
  −0.0945 in panel (a)) and it is unnamed. `frame_marginal_panel` at
  `:263-265` already writes the correct two-sided form — "→ the square keeps
  its class; ← it swaps with the perspective". Reuse it.
- `:425-426` draws the null as a **full-height** `axvline`, so it runs up
  through the legend text in all four panels. The per-class median stems at
  `:421-422` are correctly clipped to the curve height; the null is not.

**Fix.** `:427` → `MUTED`, dashed, with a label. `:43-44` → adopt the two-sided
gloss from `:263-265`. `:425` → clip the null line to the pooled density height
the way `:421` does.

### P2.4 F1(a): the legend covers the empty-square cluster; F1(b) has no legend at all

`field_panel` places its legend `loc="lower left"`
(`make_figures.py:128-132`), which is exactly where the empty-square
distribution sits (before/after ≈ (0.58, −8.56), per `results/m2_field.csv`).
The gray dispersion ellipses for the `empty` role are drawn behind the legend
text and the class's own marker sits inside the legend block.

And `f1_field` passes `legend = (col == 0)` at `:178`, so **panel (b) carries no
key at all** — no role legend, and no entry for the dashed control arrow, even
though (b) is the panel where the flipped square's arrow is the long one and
the reader most needs to know which arrow is which.

**Fix.** Move the legend out of the data region (below the panel, or
`bbox_to_anchor` outside the axes) and give it to both panels, or make it a
single figure-level legend.

### P2.5 F3(a): the "after" 50% contour is invisible inside its own cloud (check 8)

`frame_panel` draws each condition's cloud at `ms=2.0, alpha=0.25` in the
condition's colour (`geometry_figures.py:235-236`) and then draws that
condition's 50% contour in **the same colour** (`:237`, via `contour()` at
`:52-61`). §4.6 makes the contour the cue that survives grayscale precisely
because the cloud does not — but here the orange "after" contour sits inside a
dense orange cloud and cannot be picked out. Only the green "control" ellipse
is legible, and only because it sits at the cloud's edge. Panel (d) has the
same problem: three conditions are listed, two ellipses are visible.

Also §4.7 asks for the class marker **at the arrow head**. The 120 arrows at
`:227-232` have bare `-|>` heads; the head lands in the rasterized cloud rather
than on a marker.

**Fix.** In `contour()` (`:56-59`) add a white halo underneath the ellipse
(`path_effects.withStroke`) or draw the ellipse with an `INK` edge; drop the
cloud alpha to ≈0.12.

### P2.6 F2(c): a series that is zero by construction is drawn as data

`logit_panel` (`make_figures.py:222-224`) plots all three filler classes as
logits *relative to* `ℓ_empty`, so the `empty` series is identically 0 in every
condition. It is drawn as a flat gray line with a legend entry, sitting exactly
on the dotted zero rule at `:231`. §4.2 is explicit: *"If a region is zero by
construction, shade it and say so in-panel. A flat segment with no annotation
reads as a smoothing artifact."* Here it reads as "the model outputs zero for
empty", which is not what it means.

**Fix.** Label the entry `empty (reference class, 0 by construction)` at `:229`,
or drop the series and say so in the ylabel.

### P2.7 F5(a)/(b): proportion panels with no chance or null reference (check 5)

Panel (a) has the matched-control curve (`make_figures.py:324-325`) acting as
an empirical comparison, which is good. Panel (b) has nothing — `proportion of
events` for "absolute colour decoded correctly" with no chance line, no null,
no control. §3 prefers an empirical null but requires *something* labelled. A
reader cannot tell whether 0.44 at the embedding is above or below chance.

**Fix.** Carry the control curve into `absolute_panel` (`:334-351`), or draw a
labelled empirical null (shuffled labels) in both panels.

### P2.8 F1: the marginal's denominator does not match the field panel's

F1(a) legend reports `n=65,566` squares for the `empty` role
(`make_figures.py:101`, from `m2_field.csv:n_squares`). F1(c), plotting the
same role, reports `n=2,000` in the ylabel (`:166`, `len(lengths)`).
`m2_displacements.csv` has 2000 rows — one per event — so each value in the
marginal is a per-event aggregate over that event's ~33 empty squares. The
figure never says this, so the two panels appear to disagree about n by a
factor of 33.

**Fix.** At `:166`, state the unit of observation explicitly — e.g. `density
over events (n=2,000; one per-event mean per role)` — and put the per-role n in
each legend entry at `:162` the way the field panel does.

---

## Priority 3 — DRY, robustness, polish

### P3.1 Four shared-style constructs are duplicated across three files

| construct | canonical home | duplicate |
|---|---|---|
| `ROLE_TEXT` | `paper_style.py:131-137` | `src/bracket_flip/probe_space.py:58-64` — **verbatim**, five identical strings |
| `role_key()` | `paper_style.py:140-143` | `make_figures.py:64-67` |
| `PROBE_STYLE`, `BASIS_STYLE` | `paper_style.py:149-156` | `make_figures.py:46-55` (see P1.2) |
| `panel_letter()` | neither | `make_figures.py:59-61` **and** `geometry_figures.py:47-49`, with different offsets (`x=0.0` vs `x=-0.12`) — the cause of P1.4's collision |

`make_figures.py:28` imports `ROLE_TEXT` from `probe_space` rather than
`paper_style`, so the legend key and the shared style file are two independent
sources of truth for the same five sentences.

**Fix.** Delete `make_figures.py:44-55` and `:64-67`; import `role_key`,
`ROLE_TEXT`, `PROBE_STYLE`, `BASIS_STYLE` from `paper_style`. Delete
`probe_space.py:58-64` and import from `paper_style` there too. Promote one
`panel_letter` into `paper_style.py`.

Same class of issue, lower stakes: the 50% Gaussian ellipse is implemented
twice — `make_figures.py:77-83` (`dispersion_ellipse`, radius `sqrt(2·ln2)`)
and `geometry_figures.py:52-61` (`contour`, radius `sqrt(chi2.ppf(0.5, 2))`).
The two constants are mathematically identical, which is lucky rather than
guaranteed. One helper in `paper_style.py`.

### P3.2 Two hardcoded values that will break on a re-run

- `geometry_figures.py:353`: `markevery=[0, 4, 19, 99, 511]` assumes
  `D_MODEL = 512` (`act_geometry.py:27`). Any other width is an `IndexError`.
  Derive from `len(curve)`.
- `make_figures.py:369-372`: the capacity annotation places its text at
  `xytext=(worst.layer - 4.0, 0.925)`. `worst.layer` is 7 today so it lands
  inside the axes; if the worst gap moved to block 0–2 the text would be drawn
  outside `xlim` (`:303`, left edge −1.5). Anchor it in axes coords instead.

### P3.3 Per-axes titles are never gated on `camera_ready`

Every figure correctly gates its `suptitle` (`make_figures.py:188`, `:282`,
`:393`; `geometry_figures.py:387`, `:448`) — that part is right. But eleven
`ax.set_title` calls are ungated and survive into the PDF:
`make_figures.py:126`, `:214`, `:234`, `:258`, `:328`, `:348`, `:377`;
`geometry_figures.py:303`, `:431`.

Panel titles are a softer case than a figure title and I would not strip them
blindly — several carry information that exists nowhere else (F3(c)'s n, F4's
contrast and direction source). **Decide deliberately**: either accept panel
titles as camera-ready furniture and say so in `FIGURE_GUIDELINES.md` §5, or
move their content into the legend titles first and then gate them. What is not
acceptable is the current state, where the gating is accidental.

### P3.4 Typography

Several strings use TeX-style `` `quoted' `` pairs that matplotlib renders
literally as a grave accent plus an apostrophe — visible in the PNG and the
PDF: `make_figures.py:236` (`` `theirs' ``), `:350` (`` `white before, black
after' ``), `geometry_figures.py:324` (`` `before' ``), `:328` (`` `after' ``,
`` `control' ``). Use `'` … `'` or plain double quotes.

`make_figures.py:288-292` writes the F2 footer as `square outline: flipped
square; placed disc.` — two role names in muted gray with **no colour mapping**,
so the reader cannot tell that the vermillion outline is `flipped` and the pink
one is `placed`. Colour each name with its `SQUARE_ROLE_STYLE` colour, or draw
them as proper legend handles.

`make_figures.py:255-257` (F2 panel f) omits `(logits)` from the ylabel and both
`>0:` direction glosses that the equivalent F1 axes carry at `:119-125`. Same
quantity, two different labels.

---

## Check 11 — figure types the guidelines do not cover

§7's own instruction: *"When you add a figure type the document doesn't cover,
add the rule to it."* This study introduces five uncovered types. Proposed
rules, in the guidelines' voice, for whoever owns `FIGURE_GUIDELINES.md`:

**§4.8 Worked-example / single-instance panels** (F2). Nothing in the document
covers a figure whose n is 1. Proposed rule: *A single-instance panel states
`n = 1` and its selection rule in-panel, not in the caption, and the selection
rule is mechanical and stated as such ("the event closest to the median
displacement, of 34 candidates"). An unlabelled worked example is read as
cherry-picked, and correctly so. Reproduction-critical context for the instance
— position, layer, item id — lives in an annotation that survives the
camera-ready title strip, because the suptitle does not.* F2 satisfies the
mechanical-selection part in `contract_a3.csv` and satisfies none of it on the
figure.

**§4.9 Domain-state panels** (F2 a/b/d/e — the 8×8 boards). Proposed rule:
*Render domain state in the domain's own coordinates with its own labels (file
letters, rank numbers), never as an anonymous scatter. Ground truth and decoded
state sit adjacent with identical geometry, so the only visual difference is
the quantity of interest. The focal cell is marked with an outline whose colour
is keyed inside the figure. Encode the state by fill (filled/open/absent) as
well as hue, so the panel reads in grayscale without a legend.* F2 follows all
of this except the outline key (P3.4).

**§4.10 Variance / energy-share decompositions** (F3c). Proposed rule: *Shares
that partition a total are points with CIs on a log axis when they span more
than a decade — not a stacked bar, which cannot show a 0.4% component next to a
69% one. State in-panel which components sum to the total and which are scored
against the same total from outside the partition. Order rows by the
partition, put the empirical null last, and draw the isotropic/chance share as
a labelled reference line.* F3(c) does all of this well and is the model case;
the rule should be written from it.

**§4.11 Cumulative spectrum / shared-subspace curves** (F3f). Proposed rule:
*Log rank axis, cumulative share on [0, 1], and the ambient dimension d stated
so the reader can see how far along the axis the curve runs. When the point is
a comparison against a single per-item direction, draw that as a horizontal
reference and annotate the rank at which the curve crosses it — the crossing
rank is the finding, not the curve.* F3(f) does this; only the annotation
placement fails (it overlaps the curve, `geometry_figures.py:365-368`).

**§4.7 addendum — gauge-dependent probe coordinates.** The strongest design
decision in this study is not covered at all. F1 is drawn in gauge-invariant
logit differences rather than the TPR's filler coordinates, because
`logit = rᵀBf` is invariant under `f → Af, B → BA⁻¹` and the filler-plane angle
moves 131° → 105° across seeds at constant accuracy
(`docs/bracket_flip_design.md`). Proposed rule: *When a probe's coordinates
carry an unidentifiable gauge, plot an invariant function of them and say which
invariant in the axis label. Never plot raw latent/filler coordinates as an
axis a reader is asked to compare across seeds, probes, or label conventions;
report the gauge-dependent geometry as a number with the gauge stated.* F1
follows this exactly and should be cited as the example.

**Minor addendum — deterministic relabel panels.** F1(b) is F1(a) with the
`after` points reflected in x (confirmed: the `before` rows in
`results/m2_field.csv` are bit-identical between the two bases), and F5(b) is
an exact relabel of F5(a) — its own title says so. Proposed rule: *When one
panel is a deterministic relabel of another, say so in-panel, so it is not read
as independent evidence.* F5(b) does this; F1(b) does not.

---

## What is already right

Worth protecting from a careless "fix":

- **F5's §4.1 compliance is the best in the repo.** The input space is its own
  leftmost tick, separated by both a rule and a shaded span
  (`make_figures.py:298-304`), and the indexing base is written into the x-axis
  label itself — `residual stream (embed = W_E + W_pos; 0-7 = resid_post of
  that block)` — not buried in a caption. This is what §4.1 asks for and almost
  nothing in the literature does it.
- **F4's uncertainty handling is better than the papers it cites.** The KDE is
  over discs (n = 28,393) while the bootstrap resamples *events* (n = 2,000) —
  the correct clustered choice — and both n's, the kernel, Scott's rule and the
  resulting bandwidth **in cosine units** are all on the figure. §4.4 asks for
  exactly this and it is rare to see it done.
- **F3's rasterization is correct** (P1.2 above): `rasterized=True` on all three
  dense clouds, vector text over raster, 872 KB.
- **F3 ships the §4.6 nuisance panel** (`geometry_figures.py:331-335`) colouring
  the same embedding by move number `t`, and it is uninformative — which is the
  point. §4.6 says it ships even when it is uninformative. It did.
- **F1's axis labels are the best in the set**: units, the defining formula, and
  the direction gloss (`>0: owned by the player to move`).
- Every figure's `suptitle` is correctly gated on `camera_ready`.

---

## Provenance

Audited in a git worktree on branch `cosine-spade-gamma-sweep`, which predates
the `bracket_flip` work entirely — none of the audited files exist on it. All
figures, scripts, `FIGURE_GUIDELINES.md` and `docs/bracket_flip_design.md` were
read from the shared checkout at
`/n/home10/dcao/Spade_Research_David_Sumedh_Ba/`, where they are currently
untracked. Nothing outside this file was modified. Line numbers refer to the
shared-checkout copies as of the audit.
