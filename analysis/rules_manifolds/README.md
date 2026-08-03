# Rules manifolds

Do OthelloGPT activations organized by game rule cluster in 2D PCA / form
distinct manifolds in Isomap? Rule set and study logic: `docs/othello_rules.md`.

Design: matched injected-move pairs (shared legal prefix + legal and
category-stratified illegal continuations), layer-5 `resid_post` at the
injected position. Views per space:

- `combined` — legal + all six violation categories, colored by rule
  (legal / R1 placement / R2 bracketing / R3 turn), fine category, and t.
  The "do rules cluster relative to each other" figure.
- `r1_placement`, `r2_bracketing`, `r3_turn` — legal + that rule's
  violations only, own embedding, colored by category and t.
- `r5_flips` — legal moves only, colored by discs-flipped and flip
  directions: is the legal manifold *organized* by flip extent? (Every legal
  move flips, so R5 has no violation class — its causal test is separate.)

Spaces: raw standardized activations, two untrained controls (random
projection, random-init TopK SAE), and one representative checkpoint per SAE
config. Each view ships centered (per-t legal-centroid subtraction — the main
figure) and raw (`*_raw.png`) variants: t dominates uncentered geometry, so
apparent rule clusters must survive centering, and every figure carries a
t-colored panel as the diagnostic. Isomap runs on a category-balanced ~4k
subsample (shared across spaces) after PCA-50 pre-reduction; PCA on all view
items.

Pipeline (one SLURM job — `scripts/slurm/studies/rules/submit_rules.sh`):
`spade.rules.build_pairs` → `extract_acts` → `flip_labels` → `embed` →
`make_figures.py`; large artifacts on `$SPADE_RULES_DIR`, embeddings in
`embeddings/`, PNGs in `figures/<view>/<space>[_raw].png`.

Regenerate figures only (CPU): `python analysis/rules_manifolds/make_figures.py`
(`--camera-ready` for vector PDFs with the in-figure title dropped).

Reading a figure (conventions in FIGURE_GUIDELINES.md §4.6): rows are PCA (top)
and Isomap (bottom) of the same items, columns are the colorings listed above.
The cloud is drawn in one shuffled pass so no class sits on top of another; each
class is then summarized by a centroid in its own marker shape and a 50%
probability contour — a dispersion summary, not a confidence interval. Class n
is in the legend, PCA axes carry per-component variance ratios, and the last
column is always the t panel: any apparent class structure that reproduces the
t panel is the move-number confound, not a rule. The rules a view covers are
spelled out in a key under the panels (`paper_style.RULE_TEXT`), so a figure
never has to be read next to `docs/othello_rules.md`.

Caveats: SpaDE codes live on a simplex — compare cluster *structure* across
SAE families, never raw coordinate scales. Category availability correlates
with t (late-game boards rarely have detached empty squares), so per-category
comparisons need the centered variant.
