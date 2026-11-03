# Panel (b) leaves the rule_curvature figure — every document outside the study (branch `fig/panelb-docs`)

The figure lost panel (b), "one game's fork, on (a)'s scale", and the rest moved
up: **(a) stays (a), the violation-locked response (c) → (b), the board
(d) → (c)**. `analysis/violation_divergence/rule_curvature/{make_figures.py,README.md}`
belong to a sibling branch and were not touched here. This note covers everything
else in the tree that names a panel of that figure.

## The split: which documents were corrected and which were annotated

The rule I applied: a document that tells a reader what the figure *is* has to
match the figure; a document that records what an agent *did*, on a date, was
true when written and gets the `*… since — see …*` annotation
`docs/port_notes/` and `docs/finish_notes/docs-consistency.md` established,
rather than an edit to what it claimed.

### Annotated as historical (4 files)

Each is a dated finish note whose subject is a change made to the four-panel
figure. Rewriting them would erase why the figure has the shape it does.

- **`docs/finish_notes/curvature-figure.md`** — the note that built the figure.
  Its "Panels: (a) … (d)" inventory *is* the four-panel layout; that is the
  record.
- **`docs/finish_notes/rule-population.md`** — "the uncertainty display in (b)
  and (c)". The removal of the member traces genuinely happened in both of those
  panels, one of which no longer exists.
- **`docs/finish_notes/single-t0.md`** — "one dashed line serves both panels",
  "Panels (b)–(d) … are untouched". Both statements are about the layout at the
  time and are what justified the *t0* = 20 choice.
- **`docs/finish_notes/bands-divergence.md`** — the shared-helper recommendation,
  "its own panels (b) and (c) have had theirs removed".

The first three carry one annotation directly under the opening paragraph,
because the stale fact is the whole document's panel numbering rather than any
single sentence — the reader meets it before the first panel letter. The fourth
gets its annotation on the one paragraph that names panels, since that file is a
band sweep across many studies and a top-of-file note would be misfiled.

### Corrected as living: none, and that is the finding

Every live reference to this figure outside the study directory names **panel
(a) only**, and (a) did not move. Verified, not assumed:

- `analysis/straightness/feature_pca/README.md` — "the same call panel (a) of
  `rule_curvature` makes". Still true.
- `scripts/slurm/studies/violation_divergence/submit_rule_population.sbatch:26` —
  "the … curves panel (a) of the rule_curvature figure draws". Still true.
- `analysis/violation_divergence/README.md` — one run command and
  `rule_population/` in the stale-cache list. No panel letters.
- `analysis/violation_divergence/RESULTS.md`, `docs/claims_status.md` — neither
  mentions this figure at all. Its panel inventory is not in either.
- `analysis/straightness/{feature_pca,sae_profiles}/make_figures.py` — import
  `rule_curvature` for `space_key` and cite its README for the θ check. No panel
  letters.

## Left alone deliberately

- **`PROCESS.md`**, by instruction. Its two references to this figure
  (lines 52–62) are panel (a) anyway.
- **`PROCESS.md:1500`**'s "(b) says what the nine branches are, (a) says where
  they go" is `fork_and_board`, a different figure with its own panel letters.
- **The frozen tree at `/n/home10/dcao/Spade_Research_David_Sumedh_Ba`** — read
  only, not swept.

## Sweep

`rule_curvature|curvature_rules_` repo-wide, then `1235`, `nine branches`,
`four panels|four-panel`, `curvature at the fork`, and panel-letter greps across
`docs/` and `analysis/violation_divergence/`. The known list in the task
(`curvature-figure`, `rule-population`, `single-t0`) turned up one more file,
`bands-divergence.md`; `extend-branches.md` mentions the study's caches but no
panel. Nothing ambiguous was left undecided.

Nothing pushed.
