# Port notes — docs zone

What landed, what changed on the way, and every place a source document and
`docs/claims_status.md` disagreed. Written for the integrator, not for a reader
of the docs.

## What landed

| Source | Target |
|---|---|
| `docs/Spade_Pipeline_Summary.md` | `docs/pipeline.md` |
| `docs/Spade_vs_Karvonen_Comparison.md` | `docs/spade_vs_karvonen.md` |
| `docs/Karvonen_Pipeline_Summary.md` | `docs/karvonen_reference.md` |
| `docs/othello_rules.md` | `docs/othello_rules.md` |
| `docs/bracket_flip_design.md` | same name, correction header added |
| `docs/violation_divergence_design.md` | same name, correction header added |
| `docs/ReStraV_Summary.md` | `docs/restrav_summary.md` |
| `docs/othello_experiment.md`, `docs/activation_cache_plan.md` | same names |
| `analysis/analysis_spade_vs_baselines.md` | `analysis/spade_vs_baselines.md` |
| `docs/fasrc_run_commands.md` (723 lines) | `docs/fasrc_commands.md` (runbook) + 2 new session logs |
| `docs/session_logs/*.md` (3) | `docs/session_logs/`, verbatim |
| `docs/Refactor_Cleanup_Progress.md` | not ported here — see below |

Every path, module invocation, artifact directory and command was rewritten from
`ARCHITECTURE.md`'s tables. Session logs are the deliberate exception: they are
records of a date and keep their contemporary paths, with a note at the top of
each and in their README saying so.

## Conflicts between a ported doc and claims_status.md

claims_status.md won in all of these.

1. **Distance-SpaDE VE ≈ 0.43 stated as a fact.** All three of
   `Spade_Pipeline_Summary`, `Spade_vs_Karvonen_Comparison` and
   `analysis_spade_vs_baselines` quote it flatly. It is single-seed; the
   multiseed aggregate for the family is 0.849 ± 0.278. Every occurrence now
   carries the qualifier, and the multiseed table is in `pipeline.md` §7d.
2. **"The simplex ceiling is a fundamental limitation."**
   `analysis_spade_vs_baselines` overturned this in its own §1 and §3 update
   paragraphs while its executive summary, its §3 heading and its §8 conclusion
   still asserted it. A scope banner now sits above the whole document and those
   three places carry the correction, so no section reads as a claim about the
   prototype-sparsemax family.
3. **"The operating point is γ = 20."** See the next section — this is a
   conflict *with* claims_status.md, resolved against it on instruction.
4. **The SpaDE "initial" config, λ=9.77e-5 / γ=0.2 at L0 15.4, VE 0.352.** The
   source docs treat this as a distinct over-regularized configuration.
   claims_status.md attributes the same evaluation to the **γ=1e-2 seed-42
   checkpoint**, now excluded everywhere. Verified against the source tree:
   `autoencoders/trainer_5/config.json` records `lambda_init 9.77e-05`,
   `gamma_reg 0.2`, `steps 400`, `device mps` — a Mac smoke-test config carrying
   the *pre-canonicalization* class defaults. So the λ/γ label was read off a
   stale file and the weights are the production run. The row is dropped from
   the results tables in all three documents, with the reason stated in each.
5. **"BatchTopK was trained but not Karvonen-evaluated; its results directory is
   empty."** Asserted in `Spade_Pipeline_Summary` §8b′, `Spade_vs_Karvonen`
   §3 and the `analysis_spade_vs_baselines` §2 footnote. Not true since the
   2026-07-14 restore: `results/karvonen_batch_topk/all_results.csv` holds the
   seed-42 row (L0 80.16, frac_recovered 0.9999, coverage 0.6889, recon F1
   0.6300, precision 0.6563, recall 0.6660, VE 0.9871) and BatchTopK has all five
   seeds in `aggregate_baseline.csv`. claims_status.md is silent; the tables now
   carry the real numbers, re-derived from the CSVs.
6. **"Every claim about the divergence peak."** The violation-divergence design
   doc specifies `peak`, `peak_r` and `peak_layer` as contract keys and builds M3
   around them. All three are retired — the first two as max-of-noise, the third
   as vacuous under within-space normalization — and `ratio_illegal_control` is
   not a clean effect size. Marked at the header, at M3, and at each contract row.
7. **`kl_peak_L5`.** Named as a layer-5 quantity in the contract; it is a logit
   quantity. Marked in place.
8. **Bracket flip's headline decode key.** `p_s_unchanged_rel` and
   `p_s_absolute_correct` are specified as measurements; the first is degenerate
   (a board oracle scores 1.0 by construction) and the second is a relabelling of
   the first. The M3 variance decomposition rests on a global-swap premise that
   is false (0.61% of the energy against the matched control's 68.9%). All marked
   at the header and at each definition.
9. **R3's directional prediction** appears as a live prediction in
   `othello_rules.md`'s R3 entry. Falsified; the doc now says so and points at
   claims_status.md.
10. **Straightness' "SpaDE destroys the curvature signal"** is implied by
    `ReStraV_Summary`'s transfer section. Retracted twice; `restrav_summary.md`
    states the current position (every family preserves the separation,
    0.01–0.21) and that the layer-5 championship effect is not distinguishable
    from chance.

## One conflict that runs the other way — claims_status.md needs a fix

`docs/claims_status.md` line 55, under *Retired*, ends the CosineSpaDE γ-sweep
entry with: **"The operating point is γ = 20, not the γ = 1.0 default."**

That contradicts `ARCHITECTURE.md`'s Invariants, `MIGRATION.md`'s Decided
section and `PROCESS.md`'s 2026-08-03 entry, all of which record that γ=1.0 was
**kept deliberately** over the sweep's recommendation: the sweep selected γ=20 on
reconstruction F₁ (0.547 vs 0.530), coverage runs the other way (0.537 vs 0.474)
and decides it, and the F₁ margin is thin — 0.011 over γ=10, with seed dispersion
reported for L0 only.

Resolved against claims_status.md on instruction and consistently with the
Invariants: every document in this zone states that the operating point is γ=1.0
and records the γ=20 recommendation as considered and declined. **The
claims_status.md sentence should be rewritten** — what is retired is "γ is not a
sparsity lever, the sweep is flat"; the γ=20 selection is a sweep result that was
not adopted, not a governing default.

## Hyperparameter facts applied everywhere

- Default SpaDE configuration is **λ=1e-3, γ=1e-4** (`checkpoints/spade_revival/`),
  replacing γ=1e-2. Its seed-42 numbers, re-derived from
  `results/karvonen_spade_lam1e-03_gam1e-04/all_results.csv`: L0 1808.2,
  frac_recovered 0.99992, coverage 0.5697, recon F1 0.5145, precision 0.3661,
  recall 0.9559, VE 0.9952. Consistent with claims_status.md's "VE 0.995 vs
  0.434, L0 1808 vs 961" for the not-a-reproduction note, which is carried in
  `pipeline.md` §5, `spade_vs_karvonen.md` §3 and the April session log.
- The γ=1e-2 seed-42 checkpoint is excluded from every study and from the viewer,
  in one place (`spade.spaces`). The violation-divergence design doc used to make
  that exclusion locally and cite PROCESS.md 2026-07-28; it now defers to
  `spade.spaces`.
- CosineSpaDE default **γ=1.0**, as above.

## The Refactor_Cleanup_Progress fact worth preserving

Not ported to `docs/` — `ARCHITECTURE.md` routes it to `archive/old_md/`, which
does not exist yet in this worktree. The one fact from it that had to survive is
recorded in `docs/pipeline.md` §5 and in the conflict list above:

> On 2026-07-12 the SpaDE λ/γ defaults were canonicalized to named constants
> (`DEFAULT_LAMBDA_INIT=1e-3`, `DEFAULT_SPADE_GAMMA=1e-2`), replacing a stale
> `9.77e-5` / `0.2` that also disagreed with the docstring's theoretical
> `9.77e-4`. Any `config.json` written before that date therefore records the
> stale pair. The SpaDE baseline's does, which is why the source docs list a
> "SpaDE initial λ=9.77e-5, γ=0.2" configuration that never ran at those values.

The same document also records that the refactor preserved four load-bearing
contracts — module filenames, the four class names, the `trainer_class` dispatch
strings with Cosine matched first, and top-level `Ae`/`Ad`/`lambda_pre` state-dict
keys. Those are already in `ARCHITECTURE.md`'s Invariants, so they are not
duplicated.

## Judgement calls

- **λ ≤ 1e-7.** `ARCHITECTURE.md` says "λ ≤ 1e-7 collapses — below fp32 sparsemax
  resolution those runs train bitwise-identically." The eval CSVs and the source
  docs are more specific: λ ≤ 5e-8 collapses outright (L0 8192, negative
  fidelity), while 8e-8, 9e-8 and 1e-7 are functional but train
  bitwise-identically to each other. Written as "that whole region either
  collapses or trains bitwise-identically, so there is nothing to sweep there,"
  which is true of both readings. Flagging in case the Invariant wants tightening.
- **The withdrawn recommendation.** `analysis_spade_vs_baselines` §9.2 recommends
  a fine-grained λ sweep between 5e-8 and 8e-8 to characterize the transition.
  That window is a floating-point artifact. Struck through rather than deleted,
  so the reasoning stays visible.
- **Two session logs from the FASRC command log.** Written as narrative records
  rather than reproduced copy-paste blocks: the dead rsync invocations appeared
  six times identically and carried no information the one-line note at the top
  does not. Job IDs, configs, timings and the two real hazards (reused sweep
  directories, mislabeled W&B γ names) are all preserved. The April file is dated
  to the month because the log records no submission date for it; it executes the
  grid decided in `4-4-2026.md`.
- **`analysis/violation_divergence/RED_TEAM.md`** is cited at the path
  claims_status.md uses. MIGRATION.md open question 1 may move the study reports
  to `analysis/<study>/reports/`; if it does, that citation and the one in
  `violation_divergence_design.md` need updating.

## Left for someone else

- `docs/claims_status.md` line 55 (above). Parent owns the file.
- `archive/old_md/` does not exist; `Refactor_Cleanup_Progress.md` has nowhere to
  land yet.
- `scripts/slurm/main/run_all.sh` and the other stub scripts still `cd
  "$HOME/spade_sae"` while `scripts/slurm/README.md` mandates `cd
  "${SPADE_REPO:?}"`. `docs/fasrc_commands.md` documents the README's rule, so
  the scripts are what will need to change.
- Nothing in the source repo is drawn at final printed width, which blocks
  camera-ready output. Recorded in claims_status.md's open defects; no doc in
  this zone can fix it.
