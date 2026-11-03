# Finish note — the docs consistency sweep

Every `.md` under `docs/` except `claims_status.md`, plus the `analysis/` reports, checked
against `docs/claims_status.md`, MIGRATION.md's open questions, and the two provenance
findings in [acceptance-eval.md](acceptance-eval.md). Sources for every number changed are
named below; nothing was recomputed by hand.

## Corrected

**`docs/pipeline.md`**
- §5's "config-label trap" said the SpaDE config's `lambda_init 9.77e-05, gamma_reg 0.2`
  were stale pre-canonicalization defaults and "the weights are the production run". Both
  halves are wrong. The run is real (`SpaDETrainer-othello_trainer_5`, 36621 steps, cuda:0,
  finished degenerate); what is stale in all six configs is `steps: 400` and `device: mps`;
  and the `ae.pt` on disk is not that run. Now two traps instead of one, the second naming
  open question 15. Source: acceptance-eval.md.
- §5 phase-1 table, the λ=9.77e-5 row: parenthetical rewritten from "see the config-label
  trap" to the weights being a different model.
- §7a: added the ±0.014 reproduction band on coverage and reconstruction F1 for the five
  `results/main/` rows (open question 14), and noted that BatchTopK, CosineSpaDE and the
  SpaDE default reproduce to 0.000000. Removed "its `config.json` is a Mac smoke config over
  real GPU weights" as a reason for the exclusion, which is not what the config's staleness
  shows. Source: acceptance-eval.md; reference rows in the frozen
  `results/karvonen/all_results.csv` (this tree's `results/main/`, not yet copied in).

**`docs/spade_vs_karvonen.md`** — same ±0.014 caveat under the §3 table; the excluded-SpaDE
paragraph now says re-running that row from `checkpoints/main/trainer_5` will not return it.

**`docs/fasrc_commands.md`** — "Do not install from `requirements.txt` yet… four pins are
wrong" described the source repo's file. This tree's was regenerated and verified
(`docs/finish_notes/env-install.md`), and `pyproject.toml` reads it. Dropped the hand-rolled
pip list beside it, which also asked for `python-chess` where the working distribution is
`chess` (`requirements.txt` header).

**`docs/bracket_flip_design.md`** — M5 said `pair_patch` puts flip consequences in
"mlp5/mlp6"; `analysis/rules_causal/README.md:62` gives mlp4 → mlp6 (0.13 → 0.38 → 0.57),
which is also what claims_status.md carries.

**`analysis/spade_vs_baselines.md`** — Appendix A said the λ=9.77e-5 / γ=0.2 values "were
read off a `config.json` that is a Mac smoke-test config… the weights are the γ=1e-2
production run". Corrected as above, with open question 15 named. The §2 table note carried
the same claim in shorter form. Added the ±0.014 caveat there, since this document's Pareto
listings separate Gated (0.5673) from Standard (0.5679) inside that band.

**`analysis/straightness/sae_profiles/README.md`** — "PROCESS.md flags its `config.json` as
a Mac smoke config written over real GPU weights" replaced with what is actually anomalous
about that checkpoint. The exclusion decision it supports is unaffected.

**`analysis/bracket_flip/reports/SAE_ARM.md`** — "The straightness study found SpaDE
destroys the trajectory signal while CosineSpaDE keeps it" was a retired claim used as the
premise of a whole section. The section's own measurement (both prototype variants fail the
board decode together) is kept; the contrast with the retired claim is gone.

**`analysis/bracket_flip/reports/ROBUSTNESS.md`** — no correction banner, unlike its
siblings, while the document stabilizes `p_s_unchanged_rel` across 68 refits and reads its
own r ≈ 0.999 accuracy tracking as support. Banner added.

**`analysis/violation_divergence/RESULTS.md`** — no banner either, and its Headline still
carries the retired depth migration of the peak offset (0.00 → 5.68) and `peak_layer` = 7.
Banner added, also naming the `occupied_opp` no-op reading, which `README.md`,
`RED_TEAM.md` §F and `probe_readout/RESULTS.md` all answer the other way (recolor, 78.6%).

**`analysis/bracket_flip/reports/RESULTS_A3.md`** — "the number to quote" pointed at the
retired `P(theirs, theirs)`; marked in place.

**`analysis/README.md`, `analysis/linear_probe/README.md`** — both call the committed probe
labels ground truth with no mention of the 53 mislabeled rows in 10,000. Qualifier added
(claims_status.md).

## Annotated, not rewritten

Historical records get the `*Fixed since — see …*` convention the flagged-defect sweep
established, not edits to what the agent believed at the time.

- `docs/port_notes/docs.md` — conflict 4 and the Refactor_Cleanup_Progress block quote both
  conclude the λ=9.77e-5 / γ=0.2 configuration "never ran at those values". Retraction
  annotated at both. Its "Left for someone else" list is now verified item by item:
  claims_status.md's γ=20 sentence is fixed (**done**); `archive/old_md/` still does not
  exist (**open**, archive zone); the `cd "$HOME/spade_sae"` scripts are **done** — 39
  script files use `cd "${SPADE_REPO:?}"` and the only `$HOME` left is the HF-token cache
  path in two `submit_*_only_eval.sbatch` jobs; final printed width is still **open**
  (`paper_style.py` sets no figure width).
- `docs/finish_notes/env-install.md` — restated VE 0.943 as the outstanding acceptance
  number.
- `docs/finish_notes/checkpoint-data.md` — "trainer_5 is the degenerate γ=1e-2 checkpoint".
  The copy is faithful; the file is not that evaluation.
- `docs/port_notes/pipeline.md:128` and `scripts.md:130` carry VE 0.943 and L0 80.00. Left
  untouched by instruction — port notes are records of what an agent believed.

## Checked, clean

`docs/activation_cache_plan.md`, `docs/othello_experiment.md`, `docs/othello_rules.md`,
`docs/restrav_summary.md`, `docs/karvonen_reference.md` (Karvonen's own sweep numbers; the
only cross-reference to our results is the γ=1e4 VE/loss-recovered dissociation, which
holds), `docs/violation_divergence_design.md` (its correction header already covers every
retired divergence claim), `docs/session_logs/*` (historical by design, README says so, and
no retired claim is restated as current in any of the five).

In `analysis/`: the four study READMEs, `VERDICT.md`, `RED_TEAM.md`, `PROBE_REPORT.md`,
`FIGURE_AUDIT.md`, `RESULTS_A4.md`, `RESULTS_A6.md`, the manifold-capture, rules-causal,
rules-manifolds and geometry summaries, and both `violation_divergence/reports/`.

Two things that look like defects and are not. `analysis/geometry/SUMMARY.md`'s "PAnneal
wins MSE + board reconstruction" is the geometry family's own per-move reconstruction
accuracy (`per_move_accuracy_multigame/`), not Karvonen reconstruction F1, so it does not
contradict TopK being best on the latter. `analysis/bracket_flip/reports/DATA_REPORT.md`'s
0.9900 row is framed as a row-alignment check, which `VERDICT.md` explicitly endorses as the
honest use of it.

## For the owner

- **Open question 15, one observation, not a resolution.** The rerun of
  `checkpoints/main/trainer_5/ae.pt` — L0 127.76, VE 0.9729 (`results/acceptance/
  all_results.csv`) — is the number the docs report for **seeds 43–46 of the same config**
  (L0 128, VE 0.973). Every doc in this sweep that mentions the exclusion states that pair
  of numbers. Whether the seed-42 slot holds another seed's weights is yours to settle; I
  have only recorded that the two agree.
- **The ±0.014 band eats live comparisons.** In `analysis/spade_vs_baselines.md` the Pareto
  listing separates Gated and Standard on reconstruction F1 by 0.0006. Under question 14
  that ordering is not resolvable, and re-baselining on the surviving activation set would
  decide it.
- **`docs/claims_status.md`** attributes the L0 15.4 / VE 0.352 evaluation to the γ=1e-2
  checkpoint, while trainer_5's wandb record is γ=0.2. I did not touch that file, and the
  docs in this zone continue to defer to it. The two labels want reconciling in one place.
- `docs/fasrc_commands.md` still tells the reader to `pip install -e "$SPADE_REPO"` into
  `spade_env`, which PROCESS.md warns against for worktrees. On the cluster `$SPADE_REPO` is
  the real checkout, so it does not dangle; left as written.
