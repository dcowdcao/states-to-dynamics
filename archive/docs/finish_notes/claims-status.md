# Finish note — bringing claims_status.md to the current science

Scope was `docs/claims_status.md` alone. Evidence: `docs/finish_notes/acceptance-eval.md`,
`results/acceptance/all_results.csv`, the frozen source repo's `results/karvonen*/all_results.csv`
and `results/multiseed/{all_seeds,aggregate_baseline}.csv`, `checkpoints/main/trainer_5/config.json`,
`src/spade/{spaces.py,saes/registry.py,manifold_capture/spaces.py,export/viewer_weights.py}`.

## What moved

**The geometry entry (Retired) was rewritten.** It said the checkpoint's `config.json` was
"a Mac smoke config over real GPU weights" — the reading that the recorded λ=9.77e-5 / γ=0.2
was a stale label over a γ=1e-2 production run. The acceptance eval retracts it: wandb has
`SpaDETrainer-othello_trainer_5` at exactly those values, 36621 steps on cuda:0, finished
2026-03-26, and only `steps` and `device` are stale — in all six `main/` configs. The entry
now carries three separated facts (what the run was, that it was degenerate, that the `ae.pt`
is not it) and an explicit "do not restate the 'never ran' reading". The γ=1e-2 name is
recorded as the post-canonicalization class default applied backwards to a run that predates
it; I did **not** rename the checkpoint, because `spade.spaces`, `MIGRATION.md § Decided` and
three study READMEs all still call it that.

**"Distance-SpaDE is pure shattering" moved Holds → Narrowed.** Forced mechanically: the
claim's basis is `checkpoints/main/trainer_5/ae.pt`, whose weights reconstruct at VE 0.973
rather than the ≈0.43 the study's own `LOW_VE` marker names them for. Everything else in the
manifold-capture Holds entry stays; "no SAE × concept reaches capture" now says it holds for
every space but that slot. Whether the shattering headline survives is untouched — reserved.

**The `VE ≈ 0.43` narrowing got its actual shape.** It read "is single-seed … the multiseed
aggregate is 0.849 ± 0.278", which invites quoting the ± as dispersion. From
`results/multiseed/all_seeds.csv`: every distance-SpaDE config's seed-42 row is 0.35–0.44 and
every seed-43–46 row of the same config is 0.93–0.98 with near-zero spread. The aggregate is
arithmetically one outlier against four tight seeds — (4×0.973 + 0.352)/5 = 0.8488, and its
L0 twin (4×128 + 15.43)/5 = 105.5, both matching `aggregate_baseline.csv` exactly. Six of the
eight configs pin λ and γ in their names, so hyperparameters do not explain the split; I said
that and stopped, because what does explain it is a science question.

**The TopK baseline row got the q14 provenance caveat**, with the rerun numbers beside the
reference ones (coverage 0.7265 → 0.7405, F1 0.6841 → 0.6981, VE and L0 stable), and a rule:
no more than two decimals off that set.

**Preamble** now states the verification the doc rests on — three exact reproductions — and
names the two bad inputs up front. **Source paths** were pointing into the source repo;
`VERDICT.md` is at `analysis/bracket_flip/reports/` here. **Open defects** gained the two
provenance holes.

## Already done, contrary to the port note

`docs/port_notes/docs.md` § "One conflict that runs the other way" asks the parent to fix
line 55's "The operating point is γ = 20, not the γ = 1.0 default." Commit 9db3895 fixed it
before this wave; the entry now separates what the sweep recommends from what the default is,
exactly as that note asks. **The note's "Left for someone else" item is stale** — someone who
owns `docs/port_notes/docs.md` should strike it. I do not.

## Left for the owner

- Whether the manifold-capture shattering headline survives, and whether the geometry family
  is re-run or retired. Recorded as status in both places, not adjudicated.
- Renaming the checkpoint away from "γ=1e-2" across the tree: `src/spade/spaces.py:27`,
  `src/spade/saes/registry.py:12`, `src/spade/export/viewer_weights.py:39`,
  `src/spade/violation_divergence/compute_divergence.py:5`, `MIGRATION.md:124`,
  `analysis/straightness/README.md:37,87`, `analysis/straightness/sae_profiles/README.md:11`,
  `analysis/geometry/SUMMARY.md:12`. One document contradicting eight call sites is worse than
  the label being wrong everywhere, so claims_status.md states the fact and keeps the name.
- **`analysis/spade_vs_baselines.md:447` still asserts the retracted reading** — "the weights
  are the γ=1e-2 production run", the λ/γ read as pre-canonicalization defaults. Same at
  line 79. Outside my scope and outside the docs zone's port note; someone owns it.
- `src/spade/manifold_capture/manifold_views.py:26` says "SpaDE's VE~0.43 (spaces.LOW_VE)".
  That conflates two different checkpoints: 0.43 is the λ=1e-7 sweep family's seed-42 number,
  while `LOW_VE` is `main/trainer_5`, whose committed row is 0.352 and whose weights rerun at
  0.973. Out of my scope; it is a live comment on a threshold.

## Could not evaluate

- **Whether seeds 43–46 of `SpaDETrainer-othello` share seed 42's λ and γ.** They were trained
  after the 2026-07-12 canonicalization, so probably not, but nothing in the tree records
  their hyperparameters — the checkpoints are on netscratch and only `main/`'s configs are
  committed. The doc therefore says "the same config" and means the same config *name*.
- **Whose weights are in `main/trainer_5`.** Worth recording for q15: the rerun (L0 127.765,
  VE 0.972877, coverage 0.20893, recon F1 0.46989) is close to, but not equal to, seed 43 of
  the same config (L0 128.104, VE 0.972961, coverage 0.21202, recon F1 0.46559). Close enough
  that "a sibling seed of the post-canonicalization config" is the first hypothesis to test;
  not equal, so it is not seed 43's file.
