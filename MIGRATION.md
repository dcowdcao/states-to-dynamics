# Migration

How the port runs, in what order, and what must not be lost on the way. Layout and the
old→new map live in [ARCHITECTURE.md](ARCHITECTURE.md).

## Working rules

**1. All work happens in this worktree (`spade_sae`, branch `new-architecture`).** New
experiments, new figures, new analysis — everything lands here. Nothing new goes into
`../Spade_Research_David_Sumedh_Ba`.

**2. The source repo is frozen and read-only.** It is the thing being ported *from*. Read
it, copy from it, diff against it; do not edit it. The two worktrees share one `.git`, so
content can be diffed and cherry-picked between them without a remote round-trip.

**3. Commit at every coherent step, push at every wave boundary.** A wave is not done until
it is pushed. The skeleton sat uncommitted and unpushed for two weeks; that is the failure
mode this rule exists to prevent.

**4. Ported code is not ported until it runs.** Each wave below states its verification.
A wave that cannot be verified yet says so explicitly rather than being marked done.

The cost of rule 1 is real and worth naming: until wave 3 lands, this tree cannot run an
experiment, so research is blocked on the port. That argues for porting in dependency
order and getting the pipeline green early, which is what the wave order does.

## Port order

Driven by the import graph, not by tidiness. `common` → `othello` + `saes` → pipeline →
`spaces` + `rules` → the four downstream studies. Getting this backwards means porting a
study before the API it imports exists.

**Wave 0 — back up netscratch.** Independent of the port and ahead of it: copy the
irreplaceable intermediates off the 90-day tier — seeds 43–46 above all, then the branch,
event and trajectory sets. They are the error bars on every published figure and cannot be
re-derived without re-running the jobs. A `spade_checkpoint_backup` dir already exists,
which suggests this was started once and not finished.
*Verify:* a checksum listing of what was copied, committed to the repo, so a later session
can tell what survived.

**Wave 1 — packaging and foundations.** `pyproject.toml` (editable install, `>=3.11,<3.12`,
namespace-safe package discovery), `requirements.txt` regenerated from the working
`spade_env` rather than copied, `spade/common/{config,paths,device,net,study}.py`,
`FIGURE_GUIDELINES.md`, `spade/figures/paper_style.py`.
*Why paper_style this early:* every downstream figure imports it, and the guidelines forbid
color literals anywhere else. Porting it late guarantees the drift the rule exists to stop.
*Verify:* `pip install -e .` in a clean env, `import spade`, `spade.common.paths.repo_root()`
resolves to this worktree.

**Wave 2 — the shared Othello and SAE layer.** Vendor and **pin** `external/Karvonen_Research`
(capture the upstream SHA first — it has none today), vendor `external/tpr_othello` at
`cc90cea`. Then `spade/othello/{games,boards,taxonomy}.py`, `spade/saes/{spade_base,spade,
spade_cosine}.py`, `spade/saes/registry.py`.
*Verify:* `spade.saes.registry.entries()` verifies every checkpoint it lists — which means
the `checkpoints/` copy below must be done first.

**Wave 3 — the pipeline.** `extraction`, `training/{core,sweep}` + eight entrypoints,
`evaluation`, `multiseed`, `diagnostics`, the Karvonen figure entrypoints, `validation`.
*Verify:* `python -m spade.validation.final_validation`, plus one real eval reproducing a
known number. **Done, 2026-08-04** — see
[docs/finish_notes/acceptance-eval.md](docs/finish_notes/acceptance-eval.md). Use the
committed reference rows, not the numbers this line used to carry: it said variance
explained = 0.943, which appears nowhere in the tree. `results/karvonen/all_results.csv`
gives TopK on `checkpoints/main/trainer_3` as L0 79.93, VE **0.9867**, coverage 0.7265.

The strongest form of this check is not TopK. `checkpoints/{batch_topk,cosine_spade,
spade_revival}/trainer_0` reproduce their references *exactly* — 0.000000 on L0, VE,
coverage and reconstruction F1 — because their reference evals used the activation set that
still exists. The five `results/karvonen/` checkpoints cannot be reproduced exactly at all;
their activations are gone (open question 14).

**Wave 4 — spaces and the rules program.** `spade/spaces.py`, `spade/rules/*`,
`spade/rules/causal/*`. This is the layer four other studies import; nothing downstream
starts before it is green.
*Verify:* re-derive the un-flip edit number (+1.07 log-mass vs +0.10 for the random null).

**Wave 5 — the four downstream studies.** `bracket_flip`, `violation_divergence`,
`straightness`, `manifold_capture`. Independent of each other; order is free, and they can
run in parallel worktrees off this branch.
*Verify:* each study's headline number from its own report, and only the numbers that
survived red-teaming.

**Wave 6 — analysis and orchestration.** `analysis/<study>/make_figures.py` and the study
reports, then `scripts/` with every job standardized on `cd "${SPADE_REPO:?}"` and
`python -m spade.<module>`.
*Verify:* one figure regenerated byte-comparably from committed inputs.

**Wave 7 — the viewer.** `viewer/web/` plus `spade/export/viewer_weights.py`. The dead
Docker path is not ported.
*Verify:* `viewer/web/test_inference.mjs` against a freshly exported `reference.json` —
resid rel err ~7e-4, per-SAE top-10 ~2e-4, exact L0.

**Wave 8 — docs and archive.** Rewrite `docs/` to the current science, archive what is
superseded, and delete the source repo's residue only after everything above is green.

## Things git will not move for you

A `git checkout` of this branch produces a tree that cannot run. These must be copied by
hand, per wave, as the code that needs them lands:

- **SAE weights.** `*.pt` is gitignored, so ~1.5 GB of `ae.pt` across nine checkpoint dirs
  moves by filesystem copy only. `standardization_params.json` must land at the *save-dir*
  level, beside its trainers, not inside them — the adapter's parent-directory fallback
  depends on it.
- **`analysis/linear_probe/activations/`** (220 MB, gitignored). `linear_probe.pt` is a hard
  dependency of `rules/causal/unflip_edit.py` and three `bracket_flip` modules.
- **The netscratch tier**, ~92 GB under `/n/netscratch/ba_lab/Lab/$USER/spade_*`, on 90-day
  retention from 2026-07-22. It holds seeds 43–46 — the error bars on every published
  figure. Backed up in wave 0, ahead of everything else; the deadline is ~2026-10-20.
- **Three files inside `external/Karvonen_Research`** that git cannot see, two of them hidden
  by the *vendored* `.gitignore` that git still honours: `linear_probes/*.pth`,
  `circuits/resources/meta.pkl`, `circuits/dictionary_learning/pretrained_dictionary_downloader.sh`.

The `.gitignore` must also carry forward every rule that keeps ~1.2 GB out of the index —
`wandb/`, `viewer/web/weights/`, `analysis/linear_probe/activations/`, `results/*/f1_TFRRC*.pt`
— and the `*.html` negations, whose absence has already silently dropped a tracked file once.

## Decided

- **The shared layer is lifted out of `rules/`** into `spade/othello/` (games, boards,
  taxonomy), `spade/saes/registry.py` and `spade/spaces.py`. No study imports from a
  sibling study.
- **Netscratch is backed up before the port** — wave 0 above.
- **The SpaDE γ=1e-2 seed-42 checkpoint is excluded everywhere**, decided once in
  `spade.spaces`, and the default SpaDE configuration becomes **λ=1e-3, γ=1e-4**
  (`checkpoints/spade_revival/`). Two follow-ons: the geometry family's "SpaDE is broken"
  conclusion rests on the excluded checkpoint and must be re-run or retired before it is
  quoted again, and the viewer roster stays at eight with the new default taking the old
  slot.
- **The source repo carries a freeze notice** pointing here, so a session that starts there
  is told where the work moved.

## Open questions

Needing David's call. Nothing below is resolved silently.

**Structure**
1. Do the eight `bracket_flip` and two `violation_divergence` report `.md` files move to
   `analysis/<study>/reports/`, or stay beside the code that produced them?

**Data**
2. `autoencoders_spade_sweep/trainer_0–3` (λ=1e-8/1e-7, config-only, fully collapsed):
   archive as a negative result, or delete?
3. All six `autoencoders/trainer_*/config.json` are a Mac smoke config over real GPU
   weights. Regenerate them from the true run parameters during the port, or migrate as-is
   with a marker?
4. Intermediate `trainer_*/checkpoints/ae_{9155,18310,27465}.pt` — roughly 3× the final
   weight volume, no readers. Drop before migrating?
5. `archive/` (127 MB, 93 MB already in history): carry into the new repo and pay the
   history cost twice, or leave it on the old branch?
6. `wandb/` (579 MB) — delete outright?
7. Do the netscratch `spade_*` dirs keep their names after the repo is renamed?
14. **The activations behind `results/karvonen/` no longer exist.** Those five reference rows
    were committed 2026-07-13 22:02 -0700; the only surviving activation set was written
    2026-07-14 01:52–01:55 and is a later extraction, so they reproduce only to ±0.014 on
    coverage and reconstruction F1 (VE and L0 are stable). Re-baseline the reference on the
    surviving set, or reconstruct the original extraction? Reconstruction needs parameters
    nothing records — `data/activations/metadata.json` describes a 100-game smoke run
    (`activation_shape [4680, 512]`) beside 12 GB tensors holding 5,900,000 rows, which is
    question 3's problem in the activations rather than the checkpoints.
15. **`checkpoints/main/trainer_5/ae.pt` is not the run its config names.** It reruns at
    L0 127.76 / VE 0.973; its wandb record and committed eval row both say L0 15.43 /
    VE ≈ 0.36. trainer_0–4 all match their records. It is not the other recorded SpaDE
    baseline either (γ=0.1, VE 0.399), and no run in `results/Wandb_Export.csv` reports
    L0 ≈ 128 / VE ≈ 0.97. What are these weights, and what did the studies that loaded them
    actually measure? See [docs/finish_notes/acceptance-eval.md](docs/finish_notes/acceptance-eval.md).

**Science**
8. `analysis/geometry/` is the only family still built on the 7-SAE set including the now-
   excluded checkpoint. Re-run it on the current set, or retire it as a result and keep it
   as history?
9. Three violation-divergence sub-studies were computed on the superseded one-control
   branch set and never re-derived on two controls. Re-derive, or port with the caveat?
10. Tracked-figure policy: ~300 MB of committed PNGs, including 190 near-duplicate
    view-grid panels. Keep all in the new history, or track only canonical figures and
    regenerate the rest?

**Deployment**
11. Confirm the dead Docker path (`build_space.py`, `space_src/`, `hf_space/`) is deleted
    rather than ported, after its exact CPU pins are rescued into docs.
12. Keep `server.py` + `static/index.html` at all? They are the only consumers of
    fastapi/uvicorn and a second, divergent copy of the viewer UI.
13. The viewer needs a re-export and re-verification after the checkpoint consolidation and
    the roster change. Redeploy the Space then, or leave it frozen on the current build?
