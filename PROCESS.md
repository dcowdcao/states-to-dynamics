# PROCESS.md

Running log of what's being worked on and the non-obvious decisions behind it, so
humans and future agents have context the code alone doesn't show. Newest first.
Keep entries short; the code is the source of truth for *what*, this is the *why*.

---

## R^21 cannot carry the fork figure's move-index curve (2026-08-04)

Feasibility check on projecting `space_fork_grid_pca` through the ReStraV descriptor
instead of the raw activation displacement. New study in
`analysis/violation_divergence/fork_expanding_curvature/`; tables in its README.

**The blocker is dimensional, not statistical.** `restrav_features` needs 8 sampled
positions (14 plies at STRIDE=2) to fill d1..d7 and θ1..θ6. The fork window has 4 sampled
positions before *t0* (8 plies). A sliding window at W=8 therefore has zero curve points
free of the branch, so the shared prefix at the origin and the spray at *t0* — the figure's
entire content — are outside the descriptor's domain. W ≤ 4 keeps a prefix but returns
R^13 at most. No W does both.

**Gotcha worth remembering: `restrav_features` truncates silently.** `d[:, :7]` and
`theta[:, :6]` on a short window return 11/13/15/17/19 columns with no error (and an
obscure `IndexError` from `moment4` at W=2), and the column *meanings* shift — at W=5,
column 4 is θ1, not d5. Anything slicing it from a variable window must assert the width.

**The expanding window anchored at *t0* was prototyped and rendered.** It is causal and
event-locked, but it starts 14 plies after the branch and 13 of its 21 coordinates are then
frozen exactly (measured 0.000e+00) — the first steps since the branch never change again,
so only the 8 cumulative moments move, and their travel is only 1.2–2.1× a null with the
temporal order shuffled away.

**A large effect size here is the denominator, not the descriptor.** Illegal-vs-control
separation goes 0.09–0.45 (the current per-position plane) → 1.2–1.6 (R^21), but the *same
raw displacement* averaged to one point per branch gives 0.89–1.87, i.e. larger, in every
space. Collapsing the position axis removes the within-group spread that dominates the
denominator. A label-permutation null over all 28 relabellings of the 8 branches clears
nothing (best p=0.18 against a 0.036 floor). Also noted: an unscaled PCA of R^21 is a plane
of turning angles (PC1 puts 1% of its energy on the step-length block, since d is in
activation-norm units and θ in degrees), so the planes are fit after a diagonal unit-RMS
rescale that keeps zero displacement at the origin.

**Recommendation:** keep the raw-displacement plane for the fork; if R^21 belongs in the
study it belongs as one point per branch over the whole post-*t0* stretch, as a §4.6
embedding scatter over the full 13,531-branch population, not at n=1.

---

## The port runs, and two of its reference numbers do not survive contact (2026-08-04)

Three agents on `finish/*` branches — packaging, the checkpoint copy, the flagged defects —
merged here. The tree went from "code moved, nothing verified" to installable, loaded with
its seed-42 weights, and 25/25 on `final_validation` with no skips. Per-agent records in
`docs/finish_notes/`.

**The port is faithful, and the proof is exact.** `checkpoints/{batch_topk,cosine_spade,
spade_revival}/trainer_0` reproduce their committed reference rows to 0.000000 on L0,
variance explained, coverage and reconstruction F1. Identical inputs, identical outputs. That
exercises SAE loading, standardization, the Karvonen adapter and the F1/coverage math end to
end, so nothing below is a port defect.

**Wave 3's stated acceptance number is wrong.** It gave VE = 0.943 for TopK on
`main/trainer_3`; the committed reference is 0.9867 and the rerun gives 0.98688. 0.943
appears nowhere in the tree. That wrong number has been the gate on the whole port, and
anyone checking against it would have concluded the port was broken. Corrected in
MIGRATION.md and in `final_validation`'s caveat text.

**Two provenance findings, now MIGRATION.md questions 14 and 15.** The activations behind the
five `results/karvonen/` reference rows no longer exist — those rows were committed 50
minutes before the only surviving extraction was written — so they reproduce only to ±0.014
on the assignment-protocol metrics. And `main/trainer_5/ae.pt` is not the run its config
names: it reruns at VE 0.973 where its wandb record and eval row both say ≈0.36, while
trainer_0–4 all match theirs.

That second one **retracts** the 2026-08-03 entry's reading that the λ=9.77e-5, γ=0.2
configuration never ran. It ran, 36621 steps on cuda:0, and finished degenerate, which is
what the docs reporting it say. `steps=400` and `device=mps` are stale in all six configs,
not just that one. The exclusion decision's justification is unaffected; what is in doubt is
which model the studies that loaded those weights were measuring.

Worth repeating for the next session: gitignored data must be copied into the real tree, not
into an agent worktree, or it evaporates with the worktree. And two agents prepending to
PROCESS.md conflict every time — one merge write-up, per-agent notes elsewhere.

---

## `spade` is installable now; the "101 modules import" claim never went through an install (2026-08-04)

Wave 1's stated verification — `pip install -e .` in a clean env, `import spade`,
`repo_root()` resolving here — had not been run. `spade` was installed in no environment,
`pyproject.toml` declared no dependencies at all (a commented-out placeholder), and every
line of `make setup` was a comment. The 101-module claim came from per-agent worktrees with
ad-hoc `PYTHONPATH`, which is not the thing the package has to do.

Fixed and verified in a throwaway venv built `--system-site-packages` off `spade_env`, so
the torch 2.5.1+cu124 stack is inherited and only the packaging is under test. **102 of 102
modules import cleanly** — the old count was one low, nothing was broken. Also verified from
stock `/usr/bin/python3.11` + pip 22.3.1, which is the fresh-clone path.

Do not `pip install -e` into `spade_env` itself: the editable path points at whatever
worktree ran it and dangles when that worktree goes.

Three real defects behind the missing install: `requirements.txt` named `python-chess` where
the working env has the `chess` distribution; `plotly` and `umap-learn` were pinned with no
importer anywhere we call, which is the exact defect that file's header criticises in the
source repo's version; and `pydantic` was undeclared although `viewer/server.py` imports it.
`pyproject.toml` now reads `requirements.txt` as its dependency list, so there is one list.

Still data-blocked, and reported as such rather than skipped: `final_validation` is 23/24
with the checkpoint round-trip skipped, `saes.registry.entries()` raises on the missing
`ae.pt`, and the wave-3 acceptance number needs weights this tree does not have. Details,
plus the `omegaconf` gap in the vendored TPR fork, in
[docs/finish_notes/env-install.md](docs/finish_notes/env-install.md).

---

## The defects the port agents recorded but did not fix (2026-08-04)

Swept all fifteen `docs/port_notes/` files for concrete code defects and fixed five. The
one that mattered: the sweep merge's baseline filter matched the substring `"SpaDE"`, which
also matches `"CosineSpaDE"`, so every `combined_results.csv` lost its CosineSpaDE row.
Latent in the source — `results/karvonen` had no CosineSpaDE row to lose — and live now
that `results/main` carries all eight baselines. `final_validation` pins it.

The other four are small: a `TypeError` on `--save-counts` (`repo_root` divided, not
called), `branches.py` importing `classify_prefix` through the rules study after it moved
into `othello/games.py`, and two docs that contradicted their own code.

Everything the notes flagged and I did *not* touch is written up with the reason in
`docs/finish_notes/flagged-defects.md`, including the one recommendation worth acting on:
`figure_karvonen_extension` writes its "no data" placeholder over the real figure's
filename, and six shell callers depend on that not raising.

---

## The port is done: 15 parallel agents, one branch each (2026-08-03)

Every zone ported from the frozen source repo — `common`, `othello`, `saes`, `spaces` +
`rules`, `rules/causal`, `training`, the pipeline, `figures`, the four studies, `scripts`,
`docs`, and the viewer. Each agent worked in its own worktree on its own `port/*` branch;
all fifteen are merged. Per-agent records are in `docs/port_notes/`.

**Where it stands:** 101 modules import cleanly, 28 analysis scripts compile, all 38 shell
scripts parse, `final_validation` is 23/24 with the one skip being the checkpoint
round-trip (weights are gitignored). Unit tests pass: 7 taxonomy, 11 divergence kernels,
plus the manifold-capture selftests reproducing the source numbers. What is *not* verified
is any real number — no eval has been run, because neither the checkpoints nor the
activations have been copied in. The wave-3 acceptance check (TopK on
`checkpoints/main/trainer_3` → L0 80.00, VE 0.943, coverage 0.722) is still outstanding and
is the first thing to do with a GPU.

**Seams the agents could not see from inside their own worktrees, fixed on merge:**
- *Three scripts would have destroyed the checkpoint layout.* They looped the eight
  per-type training entrypoints into one `--save_dir`, which writes `trainer_0` eight times.
  `trainer_N` is assigned by position within a single run, and `checkpoints/main/trainer_3`
  being TopK is a fact the registry, every eval CSV and every figure depend on. Now one
  `spade.training.sweep` call, which is also what the source did.
- *The SAE exclusion filtered nothing.* `EXCLUDED_SAE` was ported but never applied;
  `representative_spaces()` now drops it, which is what makes "excluded everywhere" true.
- *Two agents both ported `collect_legal` and `first_illegal_pos`* — byte-identical, since
  one took them from the other. Deduplicated.
- Four figure scripts called `results_dir()` with the old no-arg signature (`TypeError`);
  the viewer exporter used a registry field name that does not exist (`AttributeError`);
  three modules re-declared `LAYER`/`D_MODEL`/`CTX_LEN` locally, which is the drift that put
  nine copies of the dictionary size in the source repo.

**Needs David — the two that affect published claims:**

1. *The manifold-capture headline rests on the checkpoint we just excluded everywhere.*
   "Distance-SpaDE is the purest tiler, support size 0, shattering on 7/8 concepts" is a
   claim **about** the degenerate γ=1e-2 seed-42 checkpoint, and `docs/claims_status.md`
   lists it under "Holds" while retiring the geometry family's conclusion for resting on
   that same checkpoint. The study knows: it carries its own `LOW_VE` marker naming it.
   I aliased `LOW_VE` to `EXCLUDED_SAE` so there is one definition, and changed no
   behaviour. Either the shattering result is re-derived on a healthy SpaDE, or it needs
   stating as a property of a degenerate run, or the exclusion carves out an exception.

2. *A configuration in three documents never ran.* `checkpoints/main/trainer_5/config.json`
   records λ=9.77e-5, γ=0.2, steps 400, device mps. Three source docs report "SpaDE
   initial, λ=9.77e-5, γ=0.2" as a distinct over-regularized run. It is not a run at all —
   it is the γ=1e-2 production run wearing a Mac smoke config with pre-canonicalization
   defaults. That also explains the L0 15.4 / VE 0.352 attribution. The ported docs say so;
   the numbers in any older writeup do not.

Smaller items, each in the relevant `docs/port_notes/` file: BatchTopK *has* been evaluated
since 2026-07-14 (three source docs say it never was — real numbers now in the tables); four
bracket-flip CSVs have no producer in the tree because the A4 driver was never committed
upstream, so those figures redraw but cannot be re-derived; the extracted baseline-merge
helper filters `sae_type` on the substring "SpaDE", which also matches CosineSpaDE; and
`autoencoders_spade_v2` has no row in the artifact table and was mapped by the general rule.

---

## Wave 0 is blocked: there is nowhere to put the netscratch backup (2026-08-03)

**Needs a decision.** The agreed first step was to back up the netscratch tier before
porting. It cannot be done as specified. The tier is ~92 GB
(`spade_multiseed` 68 G, `spade_divergence` 9.3 G, `spade_divergence_v1_onecontrol` 8.4 G,
`spade_straightness` 5.3 G, `spade_bracket` 658 M, `spade_manifold` 471 M, `spade_rules`
69 M) and `/n/home10` has **22 GB free** against a 95 GB filesystem already 77% full.
Copying within netscratch is not a backup — the same 90-day retention applies, which is
what the existing 129 MB `spade_checkpoint_backup` there already runs into.

**Done instead, tonight.** Two things. A full file manifest of every `spade_*` dir — path,
size, mtime — committed under `manifests/netscratch/`, so if the tier expires we know
exactly what was lost. And the part that does fit: everything except the `.pt` checkpoints
is only 738 MB across 1306 files, and is now copied to
`/n/home10/dcao/spade_netscratch_backup/` (outside the repo, mirroring the netscratch
layout). That covers the per-seed CSVs and derived tensors the aggregation and figures
actually read. What remains exposed is ~91 GB of `ae.pt` weights.

**The question for David:** where does 92 GB go — lab storage, tape, cold storage — and is
it all of it, or only the parts that cannot be regenerated? The distinction that matters:
the per-seed CSVs are small and irreplaceable in practice; the checkpoints are large and
*technically* regenerable by re-running training, which is 2-day GPU jobs across 5 seeds
and 14 configurations. Also worth deciding whether `spade_divergence_v1_onecontrol` (8.4 G)
is kept at all — it is the superseded one-control branch set, but three shipped
sub-studies were computed on it and never re-derived.

**Deadline:** retention started 2026-07-22, so ~2026-10-20.

---

## The plan, re-derived against the live repo (2026-08-03)

**Goal:** the 2026-07-22 skeleton was written against the source repo as it stood that
morning, and twelve days of research landed on top of it. Fold all of that drift into the
plan so the port is planned against reality rather than against a two-week-old snapshot.

**Method:** eight parallel read-only surveys of the source repo, one per zone — `src/`,
`analysis/`, top-level artifacts, `shell/` + cluster env, the feature viewer, docs and
constraints, dependencies and packaging, and an audit of the skeleton itself. Findings are
folded into [ARCHITECTURE.md](ARCHITECTURE.md) and [MIGRATION.md](MIGRATION.md).

**What the plan was missing.** Seven code packages (`rules/` and its `causal/` subpackage,
`bracket_flip/`, `violation_divergence/`, `straightness/`, `manifold_capture/`,
`cosine_gamma_range/`), eight analysis studies, a second vendored runtime dependency
(`external/tpr_othello`), the whole figure-style layer (`paper_style.py` +
`FIGURE_GUIDELINES.md`), the feature viewer, 26 of 37 shell scripts, and the netscratch
tier that six studies depend on.

**Three findings that changed the plan rather than extending it:**

- *`rules/` is not a leaf study.* The plan modelled legality as one linear sub-pipeline.
  It is now the layer four other studies import — taxonomy, game streaming, board helpers,
  the checkpoint registry and the representation-space API all live inside it. Left as-is,
  every downstream study would import from a sibling study. The genuinely shared pieces are
  lifted into `spade/othello/`, `spade/saes/registry.py` and `spade/spaces.py`.
- *Three dirs the plan queued for deletion are live inputs.* `8_spade_configs_updated/`,
  `lambda_sweep_results/` and `gamma_sweep_results/` are hard-coded in
  `aggregate_seeds.py` and `figure_all_configs_combined.py`; dropping them removes the
  seed-42 point from every sweep config in the headline figure. The question for them
  changed from *drop?* to *where?*.
- *The per-study convention should be architecture, not repetition.* Five packages
  independently re-derived the same shape: a `study_dir.py` reading `SPADE_<STUDY>_DIR`
  with a netscratch default, numbered resumable steps, and a paired
  `analysis/<study>/make_figures.py` that only draws. That is lifted into
  `spade/common/study.py` and written down as the way a study is built.

**Decisions:**
- *All work happens in this worktree.* The source repo is frozen and read-only, ported
  from and never edited. The cost — no experiment can run here until the pipeline lands in
  wave 3 — is why the port order follows the import graph.
- *Commit at every coherent step, push at every wave boundary.* The skeleton sat
  uncommitted and unpushed for two weeks; that is the failure this rule prevents.
- *`paper_style.py` and `FIGURE_GUIDELINES.md` port in wave 1*, before any figure or
  analysis code, or the port recreates the color-literal drift the guidelines exist to stop.
- *`requirements.txt` gets regenerated, not copied.* The current file cannot build a
  working environment: four pins are wrong, `transformer_lens>=2.0` is incompatible with
  the vendored `circuits`, four hard deps are undeclared, and two pins have no importer.

**Decided with David (2026-08-03):**
- *Lift the shared layer out of `rules/`* — confirmed.
- *Back up netscratch before porting anything else*, as wave 0. ~92 GB on a 90-day clock
  that expires around 2026-10-20, holding the seeds behind every error bar.
- *Exclude the SpaDE γ=1e-2 seed-42 checkpoint everywhere*, from one place, rather than
  keeping the source repo's split where one study excluded it and three others plus the
  public viewer kept reading it. Its "SpaDE is broken for cell-state mapping" conclusion in
  the geometry family therefore has to be re-run or retired before it is quoted again.
- *The default SpaDE configuration is λ=1e-3, γ=1e-4* (`checkpoints/spade_revival/`),
  replacing γ=1e-2. It takes the excluded checkpoint's slot in the viewer roster, which
  stays at eight. Noted for the record: this pair was given as "γ=1e-3, λ=1e-4"; it is
  assigned here the way the existing named checkpoint assigns it, and is a one-line flip if
  the literal reading was intended.
- *CosineSpaDE's default stays γ=1.0*, against the γ-sweep's own recommendation of γ=20.
  The sweep selected 20 on reconstruction F₁, the one metric with an interior maximum
  (0.547 vs 0.530). Coverage runs the other way and is the deciding metric here: 0.537 at
  γ=1.0 against 0.474 at γ=20. The rest of the trade is L0 173 → 98 and loss recovered
  0.991 → 0.984. Worth knowing that the F₁ peak is thin — 0.011 over γ=10, with seed
  dispersion reported for L0 only (n=5) — so the metric that picked 20 is also the one
  whose margin is unquantified.
- *Put a freeze notice in the source repo* so a session that starts there is redirected.

**Open:** thirteen questions remain in MIGRATION.md — the report-file locations, several
data-retention calls, whether the geometry family gets re-run, the one-control divergence
sub-studies, the tracked-figure policy, and the viewer's deployment.

**Status:** plan updated and pushed. No code ported yet.

---

## New repository architecture skeleton (2026-07-22)

**Goal:** clean-slate layout for the SpaDE / OthelloGPT SAE project, built in this
sibling worktree (`spade_sae`, orphan branch `new-architecture`) because the original
repo's top level had grown ~13 parallel artifact/snapshot dirs. The original `src/`
was already clean, so the churn is top-level artifact consolidation + light code
renames.

This is a **skeleton**: every `.py` is a header comment + pseudocode. Real logic ports
over from `../Spade_Research_David_Sumedh_Ba` incrementally. The two worktrees share one
`.git`, so content can be diffed and cherry-picked between them.

**Decisions (user-confirmed):**
- *Installable package.* `src/spade/` is a real package (`pip install -e .` via
  `pyproject.toml`), replacing the original's sys.path-injection hack; `import spade.*`
  resolves everywhere and `make setup` installs it.
- *One training entrypoint per SAE type.* Shared loop lives in `training/core.py`; each
  `train_<type>.py` hands its type's config to `core`. Keep shared logic in `core`, not
  copied into each entrypoint, or the files drift (this trades some DRY for per-type
  clarity — the tradeoff was chosen deliberately).
- *Analysis code + outputs co-located.* Each `analysis/<study>/` keeps its `make_figures.py`
  next to its figures + results, so a study is self-contained.
- *Consolidated artifacts.* 4 `autoencoders*/` → `checkpoints/<experiment>/`;
  `results/karvonen*/` → `results/<experiment>/`; one deduped `figures/`.

**Status:** skeleton committed (5ac74a5). Superseded by the 2026-08-03 entry above, which
keeps all four decisions and corrects the layout they were applied to.

---

## Invariants carried over from the source repo

Moved to [ARCHITECTURE.md](ARCHITECTURE.md) § *Invariants a port must not break* — dispatch
ordering, checkpoint state-dict contracts, hyperparameter canonicalization, the known-stale
configs, the figure-style contract, and the environment pins.
