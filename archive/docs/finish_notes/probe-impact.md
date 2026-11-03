# Finish note — blast radius of the linear-probe label fix

The mover's colour behind `analysis/linear_probe/activations/` was taken from move-index
parity. Othello passes when a player has no legal move, the ARENA corpus does not mark the
pass, and parity then runs out of phase with the true mover for the rest of the game. The fix
rewrites `linear_probe.pt`, `board_states.pt` and `primary_layer_activations.pt`.

This note is the consumer audit: who reads them, in what role, which committed outputs move,
and what has to be re-run. It changes no code and re-runs no study. Baseline verified before
anything moves: all seven files in `analysis/linear_probe/activations/` still match
`manifests/checkpoints/linear_probe_activations.tsv` byte-for-byte by SHA-256, checked
against the live copy at `../spade_sae`.

## Two corrections to the premise, before the table

**1. Inside these artifacts the damage is 3 games of 200, not 37.** The 37-of-200 figure is
measured over the full 60-move sequence. The artifacts hold turns 5–54 only, and forfeits
concentrate at the very end of a game, so most of those 37 games desync entirely outside the
window. The in-window count is already measured and committed
(`analysis/bracket_flip/reports/DATA_REPORT.md:130-134`):

> replaying the same 200 ARENA games through `probe_labels` reproduces `board_states.pt` on
> **9,947 / 10,000** rows. The 53 differing rows are *exactly* the rows where `t % 2` names
> the wrong mover (**3 of the 200 games**).

So: **53 rows, 3 games (1.5%), ~18 of each affected game's 50 rows.** The per-game
concentration is real — inside those three games the desync is persistent, which is why 53
rows land in 3 games and not 53 — but the affected-game fraction is 1.5%, not 18.5%. Any
severity argument built on 18.5% is a severity argument about the wrong window.

Three independent measurements agree on the shape:

| corpus | window | positions wrong | games affected | source |
|---|---|---|---|---|
| ARENA 200 games (**these artifacts**) | 5–54 | 53 / 10,000 (0.53%) | **3 / 200 (1.5%)** | `DATA_REPORT.md:130-134` |
| Karvonen 400 games | 5–54 | 70 / 20,000 (0.35%) | 12 / 400 (3.0%) | `$SPADE_BRACKET_DIR/probe/manifest.json` |
| Karvonen 2,000 games | full 59 tokens | 129 / 66,000 cells (0.20%) | 317 / 2,000 desync *somewhere*, median first desync at position **56**, only **4** at or before position 40 | `probe_readout/RESULTS.md:38-43` |

The third row is the one that explains the first: ~16% of games desync somewhere, matching the
ARENA 18.5%, but the median first desync is at position 56 — past the window's end.

**2. A parity desync swaps `mine` ↔ `theirs` and never touches `empty`.** On a mislabeled row
the occupancy channel is still perfectly correct; only the ownership contrast is inverted.
Every consumer splits on which of the two it uses:

- occupancy questions (`empty` vs disc, `p_disc`, the `empty` class frequency) — **immune**;
- ownership questions (`mine` vs `theirs`, parity/frame agreement, the un-flip edit direction,
  per-class mean projections) — **exposed**, and to the worst kind of label noise, since a
  desynced row is not noisy but exactly inverted.

## The consumer list, verified

The list I was handed named eight modules (described as nine). Two are false positives — they
matched the filename `board_states.pt` but read a different tensor:

| named module | verdict |
|---|---|
| `src/spade/extraction/extract_activations.py` | **does not read these artifacts.** Writes `data/activations/{train,test}_board_states.pt` — the SAE evaluation set, `[N,8,8,3]` one-hot, millions of rows. |
| `src/spade/evaluation/run_eval.py` | **does not read these artifacts.** Reads the `data/activations/` set written above. |

Both deserve a separate flag, not a rewrite. `data/activations/`'s labels come from upstream
`games_batch_to_state_stack_mine_yours_BLRRC`, which derives the perspective from `i % 2`
(`external/Karvonen_Research/circuits/othello_utils.py:90-95`) — **the same defect,
independently, in the artifact every SAE board-reconstruction F1 and coverage number in
`results/` is scored against.** It is also Karvonen's own convention, so "fixing" it breaks
comparability with the baseline it exists to reproduce. Owner's call, separate change.

One more non-consumer worth recording, because a reader would otherwise count it as a seventh:
`analysis/legality_probe/README.md:81` states "Probe fit set: `analysis/linear_probe/activations/`".
**The code does the opposite and says so** — `src/spade/legality_probe/spaces.py:64-70` points
`PROBE_ACTS` at `study_dir("legality")/acts/`, with the comment "NOT
analysis/linear_probe/activations/ — that set holds forfeit positions and 53 mislabeled rows".
`SPADE_LINEAR_PROBE_ACTIVATIONS` appears nowhere in that package. The README line is wrong and
should be corrected in the same pass, or the next audit re-derives this.

The six real consumers, each read end to end:

| consumer | artifact | role |
|---|---|---|
| `src/spade/rules/causal/unflip_edit.py` | all three | directions **and** labels |
| `src/spade/bracket_flip/tpr_probe.py` | `board_states.pt` (+ `all_layer_activations.pt`) | labels, train and eval |
| `src/spade/bracket_flip/tpr_intervene.py` (`--corpus arena`) | all three | directions **and** labels |
| `src/spade/bracket_flip/redteam.py` | `linear_probe.pt` | decoder |
| `src/spade/bracket_flip/act_geometry.py` (`corpus="arena"`) | `linear_probe.pt` | directions |
| `analysis/violation_divergence/probe_readout/decode.py` | `linear_probe.pt` | decoder |

Clean by construction, checked: `probe_space.py`, `geometry_tables.py`, `robustness.py`,
`sae_arm.py`, `events.py`, `extract_acts.py` all work off the bracket study's own 400-game set
at `study_dir("bracket")/probe/`, whose parity comes from engine replay
(`events.probe_labels` → `next_mover(board)`).
`analysis/geometry/linear_probe_directions_pca/…` is an unported skeleton — dormant.

## Consumer → dependence → affected outputs → severity

**S1** a published headline changes; **S2** published diagnostics move by more than their
quoted precision; **S3** a committed output refreshes but the headline survives; **S4** control
arm, nothing committed; **S5** unaffected.

| consumer | artifact / kind of dependence | committed outputs affected | recompute or cached? | severity |
|---|---|---|---|---|
| `unflip_edit.py` | all three. `linear_probe.pt` → per-square `mine−theirs` and `empty−theirs` directions; `board_states.pt` + `primary_layer_activations.pt` → per-(square, class) **mean projections `mu`**, which set the β=1 edit magnitude | `analysis/rules_causal/results/unflip_edit.csv`; `analysis/rules_causal/README.md` § Headline (+1.07 / +0.10 / +3.04 / +0.41 / +1.24 / +6.4); `docs/claims_status.md:196` § Holds; `MIGRATION.md:75` acceptance gate | recomputes; the CSV on disk will not update itself | **S1** |
| `tpr_intervene.py --corpus arena` | all three, same roles | **`analysis/bracket_flip/reports/RESULTS_A6.md` in its entirety** — job 35961417 is the ARENA run, and its head-to-head table, scale sweep and both controls are all `mu`-calibrated (+1.496 / +0.121 / +1.051 / +1.792 / +2.006). No CSV: non-primary corpus writes `*_arena.csv`, and `intervention_manifest.json` confirms the committed CSVs are `probe_corpus: karvonen` | recomputes; the **report** is the cache | **S1** |
| `tpr_probe.py` | `board_states.pt` as **labels**, for the fit (rows 0:8000) and the reported accuracy (rows 8000:10000) | `PROBE_REPORT.md` § 3 (all 8 layer rows), § 2 (class frequencies 0.461/0.290/0.249), § 4 (per-class accuracy and the class counts n = 59,000 / 37,210 / 31,790), § 6.7 (0.98838, "gap is 0.57 points"), § 5 filler geometry and λ* tables, § 7 (0.9828/0.9827); `docs/bracket_flip_design.md:262-264` "Gaps: 0.69 and 0.57 points" | **8 cached `.pt` fits in netscratch that nothing invalidates** | **S2** |
| `redteam.py` | `linear_probe.pt` as the `committed_linear_ARENA` **decoder**; plus `tpr_A1_ARENA`, which *is* `tpr_probe`'s cached fit and so label-derived at one remove | `VERDICT.md`'s `committed_linear_ARENA` and `tpr_A1_ARENA` columns throughout (held-out 0.9873 / 0.9816, contract keys 0.9900 / 0.9845 / 0.9711 / 0.9616, attack-1 marginals 0.9950², attack-4 control 0.0085, attack-6 confusion matrices, `peak_layer` 4); `ROBUSTNESS.md:129-131`; `RESULTS_A4.md`'s linear and TPR columns | recomputes, but inherits the stale TPR cache | **S2** |
| `decode.py` | `linear_probe.pt` as a **decoder** only. Labels are engine replay computed inside the module; the docstring is explicit that parity is not `t % 2` | 12 CSVs under `results/violation_divergence/probe*`; `probe_readout/figures/f7_probe_readout{,_rollout}.{png,pdf}`; that study's RESULTS.md end to end; `analysis/violation_divergence/{README,RESULTS}.md`; `RED_TEAM.md` § F3; `docs/claims_status.md:190-195` § Holds | recomputes; the CSVs and 4 figure files are caches | **S3** |
| `act_geometry.py corpus="arena"` | `linear_probe.pt` as directions; `tpr_probe.load()` for the TPR arm | none directly. `geometry_tables.py:48` calls `retain_directions(source)` with no corpus, so it takes `PRIMARY_CORPUS = "karvonen"`; every committed bracket geometry CSV is the clean corpus. `RESULTS_A4.md` predates that switch and prints the ARENA values, so it is covered by the redteam row above | recomputes | **S4** |
| `extract_activations.py`, `run_eval.py` | neither reads these artifacts | none from this fix | — | **S5** |

Largest committed *surface*: `decode.py` (12 CSVs + 4 figure files + three README/RESULTS
files). Largest committed *exposure*: `unflip_edit.py` — one CSV, but every number in it is a
published "Holds" claim, it is the wave-4 acceptance gate in `MIGRATION.md:75`, and it is the
only consumer with no clean-corpus alternative anywhere in the tree (`PROBE_DIR` is hardcoded;
there is no `--corpus` switch, unlike its bracket-flip twin).

## Where the per-game concentration actually bites

Only one consumer has game structure in it. `tpr_probe`'s eval block is rows 8000:10000 —
the last 40 games — so whether the reported accuracies move at all depends on whether any of
the three desynced games lands in games 160–199. Expected count is 3 × 40/200 = **0.6**, so
the likely answers are zero or one:

- **zero affected eval games** → the § 3 table moves only through the refit, i.e. in the
  fourth decimal;
- **one** → arithmetic straight off PROBE_REPORT § 4's own class counts, which sum to
  2,000 rows × 64 = 128,000 cells with 69,000 occupied (34.5 discs per row on average, and
  more than that on the late-game rows a desync actually hits): ~18 rows × 35–45 occupied
  squares ≈ 620–810 cells scored wrong ⇒ **0.5–0.65 points**, direction known once the games
  are known.

That is one line the fix can print and it decides the whole § 3 story: **which game indices of
the 200 desync inside turns 5–54, and are any ≥ 160?**

The good news underneath: **the same mislabeled rows are scored wrong at every layer**, so an
eval-block hit moves all eight accuracies together. `docs/bracket_flip_design.md:60-61` notes
that `probe_metadata.json`'s 0.9896 at layer 5 against 0.9870 at layer 6 is the number that
chose layer 5 for the whole bracket study — a 0.257-point margin, smaller than the shift above.
But the shift is common-mode and the margin is a difference of differences,
so the layer-5 choice is second-order safe. Say that explicitly when the table is republished,
because the naive reading is that the study's layer choice is inside the error bar.

No other consumer is event-locked on these rows. All five others consume the artifacts as
pooled aggregates — a fit over 8,000 rows, an accuracy over 2,000, a per-(square, class) mean
over thousands. The event-locked studies (bracket events, divergence branches, rules items) are
locked to *other* corpora and inherit these artifacts only as directions and calibration
constants. The amplifier has nothing to amplify through.

`probe_readout/RESULTS.md:38-43` even bounds the decoder case empirically from the other side:
on the 129 cells where the two conventions disagree the probe is at chance under either frame
(0.465 engine, 0.533 parity, n = 129), "so the choice moves no headline number."

## Derived caches that will silently not update

None of this fails loudly, which is why it is the actionable part.

**1. Eight TPR probe fits in netscratch, trained on the old labels.**
`$SPADE_BRACKET_DIR/tpr_probes/tpr_r52_f2_relative_layer{0..7}.pt` exist, dated Jul 29 /
Aug 5. `tpr_probe.train()` writes a `labels` fingerprint into each artifact —
`_fingerprint` = `shape:sum`, which *would* change, since swapping `mine`↔`theirs` changes the
label sum — but **`tpr_probe.load()` never reads that field back**. The tripwire was built and
is not armed. Three consumers load these fits without refitting: `redteam.py`
(`tpr_A1_ARENA`), `act_geometry.py(corpus="arena")`, `tpr_intervene.py(--corpus arena)`. After
the fix each reports an old-label probe against new-label data, silently.
`RESULTS_A3.md:297-303` already filed this cache path as a defect. **Delete the eight files as
part of the fix**, or have `load()` compare the fingerprint and raise.

**2. `probe_metadata.json` is not in the rewrite list and must be.** `tpr_probe.main()` reads
`layer_accuracies` from it for the `linear` and `gap` columns of PROBE_REPORT § 3. Rewrite
`board_states.pt` without it and the table compares a **new** TPR accuracy against an **old**
linear accuracy. It also holds `primary_layer` and the 8000/2000 split.

**3. Row order and row count are load-bearing across files the fix does not touch.**
`board_states.pt` row `i` is game `i//50`, turn `5 + i%50`, and that indexing is shared with
`all_layer_activations.pt` (`[8, 10000, 512]`, `all[5]` bit-identical to
`primary_layer_activations.pt`, verified in PROBE_REPORT § 2) and `move_indices.pt`. Neither is
in the rewrite list, and `tpr_probe` pairs `all_layer_activations.pt` with `board_states.pt`
directly. **If the fix drops the desynced rows, reorders games, or changes the turn window,
tpr_probe trains on misaligned pairs and says nothing.** `unflip_edit` and `tpr_intervene`
raise on a count change (`reshape(len(acts), 64)`) but not on a reorder. Rewrite labels in
place at identical count and order, or rewrite all five tensors together.

**4. The checksum manifest.** `manifests/checkpoints/linear_probe_activations.tsv` pins all
seven files; three rows change, four if `probe_metadata.json` is regenerated.

**5. Committed results are caches.** `unflip_edit.csv`, the 12 `probe*.csv`, the four F7 files
and every report table sit on disk and will not notice the artifacts moved.

## Published numbers a reader has to revisit

Ranked by how badly a reader is misled if nothing is done.

1. **`docs/claims_status.md:196-197` § Holds, "Rules, causal": "+1.07 log-mass … vs +0.10 for
   the random null"**, expanded in `analysis/rules_causal/README.md:54-60` (+3.04 / +0.41 at
   β=2, `settled` +1.24, `placed` +6.4) and used as the `MIGRATION.md:75` acceptance gate.
   Computed end to end from the three artifacts. The **contrast** is robust — `flipped` and
   `settled` share one direction and one `mu`, so a calibration shift moves them together, and
   the ~10× ratio over the random null is not `mu`-sensitive. The **absolute log-masses are
   calibration-dependent and will move**, which means the acceptance gate as written will not
   reproduce and must be restated against the new artifacts.
2. **`analysis/bracket_flip/reports/RESULTS_A6.md`.** The whole report is the ARENA run.
   Doubly stale: the label fix invalidates its calibration, and it was *already* superseded by
   the Karvonen rerun (job 37355117, commit `248d9d9`) that `analysis/bracket_flip/README.md:62`
   reports at **+2.380 nats [2.299, 2.454]**. Nothing in A6 says so.
3. **The `+1.496` / `+0.121` pair**, at `docs/claims_status.md:187-189`,
   `docs/bracket_flip_design.md:34,45`, `docs/port_notes/bracket.md:141`. `claims_status`
   already flags it as "the one number here still measured off the ARENA probe fits, pending a
   rerun of the M6 sweep on the Karvonen ones" — **that rerun has landed**: the committed
   `intervention_summary.csv` is `probe_corpus: karvonen` with `linear / flipped / β=1` at
   **2.0348 [1.9701, 2.1041]** against **0.1905** random. The label fix does not repair this;
   updating the three docs to the committed Karvonen numbers does, and should happen
   regardless.
4. **`PROBE_REPORT.md` § 7's provenance claim** that the committed labels are "derived by
   engine replay". That is the sentence the fix falsifies, and the rest of the report rests on
   it. Its cited `train_linear_probe.py:76-81` does not exist in this tree (below).
5. **The 0.9896 credential chain.** `probe_metadata.json`'s layer-5 accuracy is quoted as the
   probe's licence in at least ten places — `docs/bracket_flip_design.md:60,262-264`,
   `PROBE_REPORT.md:5,114,132,267-271`, `RESULTS_A3.md:279`, `probe_readout/RESULTS.md:6,57`,
   `docs/violation_divergence_design.md:180`, `ARCHITECTURE.md:67`, `analysis/README.md:13`,
   `analysis/geometry/SUMMARY.md:7`, `analysis/linear_probe/README.md:4`,
   `docs/session_logs/7-14-2026-multiseed-error-bars.md:175` — and in three mutually
   inconsistent roundings (0.9896 / 0.9884 / ~0.99 / ~0.97). All are accuracies against
   `board_states.pt`. Fix the number once and point the others at it.
6. **PROBE_REPORT § 3's eight-layer table**, § 6.7's 0.98838 and "the gap is 0.57 points",
   echoed at `docs/bracket_flip_design.md:264`. Movement bounded at 0.5–0.65 points if a
   desynced game is in the eval block, fourth-decimal otherwise. § 6.6's "layers 6 and 7 miss
   the linear probe by 1.5–1.9 points" survives either way.
7. **PROBE_REPORT § 2's class frequencies "0.461 / 0.290 / 0.249" and § 4's class counts
   "59,000 / 37,210 / 31,790".** The 0.461 and the 59,000 are exact — desyncs never touch
   `empty` — while the `theirs`/`mine` split shifts. Anyone re-deriving these sees two of three
   change and should not read that as a different corpus.
8. **`VERDICT.md`'s `committed_linear_ARENA` and `tpr_A1_ARENA` columns** (row 21 names the
   artifact). The red team exists to be the independent check; a check on stale weights is
   worse than no check. `VERDICT.md:405-408` is aware of the label defect and argues the ARENA
   TPR still scores 0.9816 off-corpus — that argument is fine and does not survive being left
   un-rerun.
9. **`docs/claims_status.md` § Narrowed, "53 mislabeled rows in 10,000".** Nine places to
   retire: `claims_status.md:164`, `ARCHITECTURE.md:317`, `analysis/README.md:15`,
   `analysis/linear_probe/README.md:9`, `analysis/bracket_flip/README.md:195`,
   `docs/pipeline.md:69`, `VERDICT.md:406`, `DATA_REPORT.md:130-134`,
   `docs/finish_notes/{checkpoint-data.md:102, docs-consistency.md:67}`. Plus the code comment
   at `src/spade/legality_probe/spaces.py:66`. "Any new probe derives parity from replay"
   becomes "every probe does".
10. **`docs/claims_status.md:190-195` § Holds, "Violation divergence": a disc decoded in
    98.5–99.9% of events.** This is `empty`-vs-disc, the one contrast a desync cannot reach,
    decoded by a probe whose *fit* saw 53 inverted-ownership rows. Safe — refresh the CSVs and
    confirm. The ownership metrics in the same study (`probe_parity.csv`, `p_mover_colour`, the
    78.6% recolor, R3's "agreement 0.9930") are the ones to actually look at, and they are the
    numbers `RED_TEAM.md` § F3 and `analysis/violation_divergence/RESULTS.md:9` restate.
11. **`docs/claims_status.md` § Narrowed, "`peak_layer` … 4 by the TPR probe, 5–6 by an
    independent linear probe".** Survives: the "4" is double-sourced — `robustness.py` gets it
    at 6/6 seeds on the clean Karvonen set (`ROBUSTNESS.md:45`) and `redteam.py` gets it from
    the ARENA fit. The clean source carries the claim; only the red team's copy needs the rerun.

Not affected despite naming a probe: the bracket-flip cosine headlines (+0.188 TPR / +0.205
linear, 1596/1596, agreement 0.833), the absolute-vs-relative TPR result (0.7349, 0.004 below
ceiling, +23.1), the variance shares (0.61% / 68.9%), every `robustness_*.csv`, the SAE arm,
and all of `analysis/legality_probe/` — all from the Karvonen probe set or the study's own fits.

## What actually needs re-running, prioritised

1. **Delete `$SPADE_BRACKET_DIR/tpr_probes/tpr_r52_f2_relative_layer{0..7}.pt`, then refit.**
   First, because three consumers load them silently. Regenerate `probe_metadata.json` in the
   same pass or the `gap` column is meaningless. Cost: 8 layers × 120 epochs × 8,000 rows —
   minutes on an H100, order an hour on CPU. Print the desynced game indices while you are in
   there (see below).
2. **Re-run `unflip_edit`**, replace `analysis/rules_causal/results/unflip_edit.csv`, the
   README headline block, `docs/claims_status.md:196` and the `MIGRATION.md:75` gate. The only
   published headline with no clean-corpus fallback. Cost:
   `scripts/slurm/studies/rules/submit_rules_causal.sbatch` (kempner_h100, 1 GPU, 8 h wall) —
   but `$SPADE_RULES_DIR/pairs/items.pt` is cached so `mine_forfeits` is skipped, and the
   other two stages in that job read none of these artifacts and need not be re-run.
3. **Re-run `redteam.py`** once step 1 lands; update the `committed_linear_ARENA` and
   `tpr_A1_ARENA` columns in `VERDICT.md`, `ROBUSTNESS.md:129-131` and `RESULTS_A4.md`. Fits
   nine linear probes at 400 epochs plus decodes; tens of minutes on the bracket job's GPU.
   Its report JSON is not committed — decide where it lands before running.
4. **Re-run `decode.py`** (both variants) to refresh the 12 `results/violation_divergence/probe*.csv`
   and redraw F7. All inputs cached (`$SPADE_DIVERGENCE_DIR/acts/resid_post_5.pt`, 2.1 GB), no
   model forward pass, no GPU. Expect third-decimal movement; the point is that the CSVs stop
   disagreeing with the artifact, not that the finding changes.
5. **Decide what happens to `RESULTS_A6.md`.** It is superseded on two axes at once. Either
   re-run `tpr_intervene --corpus arena` after step 1 and reissue it as the off-corpus control,
   or mark it superseded by the Karvonen run and stop citing +1.496 anywhere. The second is
   cheaper and is what `analysis/bracket_flip/README.md:62` already implies.
6. **Update the checksum manifest**, retire the nine copies of the 53-row caveat, and fix
   `analysis/legality_probe/README.md:81`.
7. **Do not touch** `data/activations/`, `run_eval.py` or `extract_activations.py` in this
   change. Same defect, different artifact, upstream's convention, and every SAE evaluation
   number in `results/` is scored against it.

## What I did not compute, and what it would cost

This environment has no `.venv` and neither `torch` nor `numpy`, and the artifacts are `.pt`
tensors, so nothing inside them could be read. Everything above is from source, from committed
CSV/JSON/markdown text, from the netscratch manifests, and from `sha256sum`. Four cheap
diagnostics would close the remaining questions, and none needs a study re-run — a throwaway
script against the artifacts is enough:

- **Which of the 200 game indices desync inside turns 5–54, and are any ≥ 160.** This is the
  one number that decides whether PROBE_REPORT § 3 moves by 0.0001 or by 0.8 points. The fix is
  already computing the desync mask; emit the per-game table beside it.
- **`cos(old, new)` per square for `linear_probe.pt`'s `mine − theirs` direction, and for
  `empty − theirs`.** One pair of numbers bounds `decode.py`, `redteam.py` and
  `act_geometry.py` at once — all three use the probe only as directions. Seconds, once both
  probes exist.
- **`max |Δmu|` relative to the class separation**, per square and edit family, from
  `unflip_edit.probe_directions`. Bounds how far +1.07 moves without running the model.
- **The refit layer-accuracy table**, only obtainable by refitting, which is step 1 anyway.

Two blockers for regenerating rather than patching: the ARENA corpus
(`reference/research_materials/ARENA/.../board_seqs_id_small.npy`) **is not in this tree** —
`reference/` holds a README and an empty `scoping/` — and
`analysis/linear_probe/train_linear_probe.py` **is a docstring-only skeleton** with its
functions commented out, so the code that produced `linear_probe.pt` does not exist here
either. Patching labels in place avoids both; regenerating needs both restored first.
