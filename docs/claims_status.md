# Claims status

What currently holds, and what was retired. Several headline claims from the source repo
were killed by red-teaming or overturned by later runs, and the retraction usually lives in
one study README while the original phrasing survives in three other places. **Read the
relevant section before writing any prose, caption or abstract about a study.**

Sources: `analysis/<study>/README.md`, `analysis/violation_divergence/RED_TEAM.md`,
`analysis/bracket_flip/reports/VERDICT.md`, and `PROCESS.md`.

The evaluation pipeline is verified. `checkpoints/{batch_topk,cosine_spade,spade_revival}/trainer_0`
reproduce their committed reference rows to 0.000000 on L0, variance explained, coverage and
reconstruction F1 ([docs/finish_notes/acceptance-eval.md](finish_notes/acceptance-eval.md)),
so a rerun that disagrees with a committed number is an input problem rather than a port bug.
Two inputs are known bad, and both are flagged below where they bite: the activations behind
`results/main/`'s reference rows no longer exist (MIGRATION.md q14), and
`checkpoints/main/trainer_5/ae.pt` is not the run its config names (q15).

## Retired — do not restate

**Bracket flip: "the flanked square is the one square whose class does not change."**
Not evidence about the model. A perfect board oracle scores 1.0 on that key by
construction; the measured 0.9900 is the product of the probe's marginal accuracies at the
square (0.9950² = 0.990025, agreeing to four decimals), tracks probe accuracy at r = 0.999
across ranks, and is false in 37.5% of individual events. Killed independently twice.
The companion number `p_s_absolute_correct` = 0.99 is a relabelling, not a second
measurement — a probe actually trained on absolute labels reaches 0.7407 held out and gets
the square right in 34.25% of events. The design doc's premise that the before→after
displacement is "a global swap nearly identical across events" is also false: the mean
direction explains 0.61% of the energy, the per-event matched control 68.9%.

**Violation divergence: "an illegal branch diverges further / persists longer than a legal
control."** Inverted in the data. The illegal/control ratio at layer 5 is 0.9793
[0.9719, 0.9854] paired — illegal branches diverge *less*. The design's fairness argument
("the null carries the same handicap") is false: the legal control has the highest KL and
the highest suffix NLL, because a legal alternative genuinely updates the board and the
shared suffix then goes wrong for it. Honest statement of the effect: *the representational
effect of an impossible move is 92–103% of an ordinary legal one.*

**Violation divergence: "the divergence peak migrates with depth."** `peak` and `peak_r`
are max-of-noise — the persistence kernel averages per-row maxima, i.e. E[max] not max E.
A within-row permutation null reproduces the peak to 0.4% at every layer and 79–83% of the
depth drift, and the population mean curve peaks at offset 0 in every layer. Those panels
were deleted as a correctness fix. `peak_layer` as defined is separately vacuous: the
metric normalizes within each space, so a cross-space "excess over the input floor" has no
meaning, and the token-embedding value exceeds every residual peak.

**Violation divergence: R3's directional prediction.** Falsified — the opponent-only
category shows no parity signature (agreement 0.9930, mid-pack).

**Straightness: "SpaDE destroys the curvature signal that every other family keeps."**
Retracted twice. First narrowed to configuration-specific, then dropped entirely when the
seed-42 SpaDE baseline was excluded for degenerate provenance. Current position: every family
preserves the violation separation, mean-θ AUROC 0.39–0.46 against raw layer 5's 0.40, every
bootstrap CI below chance. The companion "SpaDE's response cancels 79% under
whole-game averaging" died with the same checkpoint; cancellation now ranges 1–40%.

**Straightness: everything the random-token class supported.** The class is still built and
stored — it is a block of `games.pt` and `classify.py` still scores it — but as of
2026-08-05 no figure draws it, no figure reads its AUROC rows, and no PCA is fitted on it
(owner's call; the seam is `CLASS_STYLE` in `spade.figures.paper_style`). Three findings go
with it and must not be restated: "every dictionary keeps the random-token curvature shift",
the norm-and-L0 "pass-through / suppressed / inverted" regime table in
`sae_profiles/README.md`, and "PC1 is the random-token axis, everywhere" in
`feature_pca/README.md`. None of the three was refuted — the study stopped measuring the
class. Anything reinstating them has to reinstate the class first.

**Straightness: the championship effect at layer 5.** 0.509 [0.483, 0.535] — not
distinguishable from chance. Adding bootstrap CIs turned a claimed gradual rise into a
late-layer-specific effect.

**CosineSpaDE γ sweep: "γ is not a sparsity lever, the sweep is flat."** Retracted — the
original grid sat entirely in the flat left tail of a sigmoid. Across γ ∈ [0.05, 1e4], L0
runs 179 → 20.5, so γ moves the model a great deal.

What the sweep *recommends* is a separate question from what the default *is*. The sweep
selected γ = 20 on reconstruction F₁ (0.547 against the γ=1.0 default's 0.530) — the one
metric in the sweep with an interior maximum. Coverage runs the other way (0.537 against
0.474), and the default was deliberately kept at **γ = 1.0** on that basis. Do not "correct"
the default to 20 from the sweep README; the decision is recorded in PROCESS.md and
ARCHITECTURE.md § Invariants. Worth knowing that the F₁ peak is thin — 0.011 over γ=10,
with seed dispersion reported for L0 only — so the metric that picked 20 is also the one
whose margin is unquantified.

**Geometry: "SpaDE γ=1e-2 is broken."** Re-attributed, not confirmed, and the checkpoint it
names has since come apart. Three facts about `checkpoints/main/trainer_5` (config
`SpaDETrainer-othello`), the excluded seed-42 SpaDE baseline:

- **The run is λ=9.77e-5, γ=0.2, not γ=1e-2.** wandb has `SpaDETrainer-othello_trainer_5` at
  those values, 36621 steps on cuda:0, finished 2026-03-26 — before the 2026-07-12
  canonicalization that made γ=1e-2 the class default and gave the checkpoint the name it has
  been quoted under since. `config.json` is stale in `steps=400` and `device=mps` only, and
  those two are stale in all six `main/` configs.
- **The run was real and degenerate**, L0 15.43 / VE 0.352 evaluated and 0.3667 at training
  time, where seeds 43–46 of the same config give L0 ≈ 128 / VE 0.973. The exclusion decision
  stands, and is firmer than before: the other recorded SpaDE baseline (γ=0.1) is degenerate
  too, at VE 0.399.
- **The `ae.pt` in that directory is not that run.** It reruns at L0 127.76 / VE 0.973, its
  provenance is not recoverable from the tree, and trainer_0–4 all match their records
  (MIGRATION.md q15). Anything that loaded those weights measured a model other than the one
  the config and the metrics describe.

The geometry conclusion still has to be re-run on the current set or retired; it cannot be
quoted as it stands. The default SpaDE configuration is λ=1e-3, γ=1e-4.

Do not restate PROCESS.md's 2026-08-03 reading that this configuration "never ran". It ran.

**SpaDE vs baselines: "the simplex ceiling is a fundamental limitation."** Overturned
inside its own document. CosineSpaDE keeps the identical sparsemax simplex and no-bias
decoder, changes only the score to cosine, and reaches VE 0.979 — the ~0.43 ceiling is
specific to squared-distance scoring under standardized-activation norms. Sections of that
document written before the overturn must be read as distance-SpaDE only, and the ~0.43
itself carries the seed-42 qualifier under Narrowed.

## Narrowed — quote with the qualifier

- **Distance-SpaDE VE ≈ 0.43** is a seed-42 number, and seed 42 is the odd one out across the
  whole family. In `results/multiseed/all_seeds.csv` every distance-SpaDE config's seed-42 row
  sits at 0.35–0.44 while seeds 43–46 of the *same* config sit at 0.93–0.98 with near-zero
  spread — including the six sweep configs, whose λ and γ are pinned by name, so hyperparameters
  do not explain it (λ=1e-7, γ=5e-3: 0.4325 at seed 42, 0.9639 at 43). CosineSpaDE is flat at
  0.979 across all five seeds. The `SpaDETrainer-othello` aggregate of 0.849 ± 0.278 is
  therefore one outlying seed against four tight ones, not seed dispersion. Quote the
  cosine-vs-distance contrast per seed or qualitatively, never as a clean mean.
- **The revived `lam1e-03 gam1e-04` checkpoint is not a reproduction** of the lost
  original: same config, same seed, VE 0.995 vs 0.434, L0 1808 vs 961. The trainer moved
  underneath it. Label it as that config under the current trainer.
- **`peak_layer` in bracket flip is not one number**: 4 by the TPR probe, 5–6 by an
  independent linear probe. The apparent conflict with the causal depth result is not a
  conflict — board state resolves early, legality consequence late.
- **The linear probe's committed board labels have 53 mislabeled rows in 10,000**, because
  parity was taken from move index rather than engine replay and forfeits desync it. Any
  new probe derives parity from replay.
- **`kl_peak_L5` is misnamed** — it is a logit quantity, not a layer-5 one.
- **Manifold capture**: an empty support originally passed the small-support test
  vacuously and mislabeled SpaDE as capture; it is shattering. The headline regimes read
  from the curve variant because the raw and centered capture metrics are
  concept-independent by construction.
- **Manifold capture's distance-SpaDE result is about an unidentified checkpoint.** "The
  purest tiler — support size 0, shattering on 7/8 concepts" is a claim about
  `checkpoints/main/trainer_5/ae.pt`, which this study alone keeps and marks `LOW_VE` while
  every other study excludes it. Those weights reconstruct at VE 0.973, not the ≈0.43 the
  marker names them for (MIGRATION.md q15). Whether the result survives is the owner's call;
  it is not re-derived here. The geometry family and the deployed viewer build load the same
  file and carry the same caveat.

## Holds

- **Bracket flip, three surviving findings.** The model is mover-relative and barely
  encodes absolute colour (absolute TPR 0.7349, 0.004 *below* the label-only occupancy
  ceiling, while the relative probe clears its ceiling by +23.1). The flip has a specific
  direction (median cosine +0.188 by the TPR direction and +0.205 by the linear one against
  a matched non-flipping move, null ≈ 0, positive in 1596/1596 strict events, the two
  directions agreeing at 0.833). And it is causally used (editing along it moves +1.496
  log-units vs +0.121 for random — the one number here still measured off the ARENA probe
  fits, pending a rerun of the M6 sweep on the Karvonen ones).
- **Violation divergence.** The probe decodes a disc on the illegally played square in
  98.5–99.9% of events; there is no surprise signature; a Standard-SAE feature fires on
  99–100% of empty-square violations versus 25–26% on both legal branches, reproduced by a
  PAnneal feature. Headline: *the legality signal is computed and linearly available, and
  the model does not act on it.* There is no illegal-specific direction — the legal-vs-legal
  control reproduces the same cosine band.
- **Rules, causal.** The un-flip edit moves +1.07 log-mass onto the counterfactual legal
  set vs +0.10 for the random null; flip consequences are written mlp4 → mlp6; the forfeit
  conclusion is computed mid-late with attn7 largest. Nothing retracted. One trap: with a
  shared prefix a full-residual patch recovers 1.0 by construction — a labeled ceiling,
  never a result.
- **Manifold capture.** Direction SAEs dilute; Lemma 1 confirmed on the point-based SAEs, so
  SpaDE features are not 1-to-1 with concepts. CosineSpaDE at VE 0.98 destroys the
  rule-category curve while the dense revived SpaDE preserves it: reconstruction quality ≠
  concept-manifold preservation. "No SAE × concept reaches capture" holds for every space but
  the distance-SpaDE slot, which is under Narrowed.
- **Straightness.** ReStraV transfers with the sign inverted and a depth flip. Violations
  are graded and localized to t0±1; temperature-1 rollouts are at chance everywhere;
  championship play dissociates rule-OOD from policy-OOD.
- **γ sweep.** CosineSpaDE is strictly dominated at matched sparsity. VE and fidelity
  dissociate — γ = 1e4 holds VE 0.854 while recovering 0.102 of the loss, so VE alone is
  not a health check for this family.
- **Baselines.** TopK (k=80) is best overall: coverage 0.7265, reconstruction F1 0.6841,
  loss recovered 0.9996 at L0 79.9. Those come from the `results/main/` reference row, whose
  activations are gone; on the surviving extraction the same weights give coverage 0.7405 and
  reconstruction F1 0.6981 at L0 79.94, VE 0.9869. Deltas across the five rows are mixed-sign
  and within ±0.014, and TopK still leads on both, so the ranking survives — but do not print
  a coverage or F1 figure from this set past two decimals.

## Open defects

- Nothing in the source repo is drawn at final printed width, which blocks camera-ready
  output across the board.
- The bracket-flip TPR probe was never retrained on the integrated event set.
- Three violation-divergence sub-studies were computed on the superseded one-control
  branch set and never re-derived against two controls.
- `checkpoints/main/trainer_5/ae.pt` has no known provenance (MIGRATION.md q15). Three things
  loaded it: the geometry family, manifold capture, and the deployed viewer build.
- Every coverage and reconstruction-F1 number from `results/main/` is good to ±0.014 only —
  the activation set behind those rows no longer exists (q14). VE and L0 are stable.
