# Claims status

What currently holds, and what was retired. Several headline claims from the source repo
were killed by red-teaming or overturned by later runs, and the retraction usually lives in
one study README while the original phrasing survives in three other places. **Read the
relevant section before writing any prose, caption or abstract about a study.**

Sources are in the source repo: `analysis/<study>/README.md`, `analysis/violation_divergence/RED_TEAM.md`,
`src/bracket_flip/VERDICT.md`, and `PROCESS.md`.

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
γ=1e-2 checkpoint was excluded for degenerate provenance. Current position: every family
preserves the separation (0.01–0.21). The companion "SpaDE's response cancels 79% under
whole-game averaging" died with the same checkpoint; cancellation now ranges 1–40%.

**Straightness: the championship effect at layer 5.** 0.509 [0.483, 0.535] — not
distinguishable from chance. Adding bootstrap CIs turned a claimed gradual rise into a
late-layer-specific effect.

**CosineSpaDE γ sweep: "γ is not a sparsity lever, the sweep is flat."** Retracted — the
original grid sat entirely in the flat left tail of a sigmoid. Across γ ∈ [0.05, 1e4], L0
runs 179 → 20.5. The operating point is **γ = 20**, not the γ = 1.0 default.

**Geometry: "SpaDE γ=1e-2 is broken."** Re-attributed, not confirmed. That seed-42
checkpoint has a degenerate evaluation (L0 15.4, VE 0.352) where seeds 43–46 of the same
config give 128 and 0.973, and its `config.json` is a Mac smoke config over real GPU
weights. It is now excluded from every study and from the viewer, so this conclusion has to
be re-run on the current set or retired — it cannot be quoted as it stands. The default
SpaDE configuration is λ=1e-3, γ=1e-4.

**SpaDE vs baselines: "the simplex ceiling is a fundamental limitation."** Overturned
inside its own document. CosineSpaDE keeps the identical sparsemax simplex and no-bias
decoder, changes only the score to cosine, and reaches VE 0.979 — the ~0.43 ceiling is
specific to squared-distance scoring under standardized-activation norms. Sections of that
document written before the overturn must be read as distance-SpaDE only.

## Narrowed — quote with the qualifier

- **Distance-SpaDE VE ≈ 0.43** is single-seed. The multiseed aggregate is 0.849 ± 0.278.
  Quote the cosine-vs-distance contrast per seed or qualitatively, never as a clean mean.
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

## Holds

- **Bracket flip, three surviving findings.** The model is mover-relative and barely
  encodes absolute colour (absolute TPR 0.7349, 0.004 *below* the label-only occupancy
  ceiling, while the relative probe clears its ceiling by +23.1). The flip has a specific
  direction (median cosine 0.173–0.184 against a matched non-flipping move, null ≈ 0,
  positive in 1596/1596 strict events, TPR and linear directions agree at 0.822). And it is
  causally used (editing along it moves +1.496 log-units vs +0.121 for random).
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
- **Manifold capture.** No SAE × concept reaches capture. Direction SAEs dilute;
  distance-SpaDE is pure shattering; Lemma 1 confirmed, so SpaDE features are not 1-to-1
  with concepts. CosineSpaDE at VE 0.98 destroys the rule-category curve while the dense
  revived SpaDE preserves it: reconstruction quality ≠ concept-manifold preservation.
- **Straightness.** ReStraV transfers with the sign inverted and a depth flip. Violations
  are graded and localized to t0±1; temperature-1 rollouts are at chance everywhere;
  championship play dissociates rule-OOD from policy-OOD.
- **γ sweep.** CosineSpaDE is strictly dominated at matched sparsity. VE and fidelity
  dissociate — γ = 1e4 holds VE 0.854 while recovering 0.102 of the loss, so VE alone is
  not a health check for this family.
- **Baselines.** TopK (k=80) is best overall: coverage 0.7265, reconstruction F1 0.6841,
  loss recovered 0.9996 at L0 79.9.

## Open defects

- Nothing in the source repo is drawn at final printed width, which blocks camera-ready
  output across the board.
- The bracket-flip TPR probe was never retrained on the integrated event set.
- Three violation-divergence sub-studies were computed on the superseded one-control
  branch set and never re-derived against two controls.
