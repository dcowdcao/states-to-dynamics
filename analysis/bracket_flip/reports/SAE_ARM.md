# SAE arm: does the bracketing flip survive an SAE bottleneck?

A tensor-product-representation probe is trained on each SAE's **latent code**
(not the raw residual), then the flip geometry is redone in that space. Nine
spaces: raw layer-5 plus one canonical checkpoint per baseline SAE family.
Descriptive only — no causal edits (a latent-space edit would have to pass back
through the SAE decoder to move the model). Numbers in
`analysis/bracket_flip/results/sae_arm.csv`; job 35978337, ~1.5 min on one H100.

## Per-space table

| space | acc_rel | dissoc. gap | cos_flip | cos_null | p(flip>0) |
|---|---|---|---|---|---|
| raw L5 | **0.983** | +0.984 | **+0.198** | −0.002 | 1.00 |
| BatchTopK | 0.973 | +0.964 | +0.108 | −0.001 | 1.00 |
| TopK | 0.972 | +0.968 | +0.093 | −0.001 | 1.00 |
| Standard | 0.978 | +0.979 | +0.075 | −0.001 | 1.00 |
| GatedAnneal | 0.944 | +0.943 | +0.071 | −0.001 | 1.00 |
| PAnneal | 0.979 | +0.972 | +0.045 | −0.001 | 1.00 |
| GatedSAE | 0.958 | +0.955 | +0.060 | −0.001 | 1.00 |
| **SpaDE** | **0.762** | +0.455 | +0.060 | −0.001 | 1.00 |
| **CosineSpaDE** | **0.734** | +0.331 | +0.060 | −0.001 | 1.00 |

## What holds

**The flip's directional signature survives every SAE.** `cos_flip_median` is
positive in all nine spaces (raw +0.198, best SAE BatchTopK +0.108) against a
per-space null at ≈0, positive in ≥99.5% of strict events everywhere. No family
erases it. It is weaker than raw in every SAE — the bottleneck attenuates the
direction, it does not remove it — and the point-based, dense-ish top-k families
(BatchTopK, TopK) preserve it best.

**SpaDE and CosineSpaDE collapse board-state decodability.** Both drop to
acc 0.76 / 0.73 while every other family stays 0.94–0.98. That is the
occupancy-ceiling collapse F3/robustness already found for absolute colour, now
for the whole board in these two latent spaces: their `acc_sq_floor`
(0.54–0.58) shows the probe is barely above chance on occupied squares. Their
dissociation gap is correspondingly low (+0.33–0.46 vs +0.94–0.98) — but that is
**accuracy-limited, not a separate finding** (ROBUSTNESS.md: the gap scales with
accuracy²). The accuracy-robust readout, `cos_flip`, is low but positive for
both.

## The SpaDE-vs-Cosine question

**Both SpaDE variants fail together here** — same board-decode collapse
(0.76 / 0.73), same low cos_flip (0.060 / 0.060). CosineSpaDE does not rescue the
flip representation.

This section was written against the straightness claim that SpaDE destroys the
trajectory signal while CosineSpaDE keeps it. **That claim is retired** — every
family preserves the separation, 0.01–0.21 (`docs/claims_status.md`) — so there is
no split to fail to reproduce, and the "CosineSpaDE preserves what SpaDE discards"
contrast has no live counterpart to be signal-specific against. What stands is the
measurement: neither prototype variant decodes the board well enough here to carry
the flip.

## Caveats

- SpaDE codes live on a simplex; compare cluster *structure* across families,
  never coordinate magnitudes (this is why F8 reads structure, not scale).
- `p_s_unchanged` and the dissociation gap are accuracy² floors — quote
  `cos_flip` for cross-family comparison, `acc_rel` for "does this space decode
  the board at all".
- Descriptive only. The causal replication (F6) stays on raw, where the edit
  direction maps to the residual the model actually runs on.
