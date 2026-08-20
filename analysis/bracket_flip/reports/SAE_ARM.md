# SAE arm: does the bracketing flip survive an SAE bottleneck?

A tensor-product-representation probe is trained on each SAE's **latent code**
(not the raw residual), then the flip geometry is redone in that space. Eight
spaces: raw layer-5 plus one canonical checkpoint per baseline SAE family.
Descriptive only — no causal edits (a latent-space edit would have to pass back
through the SAE decoder to move the model). Numbers in
`analysis/bracket_flip/results/sae_arm.csv`; job 40535454, 2 min on one H100.

Every probe below is fit under `spade.probes.train`'s protocol — split by game,
early stop on validation accuracy — on the space's latent code. The canonical
corpus has no SAE latents, so the probes are local; the protocol is not.

## Per-space table

| space | acc_rel | dissoc. gap | cos_flip | cos_null | p(flip>0) |
|---|---|---|---|---|---|
| raw L5 | **0.975** | +0.962 | **+0.181** | −0.002 | 1.00 |
| BatchTopK | 0.968 | +0.960 | +0.082 | −0.001 | 1.00 |
| TopK | 0.965 | +0.949 | +0.072 | −0.001 | 1.00 |
| Standard | 0.977 | +0.979 | +0.068 | −0.001 | 1.00 |
| GatedAnneal | 0.930 | +0.921 | +0.066 | −0.001 | 1.00 |
| PAnneal | 0.976 | +0.975 | +0.039 | −0.001 | 1.00 |
| GatedSAE | 0.953 | +0.943 | +0.054 | −0.001 | 1.00 |
| **CosineSpaDE** | **0.941** | +0.911 | +0.049 | −0.001 | 1.00 |

`SpaDETrainer-othello` is no longer a row: `spade.spaces.EXCLUDED_SAE` drops the
degenerate seed-42 checkpoint project-wide, and the nine-row version of this
table predated that. Distance-SpaDE therefore has no measurement here at all, and
the prototype-variant claim below rests on CosineSpaDE alone.

## What holds

**The flip's directional signature survives every SAE.** `cos_flip_median` is
positive in all eight spaces (raw +0.181, best SAE BatchTopK +0.082) against a
per-space null at ≈0, positive in ≥99.8% of strict events everywhere. No family
erases it. It is weaker than raw in every SAE — the bottleneck attenuates the
direction, it does not remove it — and the point-based, dense-ish top-k families
(BatchTopK, TopK) preserve it best.

## Retracted: "SpaDE and CosineSpaDE collapse board-state decodability"

This section used to report that both prototype variants drop to acc 0.76 / 0.73
while every other family stays 0.94–0.98, and read that as an occupancy-ceiling
collapse in those latent spaces. **It was an artifact of the fitting recipe.**
The probes were fit for a fixed 120 epochs at weight decay 2.0 on 8,000 rows, a
recipe tuned for a 512-dimensional residual; an 8,192-dimensional code under it
was simply under-fit. Refit under `spade.probes.train`'s protocol — same data,
stopped on a validation plateau instead of a step budget — CosineSpaDE reaches
**0.941**, inside the 0.930–0.977 range every other family occupies.
`acc_sq_floor` follows (0.886 against the old 0.54), and the dissociation gap
with it (+0.911 against +0.33).

Nothing about CosineSpaDE is special here. It sits mid-pack on accuracy and at
the low end on `cos_flip` (+0.049), where GatedSAE and PAnneal also sit; that
ordering is the only live reading of this table. Distance-SpaDE's half of the
retracted claim cannot be rechecked in this tree, because its only seed-42
checkpoint is the excluded one.

The section it replaced was itself written against the retired straightness claim
that SpaDE destroys the trajectory signal while CosineSpaDE keeps it — every
family preserves the separation, 0.01–0.21 (`docs/claims_status.md`) — so there
was no split to reproduce even before the recipe was corrected.

## Caveats

- SpaDE codes live on a simplex; compare cluster *structure* across families,
  never coordinate magnitudes (this is why F8 reads structure, not scale).
- `p_s_unchanged` and the dissociation gap are accuracy² floors — quote
  `cos_flip` for cross-family comparison, `acc_rel` for "does this space decode
  the board at all".
- Descriptive only. The causal replication (F6) stays on raw, where the edit
  direction maps to the residual the model actually runs on.
