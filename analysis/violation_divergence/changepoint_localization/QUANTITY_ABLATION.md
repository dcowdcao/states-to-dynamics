# What the change-point score should be computed on

`quantity_ablation.py`. The study scores **step length** `d`. Two alternatives are obvious
enough that a reader will ask, and one of them is the study's own open worry:

| key | quantity | why it is a candidate |
|---|---|---|
| `d` | step length, ‖z_{t+1} − z_t‖ | the study's statistic, inherited from ReStraV |
| `norm` | the representation's own norm, latent norm in an SAE space | README: "treat this as a magnitude claim, not a geometry claim" |
| `theta` | turning angle, curvature proper | the quantity the descriptor is named for |

All three go through the identical `profile → drop → scan`, same *m* = 4, same paired MOSUM,
same corpus and arms, on **one matched band** — moves 14–46, 33 plies — so the chance level
and the scan window are held fixed and only the scored quantity changes. The three keys have
different widths and different move offsets (`d` 57 @ +2, `norm` 58 @ +1, `theta` 56 @ +2),
because the attention sink is dropped and an angle needs two consecutive displacements; the
bands undo that.

Both statistics are reported because they disagree about curvature: the **hit rate** is the
forced choice the study always had, the **per-position AUROC** is what the thresholded
read-out added.

## Result

870 violations + 1,000 held-out legal games, regenerated corpus, chance hit = 0.030.
Intervals are 95% percentile bootstrap over source games as clusters, 1,000 resamples,
seed 42. Each quantity is reported at its better sign — see below for why that is not a
free parameter.

| space | *d* AUROC | ‖z‖ AUROC | θ AUROC | *d* hit | ‖z‖ hit | θ hit | signs *d*/‖z‖/θ |
|---|---|---|---|---|---|---|---|
| GatedAnneal | 0.893 [0.882, 0.905] | 0.865 [0.851, 0.877] | 0.596 [0.578, 0.615] | 0.436 | 0.283 | 0.048 | - / - / - |
| GatedSAE | 0.855 [0.841, 0.868] | 0.833 [0.818, 0.847] | 0.570 [0.550, 0.588] | 0.359 | 0.236 | 0.040 | - / - / - |
| TopK | 0.832 [0.818, 0.846] | 0.768 [0.751, 0.784] | 0.576 [0.558, 0.596] | 0.293 | 0.157 | 0.045 | - / - / - |
| BatchTopK | 0.831 [0.817, 0.847] | 0.732 [0.713, 0.748] | 0.572 [0.552, 0.592] | 0.306 | 0.125 | 0.045 | - / - / - |
| raw L5 (ref.) | 0.761 [0.745, 0.778] | 0.544 [0.522, 0.567] | 0.577 [0.558, 0.597] | 0.156 | 0.026 | 0.029 | - / + / - |
| PAnneal | 0.680 [0.663, 0.699] | 0.630 [0.609, 0.650] | 0.567 [0.547, 0.586] | 0.102 | 0.077 | 0.023 | - / + / - |
| SpaDE | 0.583 [0.563, 0.603] | 0.691 [0.670, 0.710] | 0.527 [0.509, 0.546] | 0.038 | 0.226 | 0.016 | - / + / - |
| Standard | 0.576 [0.557, 0.595] | 0.637 [0.616, 0.657] | 0.547 [0.528, 0.565] | 0.055 | 0.121 | 0.026 | - / + / - |
| CosineSpaDE | 0.524 [0.503, 0.545] | 0.580 [0.557, 0.602] | 0.557 [0.537, 0.577] | 0.054 | 0.105 | 0.043 | - / + / - |
| **mean** | **0.726** | 0.698 | 0.566 | | | | |

Figure: `figures/quantity_ablation.png`.

## Three findings

**Step length wins where anything is detectable.** `d` is best on the six spaces that
detect at all, and on the reference space it is not close: 0.761 against 0.544 for the norm.
The README's worry that the norm-only baseline "may own the result" is **not** borne out
through this pipeline. `norm` wins only on Standard, SpaDE and CosineSpaDE — all at or near
chance on everything, so it is winning a race nobody finishes.

**The decisive argument is the sign, not the AUROC.** `d` keeps one direction in all nine
spaces: shorter step = illegal. The norm flips — smaller-is-illegal for the Gated/TopK
family, **larger**-is-illegal for raw, Standard, PAnneal and both SpaDE spaces. The method is
label-free, so a quantity whose sign moves across spaces cannot be used at all: you would
have to read the answer to choose the direction. That argument survives even if someone
disputes the numbers, which is why `sign` is a column.

**Curvature is weak, not dead — and `CURVATURE_ABLATION.md` overstates two things.** Every
θ is above 0.5 (0.527–0.596) with a consistent sign, so curvature carries a real signal.
It reads as chance only on the argmax hit rate (0.016–0.048 against 0.030), which is the
forced choice discarding every non-peak position. Two corrections follow:

- *"Curvature cannot localize at all"* is true of the hit rate and false per position.
- *"The space ordering roughly inverts, Spearman ρ ≈ −0.67"* is an artifact of the peak
  statistic. **Per position it is +0.783** — the same ordering as `d`. Do not repeat the
  inversion claim.

θ still cannot carry a dictionary comparison: its span across spaces is 0.068 against `d`'s
0.369 and the norm's 0.321. It separates illegal from legal a little, everywhere, and
separates dictionaries from each other essentially not at all.

## What this does not settle

**Regenerated corpus only.** `curvature_discriminant`'s npz carries `d` and `theta` but not
`norm`, so replicating on the spliced tree needs one GPU encode pass. The straightness npz
carries all three, which is why this runs on CPU today.

**Substitution is not explanation.** This asks whether the norm can *replace* `d`. The
README's open question is whether it *explains* `d` — `d_t` regresses on
`(‖z_t‖, ‖z_{t+1}‖)` at R² 0.87–0.88, and partialling the norm out drops GatedAnneal from
0.587 to 0.173. Both can be true at once: the norm can fail as a standalone detector while
still accounting for most of `d`'s variance. That partial-correlation check is still unrun,
and it is the one that decides whether "geometry" is the right word for the study at all.

## Reproducing

```
PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py
```

CPU-only. Writes `results/quantity_ablation.csv` and `figures/quantity_ablation.png`.
It supersedes `theta_ablation.py`, which measured only the argmax half of this.
