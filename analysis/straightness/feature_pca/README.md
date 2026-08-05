# PCA of the 21-D ReStraV descriptor, per representation space

Every figure elsewhere in the straightness study reduces a game to one number
(mean turning angle, mean step length) and scores it. This one keeps ReStraV's
whole 21-D vector — 7 step lengths, 6 turning angles, and the 4 moments of each
(`src/spade/straightness/geometry.py`) — and asks what the descriptor space
looks like before any label is used: one PCA per representation space, games
projected onto its top-2 principal components.

It is the unsupervised counterpart of the `auc_lr21` column: same 21 features,
no labels.

```
python analysis/straightness/feature_pca/make_figures.py
```

Reads the geometry npz `src/spade/straightness/compute_geometry.py` already
wrote — no re-extraction. Writes `figures/restrav_pca.png` and prints the basis
alignment table below.

## Method notes

- **Each space gets its own PCA, fit on its own z-scored features.** The 21
  components mix degrees with activation-scale lengths, and the absolute scale
  of the step-length block differs ~450× across dictionaries (SpaDE ~0.05, raw
  layer 5 ~22.8). PCA on raw covariance answers a different question: it lands
  PC1 on turning-angle variance in all nine spaces, reporting units rather than
  structure.
- **The fit runs over the four classes the study draws**, not over all of
  `games.pt`. The undrawn random-token block would otherwise set the axes and
  the variance ratios of a plane it does not appear in
  (`analysis/straightness/README.md` § Design notes).
- **Component signs are pinned**, since they are otherwise arbitrary: raw layer
  5 puts its largest loading positive, and every SAE space takes the sign that
  agrees with raw's, so panels are not mirrored against each other for no
  reason.
- Ellipses enclose 50% of a Gaussian fit per class — a dispersion summary, not a
  confidence interval.

## Results

**PC1 is a step-length axis in all nine spaces**, carrying 16–24% of the
variance against PC2's 12–14% — a shallow spectrum, as befits four classes that
mostly overlap. Its largest loading is always in the `d` block (`min d` in 4 of
9, an early step `d1`/`d4` or a `d` moment otherwise); PC2 leads on a
turning-angle component in 7 of 9. The table below is what
`make_figures.py` prints.

**Championship games are the only class that moves in the plane.** As a class
mean minus legal's, in pooled SD along each PC (`class_shift`), championship
runs 0.3–2.1; rollout and violation never exceed 0.27 in any space. That is a
statement about unsupervised structure, not about information: the supervised
LR-21 probe reaches 0.66 on violations and 0.96 on championship games in this
same 21-D space (see `analysis/straightness/README.md`).

**Only the hard-sparsity dictionaries reproduce raw layer 5's plane.** |cos|
between each space's principal component and raw layer 5's, with championship's
displacement alongside:

| space | PC1 var | PC2 var | \|cos\| PC1 | \|cos\| PC2 | champ shift (PC1, PC2) |
|---|---|---|---|---|---|
| raw L5 (reference) | 22% | 14% | 1.00 | 1.00 | +2.08, −0.05 |
| PAnneal | 22% | 14% | 0.95 | 0.88 | +1.76, +0.19 |
| BatchTopK | 23% | 14% | 0.94 | 0.92 | +1.31, −0.67 |
| TopK | 22% | 13% | 0.90 | 0.95 | +1.86, +0.12 |
| GatedAnneal | 17% | 12% | 0.85 | 0.84 | +0.64, +0.11 |
| GatedSAE | 19% | 13% | 0.83 | 0.82 | +0.01, −0.31 |
| Standard | 17% | 14% | 0.64 | 0.33 | +0.18, +0.68 |
| SpaDE (γ=1e-4) | 16% | 14% | 0.33 | 0.41 | +0.72, +0.76 |
| CosineSpaDE (γ=1.0) | 24% | 14% | 0.13 | 0.06 | +0.80, +0.04 |

The two sparsemax dictionaries are the outliers, and CosineSpaDE's plane is
essentially unrelated to raw's (0.13 / 0.06) — it re-expresses the descriptor
rather than passing it through, which is the same story the rest of the study
tells about sparsemax spaces. Standard is a milder version of it. These
alignments are much lower than a fit over all five stored classes gives: one
gross OOD direction then dominates every space's PC1, and the spaces agree
about that direction rather than about the descriptor. Restricting the fit to
the drawn classes is what exposes the disagreement.

## What the CosineSpaDE cluster is

CosineSpaDE's panel has a second cloud up and to the right, at PC1 > 2, PC2 > 3:
302 of the 8000 drawn games (violation 113, legal 112, rollout 77, championship
0). It is not a class effect — it is **delayed code recruitment**. Those games
activate far fewer codes in the opening than the rest (mean L0 at positions 2
and 3: 13 and 35, versus 44 and 114), then catch up by position 4 (~118 codes
either way). The catch-up is a large displacement: their step length at position
3 is 0.485 against 0.359, +2.60 SD, which is exactly the feature the plane loads
on. By mid-game the two groups are indistinguishable (step-length ratio 1.00
past position 10).

This explains an entry left open in PROCESS.md: CosineSpaDE's curvature profile
has a sharp dip near t=3 that no other space shows, guessed there to be
"sparsemax code turnover." It is that, and it is confined to a 4% subpopulation
rather than being a shift in the whole distribution — a per-position mean is the
wrong summary for it.

Seed 42 only, raw (uncentered) variant.
