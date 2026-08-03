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
- **Component signs are pinned**, since they are otherwise arbitrary: raw layer
  5 puts its largest loading positive, and every SAE space takes the sign that
  agrees with raw's, so panels are not mirrored against each other for no
  reason.
- Ellipses enclose 50% of a Gaussian fit per class — a dispersion summary, not a
  confidence interval.

## Results

**PC1 is the random-token axis, everywhere.** It carries 43–55% of the variance
and its largest loading is a step-length component (`min d` in 7 of 9 spaces).
The other four classes — legal, rollout, violation, championship — sit in one
overlapping cloud in every dictionary. That is a statement about unsupervised
structure, not about information: the supervised LR-21 probe separates
violations and championship games in the same 21-D space (see
`analysis/straightness/README.md`). The dominant direction of variation is
simply the gross out-of-distribution one.

**Most dictionaries reproduce raw layer 5's descriptor plane almost exactly.**
|cos| between each space's principal component and raw layer 5's:

| space | PC1 | PC2 |
|---|---|---|
| BatchTopK | 0.99 | 0.97 |
| PAnneal | 0.99 | 0.98 |
| Standard | 0.98 | 0.90 |
| GatedSAE | 0.97 | 0.96 |
| TopK | 0.96 | 0.96 |
| GatedAnneal | 0.93 | 0.95 |
| SpaDE (γ=1e-4) | 0.82 | 0.56 |
| CosineSpaDE (γ=1.0) | 0.76 | 0.12 |

The two sparsemax dictionaries are the outliers, and CosineSpaDE's second
component is essentially unrelated to raw's (0.12). Consistent with the rest of
the study: sparsemax spaces are where the trajectory geometry is re-expressed
rather than passed through.

## What the CosineSpaDE cluster is

CosineSpaDE's panel has a second cloud, ~4σ down its PC2: 399 of 10 000 games
(violation 148, legal 140, rollout 101, random tokens 10, championship 0). It is
not a class effect — it is **delayed code recruitment**. Those games activate far
fewer codes in the opening than the rest (mean L0 at positions 2 and 3: 14 and
43, versus 49 and 107), then catch up by position 4 (~115 codes either way). The
catch-up is a large displacement: their step length at position 3 is 0.366
against 0.191, +4.05 SD, which is exactly the feature PC2 loads on. By
mid-game the two groups are indistinguishable (step-length ratio 1.02 past
position 10).

This explains an entry left open in PROCESS.md: CosineSpaDE's curvature profile
has a sharp dip near t=3 that no other space shows, guessed there to be
"sparsemax code turnover." It is that, and it is confined to a 4% subpopulation
rather than being a shift in the whole distribution — a per-position mean is the
wrong summary for it.

Seed 42 only, raw (uncentered) variant.
