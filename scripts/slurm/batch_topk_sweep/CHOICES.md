# Why these k

BatchTopK is the one architecture of the eight with a single operating point and no L0
curve. Karvonen never swept it — his released sweep covers Standard, Gated, p-anneal,
gated+p-anneal and TopK — and this study has trained exactly `k = 80`, chosen in
`train_batch_topk.py` to match TopK for a head-to-head. Every cross-architecture reading
that places BatchTopK on the L0 axis is currently one point wide.

## k is the dial, and it is exact

For TopK, L0 ≡ k by construction. BatchTopK selects top-k per *batch* and switches to a
learned EMA threshold at inference (`threshold_beta=0.999`, `threshold_start_step=1000`),
so per-token L0 varies around k rather than pinning to it. Measured at k = 80 through
Karvonen's own evaluator: L0 78.00, 78.28, 79.00, 79.08 on seeds 43-46 and 80.19 on seed
42 — **0.975 to 1.002 of k**, tightest of any architecture here. No response fit is needed
and none is used: unlike SpaDE's power law in γ or CosineSpaDE's logistic, the grid can be
written straight down in L0.

## The grid: k in {10, 20, 40, 60, 80, 110, 140, 200, 280, 400}

Roughly 1.4x steps. Three things it has to do:

- **Cover Karvonen's axis.** His sweep spans L0 10-571 and his TopK arm is `k = 10..400` in
  40 steps; this is the log-spaced subset of the same range, so BatchTopK can be read
  against TopK at every decade instead of at one point.
- **Sample the comparison band densely.** Four points (60, 80, 110, 140) fall inside
  L0 60-160, where the field's coverage winners sit — his gated+p-anneal trainer14 at
  L0 61.5, his gated trainer18 at 135.7, and his best-of-sweep TopK trainer_53 at k = 140.
  The operating point is chosen inside that band, so that is where resolution is worth
  spending.
- **Show the peak is a peak.** TopK coverage in his sweep peaks near k = 140 and falls
  away above it. k = 280 and 400 are there to bracket the far side. Without them an argmax
  at the grid edge is indistinguishable from a curve still climbing.

k = 80 is retrained inside the grid rather than copied from `checkpoints/batch_topk/`.
Trainer indices restart at 0 in every sweep dir and the existing k = 80 was trained in a
different run under different standardization calibration; retraining gives the curve one
internally consistent 5-seed point instead of a seam.

The swept configs are named `BatchTopK-k<k>`, deliberately outside the
`BatchTopKTrainer-othello` name that `results/acceptance` and the registry match on, so a
swept point can never be read as the canonical baseline row.

## Cost

10 trainers share one `trainSAE()` pass, so the whole grid is one job per seed. Wall time
scales with trainers-per-GPU, not the total: 10 trainers on 3 GPUs took 8 h 06 m and 14 on
4 GPUs took 8 h 08 m, both 4 per GPU. 10 on 4 GPUs is 3 per GPU, so **~6 h per seed**,
~24 GPU-h. Five seeds: ~120 GPU-h.

Evaluation is one job, not an array — `karvonen_repro.evaluate` takes one process per
output dir, because his per-group `results.csv` is a fixed path two tasks would race on.
50 checkpoints at ~1 min each is under an hour.

**Pilot seed 42 alone (~6 h) and check the measured L0s against k before committing the
other four seeds.** The expectation is L0/k in [0.97, 1.01] at every k; a departure at the
sparse end is the thing worth knowing early.

## What may come back degenerate

- **k = 10 of 8192 atoms.** BatchTopK has an auxiliary loss (`auxk_alpha = 1/32`) that
  TopK-family SAEs use to revive dead features, so it should fare better here than SpaDE
  does at comparable sparsity — SpaDE has no resampling and no auxiliary loss and its
  L0 7.8 point came back with all 8191 atoms nominally alive but coverage 0.013, below the
  random-weight floor. Still, expect `num_alive_features` well under the dictionary size at
  k = 10 and k = 20, and report it rather than treating it as a failure.
- **The EMA inference threshold needs `threshold_start_step=1000` to have trained.** The
  SAE loader already refuses BatchTopK checkpoints whose threshold never trained; a smoke
  run at 100 steps is one of those. Real runs at 36,621 steps are past it by a factor of 36.
- **Coverage below the random-weight floor at the sparse end.** The blank-masked floor is
  0.2723. It is a result to report, not a bug — most of the SpaDE ladder sits under it.
