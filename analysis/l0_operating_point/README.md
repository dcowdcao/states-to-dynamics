# One shared L0 for all eight SAE types

**Status (2026-08-22): provisional.** BatchTopK is still a single point and two of the four
scored metrics are 25-activation estimates. Both are being fixed; see [Not yet
final](#not-yet-final) before quoting any number here.

## The problem

`docs/spade_vs_karvonen.md` compares eight SAE types at one fixed operating point each, and
those points sit at wildly different sparsities — GatedAnneal L0 31, Gated 50, TopK 80,
BatchTopK 80, CosineSpaDE 173, PAnneal 339, Standard 1095, SpaDE 1808. Any cross-architecture
claim read off that table is confounded by L0: a type can look better simply for being denser.

This study picks **one shared L0** and reports every type at the config closest to it, so the
comparison is at matched sparsity.

## Method

### 1. Assemble the points

`spade.l0_operating_point.aggregate` builds one table of 456 configs.

- **SpaDE, CosineSpaDE, BatchTopK** come from the `karvonen_repro` eval streams under
  `$SPADE_KARVONEN_REPRO_DIR`. Each checkpoint is identified from the `config.json` beside it,
  never by trainer index — indices restart at 0 in every sweep directory, and three different
  directories each hold a `trainer_0` from a different grid.
- **Standard, Gated, p-anneal, gated+p-anneal, TopK** come from
  `karvonen_othello_sweep/sweep_full.csv`: Karvonen's own 320 released checkpoints re-run
  through his own pipeline.

Our five retrained baselines are deliberately **not** on this table. Commit `8b95ec9` in the
vendored `dictionary_learning` rewrote `standard.py` and `p_anneal.py`'s reconstruction loss
from `||e||_2` to `||e||_2^2` and left `gdm.py` / `gated_anneal.py` alone, so those two share a
name with his but not a model: at `l1_penalty` 0.0587 ours lands at L0 1095 where his lands at
L0 137 for 0.0563. A type is taken from one training pipeline or the other, never mixed.

Metrics are the blank-masked board-state pair — the family his own figures plot — plus
`frac_recovered` and `frac_variance_explained`. Blank-masked coverage is mechanically capped at
128/192 = 0.6667 (`one_hot_mask_idx` is unset for Othello); rankings are unaffected, absolute
values are not comparable to the unmasked floor, and `coverage_rescaled` is the x1.5 read.

### 2. One row per config

Seeds collapse to their mean. Ours carry 4-5 seeds and his carry one, so scoring raw rows would
weight a five-seed config five times against a one-seed config. Then filter to dict 8192: his
sweep runs both expansion factors and ours are all 16x.

### 3. Score

Each metric is min-max normalized across the whole pooled table — global, not per type, so the
eight stay on one scale — then averaged with equal weight:

    score = mean( norm(coverage), norm(recon_f1), norm(frac_recovered), norm(FVE) )

### 4. Search the shared L0

41 log-spaced candidate targets from L0 10 to 600, ~11% apart. For each target:

1. **Admissibility.** Every one of the eight types must have a config within
   `sqrt(2.5) = 1.58x` of the target, or the target is rejected. This is what makes "shared L0"
   mean anything — without it a type with nothing nearby gets seated by a config three octaves
   away and the matched-sparsity claim is empty.
2. **Seat each type** at its config whose L0 is nearest the target *in log space*.
3. **Score the target** as the mean of those eight configs' scores.

The best target wins.

### 5. Where each hyperparameter comes from

It falls out of step 4.2, and the important property is that **per-type selection is by L0
distance alone, not by score**. The score only ever chooses the target. Once the target is
fixed, a type gets whichever of its configs sits nearest, good or bad. That is what makes this
a matched-sparsity comparison rather than eight independent optimizations — no type is free to
wander to its own best L0.

So Gated is at `l1_penalty` 1.1053 because that config lands at L0 112.9, the closest Gated has
to the target. Not because 1.1053 is Gated's best penalty.

### 6. Sensitivity

The whole search is repeated with mean-of-ranks in place of min-max. This matters because the
min-max ranges are set by collapsed SAEs — his dead Standard configs run down to
`frac_recovered` -0.50 and coverage 0.0005 — which compresses the two fidelity terms until the
score is in practice driven by coverage and recon-F1. `score.spread` reports that directly.

## Result

**Shared target L0 = 116.7**, mean score 0.884.

| Type | Hyperparameter | Value | L0 | L0/target | coverage | recon F1 | score |
|---|---|---|---|---|---|---|---|
| Gated | `l1_penalty` | 1.1053 | 112.9 | 0.97x | 0.481 | 0.890 | 0.966 |
| Standard + p-anneal | `sparsity_penalty` | 0.028947 | 119.3 | 1.02x | 0.431 | 0.921 | 0.949 |
| TopK | `k` | 120 | 120.0 | 1.03x | 0.467 | 0.826 | 0.938 |
| Gated + p-anneal | `sparsity_penalty` | 0.552632 | 110.5 | 0.95x | 0.488 | 0.760 | 0.934 |
| **BatchTopK** | `k` | **80** | **78.6** | **0.67x** | 0.453 | 0.843 | 0.929 |
| Standard | `l1_penalty` | 0.061053 | 116.1 | 1.00x | 0.280 | 0.906 | 0.871 |
| CosineSpaDE | `gamma_reg` | 10 | 131.1 | 1.12x | 0.352 | 0.484 | 0.773 |
| SpaDE | `lambda_init` 1e-3, `gamma_reg` | 0.01 | 130.5 | 1.12x | 0.156 | 0.620 | 0.711 |

The rank-mean twin picks the identical target and the identical eight configs, so the answer is
not an artifact of the normalization.

Karvonen's five occupy the top of the table and our two novel architectures the bottom, with
BatchTopK — his trainer, our training run — in between and ahead of his Standard.

### The alternative rule, for reference

`--rule band` instead searches for a shared *band* and gives each type its **best-scoring**
config inside it. That returns L0 63.1-143.2 (2.27x) at mean score 0.897. The extra 0.013 is
bought by letting the eight roam over a 2.2x spread in sparsity, which is the confound this
study exists to remove; `--rule nearest` is the reported answer.

The band rule needs its width capped and the cap is load-bearing, not a tuning knob: the mean of
per-type winners is monotone in width, since a wider band only ever adds candidates. Uncapped,
the search returns L0 55-506 — a 9x spread that is not a matched comparison at all.

## Not yet final

- **BatchTopK is a placeholder.** `k = 80` is its only trained config, so it seats at 0.67x the
  target when every other type is within 12%. It is the one type that cannot currently reach the
  shared L0. `scripts/slurm/batch_topk_sweep/` fills in k = 10..400; expect k=110 or k=120 to
  land near the target and to move the target itself.
- **Two of the four metrics are n = 25.** `frac_recovered` and FVE come from Karvonen's
  published protocol, where the SAE-quality columns average one batch of 25 activation vectors
  (`PROTOCOL.md` section 2). FVE carries sd ~ 0.14 there and is biased low below L0 ~ 100. The
  board-state columns are unaffected — they always see 1000 games x 59 positions. Re-measuring
  the fidelity columns at `--mode measurement` (2500 activations) for the configs near the
  target is the outstanding step before these scores are quotable.
- **SpaDE's optimum is outside any band CosineSpaDE can share.** Its score still climbs as it
  leaves the target region — 0.619 at L0 41, 0.678 at 77, 0.711 at 130, 0.737 at 232 — while
  CosineSpaDE ceilings at L0 178 for any gamma. That is a fact about the architectures, not a
  sampling gap, and more SpaDE points near 117 will not change it.
- CosineSpaDE, by contrast, is thoroughly sampled: 15 configs at 5 seeds spanning the target,
  with a score plateau of 0.754-0.774. More points there cannot move the answer.

## Reproduce

    python -m spade.l0_operating_point.aggregate          # -> results/l0_operating_point/pooled.csv
    python -m spade.l0_operating_point.pick --rule nearest # -> results/l0_operating_point/picks.csv
    python -m spade.l0_operating_point.pick --rule band     # the alternative

Use the conda interpreter directly; worktrees have no `.venv`:
`/n/home10/dcao/.conda/envs/spade_env/bin/python`, with `PYTHONPATH=<worktree>/src` so the
editable install does not resolve to a different checkout.

Any new checkpoint destined for this table must be evaluated through `karvonen_repro.evaluate`,
never `spade.evaluation.run_eval` — the latter omits his >=0.95-precision / >=10-firing feature
gate, assigns one label per feature, thresholds per-feature and macro-averages 180 cells where
he micro-pools 3.78M square-labels, and lands ~0.2 absolute F1 away.
