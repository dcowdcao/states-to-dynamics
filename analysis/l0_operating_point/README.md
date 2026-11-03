# One shared L0 for all eight SAE types

**Status (2026-08-23): the sweeps are done, the target is not settled.** All three of our
architectures now carry a 20-config grid. But with BatchTopK no longer pinning the search,
the two scoring rules stop agreeing on where the target is — see [Not yet
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

`spade.l0_operating_point.aggregate` builds one table of 516 configs.

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

**Shared target L0 = 116.7**, mean score 0.878. Every type seats within 5% of it.

| Type | Hyperparameter | Value | L0 | L0/target | coverage | recon F1 | score |
|---|---|---|---|---|---|---|---|
| **BatchTopK** | `k` | **120** | 120.3 | 1.03x | 0.538 | 0.821 | **0.956** |
| Gated | `l1_penalty` | 1.1053 | 112.9 | 0.97x | 0.481 | 0.890 | 0.954 |
| Standard + p-anneal | `sparsity_penalty` | 0.028947 | 119.3 | 1.02x | 0.431 | 0.921 | 0.938 |
| TopK | `k` | 120 | 120.0 | 1.03x | 0.467 | 0.826 | 0.926 |
| Gated + p-anneal | `sparsity_penalty` | 0.552632 | 110.5 | 0.95x | 0.488 | 0.760 | 0.921 |
| Standard | `l1_penalty` | 0.061053 | 116.1 | 1.00x | 0.280 | 0.906 | 0.864 |
| CosineSpaDE | `gamma_reg` | 13.7 | 115.6 | 0.99x | 0.348 | 0.494 | 0.764 |
| SpaDE | `lambda_init` 1e-3, `gamma_reg` | 0.0115 | 116.7 | 1.000x | 0.145 | 0.627 | 0.701 |

**BatchTopK ranks first, above all five of Karvonen's types.** That was not observable before
its grid existed: its only config was k = 80 at 0.67x the target, and a type seated a third of
the way off the comparison point cannot be read against the rest. Our two novel architectures
sit at the bottom, below every one of his.

SpaDE lands at 1.0002x the target — the grids were built by inverting each architecture's
measured L0 response, and they hit what they aimed at.

### The alternative rule, for reference

`--rule band` instead searches for a shared *band* and gives each type its **best-scoring**
config inside it. That returns L0 105.3-238.8 (2.27x) at mean score 0.888. The extra 0.010 is
bought by letting the eight roam over a 2.3x spread in sparsity, which is the confound this
study exists to remove; `--rule nearest` is the reported answer. It agrees on the shape of the
result — BatchTopK second by 0.004, our two architectures last.

The band rule needs its width capped and the cap is load-bearing, not a tuning knob: the mean of
per-type winners is monotone in width, since a wider band only ever adds candidates. Uncapped,
the search returns L0 55-506 — a 9x spread that is not a matched comparison at all.

## Not yet final

- **The target is rule-dependent, and that is new.** Min-max puts it at L0 116.7; the
  rank-mean twin puts it at **L0 215.6** and changes all eight configs. While BatchTopK was a
  single pinned point the two agreed exactly, and this README cited that agreement as evidence
  the answer was not a normalization artifact. That evidence is gone: removing the constraint
  let the rules separate. Min-max is driven by `recon_f1` and `coverage` (sigma 0.288, 0.281)
  whose ranges are set by collapsed SAEs; rank-mean weights the fidelity terms more evenly and
  prefers denser SAEs. Per the plan a pick that flips under the twin is reported as unresolved,
  so **which rule to defend is an open decision**, not something this file settles.
- **Two of the four metrics are n = 25.** `frac_recovered` and FVE come from Karvonen's
  published protocol, where the SAE-quality columns average one batch of 25 activation vectors
  (`PROTOCOL.md` section 2). FVE carries sd ~ 0.14 there and is biased low below L0 ~ 100. The
  board-state columns are unaffected — they always see 1000 games x 59 positions. Re-measuring
  the fidelity columns at `--mode measurement` (2500 activations) for the configs near the
  target is the outstanding step before these scores are quotable.
- **SpaDE's optimum is outside any band CosineSpaDE can share.** Its score still climbs as it
  leaves the target region — 0.701 at L0 117, 0.732 at 234 — while CosineSpaDE ceilings at
  L0 183 for any gamma. That is a fact about the architectures, not a sampling gap, and more
  SpaDE points near 117 will not change it.
- **Everything here is seed 42.** The three grids match Karvonen's own single-seed sweep, so
  the figure is like-for-like, but no point on our side carries a seed spread. Earlier streams
  (the gamma sweep, `l0_coverage`) do carry 4-5 seeds and are still pooled in; the `n_seeds`
  column in `picks.csv` says which is which.

## Figures

Karvonen's Figure 4 (arXiv:2408.00113v2), rebuilt with our three architectures on it.
`figures/l0_operating_point/`:

    figure4_othello_karvonen_only.{png,pdf}   his five types, his figure
    figure4_othello_with_spade.{png,pdf}      + BatchTopK, SpaDE, CosineSpaDE
    karvonen_only/, with_spade/               the four panels of each, separately

His 2x2 layout and subcaptions: top row coverage, bottom row board reconstruction, left
column the loss-recovered-vs-L0 scatter coloured by the metric, right column the metric
against L0 — (a) Coverage of Board State, (b) Coverage of Board State vs. L0,
(c) Board Reconstruction, (d) Board Reconstruction vs. L0.

Style is copied from his `circuits/f1_analysis.ipynb`, not from `paper_style.py`:
`FIGURE_GUIDELINES.md` deliberately does not apply to a figure whose point is to be his.
`KARVONEN_STYLE.md` records what was copied. His repo renders the four panels separately
and the paper composites them in LaTeX, so `combine_figure.py` composites the same PNGs
rather than re-plotting — every panel stays byte-identical to what `verify_match.py` checks.

**`karvonen_only` reproduces his published PNGs exactly** — max channel diff 0, zero pixels
differing on all four, against the vendored `saved_data/07-31_images/`. That is what makes
`with_spade` readable as data rather than as styling drift.

`with_spade` deviates from his rendering twice, both forced by our points landing where his
never did, and neither touching colour, marker, size or font:

- **The legend moves below the axes.** SpaDE and CosineSpaDE put points in the lower right
  of both right-column panels, which is where his legend sits.
- **The left-column y-axis widens** from his 0.9960–1.0005. On his window 44 of 55 SpaDE
  and 62 of 117 CosineSpaDE points fall off the bottom with nothing to say they were
  dropped. It is floored at 0 rather than at the true minimum because six of *his* Standard
  configs at L0 < 3 recover less loss than zero-ablation (down to −0.81) and would flatten
  every real point onto one line; `render()` prints what the floor excludes.

The cost of that second deviation is real: the widened left column is honest but flatter
than his, because his tight window was a zoom on the region where all his SAEs live and
ours do not. For a paper, publish the two windows as a pair rather than compromise on one.

## Reproduce

    python -m spade.l0_operating_point.aggregate          # -> results/l0_operating_point/pooled.csv
    python -m spade.l0_operating_point.pick --rule nearest # -> results/l0_operating_point/picks.csv
    python -m spade.l0_operating_point.pick --rule band     # the alternative
    python -m spade.l0_operating_point.plot_pareto          # the four panels, both variants, from pooled.csv
    python -m spade.l0_operating_point.verify_match         # holds karvonen_only to his PNGs
    python -m spade.l0_operating_point.combine_figure       # the 2x2 Figure 4 (with_spade; --variants for both)

Use the conda interpreter directly; worktrees have no `.venv`:
`/n/home10/dcao/.conda/envs/spade_env/bin/python`, with `PYTHONPATH=<worktree>/src` so the
editable install does not resolve to a different checkout.

Any new checkpoint destined for this table must be evaluated through `karvonen_repro.evaluate`,
never `spade.evaluation.run_eval` — the latter omits his >=0.95-precision / >=10-firing feature
gate, assigns one label per feature, thresholds per-feature and macro-averages 180 cells where
he micro-pools 3.78M square-labels, and lands ~0.2 absolute F1 away.
