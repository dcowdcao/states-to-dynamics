# Pipeline

> Port the full content from the source repo's `docs/Spade_Pipeline_Summary.md`.

End-to-end reference for the 8-type SAE pipeline: model + data, standardization, all 8 SAE
architectures with equations + hyperparameters, every eval metric with formulas, results
tables, and the Pareto analysis.

Stages (each maps to a `spade.*` subpackage):

1. **extract** — `spade.extraction` → `data/activations/` (acts + board labels)
2. **train** — `spade.training.train_<type>` → `checkpoints/<experiment>/`
3. **eval** — `spade.evaluation.run_eval` → `results/<experiment>/all_results.csv`
4. **aggregate** (multi-seed) — `spade.multiseed.aggregate_seeds` → `results/multiseed/`
5. **figures** — `spade.figures.*` → `figures/`

Plus `spade.diagnostics` for the SpaDE norm studies.

Standardization is part of the contract, not a preprocessing detail: SAEs are trained on
`scale*(x - mean)`, so anything that encodes without applying it is silently wrong rather
than approximately right. The parameters are written beside the checkpoints at save-dir
level and re-applied at load, at export, and again in the browser viewer.

The studies built on this pipeline — rules, bracket flip, violation divergence,
straightness, manifold capture — have their own stage sequences, each documented in its
package docstring and driven by `scripts/slurm/studies/<study>/`. They share a shape:
numbered steps that skip what already exists, writing to a netscratch working directory,
with all plotting deferred to `analysis/<study>/make_figures.py`.
