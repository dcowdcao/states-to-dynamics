# Pipeline

> Port the full content from the original `docs/Spade_Pipeline_Summary.md`.

End-to-end reference for the 8-type SAE pipeline: model + data, standardization, all 8 SAE
architectures with equations + hyperparameters, every eval metric with formulas, results
tables, and the Pareto analysis.

Stages (each maps to a `spade.*` subpackage):

1. **extract** — `spade.extraction` → `data/activations/` (acts + board labels)
2. **train** — `spade.training.train_<type>` → `checkpoints/<experiment>/`
3. **eval** — `spade.evaluation.run_eval` → `results/<experiment>/all_results.csv`
4. **aggregate** (multi-seed) — `spade.multiseed.aggregate_seeds` → `results/multiseed/`
5. **figures** — `spade.figures.*` → `figures/`

Plus `spade.diagnostics` (SpaDE norm studies) and the `spade.legality` sub-pipeline.
