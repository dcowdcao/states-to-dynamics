# results/

Evaluation outputs, one subtree per experiment (mirrors `checkpoints/`). Small CSVs/JSON are
tracked; large tensors (`f1_TFRRC.pt`) are gitignored.

- `main/`         — 8-type eval, `all_results.csv`        (was `results/karvonen/`)
- `spade_sweep/`  — SpaDE sweep evals, `combined_results.csv`  (was `results/karvonen_spade_sweep/`)
- `batch_topk/`, `cosine_spade/` — add-on evals           (was `results/karvonen_*`)
- `legality/`     — legality probe study CSVs/JSON
- `multiseed/`    — per-seed aggregates (`all_seeds.csv`, `aggregate*.csv`)
