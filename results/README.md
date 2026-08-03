# results/

Evaluation outputs, one subtree per experiment (mirrors `checkpoints/`). Small CSVs/JSON are
tracked; large tensors (`f1_TFRRC.pt`) are gitignored.

- `main/`           — 8-type eval, `all_results.csv`      (was `results/karvonen/`)
- `spade_sweep/`    — SpaDE sweep evals                    (was `results/karvonen_spade_sweep/`)
- `batch_topk/`, `cosine_spade/` — add-on evals            (was `results/karvonen_*`)
- `spade_revival/`  — the retrained chosen-best SpaDE config
- `multiseed/`      — per-seed aggregates, incl. the CosineSpaDE γ sweep

Three subtrees exist only to keep older seed-42 rows reachable. They are not archives —
each is a hard-coded input to the aggregation or to the combined figure, and dropping any
of them silently removes seed 42 from the configs it covers:

- `spade_sweep_8config/`   — the only surviving record of the chosen-best single-seed point
- `lambda_sweep/`, `gamma_sweep_lam1e-07/` — seed-42 rows for five sweep configs

`gamma_sweep_lam1e-07/` is the April distance-SpaDE λ=1e-7 sweep. It is not the CosineSpaDE
γ sweep, which lives in `multiseed/`. The name is long on purpose.

Study outputs — `straightness/`, `violation_divergence/`, `manifold_capture/` — sit
alongside. Their heavy intermediates do not: those live on netscratch under
`study_dir(<name>)`.
