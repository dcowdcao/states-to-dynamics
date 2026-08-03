# checkpoints/

Trained SAE weights, one subtree per experiment. Weight files (`ae.pt`) are gitignored;
`config.json` + `standardization_params.json` are small and tracked.

- `main/`          — the 8-type baseline+SpaDE comparison run   (was `autoencoders/`)
- `batch_topk/`    — BatchTopK add-on                            (was `autoencoders_batch_topk/`)
- `cosine_spade/`  — CosineSpaDE add-on                          (was `autoencoders_cosine_spade/`)
- `spade_sweep/`   — SpaDE λ/γ sweep checkpoints                 (was `autoencoders_spade_sweep/`)
- `spade_revival/` — the retrained chosen-best config   (was `autoencoders_spade_lam1e-03_gam1e-04/`)

Each holds `trainer_<i>/` subfolders, with `standardization_params.json` at *this* level,
not inside them — the loader searches a checkpoint dir and its parent, and the file lives
at the parent. Seeds 43-46 live on netscratch under `study_dir("multiseed")`, not here.

Two traps. Trainer indices restart at 0 in every experiment, so `trainer_0` means nothing
without its directory. And every `config.json` under `main/` records a Mac smoke-test
configuration written over real GPU weights — it is not the configuration those weights
were trained with. Read hyperparameters from the run record, not from these files.
