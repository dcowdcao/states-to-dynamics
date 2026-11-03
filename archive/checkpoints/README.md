# checkpoints/

Trained SAE weights, one subtree per experiment. Weight files (`ae.pt`) are gitignored;
`config.json` + `standardization_params.json` are small and tracked.

- `main/`          — the 8-type baseline+SpaDE comparison run   (was `autoencoders/`)
- `batch_topk/`    — BatchTopK add-on                            (was `autoencoders_batch_topk/`)
- `cosine_spade/`  — CosineSpaDE add-on                          (was `autoencoders_cosine_spade/`)
- `spade_sweep/`   — SpaDE λ/γ sweep checkpoints                 (was `autoencoders_spade_sweep/`)
- `spade_revival/` — **the default SpaDE config**, λ=1e-3 γ=1e-4
                                        (was `autoencoders_spade_lam1e-03_gam1e-04/`)
- `karvonen_saes/` — Karvonen's released roster SAEs, wholly untracked; filled by
                     `scripts/download_karvonen_saes/download.sh`, hashes in
                     `manifests/checkpoints/karvonen_saes.tsv`

Each holds `trainer_<i>/` subfolders, with `standardization_params.json` at *this* level,
not inside them — the loader searches a checkpoint dir and its parent, and the file lives
at the parent. Seeds 43-46 live on netscratch under `study_dir("multiseed")`, not here.

Two traps. Trainer indices restart at 0 in every experiment, so `trainer_0` means nothing
without its directory. And every `config.json` under `main/` records a Mac smoke-test
configuration written over real GPU weights — it is not the configuration those weights
were trained with. Read hyperparameters from the run record, not from these files.
