# checkpoints/

Trained SAE weights, one subtree per experiment. Weight files (`ae.pt`) are gitignored;
`config.json` + `standardization_params.json` are small and tracked.

- `main/`         — the 8-type baseline+SpaDE comparison run   (was `autoencoders/`)
- `batch_topk/`   — BatchTopK add-on                            (was `autoencoders_batch_topk/`)
- `cosine_spade/` — CosineSpaDE add-on                          (was `autoencoders_cosine_spade/`)
- `spade_sweep/`  — SpaDE λ/γ sweep checkpoints                 (was `autoencoders_spade_sweep/`)

Each holds `trainer_<i>/` subfolders. Seeds 43-46 live on `$SPADE_MULTISEED_DIR` (netscratch),
not here.
