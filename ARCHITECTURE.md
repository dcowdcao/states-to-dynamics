# Architecture

Clean-slate layout for the SpaDE / OthelloGPT SAE project. This is a **skeleton**:
every `.py` file holds a header comment + pseudocode describing what it will do, not
working code. Real logic gets ported from the original repo
(`../Spade_Research_David_Sumedh_Ba`) one piece at a time — see the old→new map below.

## Tree

```
spade_sae/
├── pyproject.toml            installable package `spade` (pip install -e .)
├── Makefile                  make setup / make run
├── README.md                 tiered: run it / dev setup / architecture
├── ARCHITECTURE.md           this file — layout + old→new migration map
├── PROCESS.md                running log of processes + critical decisions
├── CLAUDE.md  LICENSE.txt  requirements.txt  .gitignore
│
├── src/spade/                the code package (one import root: `import spade.*`)
│   ├── common/               config, paths, device, net  (leaf; everything imports these)
│   ├── saes/                 ALL SAE definitions live here
│   │   ├── spade_base.py         shared base: sparsemax encode + no-bias decode + trainer
│   │   ├── spade.py              distance SpaDE  (score = -λ·‖x-Aeᵢ‖²)
│   │   └── spade_cosine.py       CosineSpaDE     (score = cosine sim, γ-scaled)
│   │   # the 5 Karvonen baselines + BatchTopK are NOT redefined here — they come from
│   │   # external/Karvonen_Research (circuits.dictionary_learning).
│   ├── extraction/           stream games → cache layer-5 activations + board labels
│   ├── training/             ONE core loop + one thin entrypoint per SAE type
│   │   ├── core.py               shared: standardization, data-init, trainSAE plumbing
│   │   ├── train_standard.py     train_panneal.py  train_gated.py  train_gated_anneal.py
│   │   ├── train_topk.py         train_batch_topk.py
│   │   └── train_spade.py        train_cosine_spade.py
│   ├── evaluation/           karvonen_adapter.py (load+standardize+metrics) + run_eval.py
│   ├── multiseed/            aggregate_seeds.py  (per-seed CSVs → mean/std for error bars)
│   ├── figures/              karvonen_plot.py (shared 2×2 grid) + thin figure entrypoints
│   ├── diagnostics/          SpaDE norm / dead-prototype / per-game MSE studies
│   ├── legality/             legal-vs-illegal sub-pipeline (build_pairs→extract→metrics)
│   └── validation/           final_validation.py fast self-tests
│
├── scripts/                  shell orchestration
│   ├── run_pipeline.sh  smoke_test.sh
│   └── slurm/{main,sweeps,multiseed,legality}/   FASRC job scripts (repo-rooted paths)
│
├── analysis/                 downstream studies on trained SAEs (code + outputs co-located)
│   ├── spade_vs_baselines.md
│   ├── linear_probe/             gold-standard board probe (~0.97 acc)
│   └── geometry/                 14 self-contained studies (make_*.py + PNGs + summary)
│
├── docs/                     pipeline.md, spade_vs_karvonen.md, karvonen_reference.md,
│                             fasrc_commands.md, session_logs/
├── external/                 vendored Karvonen_Research (runtime dep: circuits.*)
├── reference/                papers, ARENA exercises, PDFs
│
│   ── generated outputs (weights/tensors gitignored; small CSVs + figures tracked) ──
├── checkpoints/{main,batch_topk,cosine_spade,spade_sweep}/   trained SAE weights
├── data/activations/         extracted .pt + metadata.json
├── results/{main,spade_sweep,batch_topk,cosine_spade,legality,multiseed}/   eval CSVs/tensors
├── figures/                  canonical figures only
└── archive/                  superseded snapshots + old runs
```

## Old → new migration map

Source repo: `../Spade_Research_David_Sumedh_Ba`.

### Code (mostly renames — `src/` was already clean)

| Old | New |
|---|---|
| `src/common/*` | `src/spade/common/*` (unchanged) |
| `src/models/spade_base.py` | `src/spade/saes/spade_base.py` |
| `src/models/spade_trainer.py` | `src/spade/saes/spade.py` |
| `src/models/spade_cosine_trainer.py` | `src/spade/saes/spade_cosine.py` |
| `src/extraction/extract_sae_training_data.py` | `src/spade/extraction/extract_activations.py` |
| `src/training/train_karvonen_matched.py` | split → `src/spade/training/core.py` + `train_<type>.py` ×8 |
| `src/evaluation/karvonen_adapter.py` | `src/spade/evaluation/karvonen_adapter.py` |
| `src/evaluation/run_karvonen_eval.py` | `src/spade/evaluation/run_eval.py` |
| `src/multiseed/aggregate_seeds.py` | `src/spade/multiseed/aggregate_seeds.py` |
| `src/figures/*` | `src/spade/figures/*` |
| `src/diagnostics/*` | `src/spade/diagnostics/*` (shorter names) |
| `src/legality/*` | `src/spade/legality/*` |
| `src/validation/final_validation.py` | `src/spade/validation/final_validation.py` |
| `shell/*` | `scripts/*` (slurm paths repointed to `spade_sae`) |

### Artifacts (consolidated)

| Old | New |
|---|---|
| `autoencoders/` | `checkpoints/main/` |
| `autoencoders_batch_topk/` | `checkpoints/batch_topk/` |
| `autoencoders_cosine_spade/` | `checkpoints/cosine_spade/` |
| `autoencoders_spade_sweep/` | `checkpoints/spade_sweep/` |
| `results/karvonen/` | `results/main/` |
| `results/karvonen_*/` | `results/{spade_sweep,batch_topk,cosine_spade}/` |
| `results/{legality,multiseed}/` | unchanged |
| `best_spade_lam1e-07_gam5e-03/` | figure → `figures/`, csv → `results/spade_sweep/` |
| `figures/` (dedup the 3 `Figure_4_karvonen_extension*` variants) | `figures/` |

### To confirm stale before dropping / archiving (decision deferred — see PROCESS.md)

`8_spade_configs_updated/`, `lambda_sweep_results/`, `gamma_sweep_results/`,
`W&B Results/`, `wandb/` (transient), `results/runs/Main_Trained_Run_All_SAE/` (empty stub),
`autoencoders/trainer_5/config.json` (stale Mac smoke-test config over real GPU weights).
