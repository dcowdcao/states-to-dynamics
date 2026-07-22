# SpaDE SAE Research

Sparse Autoencoder research on **OthelloGPT**, comparing Karvonen's baseline SAE families
against two prototype-distance architectures developed here:

- **SpaDE** (Sparsemax-Distance Encoder) — `f = sparsemax(-λ·‖x − Aeᵢ‖²)`, no-bias decode.
- **CosineSpaDE** — the cosine-similarity variant (γ-scaled, no λ).

All SAEs train and evaluate through Karvonen's `trainSAE()` loop and board-reconstruction
protocol on OthelloGPT's layer-5 residual stream (`blocks.5.hook_resid_post`, `d_model=512`,
dictionary `8192` = 16×). The 8 types: PAnneal, Standard, Gated, TopK (k=80), GatedAnneal,
SpaDE, BatchTopK, CosineSpaDE.

> **Status: skeleton.** This tree is an architecture scaffold — files hold pseudocode, not
> working code. Logic is being ported from `../Spade_Research_David_Sumedh_Ba`. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the old→new map and [PROCESS.md](PROCESS.md) for
> decisions.

## Run it

```bash
make setup                       # create venv, pip install -e ., vendor Karvonen_Research
make run                         # full pipeline: extract → train → eval → figures
```

## Dev setup

`external/Karvonen_Research` is a runtime dependency (`circuits.dictionary_learning`).
Path resolution is repo-root-anchored (`spade.common.paths` walks to the `CLAUDE.md`/`.git`
marker), so scripts work from anywhere. Cluster (FASRC SLURM) commands live in
`docs/fasrc_commands.md`.

## Making changes

Code lives under `src/spade/`; the pipeline stages map 1:1 to its subpackages
(`extraction → training → evaluation → multiseed → figures`, plus `diagnostics` and the
`legality` sub-study). Downstream analyses live in `analysis/`. Start from
[ARCHITECTURE.md](ARCHITECTURE.md).
