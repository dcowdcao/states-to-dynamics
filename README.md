# SpaDE SAE Research

Sparse Autoencoder research on **OthelloGPT**, comparing Karvonen's baseline SAE families
against two prototype-distance architectures developed here:

- **SpaDE** (Sparsemax-Distance Encoder) — `f = sparsemax(-λ·‖x − Aeᵢ‖²)`, no-bias decode.
- **CosineSpaDE** — the cosine-similarity variant (γ-scaled, no λ).

All SAEs train and evaluate through Karvonen's `trainSAE()` loop and board-reconstruction
protocol on OthelloGPT's layer-5 residual stream (`blocks.5.hook_resid_post`, `d_model=512`,
dictionary `8192` = 16×). The 8 types: PAnneal, Standard, Gated, TopK (k=80), GatedAnneal,
SpaDE, BatchTopK, CosineSpaDE.

On top of that comparison sit the interpretability studies — whether the model's layer-5
representations encode the rules of Othello, what happens to a square when it is flipped,
how an illegal move propagates, whether trajectories straighten with depth, and whether
SAE features capture concept manifolds or shatter them.

> **Status: ported and running.** All fifteen zones are ported from
> `../Spade_Research_David_Sumedh_Ba`, which stays frozen and read-only; all new work happens
> here. The package installs, `final_validation` is 25/25, and three checkpoints reproduce
> their published evaluation exactly. Two data-provenance questions are open and affect
> published numbers — MIGRATION.md questions 14 and 15. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the layout and the old→new map,
> [MIGRATION.md](MIGRATION.md) for what is left and the open questions, and
> [PROCESS.md](PROCESS.md) for decisions.

## Run it

```bash
make setup                       # create .venv, pip install -e . with the pinned deps
make run                         # full pipeline: train → extract → eval → figures
```

## Dev setup

`make setup` builds `.venv` from `python3.11` and installs the pins in `requirements.txt`
(~2 GB, mostly torch). On FASRC that stack already exists in the `spade_env` conda env, so
reuse it instead of re-downloading it:

```bash
$CONDA_PREFIX/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-deps
```

Two vendored runtime dependencies under `external/`: `Karvonen_Research` (`circuits.*` —
the baseline trainers, the Othello engine, the board-reconstruction eval) and `tpr_othello`
(the tensor-product board probe). Path resolution is repo-root-anchored — `spade.common.paths`
walks to the `CLAUDE.md` / `.git` marker — so scripts work from anywhere, but only under an
editable install. Cluster (FASRC SLURM) commands live in `docs/fasrc_commands.md`.

Heavy per-study intermediates and downloaded weights live under `$SPADE_DATA_ROOT`
(default `data/`), resolved by `spade.common.study`. On FASRC they already exist on
netscratch, so set `export SPADE_DATA_ROOT=/n/netscratch/ba_lab/Lab/dcao`. A fresh clone
cannot regenerate the figures on its own.

## Making changes

Code lives under `src/spade/`. The pipeline stages map 1:1 to its subpackages
(`extraction → training → evaluation → multiseed → figures`, plus `diagnostics`). Beneath
them sit the shared layers every study builds on — `othello/` (game corpora, board
mechanics, the legality taxonomy), `saes/` (the three SpaDE variants and the checkpoint
registry) and `spaces.py` (the representation-space API). The studies themselves —
`rules/`, `bracket_flip/`, `violation_divergence/`, `straightness/`, `manifold_capture/` —
each pair with an `analysis/<study>/` that only draws.

Figures obey [FIGURE_GUIDELINES.md](FIGURE_GUIDELINES.md); color literals live only in
`spade.figures.paper_style`. Before writing prose about any result, read
[docs/claims_status.md](docs/claims_status.md) — several headline claims were retired.
