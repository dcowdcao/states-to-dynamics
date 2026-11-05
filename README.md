# From States to Dynamics: Detecting Rule Violations in OthelloGPT

```bash
make setup && make data && make figures   # Python 3.11; redraws the paper's 16 figures on a CPU
make verify                               # byte-compares figures/ and results/ with manifests/
```

## Dev setup

`make setup` needs Python 3.11 (`make setup PYTHON=/path/to/python3.11` if `python3` is
another version). It builds `.venv`, installs `requirements.lock` (every transitive pin,
~2 GB, mostly torch 2.5.1) and then this package editable. It also clones the two
third-party trees the code imports, which carry no licence to redistribute, into `external/`
at pinned commits and applies our patches: `SAE_BoardGameEval` (Karvonen's SAE trainers,
Othello engine and board-reconstruction eval) and `tpr_othello` (the TPR board probe).

Large inputs live under `$SPADE_DATA_ROOT` (default `data/`, gitignored); every study tree
is a directory there. `make data` fills it in two steps, each sha256-checked:
- the data release (0.53 GB: our trained SAEs and probes, branch trees, cached
  intermediates; `manifests/data_release.tsv`) from the Hugging Face dataset named by
  `SPADE_RELEASE_DATASET`, or from a local copy at `SPADE_RELEASE_DATA_DIR`;
- Karvonen's released SAEs, two zips (12.3 GB) from Hugging Face at a pinned revision, of
  which six trainer directories are extracted. Point `SPADE_KARVONEN_ZIPS` at a directory
  that already holds them to skip the download.

`make figures` needs network access once, for Fig 1(a)'s games. `make test` runs the
self-tests (no GPU, no data). Recomputing a figure from the public model and corpus is
`make compute-figNN` and needs a GPU; [REPRODUCE.md](REPRODUCE.md) lists every figure's
commands, inputs, outputs and runtimes. The SLURM jobs under `scripts/slurm/` read the
cluster's account, partitions and environment from `scripts/slurm/cluster.env`: copy
`cluster.env.example` there and fill it in.

## Layout and where to start

```
figures/        the 16 paper figures, figNN_<name>.png + .pdf; written only by scripts/stage_figures.py
results/        the tables the paper draws or quotes, one folder per figure, plus shared/ and
                appendix_numbers/; each writer names which of its outputs go here
src/spade/      the package
  common/         config (model, layer, pinned HF revisions), paths, the data root (study.py)
  othello/        game corpus, board replay, the legality taxonomy
  saes/ sae_loader/ l0_operating_point/
                  SAE architectures, checkpoint loading, the L0-matched roster of eight SAEs
  <study>/        one package per measurement: violation_divergence, straightness,
                  move_legality, probes, bracket_flip, generation_legality, probe_geometry
  figures/        shared plotting: paper_style.py holds every colour and marker
analysis/       per-study scripts that measure from a study tree and draw a figure
scripts/        stage_figures.py; fetch_external/, fetch_release_data/, download_karvonen_saes/
                (what make setup and make data run); build_branch_tree.sh; slurm/ (cluster jobs);
                verify_outputs/
karvonen_repro/ our checkpoints through Karvonen's board-reconstruction eval (Fig 6)
checkpoints/    configs and standardization of the seed-42 SAEs the registry checks
manifests/      sha256 of the data release and of figures/ and results/
```

The private working repository also has `archive/`, every file the paper does not use, and
`scripts/release/`, the tools that build the public export and its data release.

To change a figure, edit the script `make figNN` runs (REPRODUCE.md, or the comment above its
entry in `scripts/stage_figures.py`), run `make figNN`, and check the result against
[FIGURE_GUIDELINES.md](FIGURE_GUIDELINES.md); colours and markers come only from
`spade.figures.paper_style`. A figure is drawn from its `results/` tables and the cached
intermediates under the data root; the `compute-*` chain upstream of it regenerates those.
After a deliberate change to an output, `python scripts/verify_outputs/verify.py --update`
rewrites the manifests, so the diff shows what moved. `make test` must stay green.
