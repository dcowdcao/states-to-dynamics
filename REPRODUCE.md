# Reproducing the paper

Two tiers. **Redraw** (`make figNN`, all 16 as `make figures`) draws each figure on a CPU
from the committed `results/` tables and the data release. **Recompute**
(`make compute-figNN`) first regenerates the figure's inputs from public sources: the
OthelloGPT model, the HF game corpus, Karvonen's SAE zips and the WTHOR archives.

```bash
make setup     # .venv from requirements.lock; external/ at pinned commits
make data      # the data release + Karvonen's SAEs into $SPADE_DATA_ROOT, sha256-checked
make figures   # all 16 figures into figures/: 5 min on one CPU core
make verify    # figures/ and results/ against manifests/figures.tsv and results.tsv
```

`make verify` compares every file by sha256, a PDF without its `/CreationDate` (the only
bytes a re-render changes). From a fresh clone, all 32 figure files and all 50 `results/`
files match. Redraws run on one thread: with several, a BLAS reduction can change between
runs, and Fig 10's PNG then moves by a few pixels.

## Data

Everything large lives under `$SPADE_DATA_ROOT` (default `data/`), at the same relative paths
on every machine.

| Tier | What | Size | Source |
|---|---|---|---|
| release | Our three roster SAEs; board, TPR and legality probes; the spliced and rollout branch trees; the rollout bank and generation scans; detector, straightness and by-space caches; bracket-flip events and probe-space fits; the L0-sweep eval tables behind `results/shared/l0_roster/`; game 1235's layer-5 activations | 527 MB, 972 files | `make data` step 1: `manifests/data_release.tsv`'s files from the HF dataset `SPADE_RELEASE_DATASET`, or a local copy at `SPADE_RELEASE_DATA_DIR` |
| Karvonen | His five roster SAEs, plus Gated `trainer32-v2`, which the figures used before the Gated erratum | 12.3 GB of zips, 0.7 GB extracted | `make data` step 2: `adamkarvonen/othello_saes` at revision `f997cba`, zips checked against `scripts/download_karvonen_saes/zips.sha256`. `SPADE_KARVONEN_ZIPS` names a directory already holding them |
| not released | Layer 0-7 activations of both branch trees (2.1 and 8.35 GB per layer), straightness trajectories, the probe corpus, bracket-flip activations | ~100 GB | `compute-*`, on a GPU |

Two kinds of release file differ from what the pipeline wrote.
`spade_straightness/games/games.pt` holds only its leading legal block, rows 0-1999 of the HF
corpus: its later classes include WTHOR championship games, which are not re-hosted. The
redraws and the detector's legal profile read only that block, and `compute-straightness`
rebuilds the full file, downloading WTHOR itself. Text files (eval tables, manifests) name
paths relative to the data root. `scripts/release/stage_data.py`
builds the release from a data root and rewrites the manifest.

## Pinned versions

- Python 3.11, torch 2.5.1 (CUDA 12.4 wheels), TransformerLens 1.17.0, numpy 1.26.4,
  datasets 4.8.4; every transitive version in `requirements.lock`.
- OthelloGPT `Baidicoot/Othello-GPT-Transformer-Lens` @ `30e9176f812fc31580671bc93ba9936bf2b7e08a`
  and the corpus `adamkarvonen/othello_45MB_games` @ `caafaff9fd273cd1783e146d039e262f97c9f5f8`,
  passed to every load from `spade.common.config.HF_REVISIONS`. The SAE training stream
  `taufeeque/othellogpt` has no revision on record.
- Karvonen's SAEs `adamkarvonen/othello_saes` @ `f997cba04444628deda521daab4c8b9a48735af6`.
- `external/`: `adamkarvonen/SAE_BoardGameEval` @ `7187d4a` with `saprmarks/dictionary_learning`
  @ `43421f5`, and `ajyl/tpr_othello` @ `cc90cea`, each plus our patch (`scripts/fetch_external/`).

## Redraw, per figure

`make figNN` runs the scripts below twice (PNG, then the camera-ready PDF) and stages
`figures/figNN_<name>.{png,pdf}`. Each also reads its own `results/figNN_*/` and
`results/shared/`; the paths below are under the data root. Times are wall time on one core
of a cluster node, from a fresh clone (setup 14 min, data 1 min).

| Fig | Script | Reads from the data release | Rewrites in `results/` | Time |
|---|---|---|---|---|
| 1 | `python -m spade.figures.figure_rules_and_generation` | nothing; panel (a) streams the HF corpus | | 20 s |
| 2 | `analysis/probe_directions_and_detection/make_figure.py` | nothing | | 5 s |
| 3 | `analysis/violation_divergence/illegal_move_detector/figure_overview.py` | `spade_divergence/branches/branches.pt`, `spade_divergence/illegal_move_detector/{meta,d}.npz`, `spade_straightness/geometry/resid_post_5_raw.npz` | | 15 s |
| 4 | `illegal_move_detector/figure_auroc_l0_matched.py`, then `figure_sae_overview.py` | Fig 3's npz, the 8 `illegal_move_detector/d_l0_*.npz` and `spade_straightness/geometry/l0_*_legal.npz` | `fig04_*/`: all three CSVs | 20 s |
| 5 | `analysis/board_probe_sweep/layer_sweep.py` | nothing | | 5 s |
| 6 | `python -m spade.l0_operating_point.plot_pareto`, then `.combine_figure` | nothing (`results/shared/l0_roster/pooled.csv`, Karvonen's published CSV in `external/`) | | 30 s |
| 7 | `analysis/bracket_flip/make_figures.py --only fig1` | nothing | | 10 s |
| 8 | `analysis/violation_divergence/changepoint_localization/quantity_ablation.py --l0-matched` | Fig 4's, `illegal_move_detector/theta_norm*.npz` (9), `geometry/l0_*_raw.npz` (8) | `fig08_*/quantity_ablation_l0_matched.csv` | 25 s |
| 9, 11 | `analysis/violation_divergence/fork_and_board/make_figures.py --l0-matched [--plane r21]` | `spade_divergence/branches/branches.pt`, `spade_divergence/fork_and_board/game_1235_resid_post_5.pt`, the eight SAEs (`karvonen_saes/`, `spade_multiseed/seed_42/`) | | 35 s each |
| 10, 12 | the same, `--variant rollout` | `spade_divergence/rollout/branches/branches.pt`, `spade_divergence/rollout/fork_and_board/game_1235_resid_post_5.pt`, the eight SAEs | | 35 s each |
| 13, 14 | `analysis/violation_divergence/divergence_by_space/make_figures.py --n_games 100 --l0-matched [--raw]` | `spade_divergence/divergence_by_space/curves_n100_l0_matched.npz` | its `divergence_by_space_*.csv` | 10 s each |
| 15, 16 | the same, `--plane r21` | `spade_divergence/divergence_by_space/curves_r21_n100_l0_matched.npz` | its `divergence_by_space_*.csv` | 10 s each |

## Recompute, per figure

`make compute-figNN` runs the shared stages it needs (next section), then the steps below,
then `make figNN`. It writes into `$SPADE_DATA_ROOT` and `results/`, and a step that finds its
output present checks it (`rollout_bank`), refuses (`build_branch_tree.sh`) or keeps it (the
geometry caches). So recompute into an empty data root, with our three SAEs
(`spade_multiseed/seed_42/`) copied in from the release for the L0-matched figures (4, 8-16).
`compute-fig01` and `compute-fig06` are the exceptions: they run where `make data` ran, the
first asserting its redraw of the released rollout bank, the second reading the eval tables. The GPU runs
torch 2.5.1's CUDA 12.4 build, which has no Blackwell kernels. On SLURM, wrap the target
(`sbatch --gres=gpu:1 --wrap "make compute-fig03"`) or use the per-study jobs in
`scripts/slurm/studies/`, which read `scripts/slurm/cluster.env` (from `cluster.env.example`).

| Fig | Shared stages, then | GPU | Time |
|---|---|---|---|
| 1 | `spade.othello.rollout_bank --n_games 1000` → `spade.generation_legality.rollout_scan` → `.aggregate --rollout-only` | yes | 3 min on one H100 |
| 2 | branches, probes; `spade.move_legality.fit_set` → `.extract_acts` → `.readout --spaces raw` → `analysis/move_legality/detection_auroc.py --space raw` → `spade.probe_geometry.embed_directions` | yes | GPU minutes per step; `embed_directions` ~15 min |
| 3 | straightness, branches; `illegal_move_detector/extract_geometry.py` → `eval_games.py` → `select_game.py` → `figure_auroc.py` → `figure_histograms.py` | yes | ≤ 1 h after the stages |
| 4 | Fig 3's, roster; `extract_geometry.py --l0-matched` → `analysis/straightness/l0_matched_profile.py` | yes | ≤ 1 h more |
| 5 | probes; `analysis/board_probe_sweep/measure.py` | yes | minutes |
| 6 | `spade.l0_operating_point.aggregate` → `.pick`, from the released eval tables | no | ~2 min |
| 7 | probes; `spade.bracket_flip.events` → `.extract_acts` → `.events --variant field --n_events 4000` → `.extract_acts --variant field` → `.probe_space` | yes | ~1 h, half of it HF streaming |
| 8 | Fig 4's; `extract_geometry.py --l0-matched --quantities` | yes | +3 min |
| 9, 11 | branches, roster; `fork_and_board/game_acts.py` | yes | seconds after the stages |
| 10, 12 | rollout tree, roster; `fork_and_board/game_acts.py --variant rollout` | yes | seconds after the stages |
| 13-16 | branches, roster; `divergence_by_space/measure.py --n_games 100 --l0-matched [--plane r21]`, on the CPU so all eight spaces share one device | yes, for the stages | ~5 min each on 8 cores |

A.6's 0.869 (`results/fig03_*/threshold_localization.csv`) is written by
`changepoint_localization/measure.py` over `curvature_discriminant/branch_geometry.py`'s
tables, which no target runs. The appendix's figureless numbers each have one CPU script in
`analysis/appendix_numbers/` (its README).

## Shared stages

| Target | Steps | Writes | GPU, time |
|---|---|---|---|
| `compute-branches` | `scripts/build_branch_tree.sh`, then `spade.violation_divergence.extract_acts` | `spade_divergence/{branches,acts}/`, 19 GB | tree on the CPU ~32 min; activations < 1 h |
| `compute-rollout-tree` | `spade.violation_divergence.rollout_branches`, then `extract_acts --variant rollout` | `spade_divergence/rollout/`, 76 GB | ~15 min sampling, then extraction (3 h limit) |
| `compute-straightness` | `spade.straightness.build_games` (downloads WTHOR) → `extract_traj` → `compute_geometry --spaces resid_post_5` | `spade_straightness/` | ≤ 4 h |
| `compute-probes` | `spade.probes.build_corpus` (20,000 games) → `train --arm board`, `train --arm tpr` (layers 0-7) → `tpr_weight_decay` | `spade_probes/`, 9.5 GB corpus | ~3 + 5 + 2 min |
| `compute-roster` | Karvonen's download, as `make data` step 2 | `karvonen_saes/` | no |

`build_branch_tree.sh` builds the paper's tree: 2003 round-robin games, then t0 = 20 appends
until 2000 rule-complete games fork there (3923 games, 34,546 rows), checked against sha256
`6e5a0253…`. 2003 rather than 2000 because the tree's games 2000-2002 came from a superseded
round-robin mode of `extend_branches`; plain `branches` plus `--t0 20` gives another,
3920-game tree. The script refuses to run over an existing tree.

## Not bit for bit

- **GPU encodes.** The same activations through the same SAE on a GPU and on a CPU differ by
  ~1e-3 relative in the latent geometry (the re-seated GatedSAE's GPU `d` matched a CPU encode
  to 6e-4). Redraws encode on the CPU and reproduce the committed figures exactly; a
  GPU recompute can move an AUROC in its third decimal.
- **Rollout sampling.** `rollout_branches` batches by fork position, and a changed batch moves
  the logits in their last bits, so a rebuilt rollout tree is a different sample. The paper's
  tree was built in three passes (`rollout_branches`, then `extend_rollout_branches` after each
  `extend_branches`); the release ships it. The 1000-game rollout bank reproduces exactly on
  the same GPU type, and `rollout_bank` asserts that against an existing bank.
- **Fig 1's aggregate.** On one H100, `compute-fig01` redraws the rollout bank exactly and its
  scan byte for byte; `rollout_by_move.csv` then differs from the committed table in 44 of
  5,220 cells, by ≤ 1.1e-16 (summation order), and the figure's PNG not at all.
- **Probe fits.** The legality probes refit on a CPU within 3e-4 of the released GPU fits, so
  A.4's fourth digits need the released weights. Board and TPR probes refit close, not
  identical. `probe_space` on a CPU matches the GPU tables to ~1e-7 relative.

## Not scripted

- **Training our three roster SAEs**, 4-10 h on three or four GPUs each: BatchTopK with
  `scripts/slurm/l0_grids/run_l0_grid.sh batch_topk`; CosineSpaDE with
  `scripts/slurm/multiseed/run_cosine_sweep_multiseed.sh "0.05,0.1,0.2,0.5,1,1.5,2,3,5" cosine_spade_sweep`
  (seed 42 is the seat). SpaDE's `autoencoders_l0_coverage/trainer_5` came from a sweep script
  that is not in this repository. The release ships all three.
- **The L0 sweep's evaluation** behind Fig 6's points: `karvonen_repro/submit_eval.sbatch` per
  stream, ~12.5 min per 20 checkpoints on one H100. The 196 checkpoints are not released; their
  eval tables are, and `compute-fig06` derives the roster from them.
- **Game 1235's selection** (B.2): `analysis/violation_divergence/game_selection/score_games.py`,
  ~4 h on a CPU; the committed ranking is `results/fig09_*/ranking.csv`.
