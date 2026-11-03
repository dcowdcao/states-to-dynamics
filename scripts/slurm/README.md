# SLURM jobs (FASRC Kempner H100)

Cluster orchestration. Every script anchors with `cd "${SPADE_REPO:?}"` — required, not
defaulted, so a run against the wrong checkout fails loudly instead of quietly producing
results from the old tree. The source repo has four competing anchoring conventions and
eleven scripts that `cd` to a path deleted in July; under `set -e` they abort while their
submitter reports success. Do not reintroduce any of that.

Common preamble: `module load python cuda/12.4.1-fasrc01`, `source activate spade_env`,
`--account=kempner_ba_lab --partition=kempner_h100`. Chains use `sbatch --parsable` +
`--dependency=afterok`. Modules are invoked as `python -m spade.<module>`.
`archive/scripts/slurm/studies/linear_probe` deviates and says why in its header: it is seven
CPU-minutes, so it asks for `ba_lab / sapphire` rather than queueing for an H100 it does not
need. Note that `kempner_requeue` includes Blackwell nodes that torch 2.5.1+cu124 has no kernels
for, so a GPU job sent there must pin `--gres=gpu:nvidia_h100_80gb_hbm3:1`.

- `l0_grids/`    — the 20-config L0 grid per architecture, trained then evaluated
- `sweeps/`      — the CosineSpaDE-only training run
- `multiseed/`   — seeds 43-46 training (gpu:4); their eval arrays are archived
- `studies/`     — one directory per study: `board_probe_sweep`, `bracket_flip`,
                   `generation_legality`, `move_legality`, `straightness`, `violation_divergence`
- `diagnostics/` — the revival retrain

Everything the paper does not run — the 8-type `main/` pipeline, the sweep evals, the rules,
manifold-capture and linear-probe studies, the norm diagnostics — is under
`archive/scripts/slurm/`.

Two things that bite. `#SBATCH --output` paths are relative to the submitting directory,
not the script, and an unwritable output path makes the job fail at launch with no log to
read — so log paths are repo-rooted. And `sbatch --export` is comma-delimited, so a
comma-separated grid passed that way is shredded into separate variables; export into the
submitting shell and use `--export=ALL`.

Big artifacts go to netscratch under `study_dir(<name>)`, never into the repo.
