# SLURM jobs

Cluster orchestration. Every script anchors with `cd "${SPADE_REPO:?}"` — required, not
defaulted, so a run against the wrong checkout fails loudly instead of quietly producing
results from the old tree. The source repo has four competing anchoring conventions and
eleven scripts that `cd` to a path deleted in July; under `set -e` they abort while their
submitter reports success. Do not reintroduce any of that.

Everything cluster-specific is in `cluster.env`: the account and partition, the modules and
the env to activate. No script names a cluster. `#SBATCH` lines cannot read a variable, so the
account and partition are `SBATCH_ACCOUNT` / `SBATCH_PARTITION`, which sbatch reads in place
of `--account` / `--partition`. Every job sources `cluster.env` for its software stack and
every `run_*.sh` sources it before submitting; to `sbatch` one script by hand, source it in
that shell first (`-A` / `-p` on the command line still win). On another cluster, edit
`cluster.env` and nothing else.

Chains use `sbatch --parsable` + `--dependency=afterok`. Modules are invoked as
`python -m spade.<module>`. `archive/scripts/slurm/studies/linear_probe` deviates and says why
in its header: it is seven CPU-minutes, so it goes to `$SPADE_CPU_PARTITION` rather than
queueing for an H100 it does not need. On FASRC, `kempner_requeue` includes Blackwell nodes that torch 2.5.1+cu124 has no
kernels for, so a GPU job sent there must pin `--gres=gpu:nvidia_h100_80gb_hbm3:1`.

- `l0_grids/`    — the 20-config L0 grid per architecture, trained then evaluated
- `sweeps/`      — the CosineSpaDE-only training run
- `multiseed/`   — seeds 43-46 training (gpu:4); their eval arrays are archived
- `studies/`     — one directory per study: `board_probe_sweep`, `bracket_flip`,
                   `generation_legality`, `move_legality`, `probes`, `straightness`,
                   `violation_divergence`
- `diagnostics/` — the revival retrain

Everything the paper does not run — the 8-type `main/` pipeline, the sweep evals, the rules,
manifold-capture and linear-probe studies, the norm diagnostics — is under
`archive/scripts/slurm/`.

Two things that bite. `#SBATCH --output` paths are relative to the submitting directory,
not the script, and an unwritable output path makes the job fail at launch with no log to
read — so log paths are repo-rooted. And `sbatch --export` is comma-delimited, so a
comma-separated grid passed that way is shredded into separate variables; export into the
submitting shell and use `--export=ALL`.

Big artifacts go under `study_dir(<name>)` (`$SPADE_DATA_ROOT`, see `spade.common.study`).
