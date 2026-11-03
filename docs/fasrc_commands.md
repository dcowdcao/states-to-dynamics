# FASRC commands

This cluster's specifics for the paper workflow in README.md and REPRODUCE.md. Private: the
public export drops this file and `scripts/slurm/cluster.env`. The pre-release runbook (the
8-type pipeline, W&B, uploads) is `archive/docs/fasrc_commands.md`.

## Where things are

| What | Path |
|---|---|
| Live data root (`SPADE_DATA_ROOT`) | `/n/netscratch/ba_lab/Lab/dcao`, 90-day retention |
| Staged data release (`SPADE_RELEASE_DATA_DIR`) | `/n/netscratch/ba_lab/Lab/dcao/release_data_staging` |
| Karvonen's two zips (`SPADE_KARVONEN_ZIPS`) | `/n/netscratch/ba_lab/Lab/dcao/karvonen_released/zips` |
| Tier A backup (v2 Gated files), git bundle | `~/spade_netscratch_backup/2026-11-03_paper_tierA/` |
| Tiers A-D copy | `/n/holylabs/LABS/ba_lab/Lab/dcao/spade_paper_data_2026-11-03/` |

`scripts/slurm/cluster.env` holds the account (`kempner_ba_lab`), the GPU partition
(`kempner_h100`), the CPU account and partition (`ba_lab`, `sapphire`), and the stack
(`module load python cuda/12.4.1-fasrc01; source activate spade_env`).

## The login node is 1 CPU / 8 GB

Run anything heavier than a git command as a job. A make target on a CPU node:

```bash
sbatch -A kempner_ba_lab -p test -c 8 --mem=48G -t 06:00:00 --wrap "cd $PWD && make figures"
```

and a GPU recompute on one H100 (torch 2.5.1+cu124 has no Blackwell kernels, so pin the card
on a partition that has them):

```bash
sbatch -A kempner_ba_lab -p kempner_h100 --gres=gpu:nvidia_h100_80gb_hbm3:1 -c 8 --mem=96G \
    -t 02:00:00 --wrap "cd $PWD && make compute-fig01"
```

`make data` against the staged release and the local zips, into a fresh root:

```bash
make data SPADE_DATA_ROOT=<empty dir> \
    SPADE_RELEASE_DATA_DIR=/n/netscratch/ba_lab/Lab/dcao/release_data_staging \
    SPADE_KARVONEN_ZIPS=/n/netscratch/ba_lab/Lab/dcao/karvonen_released/zips
```

## Two traps

- `spade_env`'s editable install points at `~/spade_sae`. A worktree imports its own tree
  only through its own venv: `$CONDA_PREFIX/bin/python -m venv --system-site-packages .venv`,
  then `.venv/bin/pip install -e . --no-deps`. Check with
  `.venv/bin/python -c "import spade; print(spade.__file__)"`.
- HF's file lock fails on NFS when array tasks share a cache: set
  `HF_DATASETS_CACHE="$TMPDIR/hf_datasets"` in jobs that stream the corpus concurrently.

## Re-staging the release

After any change to a released file in the live root:

```bash
sbatch -A kempner_ba_lab -p test -c 2 --mem=16G -t 01:00:00 --wrap \
    "cd $PWD && SPADE_DATA_ROOT=/n/netscratch/ba_lab/Lab/dcao .venv/bin/python scripts/release/stage_data.py <empty dir>"
```

It rewrites `manifests/data_release.tsv`; commit that, then upload the directory to the
anonymous HF dataset and set `SPADE_RELEASE_DATASET` in `scripts/fetch_release_data/fetch.py`.
