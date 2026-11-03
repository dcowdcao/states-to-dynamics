# FASRC run commands

The cluster workflow end to end: create the environment once, upload, submit,
collect. Kempner H100, account `kempner_ba_lab`.

Two conventions this runbook assumes, both enforced by the job scripts:

- **`SPADE_REPO` names the checkout.** Every SLURM script starts
  `cd "${SPADE_REPO:?}"` — required, not defaulted, so a run against the wrong
  tree fails loudly instead of quietly producing results from an old one.
- **Modules are invoked as `python -m spade.<module>`**, never as a script path.

For the job layout, the log-path and `--export` gotchas, and which directory
holds which program, see [`scripts/slurm/README.md`](../scripts/slurm/README.md).

```bash
export SPADE_REPO="$HOME/spade_sae"      # put both in ~/.bashrc
export SPADE_DATA_ROOT=/n/netscratch/ba_lab/Lab/dcao   # study trees + weights (spade.common.study)
```

## 1. Create the environment (once)

```bash
module load python cuda/12.4.1-fasrc01
mamba create -n spade_env python=3.11 -y
source activate spade_env

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e "$SPADE_REPO"
```

`transformer_lens` is pinned at 1.17.0 — 2.x is incompatible with the vendored
`circuits`. The install must be **editable**: `spade.common.paths.repo_root()`
walks up for a `CLAUDE.md` / `.git` marker, and a non-editable install resolves
into site-packages and then fails late, at the first `karvonen_root()` call.

The dependencies come from `requirements.txt`, which `pyproject.toml` reads as the
package's requirement list. It was regenerated off the working `spade_env` and
verified (`docs/finish_notes/env-install.md`); the source repo's version, which
could not build a working environment, is not what ships here. Do not hand-install
a package list beside it.

`plotly` and `umap-learn` are deliberately not in it — only the older
`analysis/geometry/` scripts import them.

### W&B

Once, on the login node, so SLURM jobs authenticate from `~/.netrc`:

```bash
source activate spade_env
wandb login          # key from https://wandb.ai/authorize
```

Training logs to project `spade-vs-karvonen-othello`.

### Pre-cache HuggingFace models and datasets

Compute nodes may have no internet:

```bash
python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('Baidicoot/Othello-GPT-Transformer-Lens')"
python -c "from datasets import load_dataset; next(iter(load_dataset('taufeeque/othellogpt', split='train', streaming=True)))"
```

Point the datasets cache at node-local scratch and leave `HF_HOME` on NFS —
concurrent array tasks otherwise collide on an NFS flock. `HF_USE_SOFTFILELOCK=1`
does nothing in this `datasets` version:

```bash
export HF_DATASETS_CACHE="$TMPDIR/hf_datasets"
```

## 2. Upload

`external/Karvonen_Research` and `external/tpr_othello` are runtime dependencies
and must go up with the code; three files inside `Karvonen_Research` are hidden
from git by its own vendored `.gitignore` and by the root ignore, so a
`git clone` alone yields a broken `circuits`.

```bash
# locally; SPADE_LOCAL is your working copy
rsync -avz --progress \
  --exclude='*.pyc' --exclude='__pycache__' --exclude='.DS_Store' \
  --exclude='wandb/' --exclude='*.pt' \
  "${SPADE_LOCAL:?}"/{src,scripts,external,pyproject.toml,requirements.txt} \
  dcao@login.rc.fas.harvard.edu:spade_sae/
```

`*.pt` is excluded deliberately: SAE weights, the linear probe and the
activation caches move by filesystem copy, not through this path.

## 3. Smoke test before committing a long job

```bash
salloc -p kempner_requeue -A kempner_ba_lab -c 2 --gres=gpu:1 -t 00:30:00
source "$SPADE_REPO/scripts/slurm/cluster.env"
python -m spade.validation.final_validation
bash "$SPADE_REPO/scripts/smoke_test.sh"
exit
```

## 4. Submit

```bash
cd "$SPADE_REPO"
mkdir -p logs/{training,extraction,eval,figures}
bash scripts/slurm/main/run_all.sh
```

Account, partition, modules and env come from `scripts/slurm/cluster.env`, which every
`run_*.sh` and job sources; to `sbatch` a single script by hand, `source` it first.
`run_all.sh` submits train and extract in parallel, then chains eval on both with
`--dependency=afterok`, and prints the three job IDs. Roughly: training 8–16 h,
extraction 1–2 h, evaluation 4–8 h on one H100.

Other programs, same shape:

```bash
bash scripts/slurm/sweeps/run_spade_sweep.sh        # SpaDE lambda/gamma, CosineSpaDE gamma
bash scripts/slurm/multiseed/run_multiseed.sh       # seeds 43-46, then aggregate
bash scripts/slurm/studies/<study>/                 # one dir per downstream study
```

## 5. Monitor

```bash
squeue -u dcao
sacct -u dcao --format=JobID,JobName%25,Elapsed,State,ExitCode --starttime=today

tail -f "$SPADE_REPO"/logs/training/train_JOB_ID.out
tail -f "$SPADE_REPO"/logs/training/train_JOB_ID.err
```

tqdm writes to stderr, so progress is in `.err` and print output in `.out`.
Check both.

## 6. Collect

Results and figures are small; weights are not.

```bash
# locally
scp -r dcao@login.rc.fas.harvard.edu:spade_sae/{results,figures,logs} "${SPADE_LOCAL:?}/"
scp -r dcao@login.rc.fas.harvard.edu:spade_sae/checkpoints "${SPADE_LOCAL:?}/"   # ~2 GB
```

Sweeps write to their own `results/<experiment>/` and `checkpoints/<experiment>/`
subdirectory, so nothing overwrites the main run — the older workflow's
back-up-before-eval dance is gone. **Do not merge sweep outputs by trainer
index**: sweep directories restart `trainer_N` at 0, and three different
experiments each contain a `trainer_0` from a different λ/γ grid.

Big study intermediates never come back over scp. They live on netscratch under
`$SPADE_DATA_ROOT/spade_<study>`, resolved by `spade.common.study.study_dir()` and
overridable with `SPADE_<STUDY>_DIR`. That tier is on **90-day retention**; back up
anything you need to keep, seeds 43–46 above all.

---

Historical one-off sessions — the March/April sweep submissions, their job IDs
and their resubmit dances — are in [`session_logs/`](session_logs/), not here.
