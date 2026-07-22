# FASRC run commands

> Rewrite from the original `docs/fasrc_run_commands.md`, fixing the stale paths.

Copy-paste FASRC workflow: rsync upload, conda env creation (`spade_env`), wandb login, HF
pre-caching, smoke test, then the SLURM chains in `scripts/slurm/`.

**Repo path is `$HOME/spade_sae`** (the original doc hardcoded the old `~/Spade_SAE_Research`
everywhere — the single biggest staleness bug; do not carry it over). Drop the one-off dated
job IDs and the pre-multiseed hyperparameters when porting.
