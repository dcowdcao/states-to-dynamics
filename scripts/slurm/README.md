# SLURM jobs (FASRC Kempner H100)

Cluster orchestration. All scripts `cd $HOME/spade_sae` (the new repo name — the original's
hardcoded `$HOME/Spade_SAE_Research` paths were a known staleness bug; do not reintroduce
them). Common preamble: `module load python cuda/12.4.1-fasrc01`, `source activate spade_env`,
`--account=kempner_ba_lab --partition=kempner_h100`. Chains use `sbatch --parsable` +
`--dependency=afterok`.

- `main/`      — train → extract → eval → figures for the 8-type baseline run (`run_all.sh`).
- `sweeps/`    — SpaDE λ/γ, CosineSpaDE γ, BatchTopK add-on sweeps.
- `multiseed/` — seeds 43-46 chain (train gpu:4 → per-seed eval array), then aggregate.
- `legality/`  — build_pairs → extract_acts → compute_metrics.

Big artifacts go to `$SPADE_MULTISEED_DIR` / netscratch, not the repo.
