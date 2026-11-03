# 3-28-2026 — SpaDE retrain and the λ / γ sweeps on FASRC

Historical record, reconstructed from the append-only cluster command log. It
covers the SpaDE-only retrain and the three sweep submissions that produced
`results/lambda_sweep/` and `results/gamma_sweep_lam1e-07/`.

**Paths here are as they were.** At the time the repo lived at
`$HOME/Spade_SAE_Research` on the cluster and was rsync'd up from a Mac working
copy; job scripts sat in `shell/slurm/` and were run as script paths. That
directory was deleted in the July 2026 reorg, the Mac copy is gone, and the
current workflow is [`../fasrc_commands.md`](../fasrc_commands.md). Nothing below
runs as written.

## SpaDE-only retrain

Retrained just the SpaDE SAE against updated hyperparameters, reusing the
existing extraction data, via `run_spade_only.sh` — a training job plus a
dependent eval that merges the result with the five baselines and regenerates
`figures/Figure_4_karvonen_extension.png`. Hyperparameters at the time:
`lambda_init=5e-6`, `gamma_reg=0.1`. Roughly 15–20 min training, 5 min eval.

Outputs went to `autoencoders_spade_v2/` and `results/karvonen_spade_v2/`, and
the figure was renamed on download to `_spade_v2` so it would not clobber the
baseline figure. That rename-on-scp pattern is why several
`Figure_4_karvonen_extension_*.png` variants exist with no distinct generator
script: same script, different input CSV.

## λ sweep — 3 configs at fixed γ

Probing the boundary between collapse and a working SAE:
**λ ∈ {5e-8, 8e-8, 9e-8}, γ = 0.001**. Training 20–30 min, eval 10–15 min.

Downloaded into a dedicated `lambda_sweep_results/` folder rather than into the
repo, because the eval script writes to `results/karvonen_spade_sweep/` and
`figures/Figure_4_karvonen_extension.png` — the same paths every sweep uses.

Result: λ=5e-8 collapsed outright (L0 8192, frac_recovered −0.222), and 8e-8 and
9e-8 came out identical. Later work established why — that whole λ region is
below fp32 resolution in the sparsemax scores, so runs there collapse or train
bitwise-identically. See [`../pipeline.md`](../pipeline.md) §5.

## γ sweep — 4 configs at λ=1e-7

Tuning γ at the best λ: **γ ∈ {0.005, 0.008, 0.015, 0.025}**. Submitted
2026-03-28 as training job 2137450 with dependent eval 2137452, after a first
attempt timed out at 4 hours around 25% complete and the SLURM limit was raised
to 20 hours. A stale dependent eval (2071074) from the earlier attempt had to be
cancelled first.

Two hazards this session hit, both now designed out:

- **Sweep directories were reused.** `autoencoders_spade_sweep/` still held
  `ae.pt` files from the λ sweep; only the `ae_9155.pt` intermediate checkpoints
  were from the γ run. Verifying a run meant comparing timestamps between
  `*/ae.pt` and `*/checkpoints/`. The λ-sweep results and figure had to be copied
  aside before the γ eval overwrote them.
- **Two runs carry misleading W&B names.** The run labeled `gam1e-02`
  (trainer_2) actually has γ=0.015, and `gam3e-02` (trainer_3) actually has
  γ=0.025. The config fields are authoritative.

Training 3–4 h for the four configs at 300M tokens, eval 15–20 min.

Both sweeps now have their own artifact directories — `results/lambda_sweep/`
and `results/gamma_sweep_lam1e-07/`, the second named to disambiguate it from the
CosineSpaDE γ sweep — so no backup dance is needed. Both are live inputs:
`spade.multiseed.aggregate_seeds` reads them for the seed-42 point of five of the
six sweep configs, and dropping either silently removes seed 42 from the headline
figure.
