# Session logs

Dated narrative logs of work sessions: planning, gotchas, submitted job IDs.
Historical context, not current-state docs — the current state lives in
`PROCESS.md` and the other `docs/`.

**Read these as records of their date.** Paths, job IDs, hyperparameter defaults
and conclusions are as they stood then, and several have since moved or been
retired — the repo path, `shell/slurm/`, the SpaDE γ=1e-2 default, the geometry
family's SpaDE conclusion. Check `docs/claims_status.md` before quoting a result
from one of these, and `ARCHITECTURE.md` before running a command.

- `3-28-2026-spade-sweeps.md` — the SpaDE-only retrain and the λ and γ sweeps.
- `4-4-2026.md` — PI session: the λ/γ grid, dead-neuron tracking, the
  distance-weighted L1 verification.
- `4-2026-spade-sweep-v2-8-configs.md` — the 8-config sweep that grid produced.
- `5-10-2026.md` — the per-move board reconstruction color-flip bug, per-move L0,
  PCA/UMAP of per-square feature directions.
- `7-14-2026-multiseed-error-bars.md` — how seeds flow through training, and the
  five-seed run behind every error bar.
