# Analysis

Downstream studies on the trained SAEs. Code and outputs stay co-located: each study is a
self-contained folder you can rerun in place. Nothing here retrains — analyses read
precomputed eval tensors (`f1_TFRRC.pt` under `results/`) or stream activations through
OthelloGPT once.

- `spade_vs_baselines.md` — the headline results writeup (SpaDE/CosineSpaDE vs baselines).
- `linear_probe/` — the gold-standard board-state probe (~0.97 accuracy) reused as ground
  truth across the geometry studies.
- `geometry/` — 14 studies of SAE feature geometry; see `geometry/SUMMARY.md`.
