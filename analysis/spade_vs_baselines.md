# SpaDE vs baselines (headline results)

> Placeholder — port the full writeup from the original `analysis/analysis_spade_vs_baselines.md`.

Compares SpaDE / CosineSpaDE against the 5 Karvonen baselines + BatchTopK on OthelloGPT
layer-5 board-state reconstruction (Karvonen protocol). Core narrative to carry over:

- Distance-SpaDE underperforms every baseline; variance-explained capped ≈0.43 regardless
  of L0 (baselines 0.97–1.00). TopK (k=80) best overall.
- Originally attributed to the sparsemax **simplex ceiling**.
- **CosineSpaDE overturns this**: same simplex + no-bias decode, cosine score instead of
  squared distance → variance-explained ≈0.979. Real culprit is the **distance-norm
  pathology** (magnitude-driven prototype assignment on heavy-tailed standardized acts).
- Behavioral-vs-representational gap: SpaDE keeps frac-recovered ≈0.95 while explaining
  ~43% variance ("dark" variance not needed for next-move logits).
