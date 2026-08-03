# Geometry studies — index

13 self-contained studies of SAE feature geometry on OthelloGPT. Each folder holds its
`make_<study>.py` + a `README.md` + `summary.json` + its PNGs. Cross-cutting takeaways to
carry over: "best SAE" is metric-dependent (TopK owns the single sharpest feature; PAnneal
wins MSE + board reconstruction; SpaDE-1e-4 wins per-square avg coverage); the linear probe
(~0.99) is the gold standard, so the bottleneck is SAE design rather than layer-5
information; move 0 is uniformly hard.

Two caveats before quoting any of this. The `legal_vs_illegal/` study was deleted and
became `analysis/rules_manifolds/`. And this family is the last one still built on the
7-SAE set, including the SpaDE γ=1e-2 checkpoint that was later excluded elsewhere for
degenerate provenance — its "SpaDE γ=1e-2 is broken for cell-state mapping" conclusion is
plausibly an artifact of that checkpoint rather than of the architecture. See
`docs/claims_status.md`.

| Study | What it shows |
|---|---|
| `feature_locality_heatmaps/` | 3×3 board-F1 heatmaps of each SAE's sharpest / min / max-locality feature |
| `feature_locality_distributions/` | per-feature locality histograms (bimodal baselines vs unimodal Standard/SpaDE) |
| `feature_locality_deep_dive/` | drill into TopK ≥0.7 and SpaDE-1e-4 ≥0.5 locality buckets |
| `per_square_best_feature_heatmaps/` | per-square avg-state coverage; ranking inverts locality |
| `linear_probe_directions_pca/` | PCA-3 of the 180 linear-probe direction vectors |
| `sae_directions_pca/` | PCA-3 of each SAE's 180 best-per-cell decoder directions |
| `mse_per_token_heatmaps/` | per-token reconstruction MSE across 100 games (SpaDE pathological at token 0) |
| `per_move_board_reconstructions/` | reconstruct the board every move of game 42, 8 reconstructors |
| `per_move_accuracy_multigame/` | per-move reconstruction accuracy pooled over 100 games |
| `activations_pca_probe_overlay/` | PCA of raw layer-5 acts colored by move, probe directions overlaid |
| `embedding_pca_umap/` | 2D PCA + UMAP of per-square feature directions |
| `l0_per_move/` | L0 sparsity vs game move per SAE |
| `legal_vs_illegal/` | legality geometry (consumes the `spade.legality` outputs) |
| `local_squares_comparison/` | where on the board TopK vs SpaDE-1e-4 sharp features live |
