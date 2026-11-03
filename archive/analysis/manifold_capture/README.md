# Manifold capture: does SpaDE tile Othello concept manifolds or capture them 1-to-1?

Applies "Do Sparse Autoencoders Capture Concept Manifolds?" (Bhalla, Fel et al.
2026, arXiv:2604.28119) to OthelloGPT layer-5 and the repo's 8 seed-42 SAEs.
SpaDE/CosineSpaDE are the paper's App C "concepts as points" (simplex) paradigm;
direction SAEs (TopK, BatchTopK, Standard, Gated, PAnneal, GatedAnneal) are
"concepts as directions". Concepts: t, n_flipped, n_legal, disc_diff, three
board squares (mover-relative), and the 7-way rule category.

Run: `sbatch scripts/slurm/studies/manifold_capture/submit_manifold_capture.sh`
(build corpus → extract → 7 analyses → figures; resumable, ~all artifacts in
`results/manifold_capture/`, corpus and activations under `study_dir("manifold")`).

## Headline results

Othello concepts are real low-dimensional manifolds: the per-concept centroid
curves need only k_M = 2–13 PCA dims for 90% VE (t 4, n_flipped 4, n_legal 3,
disc_diff 13, squares/rules 2). Yet **no SAE × concept lands in the capture
regime** (regimes.png, curve variant, τ=0.3) — the paper's dilution finding
replicates on OthelloGPT:

- **Direction SAEs dilute.** Corpus support sizes 44 (Gated) – 5568 (ReLU) vs
  k_M ≤ 13; capture curves plateau far beyond k_M and below the PCA ceiling.
  ReLU/p-Anneal are "mixed" (huge support but cohesive ρ ≈ 0.7–0.9); TopK /
  BatchTopK / Gated are dilution-with-exclusion.
- **SpaDE does NOT have 1-to-1 features — it is the purest tiler.** Baseline
  distance-SpaDE is "shattering" on 7/8 concepts: support size 0 under the
  ≥10%-of-rows rule while ~8k of 8192 prototypes fire (each on <10% of
  positions — place-cell tiling), sparsemax making atoms mutually exclusive.
- **Lemma 1 confirmed for point-based SAEs.** Active landmarks form ONE
  kNN-connected component (no per-concept clusters), sit 1.9–5.6× the manifold
  scale off the data, are no closer to any concept class than to the joint
  corpus, and have near-maximal firing-class entropy (e.g. rule_category:
  median 1.81/1.95; ~0% of landmarks below 0.1). SpaDE prototypes tile the
  joint game-state manifold; individual concepts are not factored out.
- **Reconstruction quality ≠ concept-manifold preservation.** CosineSpaDE
  (VE 0.98) destroys the rule-category centroid curve (curve R² 0.28); the
  dense revived SpaDE-lam1e-03-gam1e-04 (VE 0.995, L0 ~1808) preserves it
  (0.91) and is the only SpaDE variant escaping shattering — mostly "mixed",
  the closest thing to a compact point-based representation in the set.
- Latent-vs-ambient PCA (latent_<concept>.png): the ambient t-manifold's two
  parity arcs survive in TopK's latent, collapse to a simplex edge in SpaDE's,
  and bend into a V in CosineSpaDE's — architecture dictates latent geometry.

## Figure → paper map

| figures/ | paper | reading it |
|---|---|---|
| manifold_pca_<concept> | Fig 2 | smooth value gradient = manifold structure exists |
| piecewise_<concept>[_c] | Fig 5 | SAE top-k recons approximate the manifold piecewise-linearly |
| latent_<concept> | (ours) | activation-space PCA vs each SAE's latent PCA, same concept |
| capture_<concept>[_c] | Fig 6 | R² plateau at ~k_M = compact capture; far beyond = dilution |
| tuning_<concept> | Fig 7 | localized bumps = tiling; one monotone curve = 1-to-1 |
| fourviews_<concept>_<space> | Fig 9L | four coordinate systems for the same manifold |
| similarity_<concept>_<space> | Fig 9R | block-diagonal = coherent feature group |
| receptive_<concept>_<space> | Fig 8 | how argmax features shatter the ambient space |
| landmarks_<space> | App C | Lemma 1 test + firing-class entropy (0 = 1-to-1, ln k = joint tiling) |
| regimes | Def 6 | capture / shattering / dilution per (SAE, concept) |
