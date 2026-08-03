# Straightness study: ReStraV's trajectory hypothesis on OthelloGPT

Does the perceptual-straightening result (ReStraV, arXiv:2507.00583; summary in
`docs/ReStraV_Summary.md`) transfer to OthelloGPT? There, natural videos trace
straighter trajectories in DINOv2 space than AI-generated ones. Here, the
"video" is a game's 59-move sequence, the "frame embedding" is the activation
at each position, and the classes are: legal games (training distribution, the
baseline every comparison is scored against), OthelloGPT temp-1 rollouts, games
with one injected illegal move (no-flip placement at t0, then random legal
continuation), uniform random token sequences, and real human championship games
(WTHOR 2014-2023). Metrics are ReStraV's verbatim: step lengths d, turning
angles θ, and the 21-D feature vector (`src/spade/straightness/geometry.py`).

## Pipeline

```
python -m spade.straightness.build_games       # 5 x 2000 games -> $DIR/games/
python -m spade.straightness.extract_traj      # W_E + resid_post_0..7 [N,59,512] -> $DIR/traj/
python -m spade.straightness.compute_geometry  # d/theta per space+variant -> $DIR/geometry/
python -m spade.straightness.classify          # AUROCs -> results/straightness/
python analysis/straightness/make_figures.py   # -> analysis/straightness/figures/
```

`$DIR` = `$SPADE_STRAIGHTNESS_DIR` (default `/n/netscratch/ba_lab/Lab/$USER/spade_straightness`).
One GPU job runs the whole chain:
`sbatch scripts/slurm/studies/straightness/submit_straightness.sh`.

## Design notes

- Position 0 is an attention sink (norm ~300 vs ~22) and is dropped; trajectories
  cover positions 1..58.
- Every space ships raw + centered (minus the mean legal train-split trajectory)
  because move index t dominates activation geometry; a θ signal that vanishes
  under centering and is matched by mean-d alone is a norm artifact
  (norm_diagnostics.png is the check).
- SAE spaces encode the layer-5 trajectory with 8 seed-42 families (the SpaDE
  γ=1e-2 baseline is excluded repo-wide by `spade.spaces.EXCLUDED_SAE`, see
  sae_profiles/README.md); step
  lengths are never compared across families (SpaDE codes live on a simplex);
  zero-length steps give NaN angles, moments are nan-aware.
- Violation games reuse legal-class games as prefixes (src_game), so the
  curvature spike is measured against the matched source game
  (curvature_vs_t.png, right panel).
- Built-in checks (fail loudly): legal-vs-legal null AUROC in [0.42, 0.58] per
  space, sink-norm ratio, prefix invariance of violation activations before t0,
  spike localized within +-2 of t0, θ/d range and NaN-fraction bounds, SAE L0
  plausibility.

## Results (n=2000/class, seed 42)

The hypothesis transfers with the **sign inverted and a depth flip**. AUROCs are
mean-turning-angle scores of the non-legal side (>0.5 = that side curvier), raw
variant, n=2000/class (championship = real WTHOR tournament games 2014-2023):

| pair | W_E | L0 | L5 | L7 | LR-21 @ L7 |
|---|---|---|---|---|---|
| legal vs random_tokens | 0.81 | 1.00 | 0.01 | 0.02 | 1.00 |
| legal vs violation | 0.52 | 0.56 | 0.40 | 0.37 | 0.84 |
| legal vs rollout | 0.49 | 0.49 | 0.50 | 0.51 | 0.50 |
| legal vs championship | 0.23 | 0.34 | 0.51 | 0.73 | 0.92 |

- Early layers see OOD input as *curvier*; layers 4-7 flip: OOD trajectories
  become much *straighter* with near-zero step lengths — the deep layers only do
  large, direction-changing computation on in-distribution games and collapse to
  small straight steps (toward an elevated-norm state) on garbage. ReStraV's
  natural-video signature (low mean / high variance) is likewise inverted.
- The violation effect is exactly localized: Δθ against the matched source game
  is 0.00 at every position before t0 and peaks in magnitude at t0±1 (positive
  at L0, negative at L5/L7), decaying slowly with a mover-parity oscillation.
- Temp-1 rollouts are indistinguishable from real games in every space — unlike
  video generators, OthelloGPT has essentially converged to its training
  distribution.
- Championship (real human WTHOR) games dissociate the two kinds of OOD: they
  are *straighter* than synthetic games early (stereotyped strategic move
  statistics; W_E step lengths much smaller, d-AUROC 0.02) but *curvier with
  larger steps* at L6-L7 (θ 0.73, d 0.72) — deep layers work harder on legal
  but unusual board states instead of collapsing. The straightness collapse is
  therefore specific to rule violations, not distribution shift per se.
- Bootstrap 95% CIs over games are ~±0.025 wide. They matter in one place: the
  championship effect at L5 (0.509 [0.483, 0.535]) is *not* distinguishable from
  chance — the human-game signal is genuinely a late-layer phenomenon (L7 0.730
  [0.707, 0.752]), not a gradual rise.
- SAE latents (layer 5): **every** family preserves the random-token separation
  (PAnneal 0.01, BatchTopK 0.04, TopK 0.05, GatedSAE 0.11, GatedAnneal 0.14,
  CosineSpaDE 0.16, SpaDE γ=1e-4 0.20, Standard 0.21), and every one localizes
  the violation at t0±1. On violations CosineSpaDE (0.39) matches raw L5.
  All results hold under per-t centering. The SpaDE γ=1e-2 baseline is excluded
  — see `sae_profiles/README.md` for why.

## Figures

| file | question |
|---|---|
| theta_dist_layer5.png | only rule-breaking games shift the layer-5 curvature distribution: random tokens sit ~4° left of legal with little overlap, while rollout and championship sit on top of it. Gaussian KDE per class with a bootstrap 95% band; triangles mark class medians; raw + per-t centered |
| auroc_vs_depth.png | **headline.** where in depth does straightening emerge? mean-θ AUROC per class across W_E (input-space control) and layers 0-7, raw + per-t centered. Gray band = legal-vs-legal empirical null (the measured noise floor); shaded bands are stratified bootstrap 95% CIs over games; >0.5 = that class is curvier than legal, <0.5 = straighter |
| curvature_vs_t.png | (a) curvature profile over the game per class, 95% CI bands &mdash; note all classes share a strong t-dependent "clock"; (b) violation-locked response: Δ turning angle vs. the matched source legal game, aligned at t0. Δ ≡ 0 before t0 by construction (shaded), troughs −4.3° at t0+1, decays with a mover-parity oscillation |
| step_vs_t.png | the displacement-magnitude counterpart of curvature_vs_t, raw layer 5: (a) step-length ‖Δz‖ profile per class — random tokens sit ~20% below every other class across the whole game, and championship slightly *above* legal; (b) violation-locked Δ step length, −1.73 at t0+1 with a monotone decaying tail (no parity oscillation, unlike the angle) |
| mean_vs_var_theta.png | ReStraV's signature is inverted here — the in-distribution class has HIGH mean and LOW variance (legal 142.0°/28.1 vs random 138.1°/35.8). One point per game, large markers are class medians |
| sae_families_auroc.png | every SAE family preserves the curvature separation: on random tokens the bars run 0.01 (PAnneal) to 0.21 (Standard), all far below chance, and every family reaches LR-21 = 1.00. Rows are representation spaces, each panel one legal-vs-X comparison. Bars are mean-θ AUROC with bootstrap 95% CI, dots are LR-21; our methods outlined |
| norm_diagnostics.png | step lengths corroborate the angle result with the same signs, so the effect is not a pure magnitude artifact: (a) random-token norms swell to 26.7 while other classes sit flat at ~23.4; (b) step-length AUROC by depth, with the legal-vs-legal null band |

Two sibling studies extend these figures into the SAE latent spaces:
`sae_profiles/` (per-position profiles and violation-locked responses per family,
plus the curvature distribution across all nine spaces) and `feature_pca/` (the
21-D ReStraV descriptor viewed unsupervised, one PCA per space).
