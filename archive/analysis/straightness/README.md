# Straightness study: ReStraV's trajectory hypothesis on OthelloGPT

Does the perceptual-straightening result (ReStraV, arXiv:2507.00583; summary in
`docs/ReStraV_Summary.md`) transfer to OthelloGPT? There, natural videos trace
straighter trajectories in DINOv2 space than AI-generated ones. Here, the
"video" is a game's 59-move sequence, the "frame embedding" is the activation
at each position, and the classes drawn are: legal games (training distribution,
the baseline every comparison is scored against), OthelloGPT temp-1 rollouts,
games with one injected illegal move (no-flip placement at t0, then random legal
continuation), and real human championship games (WTHOR 2014-2023). A fifth
class of uniform random token sequences is built and stored but neither drawn nor
scored (see Design notes). Metrics are ReStraV's verbatim: step lengths d,
turning angles θ, and the 21-D feature vector
(`src/spade/straightness/geometry.py`).

## Pipeline

```
python -m spade.straightness.build_games       # 5 x 2000 games -> $DIR/games/
python -m spade.straightness.extract_traj      # W_E + resid_post_0..7 [N,59,512] -> $DIR/traj/
python -m spade.straightness.compute_geometry  # d/theta per space+variant -> $DIR/geometry/
python -m spade.straightness.classify          # AUROCs -> results/straightness/
python analysis/straightness/make_figures.py   # -> analysis/straightness/figures/
PYTHONPATH=src python analysis/straightness/l0_matched_profile.py   # legal-block d per L0 pick
```

`l0_matched_profile.py` is not part of this study's own chain. It writes `l0_<config>_legal.npz`
beside the geometry above, because `changepoint_localization` and `illegal_move_detector` score
against a per-move-index profile fit on these legal games and `compute_geometry.py` only ever
wrote one for the seed-42 roster. Legal block only -- that is all the profile reads.

`$DIR` = `$SPADE_STRAIGHTNESS_DIR` (default `$SPADE_DATA_ROOT/spade_straightness`).
One GPU job runs the whole chain:
`sbatch scripts/slurm/studies/straightness/submit_straightness.sh`.

## Design notes

- The random-token class stays in the data and is out of every figure. `CLASSES`
  in `build_games.py` fixes the block layout of `games.pt` and every geometry npz
  indexes into it by position, so dropping it there would silently re-slice every
  array. The figures take their class set from `CLASS_STYLE` instead, through
  `spade.figures.straightness_panels.class_blocks`, and that dict has no entry for
  it: nothing draws it, scores it, or fits on it. `classify.py` still writes its
  `legal_vs_random_tokens` rows into `auroc.csv` and nothing reads them. What the
  class supported is under *Retired* in `docs/claims_status.md`.
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

The hypothesis transfers with the **sign inverted**, and the two out-of-distribution
classes flip against each other with depth. AUROCs are mean-turning-angle scores
of the non-legal side (>0.5 = that side curvier), raw variant, n=2000/class
(championship = real WTHOR tournament games 2014-2023):

| pair | W_E | L0 | L5 | L7 | LR-21 @ L7 |
|---|---|---|---|---|---|
| legal vs violation | 0.52 | 0.56 | 0.40 | 0.37 | 0.84 |
| legal vs rollout | 0.49 | 0.49 | 0.50 | 0.51 | 0.50 |
| legal vs championship | 0.23 | 0.34 | 0.51 | 0.73 | 0.92 |

- Rule violations are indistinguishable from legal games through layer 3 — the
  0.52-0.56 they score there is the legal-vs-legal null's own reading (0.52-0.56;
  the empirical noise floor is not centered on 0.5 early, which is why the figure
  draws it as a band). They fall away from it at layer 4 (0.454, CIs still
  touching) and are clear of it from layer 5 (0.397 [0.372, 0.421] against a null
  of 0.504 [0.469, 0.541]), reaching 0.370 at L7, with step lengths following the
  angle (mean-d 0.37 at L5, 0.20 at L7). That is ReStraV's result with the sign
  inverted: here the out-of-distribution class is the straighter one, and only in
  the deep layers.
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
  but unusual board states. Late-layer straightening is therefore specific to
  rule violations; policy shift alone moves the geometry the other way.
- Bootstrap 95% CIs over games are ~±0.025 wide. They matter in one place: the
  championship effect at L5 (0.509 [0.483, 0.535]) is *not* distinguishable from
  chance — the human-game signal is genuinely a late-layer phenomenon (L7 0.730
  [0.707, 0.752]), not a gradual rise.
- SAE latents (layer 5): **every** family preserves the violation separation
  (CosineSpaDE 0.39, SpaDE γ=1e-4 0.42, PAnneal 0.41, GatedAnneal 0.43, TopK
  0.44, Standard 0.44, BatchTopK 0.45, GatedSAE 0.46, against raw L5's 0.40) and
  every one localizes the violation at t0±1. On championship games 7 of 8
  separate where raw L5 does not (0.23-0.54, `sae_profiles/README.md`).
  All results hold under per-t centering. The SpaDE γ=1e-2 baseline is excluded
  — see `sae_profiles/README.md` for why.

## Figures

| file | question |
|---|---|
| theta_dist_layer5.png | only games carrying an illegal move shift the layer-5 curvature distribution: the violation median sits 0.34° left of legal (mean-θ AUROC 0.40), while rollout and championship sit on top of it. Gaussian KDE per class; triangles mark class medians; raw + per-t centered. The curves carry no interval — the separation is scored with a bootstrap CI in auroc_vs_depth.png and sae_families_auroc.png |
| auroc_vs_depth.png | **headline.** where in depth does straightening emerge? mean-θ AUROC per class across W_E (input-space control) and layers 0-7, raw + per-t centered. Gray band = legal-vs-legal empirical null (the measured noise floor); per-class error bars are stratified bootstrap 95% CIs over games; >0.5 = that class is curvier than legal, <0.5 = straighter |
| curvature_vs_t.png | (a) curvature profile over the game per class, 95% CI error bars at the drawn markers &mdash; note all classes share a strong t-dependent "clock"; (b) violation-locked response: Δ turning angle vs. the matched source legal game, aligned at t0. Δ ≡ 0 before t0 by construction (shaded), troughs −4.3° at t0+1, decays with a mover-parity oscillation |
| step_vs_t.png | the displacement-magnitude counterpart of curvature_vs_t, raw layer 5: (a) step-length ‖Δz‖ profile per class — championship steps run ~5% above legal through the opening and cross below it at t≈36, while rollout and violation track legal; (b) violation-locked Δ step length, −1.73 at t0+1 with a monotone decaying tail (no parity oscillation, unlike the angle) |
| mean_vs_var_theta.png | the two OOD classes leave legal on different axes: violations shift the mean turning angle, championship games its variance (medians legal 142.0°/27.7, violation 141.7°/27.3, championship 142.1°/30.6). One point per game, large markers are class medians |
| sae_families_auroc.png | every SAE family keeps the violation separation (bars 0.39-0.46 against raw L5's 0.40, all below chance) and 7 of 8 separate championship games, which raw L5 does not. Rows are representation spaces, each panel one legal-vs-X comparison. Bars are mean-θ AUROC with bootstrap 95% CI, dots are LR-21; our methods outlined |
| norm_diagnostics.png | the classes share one layer-5 norm profile (means 23.35-23.46), so the angle result is not a magnitude artifact, and step lengths carry the same signs as the angles: (a) per-position norms per class; (b) step-length AUROC by depth, with the legal-vs-legal null band |

Two sibling studies extend these figures into the SAE latent spaces:
`sae_profiles/` (per-position profiles and violation-locked responses per family,
plus the curvature distribution across all nine spaces) and `feature_pca/` (the
21-D ReStraV descriptor viewed unsupervised, one PCA per space). `feature_pca/`
draws the illegal move as the three constraint rules rather than as this study's
violation class, which cannot be split that way — it carries no R1 — so those
three points per game come from the violation-divergence branch population
instead, under the same per-ply geometry.
