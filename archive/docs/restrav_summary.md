# ReStraV: AI-Generated Video Detection via Perceptual Straightening

Internò, Geirhos, Olhofer, Liu, Hammer, Klindt — NeurIPS 2025, arXiv:2507.00583.
Code: github.com/ChristianInterno/ReStraV. Read for the straightness study
(`spade.straightness`), which transfers the hypothesis to OthelloGPT.

## Hypothesis

Neuroscience's "perceptual straightening" (Hénaff et al. 2019/2021): the visual
system transforms natural video so temporal trajectories become *straighter* in
neural representation space — straight paths are linearly extrapolatable, which
predictive coding favors. ReStraV's transfer: DINOv2, trained on natural data,
straightens natural videos too, so AI-generated videos — with their subtle
temporal inconsistencies — trace *curvier, more irregular* trajectories in its
representation space. In pixel space the two classes overlap; in representation
space they separate.

## Method

1. Sample T=24 frames over a 2 s window; resize 224×224; divide by 255 (notably
   no ImageNet mean/std normalization).
2. Embed each frame with DINOv2 ViT-S/14, final block: CLS + all 196 patch
   tokens concatenated → z_i ∈ R^75648. Trajectory Z = (z_1..z_T).
3. Geometry: displacements Δz_i = z_{i+1} − z_i; step length d_i = ‖Δz_i‖₂;
   turning angle θ_i = arccos(cos_sim(Δz_i, Δz_{i+1})) in degrees.
4. 21-D feature vector: first-7 d, first-6 θ, and {mean, min, max, var} of each
   full sequence. Z-scored → 2-layer MLP (64→32), BCE. (RF/GB/LR nearly as good.)

## Key results

- 94.17% acc / 98.63% AUROC internal test (VidProM); 97.17% avg across the four
  VidProM generators.
- Generalization vs a trained VideoSwin detector: unseen generators 89.45% vs
  62.44%; future model (Sora) 80.05% vs 60.70%.
- Physics-IQ matched-pair 2AFC using *only mean curvature μ_θ*, zero training:
  Pika 97.5%, Runway 94.9%, Sora 99.0%. The raw curvature scalar carries almost
  the whole signal; the classifier is calibration.
- Encoder ablation: DINOv2 has the largest natural-vs-AI mean-curvature gap
  among 8 encoders (AlexNet → CLIP); pixel-space trajectories don't separate.
- Signature nuance: natural videos have *lower mean* curvature but *higher
  variance* (mostly straight, punctuated by sharp transitions); AI videos are
  uniformly jittery.

## What the paper does NOT report (checked against the full text)

- **No activation norms.** Only the displacement magnitude d_i = ‖z_{i+1} − z_i‖
  is ever used; ‖z_i‖ itself never appears. Our norm analyses
  (`spade.diagnostics.norm_diagnostics`, the per-SAE `norm_vs_t_*` figures) have
  no counterpart here — they are an addition, and cannot be checked against
  ReStraV.
- **No direction for step length.** Sec. 5 says only "Both distance- and
  curvature-based features provide discriminative signal"; whether d is higher
  or lower for AI videos is never stated. Only curvature gets a direction.
- **No layer sweep.** A single fixed layer (DINOv2 block 11). The ablation axis
  is *encoder*, not depth — so the depth flip we find (curvier early, straighter
  from layer 4) is outside what their protocol could have seen.

## Limitations they state

Goodhart (generators can optimize against DINOv2 curvature); future generators
may converge to natural statistics; curvature is time-reversal invariant, so
arrow-of-time violations are invisible; encoder bias.

## Transfer to OthelloGPT (this repo's straightness study)

| ReStraV | Othello analog |
|---|---|
| video | one game's 59-token move sequence |
| frame embedding (DINOv2) | activation at position t (resid_post, layer ℓ) |
| natural video | legal game from the training distribution |
| AI-generated video | OOD game: injected rule violation; also OthelloGPT's own temp-1 rollouts and real human championship play |
| pixel-space control | token-embedding-space (`W_E`) trajectories |
| per-encoder comparison | per-layer sweep + SAE latent spaces (SpaDE/CosineSpaDE vs baselines) |

Hypothesis under test: legal games trace straighter layer-ℓ trajectories than
OOD games, mean turning angle alone separates the classes, and the separation
emerges with depth (absent in `W_E`, present in late layers) — plus whether SAE
latents preserve or amplify it.

**What the transfer found, and what was retracted.** ReStraV transfers with the
sign inverted and a depth flip; violations are graded and localized to t0±1;
temperature-1 rollouts are at chance everywhere; championship play dissociates
rule-OOD from policy-OOD. Two claims did not survive and must not be restated:
"SpaDE destroys the curvature signal that every other family keeps" (every
family preserves the violation separation, 0.39–0.46 against raw layer 5's 0.40)
and the layer-5 championship effect (0.509 [0.483, 0.535], not distinguishable
from chance). A third set of claims was retired with the random-token class,
which the study no longer draws or scores. See `docs/claims_status.md`.

## Related: tpr_othello (arXiv:2605.09967, Lee/Viégas/Wattenberg)

Factorizes OthelloGPT's 192 linear board-probe directions as a tensor product
representation (square ⊗ color via a learned binding map; 99% acc at 57.5% of
linear-probe params; causally valid under interventions). Its "OOD/random"
baselines are entirely *probe-side* — TPR probes trained on i.i.d.-random board
labels or on random square-color encodings — used to show board-aligned probe
geometry comes from OthelloGPT's activations (gap-geometry R² 0.54 vs 0.03 OOD).
It never feeds random/illegal move sequences to the model, so the model-side
OOD-trajectory question this study asks is untouched there. Useful to us mainly
as evidence that OthelloGPT's representation space is highly structured, and as
a second OthelloGPT codebase (synthetic-trained Li et al. 8L/512d, like ours).
It is vendored at `external/tpr_othello` (pinned `cc90cea`) and is the decoder
behind the bracket-flip study.
