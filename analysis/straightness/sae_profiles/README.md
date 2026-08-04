# Trajectory profiles per SAE latent space

The straightness study's per-position figures, computed in each of the 8 seed-42
SAE latent spaces instead of the raw layer-5 residual stream: curvature, step
length, and latent norm. Each is a per-class profile over the game plus a
violation-locked response against each violation game's matched source legal
game, aligned at the injected move t0.

Family labels carry the sparsity setting (`src/spade/figures/paper_style.sae_label`).

**Dropped (2026-07-28, and now repo-wide):** the SpaDE γ=1e-2 baseline
(`SpaDETrainer-othello`). Its recorded Karvonen eval is degenerate — L0 15.4,
frac_recovered 0.597, VE 0.352 — where seeds 43–46 of the same config give
L0 128, frac_recovered 0.996, VE 0.973, and that bad row is what gives the
published SpaDE aggregate its VE 0.849 ± 0.278. The checkpoint's provenance is
independently suspect — the `ae.pt` in `checkpoints/main/trainer_5` does not
reproduce that degenerate row at all, rerunning at L0 127.76 / VE 0.973
(MIGRATION.md open question 15) — so it is excluded rather than carrying a result.
This study made the call first and enforced it locally; the exclusion is now a
global decision, held in one place as `spade.spaces.EXCLUDED_SAE` and read from
there by every study and by the viewer. Registry membership is untouched. SpaDE
is still represented by the γ=1e-4 point.

Angles are scale-free, so latent spaces are comparable to each other and to the
raw residual stream even though their step lengths are not (SpaDE codes live on
a simplex; see `analysis/straightness/README.md`).

```
python analysis/straightness/sae_profiles/make_figures.py
```

Reads the geometry npz already written by
`src/spade/straightness/compute_geometry.py` — no re-extraction. Panels come from
`src/spade/figures/straightness_panels.py`, shared with
`analysis/straightness/make_figures.py` so the two cannot drift.

## Result: the whole-game AUROC reads the net response, not its magnitude

Every space localizes the violation, and none inverts — but how much of the
local response survives a whole-game average varies a lot, and that is what the
AUROC actually tracks.

| space | trough | net Δ over t ≥ t0 | mean \|Δ\| | sign cancellation | violation AUROC |
|---|---|---|---|---|---|
| raw L5 | −4.26° @ t0+1 | −0.864 | 0.873 | 1% | 0.397 |
| GatedAnneal | −2.41° @ t0+1 | −0.704 | 0.719 | 2% | 0.429 |
| PAnneal | −3.67° @ t0+1 | −0.745 | 0.772 | 3% | 0.414 |
| BatchTopK | −2.39° @ t0+1 | −0.531 | 0.565 | 6% | 0.448 |
| TopK | −2.44° @ t0+1 | −0.508 | 0.558 | 9% | 0.443 |
| GatedSAE | −1.88° @ t0+1 | −0.440 | 0.485 | 9% | 0.456 |
| CosineSpaDE | −4.25° @ t0−1 | −1.044 | 1.194 | 13% | 0.392 |
| Standard | −3.16° @ t0+1 | −0.493 | 0.652 | 25% | 0.440 |
| SpaDE (γ=1e-4) | −4.03° @ t0+1 | −0.461 | 0.761 | 40% | 0.419 |

- **Every dictionary localizes the violation** at t0±1 and every net is negative
  — no space inverts the effect.
- **Net, not magnitude, drives the AUROC.** Across the 9 spaces (8 SAE + raw) net post-t0 Δ
  correlates with (AUROC − 0.5) at r = 0.865, while mean |Δ| alone does not
  order them. SpaDE γ=1e-4 is the clearest case: its local response is large
  (mean |Δ| 0.761, trough −4.03°, second deepest of any space) but 40% of it
  cancels across offsets, leaving a net of −0.461 and a middling AUROC.
- **CosineSpaDE peaks one ply earlier** (t0−1 rather than t0+1) and carries the
  largest net response of any space, raw included. Its profile panel is also
  visibly non-smooth (sharp dips near t=3 and t=25) where every other space
  traces a smooth arc — worth a look if the sparsemax code turnover matters.
- Practical consequence: a whole-game summary can understate a dictionary that
  preserves the event but scrambles its sign structure. Read the event-locked
  panel before concluding a latent space has lost a signal.

## Result: the trajectory metrics agree across dictionaries, the point metric does not

Three per-position quantities, and they are not equally robust. `d = ‖Δz‖` and
the turning angle are properties of the *path*; `‖z‖` is a property of the
*point* and is only here as a confound diagnostic — ReStraV never uses it (see
`docs/ReStraV_Summary.md`). Random/legal ratios per family:

| family | step ‖Δz‖ | norm ‖z‖ |
|---|---|---|
| Standard | 0.884 | 1.252 |
| PAnneal | 0.843 | 1.211 |
| SpaDE-lam1e-03-gam1e-04 | 0.818 | 1.153 |
| CosineSpaDE | 0.814 | 1.143 |
| TopK | 0.788 | 1.033 |
| GatedSAE | 0.780 | 1.029 |
| BatchTopK | 0.763 | 1.089 |
| GatedAnneal | 0.742 | 0.993 |

**Step length is unanimous**: every dictionary, without exception, gives
random-token games shorter steps (0.74–0.88×), reproducing the raw-space result.
**Norm is not**: it ranges 0.99–1.25 with no consistent direction, and splits
into three mechanistically different regimes (below). So the trajectory-geometry
metrics transfer through every dictionary, while the magnitude diagnostic is
dictionary-dependent — a reason to lead with d and θ and treat ‖z‖ as the
supporting check it is.

One refinement to the cancellation result above: **the sign structure that
cancels is specific to the turning angle.** Every dictionary's step-length
violation response is a sustained negative decaying monotonically, with none of
the offset-to-offset sign flips the angle shows. What varies across dictionaries
is angular direction, not displacement magnitude.

## Result: each dictionary re-expresses the norm elevation differently

Layer-5 residual norms are 1.141× higher on random-token games than on legal ones
(see `analysis/straightness/figures/norm_diagnostics.png`). The `norm_vs_t_*`
figures ask what each dictionary does with that. Ratios are random/legal, averaged
over all positions; L0 is the per-position active-code count, also stored in the
npz:

| family | norm r/l | L0 r/l | mean L0 | what it does |
|---|---|---|---|---|
| Standard | 1.252 | 1.220 | 1163 | passes through: more codes fire, norm rises to match |
| PAnneal | 1.211 | 1.223 | 361 | same |
| SpaDE-lam1e-03-gam1e-04 | 1.153 | 0.830 | 1776 | *fewer* codes, each larger |
| CosineSpaDE | 1.143 | 0.910 | 172 | *fewer* codes, each larger |
| BatchTopK | 1.089 | 1.730 | 88 | many more codes, each much weaker |
| TopK | 1.033 | 1.000 | 80 | hard-constrained: L0 cannot move, norm barely does |
| GatedSAE | 1.029 | 1.672 | 58 | many more codes, each much weaker |
| GatedAnneal | 0.993 | 1.437 | 35 | elevation eliminated |
| raw L5 | 1.141 | — | — | the residual-stream reference |

Three regimes, and they do not line up with the usual architecture families:

- **Pass-through (Standard, PAnneal).** Norm and L0 rise together by the same
  ~22%, so the elevation is entirely "more features fire." These are the two
  least sparsity-constrained dictionaries.
- **Suppressed (GatedAnneal, GatedSAE, TopK, BatchTopK).** Norm elevation is
  gone or nearly so. For TopK the reason is mechanical — L0 ratio is exactly
  1.000 because k is fixed. For the Gated families and BatchTopK the mechanism
  is different and more interesting: L0 goes *up* a lot (1.4–1.7×) while the
  norm stays flat, so many more codes fire but each carries a much smaller
  coefficient.
- **Inverted (CosineSpaDE, SpaDE-lam1e-03).** L0 goes *down* (0.83–0.91×) while
  the norm goes *up*. Off-distribution input makes these dictionaries commit to
  fewer, larger codes — the opposite sparsity response to every other family,
  and presumably the simplex geometry.

So "activation norm is higher on random tokens" survives into latent space for
some dictionaries and is erased by others, and knowing the norm ratio alone does
not tell you which mechanism produced it — L0 is needed to separate "more codes"
from "bigger codes." Worth remembering when a norm-based metric is used to
compare SAEs.

## Result: the two path quantities agree everywhere, the norm does not

`quantity_comparison.png` puts curvature, step length and latent norm on one
axis per space, each as a percentage of that space's own mean level. Peak Δ and
the offset it occurs at:

| space | θ | ‖Δz‖ | ‖z‖ |
|---|---|---|---|
| raw L5 | −3.00% @ +1 | −5.95% @ +1 | −3.78% @ 0 |
| BatchTopK | −1.67% @ +1 | −8.11% @ +1 | −11.73% @ 0 |
| GatedAnneal | −1.71% @ +1 | −11.72% @ +1 | −14.91% @ 0 |
| GatedSAE | −1.34% @ +1 | −10.01% @ +1 | −15.12% @ 0 |
| TopK | −1.71% @ +1 | −8.23% @ +1 | −10.99% @ 0 |
| PAnneal | −2.62% @ +1 | −4.73% @ +1 | **+5.76%** @ +1 |
| Standard | −2.31% @ +1 | −2.82% @ +1 | **+7.14%** @ +1 |
| CosineSpaDE | −3.21% @ −1 | −2.07% @ +1 | **+3.93%** @ +1 |
| SpaDE (γ=1e-4) | −2.99% @ +1 | −1.95% @ +1 | **+3.92%** @ +1 |

The two *path* quantities behave identically everywhere: an illegal move makes
the next step shorter and the next turn straighter, in all nine spaces, peaking
at t0+1 (CosineSpaDE's angle peaks a ply earlier, as it does in every figure).

The *point* quantity does not. The norm falls with them in raw layer 5 and the
four hard-sparsity dictionaries, but **rises** in Standard, PAnneal and both
sparsemax spaces — the same split the random-token table above shows, where
those four are the "pass-through" and "inverted" regimes and the others are
"suppressed". So the sign of a norm response is a property of the dictionary,
not of the event, and a norm-based OOD score inherits it. Note also that a
single illegal move *lowers* the raw layer-5 norm (−3.78%) while a fully random
game *raises* it (+14% overall): local violation and global distribution shift
are not the same phenomenon for ‖z‖.

## Result: SAE latents expose the human-game shift that raw layer 5's angle misses

`theta_dist_comparison.png` plots the whole-game distribution the AUROCs
summarize, and it turns up something the per-position figures do not. In raw
layer-5 activations, championship (real WTHOR) games are indistinguishable from
legal games by mean turning angle — 0.509 [0.483, 0.535]. Encode the same
activations and 7 of 8 dictionaries separate them (mean-θ AUROC, raw variant,
bootstrap 95% CI, n=2000/class):

| space | championship AUROC | direction | survives per-t centering |
|---|---|---|---|
| Standard | 0.230 [0.208, 0.250] | straighter | yes (0.235) |
| CosineSpaDE | 0.282 [0.259, 0.304] | straighter | yes (0.280) |
| SpaDE (γ=1e-4) | 0.293 [0.269, 0.316] | straighter | yes (0.312) |
| GatedSAE | 0.413 [0.388, 0.437] | straighter | weakens (0.462) |
| GatedAnneal | 0.418 [0.392, 0.443] | straighter | weakens (0.455) |
| TopK | 0.465 [0.441, 0.491] | straighter | no (0.511) |
| PAnneal | 0.539 [0.514, 0.564] | *curvier* | yes (0.547) |
| BatchTopK | 0.504 [0.479, 0.531] | — | flips to curvier (0.543) |
| raw L5 | 0.509 [0.483, 0.535] | — | flips to curvier (0.546) |

Read it as a dictionary property, not a fact about the games: the same
activations carry the same information, and what changes is whether a whole-game
mean angle can see it. The three dictionaries that survive centering are the
strongest, and their direction is consistent (human games trace *straighter*
latent trajectories than synthetic legal ones), while the weak ones are position
clock effects that centering removes.

Two caveats before this goes in a writeup. Step length separates championship in
most spaces too (mean-d AUROC 0.55–0.85, and 0.24 for Standard), so this is not
an angle-only effect — `norm_diagnostics.png`'s logic applies. And raw layer 5's
own centered variant does separate championship (0.546), in the *opposite*
direction from the SAE latents. Seed 42 only.

## Files

| file | contents |
|---|---|
| `curvature_vs_t_<Family>.png` | per-family, two panels: (a) curvature profile over the game, (b) violation-locked Δ turning angle with the peak labeled |
| `step_vs_t_<Family>.png` | per-family, two panels: (a) step-length ‖Δz‖ profile over the game, (b) violation-locked Δ step length. The displacement-magnitude counterpart of curvature, and the quantity ReStraV pairs with the turning angle |
| `norm_vs_t_<Family>.png` | per-family, two panels: (a) latent-norm ‖z‖ profile over the game, (b) violation-locked Δ latent norm. A confound diagnostic, not a trajectory metric |
| `violation_spike_comparison.png` | small multiples of the curvature panel (b) across all spaces on a shared y-axis, raw layer 5 ghosted in gray, each titled with its trough and net |
| `theta_dist_comparison.png` | small multiples of the whole-game curvature distribution across all spaces on shared axes: per-class KDE with a bootstrap 95% band, class medians on the baseline, each titled with its random−legal and championship−legal median gap. The distribution behind `analysis/straightness/figures/sae_families_auroc.png` |

| `quantity_comparison.png` | all three quantities at once, small multiples across all spaces on a shared axis. Each Δ is a percentage of that space's own mean level, which is what makes the nine dictionaries commensurable |

Norms are only comparable across dictionaries once normalized — the raw latent
norms differ by two orders of magnitude (SpaDE ~0.13, TopK ~12.9), which is why
`quantity_comparison.png` plots percentages and there is no absolute-scale norm
comparison.

## Caveats

- The prefix Δ ≡ 0 identity is asserted per family in `make_figures.py`; it held
  exactly in all 8 latent spaces, which is the alignment check.
- Absolute mean angles differ across families (CosineSpaDE ~132°, SpaDE ~135°,
  BatchTopK ~142°) because sparsity and geometry differ. Compare Δ, not levels.
- These are single-seed (42) checkpoints. The cancellation result would be worth
  confirming on seeds 43–46, which exist on netscratch for the sweep configs.
