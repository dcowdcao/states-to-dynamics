# What the change-point score should be computed on

`quantity_ablation.py`. The study scores **step length** `d`. Two alternatives are obvious
enough that a reader will ask, and one of them is the study's own open worry:

| key | quantity | why it is a candidate |
|---|---|---|
| `d` | step length, ‖z_{t+1} − z_t‖ | the study's statistic, inherited from ReStraV |
| `norm` | the representation's own norm, latent norm in an SAE space | README: "treat this as a magnitude claim, not a geometry claim" |
| `theta` | turning angle, curvature proper | the quantity the descriptor is named for |

All three go through the identical `profile → drop → scan`, same *m* = 4, same paired MOSUM,
same corpus and arms, on **one matched band** — moves 14–46, 33 plies — so the chance level
and the scan window are held fixed and only the scored quantity changes. The three keys have
different widths and different move offsets (`d` 57 @ +2, `norm` 58 @ +1, `theta` 56 @ +2),
because the attention sink is dropped and an angle needs two consecutive displacements; the
bands undo that.

Both statistics are reported because they disagree about curvature: the **hit rate** is the
forced choice the study always had, the **per-position AUROC** is what the thresholded
read-out added.

## Result

Both suffix regimes, one matched band of 33 plies (moves 14-46). **Regenerated**: 870
violations + 1,000 held-out legal games, whole band scored. **Spliced**: 575 illegal branches
+ 1,725 legal arms, cut at *t0* for every arm alike because ground truth stops there.
Per-position AUROC at each quantity's better sign; intervals in the CSVs are 95% percentile
bootstrap over source games, 1,000 resamples, seed 42.

| | regenerated | | | spliced | | |
|---|---|---|---|---|---|---|
| GatedAnneal | 0.893 | 0.865 | 0.596 | 0.855 | 0.830 | 0.617 |
| GatedSAE | 0.855 | 0.833 | 0.570 | 0.825 | 0.803 | 0.586 |
| TopK | 0.832 | 0.768 | 0.576 | 0.806 | 0.746 | 0.598 |
| BatchTopK | 0.831 | 0.732 | 0.572 | 0.807 | 0.714 | 0.599 |
| raw L5 (ref.) | 0.761 | 0.544 | 0.577 | 0.744 | 0.511 | 0.595 |
| PAnneal | 0.680 | 0.630 | 0.567 | 0.678 | 0.611 | 0.587 |
| SpaDE | 0.583 | 0.691 | 0.527 | 0.600 | 0.651 | 0.553 |
| Standard | 0.576 | 0.637 | 0.547 | 0.589 | 0.611 | 0.563 |
| CosineSpaDE | 0.524 | 0.580 | 0.557 | 0.533 | 0.570 | 0.560 |
| **mean** | **0.726** | 0.698 | 0.566 | **0.715** | 0.672 | 0.584 |

Figures: `figures/quantity_ablation_regenerated.png`, `figures/quantity_ablation_spliced.png`.

**Every claim below holds in both regimes**, and the two orderings of `d` agree at Spearman
+0.983, so none of this is a property of one suffix convention.

## Three findings

**Step length wins where anything is detectable.** `d` is best on the six spaces that
detect at all, and on the reference space it is not close: 0.761 against 0.544 for the norm.
The README's worry that the norm-only baseline "may own the result" is **not** borne out
through this pipeline. `norm` wins only on Standard, SpaDE and CosineSpaDE — all at or near
chance on everything, so it is winning a race nobody finishes.

**The decisive argument is the sign, not the AUROC.** `d` keeps one direction in all nine
spaces: shorter step = illegal. The norm flips — smaller-is-illegal for the Gated/TopK
family, **larger**-is-illegal for raw, Standard, PAnneal and both SpaDE spaces. The method is
label-free, so a quantity whose sign moves across spaces cannot be used at all: you would
have to read the answer to choose the direction. That argument survives even if someone
disputes the numbers, which is why `sign` is a column.

**Curvature is weak, not dead — and `CURVATURE_ABLATION.md` overstates two things.** Every
θ is above 0.5 in both regimes (0.527–0.596 regenerated, 0.553–0.617 spliced) with a
consistent sign, so curvature carries a real signal. It reads as chance only on the argmax
hit rate (0.016–0.048 and 0.030–0.064, against a chance of 0.030), which is the forced choice
discarding every non-peak position. Two corrections follow:

- *"Curvature cannot localize at all"* is true of the hit rate and false per position.
- *"The space ordering roughly inverts, Spearman ρ ≈ −0.67"* is an artifact of the peak
  statistic. **Per position it is +0.783 in *both* regimes** — the same ordering as `d`. Do
  not repeat the inversion claim.

θ still cannot carry a dictionary comparison: its span across spaces is 0.068 (regenerated)
and 0.063 (spliced) against `d`'s 0.369 and the norm's 0.321. It separates illegal from legal a little, everywhere, and
separates dictionaries from each other essentially not at all.

## What this does not settle

**The SpaDE norms have almost no dynamic range.** Mean latent norm on the spliced tree is
0.036 (SpaDE) and 0.133 (CosineSpaDE) against 5.4-23.5 everywhere else -- the sparsemax code
lies on the simplex and sums to 1. Every quantity is standardized per move index before
scoring, so scale cancels and the comparison is still valid, but those two norm columns are
measuring a quantity with very little room to move.

**Substitution is not explanation.** This asks whether the norm can *replace* `d`. The
README's open question is whether it *explains* `d` — `d_t` regresses on
`(‖z_t‖, ‖z_{t+1}‖)` at R² 0.87–0.88, and partialling the norm out drops GatedAnneal from
0.587 to 0.173. Both can be true at once: the norm can fail as a standalone detector while
still accounting for most of `d`'s variance. That partial-correlation check is still unrun,
and it is the one that decides whether "geometry" is the right word for the study at all.

## At matched L0

The roster above is the seed-42 one, whose L0s span 32.5 to 1,808, so its **space ordering**
confounds the architecture with its sparsity — the confound the by-space figures in this
directory were redrawn for. `--l0-matched` re-runs everything over the eight
`l0_operating_point` picks, one checkpoint per architecture near L0 220. Figures:
`quantity_ablation_{regenerated,spliced}_l0_matched.png`.

Per-position AUROC, regenerated regime, ordered by `d`:

| space | `d` | `norm` | `theta` |
|---|---|---|---|
| GatedSAE | 0.848 | 0.809 | 0.570 |
| BatchTopK | 0.779 | 0.577 | 0.588 |
| TopK | 0.778 | 0.515 | 0.578 |
| raw L5 (ref.) | 0.761 | 0.544 | 0.577 |
| GatedAnneal | 0.758 | 0.510 | 0.585 |
| PAnneal | 0.717 | 0.558 | 0.570 |
| SpaDE | 0.662 | 0.531 | 0.517 |
| Standard | 0.645 | 0.556 | 0.524 |
| CosineSpaDE | 0.528 | 0.619 | 0.506 |
| **mean** | **0.719** | 0.580 | 0.557 |

**The ablation's three findings survive.** `d` beats `norm` in 8 of 9 spaces and `theta` in
all 9 (8 of 9 spliced); the means barely move (0.726 → 0.719 for `d`). `norm` still flips
sign, 5 spaces positive against 4 negative, which is the argument that rules it out. And
curvature is still weak-but-real and still *not* inverted: ρ(`d`, `theta`) is +0.767
regenerated and +0.733 spliced, the same sign as on the seed-42 roster. That is what a
within-space comparison should do — three bars in one group off one corpus cannot be moved by
seating the architectures at a common L0.

**The space ordering does not survive, exactly as expected.** GatedAnneal was first at 0.893
and is now fifth at 0.758, *below* raw layer 5 at 0.761 — its lead was its sparsity, not its
architecture, which is the same reversal `36f452c` found for the by-space bars. GatedSAE takes
the top. Read the seed-42 figure for the quantity comparison and this one for anything about
which dictionary is better.

**One thing gets worse, and it is worth saying.** On the seed-42 roster `d` keeps one
direction in all nine spaces. Here it flips positive on CosineSpaDE — the one space where `d`
is at chance (0.528 regenerated, 0.514 spliced), so the direction is being fit on noise rather
than on a signal that reversed. The sign argument against `norm` is unaffected: `norm` flips
between spaces that *do* detect.

## Reproducing

```
# once, on a GPU: per-position norms for the spliced tree, written beside the existing npz
sbatch scripts/slurm/studies/violation_divergence/submit_branch_norms.sbatch

PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py --variant regenerated
PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py --variant spliced

# the L0-matched roster needs two more GPU steps, then the same two runs with --l0-matched
sbatch scripts/slurm/studies/violation_divergence/submit_branch_norms.sbatch --l0-matched
sbatch scripts/slurm/studies/straightness/submit_l0_matched_profile.sbatch
```

The ablation itself is CPU-only. `branch_norms.py` is the one GPU step, and it is
non-destructive: it writes `<space>_norm.npz` beside `<space>.npz` rather than rewriting it,
because re-running `branch_geometry.py` would recompute `d` and `theta` through an encode that
is not bitwise reproducible and every committed number here is byte-identical against the
current files. Its rows come from the committed `meta.npz`'s `row` column, never re-derived --
`extend_branches` grew the tree after that npz was written, so re-deriving now yields 2,207
games where the npz holds 575.

`--l0-matched` needs two artifacts the picks did not have: `l0_<config>_norm.npz` on the
branch tree (`branch_norms.py --l0-matched`) and `l0_<config>_raw.npz` on the straightness
corpus (`analysis/straightness/l0_matched_profile.py`, all five classes so the row layout
matches the canonical file — the regenerated regime scores the *violation* block, which the
profile-only `_legal.npz` does not carry). Neither rewrites a file already on disk.

This supersedes `theta_ablation.py`, which measured only the argmax half of this.
