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
read-out added. The figures plot the AUROC only; the hit rate is a column in every CSV, and
the numbers this document quotes from it are read there.

## Result

The spliced branch tree, one matched band of 33 plies (moves 14-46): 575 illegal branches +
1,725 legal arms, cut at *t0* for every arm alike because ground truth stops there. The suffix
after *t0* is the source game's own, never regenerated. Per-position AUROC at each quantity's
better sign; intervals in the CSV are 95% percentile bootstrap over source games, 1,000
resamples, seed 42.

| space | `d` | `norm` | `theta` |
|---|---|---|---|
| GatedAnneal | 0.855 | 0.830 | 0.617 |
| GatedSAE | 0.825 | 0.803 | 0.586 |
| BatchTopK | 0.807 | 0.714 | 0.599 |
| TopK | 0.806 | 0.746 | 0.598 |
| raw L5 (ref.) | 0.744 | 0.511 | 0.595 |
| PAnneal | 0.678 | 0.611 | 0.587 |
| SpaDE | 0.600 | 0.651 | 0.553 |
| Standard | 0.589 | 0.611 | 0.563 |
| CosineSpaDE | 0.533 | 0.570 | 0.560 |
| **mean** | **0.715** | 0.672 | 0.584 |

Figure: `figures/quantity_ablation_spliced.png`. One panel, no title: the bars are the AUROC
column of this table, spaces ordered by `d`. This document is its caption.

`d`'s column here is `threshold_localization.csv` exactly, all nine spaces — this is the
parent study's own per-position statistic, read three ways.

## Three findings

**Step length wins where anything is detectable.** `d` is best on the six spaces that
detect at all, and on the reference space it is not close: 0.744 against 0.511 for the norm.
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
θ is above 0.5 (0.553–0.617) with a consistent sign, so curvature carries a real signal. It
reads as chance only on the argmax hit rate (0.030–0.064, against a chance of 0.030), which is
the forced choice discarding every non-peak position. Two corrections follow:

- *"Curvature cannot localize at all"* is true of the hit rate and false per position.
- *"The space ordering roughly inverts, Spearman ρ ≈ −0.67"* is an artifact of the peak
  statistic. **Per position it is +0.783** — the same ordering as `d`. Do not repeat the
  inversion claim.

θ still cannot carry a dictionary comparison: its span across spaces is 0.063 against `d`'s
0.321 and the norm's 0.319. It separates illegal from legal a little, everywhere, and
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

## At matched L0, on the detector's corpus

The roster above is the seed-42 one, whose L0s span 32.5 to 1,808, so its **space ordering**
confounds the architecture with its sparsity. `--l0-matched` re-runs the ablation over the
eight `l0_operating_point` picks — and on a different corpus and statistic from the table
above, on purpose.

**It is `illegal_move_detector`'s figure with two more quantities on it.** Same 500
t0-stratified games (`illegal_move_detector/results/eval_games.csv`), same at-t0 paired
statistic: the pooled illegal branch against the SAME source game's legal control, ranked on
the score at that game's own t0. `d` is therefore not merely close to panel (a) of
`sae_detector_overview.png` — it **is** that bar, and `assert_matches_detector` fails the run
if the two ever drift. Figure: `quantity_ablation_l0_matched.{png,pdf}`, the PDF being the
camera-ready twin.

| space | `d` | `norm` | `theta` |
|---|---|---|---|
| GatedSAE | 0.723 | 0.595 | 0.556 |
| BatchTopK | 0.710 | 0.553 | 0.569 |
| TopK | 0.708 | 0.554 | 0.566 |
| GatedAnneal | 0.703 | 0.501 | 0.564 |
| raw L5 (ref.) | 0.681 | 0.559 | 0.548 |
| PAnneal | 0.677 | 0.546 | 0.558 |
| SpaDE | 0.634 | 0.521 | 0.503 |
| Standard | 0.612 | 0.533 | 0.530 |
| CosineSpaDE | 0.498 | 0.572 | 0.501 |

**The ablation's case is stronger here than on the per-position table.** `d` beats both
challengers in 8 of 9 spaces, and the exception is the one space where `d` is at chance
(CosineSpaDE, 0.498). `norm` clears 0.56 only twice; `theta` never clears 0.57. The sign
argument is unchanged and now starker: `norm`'s better direction is *positive* in 6 of 9
spaces and negative in 3, so no fixed direction serves it.

**Two asymmetries a reader is owed.** `d` is held at the direction this study fixed for it —
shorter is illegal — because that is what makes the bar the detector's own number; each
challenger keeps its best sign, which favours the challengers. And each quantity keeps its own
band: the statistic reads one position per game, its own t0, and every band covers the t0
range, so unlike the per-position AUROC nothing here depends on the bands agreeing.

**The space ordering also changes, as it should.** GatedAnneal is fourth here, a point above
raw, not the runaway first it is on the seed-42 roster — its lead there was sparsity. GatedSAE
takes the top by point estimate, not separably from BatchTopK (`d` +0.013 [-0.000, +0.027],
paired; `illegal_move_detector/results/auroc_by_space_l0_matched_paired.csv`). That reversal is `36f452c`'s, reproduced on a third corpus.

## Reproducing

```
# once, on a GPU: per-position norms for the spliced tree, written beside the existing npz
sbatch scripts/slurm/studies/violation_divergence/submit_branch_norms.sbatch

PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py

# the L0-matched figure scores the detector's corpus, so it needs that study's theta/norm
sbatch scripts/slurm/studies/violation_divergence/submit_detector_sae_geometry.sbatch \
    --l0-matched --quantities
PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py --l0-matched
PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py --l0-matched --camera-ready
```

The ablation itself is CPU-only. `branch_norms.py` is the one GPU step, and it is
non-destructive: it writes `<space>_norm.npz` beside `<space>.npz` rather than rewriting it,
because re-running `branch_geometry.py` would recompute `d` and `theta` through an encode that
is not bitwise reproducible and every committed number here is byte-identical against the
current files. Its rows come from the committed `meta.npz`'s `row` column, never re-derived --
`extend_branches` grew the tree after that npz was written, so re-deriving now yields 2,207
games where the npz holds 575.

`--l0-matched` needs `theta_norm[_<space>].npz` on the detector's corpus, written beside its
`d` files by `extract_geometry.py --quantities` rather than into them, because every committed
figure in that study is pinned to the existing `d`. Its profile for the two challengers comes
from `l0_<config>_raw.npz` (`analysis/straightness/l0_matched_profile.py`); `d` takes that
study's own `_legal.npz`, since a pick's two legal-block files differ by encode noise and `d`
has to reproduce its bar exactly.

This supersedes `theta_ablation.py`, which measured only the argmax half of this.
