# Finish note — the wave-3 acceptance eval

All nine seed-42 checkpoints re-evaluated on GPU against the committed reference rows in the
frozen source repo. Results in `results/acceptance/`.

Command shape (per checkpoint), activations read from netscratch because 28 GB does not fit
in home:

```
python -m spade.evaluation.run_eval \
    --checkpoint_dir <ckpt> \
    --test_data_dir /n/netscratch/ba_lab/Lab/dcao/spade_multiseed/data/activations \
    --output_dir results/acceptance --n_samples 59000 --skip_fidelity
```

`--skip_fidelity` because loss_recovered is not one of the acceptance metrics and costs an
OthelloGPT load per SAE.

## The port is faithful

Three checkpoints reproduce their reference **exactly** — delta 0.000000 on L0, variance
explained, coverage and reconstruction F1:

| checkpoint | reference dir |
|---|---|
| `checkpoints/batch_topk/trainer_0` | `results/karvonen_batch_topk/` |
| `checkpoints/cosine_spade/trainer_0` | `results/karvonen_cosine_spade/` |
| `checkpoints/spade_revival/trainer_0` | `results/karvonen_spade_lam1e-03_gam1e-04/` |

Identical inputs, identical outputs to the last decimal. Whatever else is below, the ported
SAE loading, standardization, reconstruction and F1/coverage math are correct.

## MIGRATION.md's acceptance target is wrong

Wave 3 states TopK on `checkpoints/main/trainer_3` gives L0 80.00, **variance explained
0.943**, coverage 0.722. The committed reference row is VE **0.9867**, and the rerun gives
0.98688. Nothing in the tree produces 0.943. Anyone running the stated check would have
concluded the port was broken.

## The five `results/karvonen/` checkpoints were evaluated on activations that no longer exist

They differ from their reference by small mixed-sign amounts: VE always within 0.00056,
coverage and reconstruction F1 wobbling within ±0.014 in both directions. Subsampling is a
deterministic prefix slice (`train_acts[:n_samples]`), so identical tensors give identical
numbers — the three exact reproductions above prove that. The inputs therefore differ.

The reference CSV was committed 2026-07-13 22:02 -0700; the only surviving activation
tensors were written 2026-07-14 01:52–01:55, after the fact. They are a later extraction.
The set behind the published `results/karvonen/` coverage numbers is gone, and
`data/activations/metadata.json` — which would record the extraction parameters — describes
a 100-game smoke run (`activation_shape [4680, 512]`) while sitting beside 12 GB tensors
holding 5,900,000 rows.

VE is an aggregate over millions of rows and is stable under resampling; coverage depends on
feature-label assignments learned by argmax over thresholds, which flip under small input
changes. The pattern of deltas matches that and nothing else.

## `checkpoints/main/trainer_5` — the weights are not the run

The `ae.pt` in this directory does not correspond to its own config, its wandb run, or its
committed eval row. Everything except the weights agrees:

| source | L0 | variance explained |
|---|---|---|
| wandb training record | 15.43 | 0.3667 |
| committed eval row | 15.43 | 0.352 |
| **rerun of the ae.pt on disk** | **127.76** | **0.973** |

Checked against the wandb export for all six `main/` trainers, comparing each run's
training-time `l0` / `frac_variance_explained` against the rerun of the weights in that
directory. **trainer_0 through trainer_4 match. trainer_5 alone does not.**

This corrects the finding recorded in PROCESS.md on 2026-08-03, which read the same
discrepancy as "a configuration in three documents never ran". It did run:
`SpaDETrainer-othello_trainer_5`, λ=9.77e-5, γ=0.2, 36621 steps, cuda:0, finished,
2026-03-26. The docs reporting "SpaDE initial, λ=9.77e-5, γ=0.2" as a distinct
over-regularized run are correct, and its degeneracy is real.

What is actually stale in `config.json` is `steps=400` and `device=mps`, and those are stale
in **all six** configs, not just this one — a general artifact of the Mac smoke config, and
exactly what MIGRATION.md open question 3 is about. λ and γ are correct.

**So the exclusion decision's justification stands**, and is if anything firmer: both
recorded SpaDE baseline runs are degenerate (γ=0.2 → VE 0.367, γ=0.1 → VE 0.399). The
anomaly is that the file on disk is neither of them.

Provenance of those weights is not recoverable from what is in the tree. No run in
`results/Wandb_Export.csv` reports L0 ≈ 128 / VE ≈ 0.97. All six `ae.pt` share one mtime
(2026-04-09 05:52), so trainer_5 was not overwritten later than its siblings; the local
`wandb/` runs that do report L0 ≈ 128 are from July and postdate the file.

**Owner's call.** Anything that loaded `main/trainer_5/ae.pt` was analysing a model other
than the one its config and metrics describe — that includes the viewer roster, the geometry
family, and `manifold_capture`'s `LOW_VE` marker, which names this checkpoint as low-VE
while the file itself reconstructs at 0.973.
