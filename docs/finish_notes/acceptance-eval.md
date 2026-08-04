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

## `checkpoints/main/trainer_5` — the reference row does not describe the weights

| metric | reference | rerun |
|---|---|---|
| L0 | 15.43 | 127.76 |
| variance explained | 0.352 | 0.973 |
| coverage | 0.0614 | 0.209 |
| reconstruction F1 | 0.256 | 0.470 |

This is not a resampling effect. It confirms, with numbers, the finding recorded in
PROCESS.md on 2026-08-03: `trainer_5/config.json` records λ=9.77e-5, γ=0.2, 400 steps,
device mps — a Mac smoke config that was never the run that produced these weights, and
which "also explains the L0 15.4 / VE 0.352 attribution".

**The consequence is not yet drawn, and is the owner's call.** The exclusion of this
checkpoint, and the geometry family's retired "SpaDE is broken for cell-state mapping"
conclusion, both rest on it being degenerate. On its actual weights it reconstructs at
VE 0.973, which is not degenerate. Its coverage of 0.209 *is* the worst of the eight by a
wide margin (the others run 0.51–0.74), so "poor at board-state coverage" survives; "broken
reconstruction, VE 0.352" does not.

Note also that nothing in the tree pins which run these weights came from: the config.json
that would say is the known-wrong smoke config.
