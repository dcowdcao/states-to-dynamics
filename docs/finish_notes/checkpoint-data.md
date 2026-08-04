# Finish note — the binary tier git could not move

`*.pt` is gitignored, so the SAE weights and the probe activations had to be copied by
filesystem from the frozen source repo. This note records what landed, how it was
verified, and what was left behind on purpose.

Source (read-only): `../Spade_Research_David_Sumedh_Ba`, branch `cosine-spade-gamma-sweep`.
Destination: the **main tree** `spade_sae`. Nothing here is committable — the files are
gitignored by design. The committed record is this note plus `manifests/checkpoints/`.

## What landed

| source | here | files | bytes |
|---|---|---|---|
| `autoencoders/` | `checkpoints/main/` | 13 | 201,661,723 |
| `autoencoders_batch_topk/` | `checkpoints/batch_topk/` | 3 | 33,605,538 |
| `autoencoders_cosine_spade/` | `checkpoints/cosine_spade/` | 3 | 33,569,853 |
| `autoencoders_spade_lam1e-03_gam1e-04/` | `checkpoints/spade_revival/` | 3 | 33,570,058 |
| `autoencoders_spade_sweep/` trainer_4, trainer_6 | `checkpoints/spade_sweep/` | 5 | 67,126,848 |
| `autoencoders_spade_sweep/` trainer_0–3 (config-only) | `archive/spade_sweep_collapsed/` | 5 | 16,746 |
| `analysis/linear_probe/activations/` | same path | 7 | 189,920,178 |

**39 files, 559,470,944 bytes (534 MiB).** The 810 / 137 / 159 MB sizes quoted in
ARCHITECTURE.md are the source dirs *including* their intermediate checkpoints; those were
not copied (below), which is where the difference goes.

Both load-bearing layout rules hold:

- `standardization_params.json` sits at the **save-dir** level in every destination, beside
  the trainers and not inside them, which is what
  `load_standardization_params`' parent-directory fallback needs.
- `trainer_N` numbering is carried over verbatim. `checkpoints/main/trainer_3` is TopK,
  `trainer_5` is SpaDE γ=1e-2; the sweep keeps the gaps in its numbering (`trainer_4` and
  `trainer_6` in `checkpoints/spade_sweep/`, `trainer_0`–`trainer_3` in the archive) rather
  than being resequenced.

`checkpoints/main/trainer_5` is the degenerate γ=1e-2 seed-42 checkpoint. It is copied and
the registry lists it; the exclusion is a downstream decision made once in `spade.spaces`.

## Verification

Every file was hashed with `sha256sum` on both sides and the two digest lists compared:
all 39 match. The digests are committed under `manifests/checkpoints/`, one `.tsv` per
destination directory. Format follows `manifests/netscratch/` — headerless TSV, size, date
(file mtime, preserved from the source by `rsync -a`) — with the sha256 prepended:

```
sha256 <TAB> size <TAB> YYYY-MM-DD <TAB> path
```

Paths are repo-relative, where the netscratch manifests are absolute: that tier lives
outside the repo and has nowhere else to anchor, this one does.

The only pair of files sharing a digest is the `standardization_params.json` deliberately
placed in both `checkpoints/spade_sweep/` and `archive/spade_sweep_collapsed/` — the two
halves of one save dir. No weights are accidental duplicates of each other.

Then the registry's own verifier — `ae.pt` exists and each checkpoint's own `wandb_name`
maps to the expected config:

```
$ cd /n/home10/dcao/spade_sae && PYTHONPATH=src python -c "from spade.saes.registry import seed42_entries; ..."
9 verified
  PAnnealTrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/main/trainer_0
  StandardTrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/main/trainer_1
  GatedSAETrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/main/trainer_2
  TopKTrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/main/trainer_3
  GatedAnnealTrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/main/trainer_4
  SpaDETrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/main/trainer_5
  BatchTopKTrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/batch_topk/trainer_0
  CosineSpaDETrainer-othello_seed42 /n/home10/dcao/spade_sae/checkpoints/cosine_spade/trainer_0
  SpaDE-lam1e-03-gam1e-04_seed42 /n/home10/dcao/spade_sae/checkpoints/spade_revival/trainer_0
```

All 9 seed-42 entries verify, no assertion fired. `PYTHONPATH=src` stands in until the
editable install lands.

## Deliberately not copied

**Intermediate `trainer_*/checkpoints/ae_{9155,18310,27465}.pt`** — 33 of them, ~1.1 GB,
roughly 3× the final weight volume, no readers anywhere in the tree. Skipped.
**This is MIGRATION.md open question 4 and it is still open**: the frozen source keeps
every one of them, so copying them later is a `rsync` away, whereas copying them now and
deciding against them later means paying the space for nothing.

`disk`: `/n/home10` went 22 GB → 21 GB free of 95 GB across the copy.

## For the owner

- **Open question 4 stands.** Skipping the intermediates is the reversible half of that
  decision, not an answer to it.
- **Open question 2** (archive the four collapsed λ=1e-8/1e-7 sweep configs, or delete)
  is likewise unanswered. They are archived here because that is what ARCHITECTURE.md's
  map says; the source still holds them either way. They are config-only — VE ≈ −3e-05,
  L0 = 8192, no weights were ever saved.
- The archived collapsed configs got a copy of their save dir's
  `standardization_params.json` so the archive describes itself. It is 13 KB and nothing
  loads from there; drop it if the archive should be configs only.
- `analysis/linear_probe/activations/board_states.pt` came over as-is, including the 53
  mislabeled rows in 10,000 that ARCHITECTURE.md records. Copying it is not a claim that
  it is correct.
