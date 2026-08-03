# Vendored: tpr_othello

This file is ours; everything else in this directory is upstream and unmodified.

- Upstream: https://github.com/ajyl/tpr_othello
- Pinned commit: `cc90cead6cf66a8d3e25ceec6c0951112066fbdf` (2026-04-29)
- Paper: Lee, Viégas & Wattenberg, *Tensor Product Representation Probes Reveal
  Shared Structure Across Linear Directions*, arXiv:2605.09967
- Used by: `src/bracket_flip/` (see `docs/bracket_flip_design.md`)

Omitted from the vendored copy, recoverable by cloning at the pinned commit:
`.git` (29 MB), `train_data/` (26 MB) and `test_data/` (8.6 MB) — their own
Othello game pickles. We train on our corpus and our checkpoint
(`Baidicoot/Othello-GPT-Transformer-Lens`), so their data is dead weight.

`ckpts/` and `probes/` upstream contain only placeholder files: no pretrained
probe and no model checkpoint ship with the repo, so the TPR is trained here
rather than loaded. That also rules out reusing their probes directly, which
would have been unsafe anyway — TransformerLens folds LayerNorm and centers
writing weights, so our residual stream is not in the same basis as the raw
HuggingFace model theirs was fit on.

Upstream carries **no LICENSE file**. Treat it as all-rights-reserved research
code: fine to read and run for our own research, not to redistribute.
