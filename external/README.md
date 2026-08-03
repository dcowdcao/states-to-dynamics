# external/

Two vendored forks, both hard runtime dependencies. `spade.common.paths` puts each on
`sys.path`; neither is pip-installed, because their own pin sets (torch, TransformerLens,
nnsight) would fight ours.

**`Karvonen_Research/`** — Adam Karvonen's board-game SAE repo. Supplies
`circuits.dictionary_learning` (the baseline trainer/architecture classes: Standard,
PAnneal, Gated, GatedAnneal, TopK, BatchTopK), the Othello engine and ground-truth
utilities, the board-reconstruction eval, and NNsight activation capture. SpaDE plugs in as
a `circuits` `Dictionary` subclass.

**`tpr_othello/`** — the tensor-product board probe used by the bracket-flip study, pinned
at `cc90cea`. No upstream license; treat as all-rights-reserved and do not redistribute.

Neither is vendored into this skeleton yet. When they are, three things must be got right,
because the source repo got them wrong:

- **Pin `Karvonen_Research`.** It carries no commit, no provenance file, a flattened
  submodule and a dangling `.git` pointer. Capture the upstream SHA before copying.
- **Bring the files git cannot see.** The vendored `.gitignore` is still honoured inside
  our repo and hides `circuits/resources/meta.pkl` and a downloader script;
  `linear_probes/*.pth` is hidden by ours. A fresh clone yields a `circuits` that imports
  and then fails at runtime.
- **Do not re-clone from upstream to "clean it up"** without diffing first — the fork has
  been instrumented in place before, and a fresh clone silently reverts that.
