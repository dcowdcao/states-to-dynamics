# external/

Two third-party trees, both hard runtime dependencies. `spade.common.paths` puts each on
`sys.path`; neither is pip-installed, because their own pin sets (torch, TransformerLens,
nnsight) would fight ours.

**`Karvonen_Research/`** — Adam Karvonen's board-game SAE repo. Supplies
`circuits.dictionary_learning` (the baseline trainer/architecture classes: Standard,
PAnneal, Gated, GatedAnneal, TopK, BatchTopK), the Othello engine and ground-truth
utilities, the board-reconstruction eval, and NNsight activation capture. SpaDE plugs in as
a `circuits` `Dictionary` subclass.

**`tpr_othello/`** — the tensor-product board probe used by the bracket-flip study.

Neither upstream has a licence, so neither is redistributed. `make setup` runs
`scripts/fetch_external/fetch.sh`, which clones each at its pinned commit and applies our
patch from `scripts/fetch_external/patches/`. A tree that is already here is left alone.

| tree | upstream | commit | our patch |
|---|---|---|---|
| `Karvonen_Research` | github.com/adamkarvonen/SAE_BoardGameEval | `7187d4a` | drops `.gitmodules` and 18 `ae.pt`; re-run walkthrough notebook |
| `Karvonen_Research/circuits/dictionary_learning` | github.com/saprmarks/dictionary_learning (MIT) | `43421f5` | per-trainer device moves, dead-neuron logging |
| `tpr_othello` | github.com/ajyl/tpr_othello | `cc90cea` | adds `PROVENANCE.md`; drops upstream's game pickles |

A fetched tree is a git checkout of its pin, so after editing one, regenerate its patch from
inside it: `git add -A && git diff --cached --full-index > <its patch>`. Keep
`--full-index` and leave out `--binary`: a deleted binary then carries only its blob id, no
upstream bytes, and `git apply` still checks and deletes it.
