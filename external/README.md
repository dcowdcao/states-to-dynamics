# external/

`Karvonen_Research/` — vendored copy of Adam Karvonen's board-game SAE repo. Hard runtime
dependency: supplies `circuits.dictionary_learning` (baseline SAE trainer/architecture
classes: Standard, PAnneal, Gated, GatedAnneal, TopK, BatchTopK), the Othello engine +
ground-truth utilities, the board-reconstruction eval, and NNsight activation capture. SpaDE
plugs in as a `circuits` `Dictionary` subclass.

Not yet vendored into this skeleton — copy it in (or add as a git submodule of
`saprmarks/dictionary_learning` under it) during migration. `spade.common.paths` puts it on
`sys.path` so `import circuits.*` resolves.
