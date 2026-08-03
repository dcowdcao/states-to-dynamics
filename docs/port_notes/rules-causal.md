# Port notes — `spade/rules/causal` + `analysis/rules_causal`

Source: `src/rules/causal/{patch_battery,mine_forfeits,forfeit_patch,pair_patch,unflip_edit}.py`
and `analysis/rules_causal/{make_figures.py,README.md}`. A move, not a rewrite: outside the
import block and the `Usage:` lines, `diff` against the source is empty.

## What changed

Imports become `spade.*`; the four `sys.path.insert(..., parents[2])` lines are gone (this
package sits one level deeper than the rest of `src/`, so it walked up two, not one).
`from rules.study_dir import rules_dir` → `from spade.common.study import study_dir` with
`study_dir("rules")` at each call site — same `SPADE_RULES_DIR`, same
`/n/netscratch/ba_lab/Lab/$USER/spade_rules`. `boards` and `illegal_taxonomy` now resolve
to `spade.othello.boards` / `spade.othello.taxonomy`.

`study_dir` mkdirs where `rules_dir` did not. It makes no difference here: `mine_forfeits`
mkdirs its own output dir anyway, and the three readers only join a path onto it.

## Carried unchanged, on purpose

`PATCH_HOOKS` keeps its names and insertion order — `resid0`, then `attn{L}`/`mlp{L}` for
L=0..7. `make_figures.component_curves` slices the layer index off the key with
`patch.str[len(comp):]` and `paper_style.COMPONENT_STYLE` is keyed on the `attn`/`mlp`
prefixes, so renaming a hook silently changes what the figure labels.

`unflip_edit.set_logmass(logprobs, token_sets) -> Tensor` keeps its name and signature:
`spade/bracket_flip/tpr_intervene.py` imports it so both studies score identically.

The by-construction ceiling survives in all four places it was written down: the
`patch_battery` module docstring, `rules/causal/__init__.py` (already carried it in the
skeleton), the `pair_patch` figure's `axhline(1, label="full-residual patch ≡ 1 (shared
prefix)")`, the forfeit figure's "clean run (= full-residual patch, sanity ✓)" line, and
the README paragraph. Nothing needed to be added.

## Questionable things ported as-is

`pair_patch.py` and `unflip_edit.py` do `from circuits.othello_engine_utils import
to_string` at module level without ever calling `add_karvonen_to_path()`. They only work
because importing `spade.othello.boards` a few lines earlier calls it as an import side
effect. Reordering the import block, or `boards` dropping that call, breaks both files.

`mine_forfeits.py` puts `from spade.othello.taxonomy import flips` in the post-path-
injection block, below the third-party `circuits` import — a first-party import sitting out
of order because `taxonomy` itself needs `circuits` on the path.

`forfeit_patch.py` shadows the loop variable `e` (the event dict) over the outer `e` from
the `by_len` build, and `mine_forfeits` uses `e` for the mover color. Both are correct as
written; neither is pleasant.

`patch_battery.battery_logprobs` binds `src=cache[name][:, -1, :]` as a default argument to
capture the loop variable. Intentional, and the only reason the closure is correct.

## Cannot be verified in this worktree

- `unflip_edit.probe_directions` loads `analysis/linear_probe/activations/linear_probe.pt`
  (plus `primary_layer_activations.pt` and `board_states.pt`) from `repo_root() / PROBE_DIR`.
  That directory is gitignored and absent here; it must be copied by hand. The path logic
  is repo-root-anchored and was left without a fallback, as specified. The module cannot run
  in this tree.
- Nothing here runs without a GPU, the model download, or the netscratch study dir.
- `spade.common.{config,device,net,paths,study}`, `spade.othello.{boards,taxonomy}` and
  `spade.figures.paper_style` are still skeletons owned by other port agents, so a plain
  `import spade.rules.causal.pair_patch` fails today on their bodies, not on this code.
  Verification was: `py_compile` on all six files, then importing all five modules and
  `analysis/rules_causal/make_figures.py` with those seven modules and `circuits` stubbed
  out (`PYTHONPATH=$WT/src`, python 3.11.15 from `spade_env`, no install). That also asserts
  the `PATCH_HOOKS` order and the `set_logmass` signature.
- `spade.othello.taxonomy`'s skeleton advertises `flips(board, move)`; the source is
  `flips(state, square, color)` and all three call sites here pass three arguments. If the
  taxonomy port follows the skeleton comment rather than the source, this package breaks.

## Judgement call

`analysis/rules_causal/results/*.csv` (9.5 MB, git-tracked in the source repo) came along.
`make_figures.py` reads nothing else, and wave 6 verifies "one figure regenerated
byte-comparably from committed inputs". The three `*.png` did not: they regenerate from
those CSVs, and MIGRATION open question 10 has not settled the tracked-figure policy.
