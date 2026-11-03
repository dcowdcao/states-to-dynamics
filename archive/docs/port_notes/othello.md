# Port notes — `spade.othello`

The shared game layer, lifted out of the source repo's `rules/` so no study imports from
a sibling study. Four studies read it.

## What moved where

| Source | New |
|---|---|
| `src/rules/illegal_taxonomy.py` | `spade/othello/taxonomy.py` |
| `src/rules/test_illegal_taxonomy.py` | `spade/othello/test_taxonomy.py` |
| `src/rules/causal/boards.py` | `spade/othello/boards.py` |
| `src/rules/build_pairs.py` — `stream_games`, `classify_prefix`, `T_MIN`/`T_MAX`, the token-index assert | `spade/othello/games.py` |
| `src/straightness/build_games.py` — `collect_legal`, `engine_legal`, `first_illegal_pos` | `spade/othello/games.py` |

Bodies are byte-identical to the source. The only edits are the import block: the
`sys.path.insert` lines are gone and `common.*` / `rules.*` became `spade.*`.

## `collect_legal` and `first_illegal_pos` live in `games.py`

The straightness agent should import them from `spade.othello.games`, not keep them local.
Three studies call them, which is the "more than one study" bar:

- `src/straightness/build_games.py` (their owner)
- `src/manifold_capture/build_corpus.py`
- `src/violation_divergence/branches.py`

`engine_legal` came with them; it is `first_illegal_pos`'s only helper and has no other
caller. `CLASSES`, `EMPTY_CATEGORIES`, `T0_MIN`/`T0_MAX`, the rollout sampler, the WTHOR
loader and the violation builder are straightness-only and stay in that study.

## `MAX_LEN` is not here

`MAX_LEN = T_MAX + 1` is the pair-item padding width, used only inside `build_pairs.main`.
`spade/othello/games.py` exports `T_MIN`/`T_MAX`; `spade/rules/build_pairs.py` defines its
own `MAX_LEN` from `T_MAX`. Nothing outside build_pairs ever read it.

## Import side effects are eager, deliberately

`spade/othello/games.py` calls `force_ipv4()` and `add_karvonen_to_path()` at module level,
as `rules/build_pairs.py` did. `bracket_flip/events.py` and `violation_divergence/branches.py`
imported that module for the side effect alone (their source lines say so). `taxonomy.py` and
`boards.py` call `add_karvonen_to_path()` at module level for the same reason. Do not make
any of these lazy.

`boards.py` imports `flips` from `taxonomy.py`, so the taxonomy is on the karvonen path
before `circuits` is imported either way.

## Symbols downstream code binds to, by importer

Verified by grepping the source repo. These names cannot be renamed.

- `taxonomy`: `CATEGORIES` (rules/embed, rules/extract_acts, rules/build_pairs,
  violation_divergence/branches, three `analysis/*/make_figures.py`, `figures/paper_style.py`
  keys its category colors on this order), `LEGAL`, `classify`, `classify_all`, `flips`
  (`rules/causal/{boards,mine_forfeits}`).
- `boards`: `replay`, `flip_set`, `regular_moves`, `rays_touch`, `tokens_of` — read by
  `rules/causal/{pair_patch,mine_forfeits,unflip_edit}`, `bracket_flip/{events,redteam,
  tpr_intervene}`, `manifold_capture/build_corpus`, `analysis/violation_divergence/*`.
  `VOCAB_SQUARES` has no external reader but is `regular_moves`'s domain.
- `games`: `stream_games` (rules/build_pairs, straightness/build_games,
  manifold_capture/build_corpus, bracket_flip/events), `classify_prefix`
  (violation_divergence/branches), `T_MIN`/`T_MAX` (bracket_flip/events),
  `collect_legal` + `first_illegal_pos` (three studies, above).

## Verification

`python -m py_compile` passes on all five files. The taxonomy test suite runs green:
7 tests, including `test_played_games_match_engine`, which replays 20 random games and
checks `classify_all` against the engine's `get_valid_moves()` on non-forfeit positions.

Import could not be checked the plain way — `spade/common/{paths,net,config}.py` are still
skeletons on this branch, so `add_karvonen_to_path`, `force_ipv4` and `CTX_LEN` do not
exist yet. Verification stubbed those three symbols, then imported all three modules and
ran the tests. Re-run `PYTHONPATH=src python -c "import spade.othello.games"` once
`spade.common` lands; nothing else about this port depends on it.

## Ported as-is, worth a second look

- `taxonomy.classify_all` returns a `list[int]`. The skeleton docstring said `-> dict` and
  `classify(board, move) -> str`; the real functions return ints and a list. Ported as the
  source has it.
- `boards.rays_touch(square, targets)` walks the eight full rays and ignores blocking discs,
  so it answers "is `square` on a line with a target", not "does a flip ray reach it". The
  name reads as the stronger claim. `bracket_flip/redteam.py` and `events.py` depend on the
  weaker behaviour, so it was not changed.
- `games.collect_legal` truncates to `CTX_LEN` before testing `len(seq) < CTX_LEN`, so a
  game longer than 59 tokens is accepted with its tail dropped. Intended, given the class
  definition in the straightness docstring, but not stated anywhere.
