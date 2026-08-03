# Bracket-flip port notes

What moved, what changed, and what a later session should not be surprised by.
Source: `../Spade_Research_David_Sumedh_Ba/src/bracket_flip/` and
`analysis/bracket_flip/`, both frozen and read-only.

## What landed

| source | here |
|---|---|
| `src/bracket_flip/{events,extract_acts,tpr_probe,probe_space,act_geometry,sae_arm,tpr_intervene,robustness,redteam}.py` | `src/spade/bracket_flip/` |
| `src/bracket_flip/study_dir.py` | dropped — `spade.common.study.study_dir("bracket")` |
| `src/bracket_flip/*.md` (9 write-ups) | `analysis/bracket_flip/reports/` |
| `analysis/bracket_flip/{make_figures,sae_pca_figure}.py`, `README.md`, `results/`, `figures/` | unchanged path |

`src/bracket_flip/` had no `__init__.py`; the skeleton's
`src/spade/bracket_flip/__init__.py` already described the ten modules accurately and
was kept as written.

Algorithms are untouched. Every change is an import, a path, or a docstring citation.

## Import rewiring

- `bracket_flip.study_dir.bracket_dir()` → `study_dir("bracket")` at every call site.
  `$SPADE_BRACKET_DIR` and the on-disk `.../spade_bracket` are byte-identical, so the
  real data on netscratch is still found.
- `rules.build_pairs` → `spade.othello.games` for `stream_games` **and for `T_MIN`,
  `T_MAX`**. The source defined those two constants in `build_pairs.py`; importing them
  from `spade.rules.build_pairs` would make bracket_flip depend on a sibling study,
  which the architecture forbids, and they belong to the legal-prefix corpus that
  `othello.games` owns. If the games agent puts them elsewhere, this is the one import
  to fix.
- `rules.causal.boards` → `spade.othello.boards`; `rules.embed` → `spade.spaces`;
  `multiseed.aggregate_seeds` → `spade.multiseed.aggregate_seeds`; `figures.*` →
  `spade.figures.*`.
- `rules.causal.unflip_edit.set_logmass` stays an import in `tpr_intervene.py`, so the
  TPR edit and the linear un-flip edit keep scoring with the same function.

## Path handling

- `tpr_probe.py` no longer hand-rolls two `sys.path.insert`s into
  `external/tpr_othello/src` and `.../src/hook_utils`; it calls
  `spade.common.paths.add_tpr_to_path()`. Verified against the vendored fork: `hook_utils`,
  `intervene_tpr_probe` and `train_tpr_probe` all resolve and
  `TensorProductBoardProbe(d_model=512, role_dim=52, filler_dim=2)` builds at 56,582
  parameters, the count the study's README quotes.
- `redteam.py` keeps `REPO = Path(os.environ.get("SPADE_REPO", repo_root()))` and its own
  insert of `REPO/external/Karvonen_Research`. This is deliberate: `repo_root()` resolves
  to whatever worktree the red team runs in, and `SPADE_REPO` is how it is pointed at a
  checkout that actually carries `external/`, `reference/` and
  `analysis/linear_probe/activations/`. Its `REPO/"src"` insert is gone. Note the ordering
  quirk, present in the source too: importing `spade.othello.boards` calls
  `add_karvonen_to_path()`, whose repo-root-relative entry is inserted at position 0 *after*
  redteam's `SPADE_REPO` one, so the repo-root fork wins when both exist.
- `sae_arm.py` wrote its CSV through `Path(__file__).resolve().parents[2]`; that is now
  `repo_root()`, matching how `probe_space.py` and `tpr_intervene.py` already resolved it.
- `act_geometry.py` and `tpr_intervene.py` keep `$SPADE_LINEAR_PROBE_ACTIVATIONS`.

## Cannot be run in this worktree

- `analysis/linear_probe/activations/` is gitignored and absent. `tpr_probe`,
  `act_geometry`, `tpr_intervene` and `redteam` all read `linear_probe.pt`,
  `board_states.pt`, `all_layer_activations.pt` or `primary_layer_activations.pt` from it.
  Copy it by hand (MIGRATION.md lists it) or point the env var at a checkout that has it.
- Everything else needs the netscratch study dir (events, probe set, activations, fitted
  probes) and, for `extract_acts` / `tpr_intervene` / `probe_space`'s embedding fetch, the
  model. No probe was fitted and no figure was regenerated here.
- Verification done instead: `py_compile` on every ported file, and an import check of all
  nine modules with the still-skeleton `spade.common` / `othello` / `spaces` / `multiseed`
  / `rules` stubbed out but `paths` and the two vendored forks real. 9/9 import.

## Judgement calls

- **Fig 1's suptitle carried the retired headline.** `make_figures.py` said *"A flanking
  move swaps every disc's read-out except the one it flips; relabeling to absolute colour
  inverts that"* — the sentence `docs/claims_status.md` retires and the study's own README
  leads by retracting. Replaced with a description of what the panels show: *"Where each
  square's read-out moves when a flanking move is played, and how relabeling to absolute
  colour inverts it."* The panel titles and Fig 2's suptitle state the three surviving
  findings and were left alone. `FIGURE_GUIDELINES.md` has no rule about figure text
  restating a retired claim; it probably should, but it is shared with every other
  in-flight port, so the rule is proposed here rather than written there.
- **`results/` and `figures/` came along.** Not named in the port brief, but the README
  cites six CSVs by path, `make_figures.py` reads five, and wave 6 wants a figure
  regenerated from committed inputs. 5.4 MB of CSV and 3.3 MB of image. These are the
  canonical three figures left after the source repo's own 8→3 prune, not the near-duplicate
  panels behind MIGRATION.md's open question 10; revert the figure PNGs alone if that
  question resolves the other way.
- **Nine reports, not eight.** The brief lists nine filenames as "the eight .md reports".
  All nine moved.

## Ported as-is, worth a second look

- `RESULTS_A3.md` and the other write-ups still cite `src/bracket_flip/*.py` and
  `shell/slurm/*.sh` in their reproduce blocks. They are dated records of what was run, so
  only RESULTS_A3.md was touched, and only to add the correction header. Module docstrings
  and the README were updated to the new paths.
- `make_figures.py` reads `f2_event_meta.csv` and `flip_cos_per_event.csv`, and the README
  cites `depth_reconciliation.csv`, `variance_shares.csv`, `sanity.csv` and `summary.csv`.
  **Nothing in the ported tree writes any of them.** `act_geometry.py` has no `main()` — it
  is a library the A4 driver called, and that driver is not in the source repo either. The
  CSVs are committed, so the figures redraw, but Fig 1 and Fig 2(a) cannot currently be
  re-derived from scratch. Same for `f2_event_meta.csv`, which `probe_space.py` does not
  write although it writes `f2_event.csv` beside it.
- `sae_pca_figure.py`'s suptitle still reads "F8 — …", the figure's number before the prune
  to three; the README calls it Fig 3. Left as found. It also imports `INK` without using it.
- `sae_arm.py` communicates through six module-level globals declared inside `main()`
  (`TABLE`, `raw`, `CODES`, `STRICT`, `PROBE`, `CACHE`), one of them lowercase. `measure()`
  reads all six. Untouched, but it is the one module here that would not survive a review.
- `study_dir()` creates its directory (`mkdir parents`, per the skeleton's contract) where
  the source's `bracket_dir()` was pure. `probe_space.py` and `robustness.py` call it at
  module scope, so importing them will now touch netscratch. Harmless, but it is a
  behaviour change the shared study-dir policy introduces, not one this study asked for.
- `robustness.py`'s docstring pointed at `shell/slurm/submit_bracket_robustness.sh`; it now
  points at `scripts/slurm/studies/bracket_flip/`, where ARCHITECTURE.md says the five
  `submit_bracket_*.sh` jobs land. Those scripts are another agent's zone and are not here
  yet.

## The claim discipline

`docs/claims_status.md` retires the original headline (*the flanked square is the one
square whose class does not change*) and keeps three findings: the model is
mover-relative and barely encodes absolute colour, the flip has a specific direction
(median cos 0.173–0.184 against a matched non-flipping move), and that direction is
causally used (+1.496 log-units vs +0.121 random). Nothing in the ported docstrings,
README or figure titles restates the retired sentence. `reports/RESULTS_A3.md` still
contains it in full, under a correction header, because it is the record of what was
measured.
