# Port notes: violation_divergence

What moved, what had to change, and what a reader should not trust. Source is the
frozen repo's `src/violation_divergence/` and `analysis/violation_divergence/`.

## Map

| Source | Here |
|---|---|
| `src/violation_divergence/{branches,extract_acts,divergence,compute_divergence,test_divergence}.py` | `src/spade/violation_divergence/` |
| `src/violation_divergence/study_dir.py` | deleted — `spade.common.study.study_dir("divergence")` |
| `src/violation_divergence/{DATA,METRICS}_REPORT.md` | `analysis/violation_divergence/reports/` |
| `analysis/violation_divergence/**` | same path, imports rewritten |

`study_dir("divergence")` must keep resolving `$SPADE_DIVERGENCE_DIR`, default
`/n/netscratch/ba_lab/Lab/$USER/spade_divergence` — the real branch set, activations
and npz live there and are not regenerable inside a 90-day window.

## Changes forced by the new layout

- Every intra-repo `sys.path.insert` is gone; imports are absolute `spade.*`.
- `divergence_dir()` → `study_dir("divergence")` at each of its six call sites.
- `EXCLUDED_SAE` is no longer defined in `compute_divergence.py`. It is imported from
  `spade.spaces`, and `game_trajectory/make_figures.py` now imports it from there too
  rather than re-exporting it through `compute_divergence`.
- Three analysis scripts computed `REPO = Path(__file__).resolve().parents[3]` only to
  put `src/` on the path and to build `REPO / "results" / "violation_divergence"`. The
  path insert is gone and the results path is `results_dir() / "violation_divergence"`,
  which is what the study's own top-level `make_figures.py` already used. Same
  directory, one fewer hardcoded parent count.
- `sae_spaces/make_figures.py` likewise: `CSV_DIR` is `results_dir() / ...`.
- Usage lines in package docstrings and in the study README are `python -m
  spade.violation_divergence.<module>`; the analysis scripts stay
  `python analysis/violation_divergence/.../make_figures.py`.
- SLURM lines in the README point at `scripts/slurm/studies/violation_divergence/`,
  per ARCHITECTURE's script map. Those scripts are not in my zone and are not ported here.

## Cross-package imports, written against modules that are still skeletons

Everything below is a name this study calls and another agent owns. Only
`divergence.py` imports cleanly today; the rest fail at import until their
dependencies land, which is expected at this point in the port.

- `spade.othello.games`: `stream_games`, `collect_legal`, `first_illegal_pos`.
  `collect_legal`/`first_illegal_pos` came from `straightness/build_games.py` in the
  source and `stream_games` from `rules/build_pairs.py`; ARCHITECTURE puts all three
  in `othello/games.py`. Call signatures are unchanged: `stream_games(dataset_name,
  seed)`, `collect_legal(games, n, skips)`, `first_illegal_pos(seq)`. Note that the
  skeleton's pseudocode signatures for these differ from the source's; the source's
  are what this study calls.
- `spade.rules.build_pairs.classify_prefix` — a sibling-study import. The skeleton
  lists `classify_prefix` under `rules/build_pairs.py` and not under `othello/games.py`,
  so that is where it is imported from. If it moves to `othello/games.py`, this is a
  one-line change. *It moved; the import follows it — see
  `docs/finish_notes/flagged-defects.md`.*
- `spade.rules.flip_labels.flip_stats`, `spade.straightness.extract_traj.check_sink` —
  two more sibling-study imports, both carried over as-is on instruction.
  `spade/straightness/extract_traj.py` does not exist yet, so `extract_acts.py` is the
  one module whose import graph is currently unresolvable for a reason other than a
  skeleton stub.
- `spade.othello.boards` (`replay`, `flip_set`, `regular_moves`, `tokens_of`),
  `spade.othello.taxonomy` (`CATEGORIES`, `LEGAL`, `classify`),
  `spade.saes.registry.seed42_entries`, `spade.evaluation.karvonen_adapter`
  (`load_karvonen_sae`, `load_standardization_params`, `StandardizedDictionary`),
  `spade.spaces.EXCLUDED_SAE`, `spade.figures.paper_style`, `spade.common.{config,
  device,net,paths,study}`.
- `compute_divergence.sae_encoder` reads `entry.ckpt_dir`, which is the source
  registry's field name. The skeleton's `SAEEntry` sketch calls it `checkpoint_dir`.
  Whoever ports `saes/registry.py` decides; if the field is renamed, two lines here
  change.

## Karvonen fork on sys.path — implicit, as in the source

`branches.py`, `behavioral/make_figures.py` and `probe_readout/decode.py` all
`from circuits.othello_engine_utils import ...` without calling
`add_karvonen_to_path()` themselves. They work because an earlier `spade.othello`
import already did it, exactly the way the source relied on `rules.illegal_taxonomy`
and `rules.build_pairs` doing it. This is fragile — reordering the import block
breaks it with an unhelpful `ModuleNotFoundError` — but it is the source's behaviour
and porting it faithfully means keeping it. `spade/othello/games.py`'s own docstring
already commits to holding that side effect, so the contract is at least written down
now.

## Verification done

- `python -m py_compile` on all 5 package modules and all 6 analysis scripts: clean.
- `python -m spade.violation_divergence.test_divergence` with `PYTHONPATH=src`:
  **11/11 kernel tests pass**, no data, no model, no GPU.
- `import spade.violation_divergence.divergence`: clean.
- A static sweep of every `from spade.* import` target in the zone finds one missing
  module, `spade.straightness.extract_traj` (above). No other typo'd module path.
- Nothing else can be run here: the pipeline needs netscratch tensors and a GPU, and
  the figure scripts need `paper_style` plus the npz set.

## Things ported as-is that a reader should question

- `probe_readout/decode.py` defaults `--probe` to
  `repo_root()/analysis/linear_probe/activations/linear_probe.pt`, which is gitignored
  and absent from this worktree. The path logic is unchanged; **the script cannot run
  here** until that 220 MB directory is copied by hand (MIGRATION, "Things git will not
  move for you").
- `analysis/violation_divergence/make_figures.py`'s docstring still advertises F1 as
  "event-locked delta-hat and M2's cosines". The two cosine panels were deleted; the
  code comment inside `f1_divergence_layer5` says so and the docstring was never
  updated. Left alone — it is a source-repo defect, not a port artifact.
  *Fixed since — see `docs/finish_notes/flagged-defects.md`.*
- `game_trajectory/make_figures.py` re-imports torch inside `space_features` as
  `import torch as _t` while the module already imports torch at the top. Pointless,
  and it violates the house rule on function-level imports. Left as-is.
- `game_trajectory/README.md` documents flags `--n_grid` and `--embedding both` that
  the script's argparse does not accept (`--embedding` takes `pca|isomap`). Stale in
  the source; not repaired.
  *Fixed since — see `docs/finish_notes/flagged-defects.md`.*
- The behavioural figure's suptitle, "An illegal move changes what OthelloGPT predicts,
  immediately and for the rest of the game", is about the branch-vs-reference KL and is
  not one of the retired claims — but the legal control has the *highest* KL of all, and
  the figure draws it as the null, so the title reads as more of a contrast than the
  data supports. Flagging rather than editing: rewriting a figure title is a science
  call, not a port call.
- `compute_divergence.py`'s round-robin shard comment cites `PROCESS.md` in the source
  repo. Kept verbatim; the incident it refers to should survive into this repo's
  PROCESS.md, which I am not editing.

## METRICS_REPORT.md is internally inconsistent

Recorded because it was found, not fixed. Judgement 1 carries a "Superseded at
integration (commit e93f9bb)" note saying `half_life` is now **elapsed since the peak**;
"Requests for the integrator" item 3, later in the same file, still asserts the old
**absolute-offset** definition and asks the integrator to change it. `RED_TEAM.md` F6
flags exactly that paragraph as stale and says to fix the report, not the code — the
code is elapsed and verified (`max(peak_r + half_life)` over finite rows is exactly 24).

The prose is left as written. A correction header at the top of the file names the
contradiction and points at `docs/claims_status.md` and `RED_TEAM.md`; the body is a
record of a review at a point in time and silently rewriting it would destroy that.
The same reasoning applies to both reports' "For the integrator" sections, which
describe the source repo's then-current file layout (`study_dir.py` as A1's file, the
`shell/slurm/` submit-script default). Those sentences are history and were not
rewritten to the new tree; only paths that point at ported code a reader would want to
open were updated.

## Not ported

`analysis/violation_divergence/**/figures/` — 9 committed PNG/PDF files. Tracked-figure
policy is MIGRATION open question 10 and is undecided; every one of them is regenerable
from the netscratch npz set by the scripts in this zone. Say the word and they are a
`cp` away.
