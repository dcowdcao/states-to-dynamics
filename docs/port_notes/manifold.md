# Port notes — manifold_capture

Source: `../Spade_Research_David_Sumedh_Ba/src/manifold_capture/` (13 modules) and
`analysis/manifold_capture/{make_figures.py,README.md}`. Twelve modules ported to
`src/spade/manifold_capture/`; `study_dir.py` dropped in favour of
`spade.common.study.study_dir("manifold")`. Logic is unchanged — the only edits are
imports, the `study_dir` / `results_dir` call shapes, and `Usage:` lines rewritten to
`python -m spade.manifold_capture.<module>`. A line-by-line diff against the source shows
nothing else.

## Verification

No install; `PYTHONPATH=src` under `spade_env` (CPython 3.11.15).

- `python -m py_compile` clean on all 13 files (12 modules + `make_figures.py`).
- Every module resolves its imports down to the still-stubbed foundation modules. The
  remaining `ImportError`s are all of the form "cannot import name X from
  spade.common.<mod>" — the module path is right, the name is not ported yet. `ising`
  imports cleanly today because it has no repo dependencies.
- `python -m spade.manifold_capture.ising --selftest` passes:
  `rho={0: 1.0, 1: 1.0}, mean intra J=0.193, mean |cross J|=0.0053` — byte-identical to
  the source module's output.
- The other five selftests (`subspace_capture`, `landmarks`, `receptive_fields`,
  `manifold_views`, `tuning_curves`) were run against scratchpad stand-ins for
  `spade.common.*`, `spade.spaces` and `spade.saes.registry`. All five pass and print
  numbers identical to the source repo's, e.g. `greedy r2@3=1.0000 ... rf_spread=0.976
  k_M=3` and `k16 coord gap=0.00049`. The harness lives in the scratchpad, not the repo;
  once the foundations land these run directly as `--selftest`.
- `networkx` (3.6.1) is in `spade_env` and `nx.community.louvain_communities` resolves;
  the Ising selftest exercises it. It was undeclared in the source `requirements.txt` and
  must be declared in the regenerated one — `feature_geometry` and `capture_regimes` are
  dead without it. `pandas` (`landmarks.merge_csv`) and `scipy.sparse.csgraph`
  (`landmarks.knn_components`) are the other non-obvious hard deps.

## The `spaces` name collision

`spade/manifold_capture/spaces.py` (this study's per-space encode/decode adapters) and
the new top-level `spade/spaces.py` (the shared `load_space` / `representative_spaces` /
`batched_apply` / `center_by_t` API) are different modules. All imports are absolute, so
`from spade.spaces import batched_apply, load_space` inside the local `spaces.py` reaches
the top-level module and the two never shadow. Callers that want the local one use
`from spade.manifold_capture import spaces` or `from spade.manifold_capture.spaces import ...`.

**Does the local one still earn its name?** Marginally, and I would rename it. It holds
four things: `spaces()` (which checkpoints exist on disk), `encode_fn` (a one-line wrapper
over `load_space`), `decoder` (decoder matrix + bias per SAE family), and `check_codes`
(simplex/non-negativity invariants). Only `spaces()` is about spaces-as-a-list; the rest is
decoder access and validation. `spade/manifold_capture/sae_access.py` or `roster.py` would
say what it does and end the collision. That is a rename of a module the whole study
imports, so I left it alone — it needs your call.

## Cross-package coupling to flag

**Private registry internals.** `manifold_capture/spaces.py` imports `_SEED42_DIRS`,
`_verify` and `SAEEntry` from `spade.saes.registry`, exactly as the source did from
`rules.sae_registry`. The reason is in a comment there: `seed42_entries()` verifies every
config at once and crashes on any registered-but-not-yet-trained checkpoint, while this
study wants to verify per config and skip the missing ones loudly. Two things follow.

1. The port uses `entry.ckpt_dir`. The registry skeleton's pseudocode names that field
   `checkpoint_dir`. If the registry lands with the longer name, `spaces.decoder` and
   `landmarks` break at runtime, not at import. One of the two has to move; I kept the
   source's `ckpt_dir`.
2. `_SEED42_DIRS` keys are also *data*: they become the `space` column of
   `capture_{curves,metrics}.csv` / `regimes.csv` / `landmarks*.csv` and the `<space>`
   half of `geometry_*.npz`, `receptive_*.npz`, `landmark_entropy_*.npz`. The 121 MB of
   already-committed `results/manifold_capture/` is keyed on the source repo's config
   names. Renaming a config key silently orphans those rows and re-runs everything.
   Its *values* are repo-relative checkpoint paths (`autoencoders/trainer_5`), which the
   new `checkpoints/{main,batch_topk,...}` layout does change — that part is fine, the
   paths are resolved against `repo_root()` and never written out.

**The excluded γ=1e-2 checkpoint is not excluded here.** MIGRATION decides the
`SpaDETrainer-othello` seed-42 exclusion "once, in `spade.spaces`". This study never goes
through `representative_spaces()`; `manifold_capture.spaces.spaces()` enumerates
`_SEED42_DIRS` itself and takes every config with an `ae.pt` on disk. So an `EXCLUDED_SAE`
set applied inside `spade.spaces.representative_spaces()` has no effect on manifold
capture. Two consequences worth your attention:

- Mechanically: the exclusion has to be enforced in `_SEED42_DIRS` (or the study needs its
  own filter) or it does not hold for this study.
- Scientifically: `LOW_VE = {"SpaDETrainer-othello"}` *is* that checkpoint, and the
  study's headline "distance-SpaDE is the purest tiler — shattering on 7/8 concepts,
  support size 0" is computed on it. `docs/claims_status.md` retired the geometry family's
  "SpaDE γ=1e-2 is broken" for resting on exactly this checkpoint, while listing the
  manifold-capture shattering result under "Holds". I have not touched either file, but
  the two entries look like they are about the same checkpoint and should be reconciled
  before the shattering claim is quoted.

**`collect_legal` has two incompatible definitions in the source.** `build_corpus.py`
calls `collect_legal(stream_games(...), n_games, skips)` — the *straightness* version
(`straightness/build_games.py:48`: take fixed 59-token prefixes, count skips). The
`spade.othello.games` skeleton docstring describes a different one,
`collect_legal(n, t_min, t_max)  # legal prefixes, round-robin over t`, which is the
`rules/build_pairs.py` prefix sampler. These are not the same function. `build_corpus`
(and `violation_divergence/branches.py`) need the straightness signature; if
`spade.othello.games` ships only the rules one, `build_corpus` fails at its first call.
Written here as an import against the straightness semantics.

**Karvonen path injection by side effect.** `build_corpus.py` has a bare
`from circuits.othello_engine_utils import ...` at module level with no
`add_karvonen_to_path()` of its own — in the source it worked because
`rules.build_pairs` / `rules.causal.boards` called it at import time. The port relies on
`spade.othello.games` doing the same, which its docstring promises ("puts the Karvonen
fork on the path ... do not make those effects lazy"). Faithful to the source, but it is a
load-bearing import-order dependency, not an obvious one.

## Adaptations forced by the new signatures

- `manifold_dir()` → `study_dir("manifold")`, `rules_dir()` → `study_dir("rules")`.
  `$SPADE_MANIFOLD_DIR` and `/n/netscratch/ba_lab/Lab/$USER/spade_manifold` are unchanged,
  as is the `corpus/corpus.pt` + `acts/acts.pt` layout under it — the real data is there.
- `results_dir() / "manifold_capture"` → `results_dir("manifold_capture")`, matching the
  skeleton's `results_dir(experiment)`. Same directory on disk.
- `make_figures.py` keeps its own `Path(__file__).resolve().parents[2] / "results" /
  "manifold_capture"` rather than switching to `results_dir` — that is what the source
  does and the layout is unchanged. It could reasonably move to `results_dir` now that
  the file imports `spade` anyway.

## Ported as-is, questionable

- **`make_figures.py` does not follow FIGURE_GUIDELINES.md and was ported unchanged.**
  It sets rcParams inline instead of calling `paper_style.apply_style`, hardcodes color
  literals (`SQUARE_COLORS`, the `regime_color` map, `#377eb8`, `#ff7f00`), pulls hues
  from `tab10`/`tab20`/`plasma`/`magma`, saves at `DPI = 100` where the guidelines want
  ≥300 and vector output, draws no uncertainty on any capture curve, and distinguishes
  spaces by color plus a two-way dash so it will not survive grayscale. It also imports
  `SAE_CONFIG` / `TRAINER_CLASS_MAP` from `spade.figures.karvonen_plot` for color, which
  is a second style source alongside `paper_style`. All of this is pre-existing and none
  of it changes a number, so it is a rewrite for whoever owns the figure pass, not a port
  bug. Flagging it because the guidelines say to.
- **`batched_apply` is imported and never used** in `manifold_capture/spaces.py`. Dead in
  the source too — callers import it from the shared module directly. Kept for fidelity;
  safe to delete.
- **`subspace_capture.main` reads `ve` and `recon` across loop iterations.** In the
  centroid-curve block, `ve` is only assigned when `entry is not None`, and the
  `metrics.append` that follows reads `f"{ve:.6f}" if entry is not None else ""` — correct,
  but it leans on `recon`/`ve` staying bound from earlier in the same space's iteration.
  Untouched.
- **`capture_regimes` writes a header-only CSV when `out_rows` is empty**, using a
  hardcoded fallback field list that duplicates the row keys. Untouched.

## Not ported

- `analysis/manifold_capture/figures/` (~190 PNGs). Generated output, and MIGRATION open
  question 10 (tracked-figure policy) has not been answered. `make_figures.py` recreates
  them from `results/manifold_capture/`.
- `analysis/manifold_capture/.gitkeep`, deleted now that the directory has real contents.
- The SLURM driver. The README's run line now points at
  `scripts/slurm/studies/manifold_capture/submit_manifold_capture.sh`, the path
  ARCHITECTURE assigns it; that file belongs to the scripts wave and does not exist yet.
  The README's findings are otherwise untouched.
