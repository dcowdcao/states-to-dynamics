# Port notes — `spade.spaces` + the rules study

Zone: `src/spade/spaces.py`, `src/spade/rules/{build_pairs,extract_acts,flip_labels,embed}.py`,
`analysis/rules_manifolds/`. Source: `src/rules/*` and `analysis/rules_manifolds/*` in
`Spade_Research_David_Sumedh_Ba`. A move, not a rewrite: bodies are byte-identical to the
source apart from the import block and the four path/name changes listed below.

## The split of `rules/embed.py`

The source file mixed a generic space API with the rules-specific embedding run. Grepping
the source repo for importers of `rules.embed` gives ten call sites, all of which want only
the generic half:

- `src/manifold_capture/`: `spaces.py` (`load_space`, `batched_apply`), `subspace_capture`,
  `manifold_views`, `landmarks`, `tuning_curves`, `feature_geometry`, `receptive_fields`
  (`batched_apply`), `concepts` (`center_by_t`) — eight modules.
- `src/bracket_flip/sae_arm.py` and `analysis/bracket_flip/sae_pca_figure.py`
  (`load_space`, `representative_spaces`, `batched_apply`).

Nothing outside the rules study imports `VIEWS`, `view_masks`, `isomap_subsample`,
`embed_view` or `main`, so the split leaves nothing load-bearing behind.
`manifold_capture/manifold_views.py` re-implements a balanced subsample "in the spirit of"
`isomap_subsample` but does not import it; that stays a separate implementation.

`MIN_T_BIN` and `ENCODE_BATCH` moved with `center_by_t` / `batched_apply`;
`SUBSAMPLE_SEED`, `ISOMAP_NEIGHBORS`, `ISOMAP_PRE_DIMS` and `VIEWS` stayed with the run.

## Contract other agents depend on

```
spade.spaces.load_space(name: str, entry, device: str) -> callable   # raw batch -> features
spade.spaces.representative_spaces() -> list[(name, SAEEntry | None)]
spade.spaces.batched_apply(fn, acts: Tensor, device: str) -> Tensor  # returns on CPU
spade.spaces.center_by_t(feats, t, legal) -> Tensor
spade.spaces.EXCLUDED_SAE = {"SpaDETrainer-othello"}
```

`load_space` reads `entry.ckpt_dir`. The source `SAEEntry` field is `ckpt_dir`; the
skeleton stub for `spade/saes/registry.py` names it `checkpoint_dir`. The registry port must
keep `ckpt_dir` or `load_space` breaks. Same for `entries()`, which `representative_spaces`
calls and which must keep returning entries carrying `.config` and `.seed`.

`EXCLUDED_SAE` is defined here and applied nowhere in this zone: `representative_spaces()`
still returns the excluded config, exactly as the source did. Straightness and
violation-divergence import the constant and filter their own rosters. **Open for David:**
if "excluded everywhere" is meant to include the rules embeddings, the viewer and
`bracket_flip/sae_arm.py`, then `representative_spaces()` should filter — that is a
behaviour change, so it was not made silently.

## Changes beyond imports

1. `study_dir("rules")` replaces the source's `rules_dir()`. `$SPADE_RULES_DIR` and the
   default `/n/netscratch/ba_lab/Lab/$USER/spade_rules` are unchanged, byte for byte,
   because the real intermediates are already on disk there.
2. `load_space` reads the standardization params for `raw` / `random_proj` / `random_topk`
   from `checkpoints/main/trainer_0`, where the source read `autoencoders/trainer_0`. This
   is the ARCHITECTURE.md artifact rename, not a logic change; `standardization_params.json`
   must land at the `checkpoints/main/` level (the adapter's parent-dir fallback).
3. `T_MIN`, `T_MAX`, `stream_games` and `classify_prefix` come from `spade.othello.games`.
   `MAX_LEN = T_MAX + 1` stays in `build_pairs`, as does the module-level
   `assert all(stoi[sq] == i + 1 ...)`: it guards the `token = vocab index + 1` mapping that
   only `build_pairs`' `by_cat` relies on.
4. Usage lines in docstrings now say `python -m spade.rules.<step>`.

Import side effects are preserved and eager, not lazy: `build_pairs` calls `force_ipv4()`
and `add_karvonen_to_path()` at module level before importing `circuits`, `flip_labels`
calls `add_karvonen_to_path()`, `extract_acts` calls `force_ipv4()` before the HF model
download. These stay even though importing `spade.othello.games` is documented to do the
same thing — the source ran them explicitly and the ordering matters.

## Ported as-is, questionable

- `embed.py` imports `CATEGORIES` from the taxonomy and never uses it. Dead in the source
  too; kept rather than silently cleaned.
- `center_by_t` is visualization-grade: no leave-own-prefix-out, so it must not be reused
  for distance metrics. The source docstring says so and is kept verbatim.
- `load_space` takes the literal string `"sae"` for any checkpoint space, with the caller
  writing `load_space(name if entry is None else "sae", entry, device)` at all four call
  sites. Ugly, and reproduced exactly.
- `representative_spaces()` calls `entries()`, which verifies all 65 checkpoints including
  the netscratch seeds 43–46. It therefore fails if netscratch is unmounted, even for the
  `raw` space. `manifold_capture/spaces.py` works around this with private registry
  internals; that stays the manifold-capture agent's problem.

## `analysis/rules_manifolds/`

`make_figures.py` and `README.md` are ported; the README's findings and caveats are
unchanged, only the pipeline paths (`scripts/slurm/studies/rules/submit_rules.sh`,
`spade.rules.*`).

Not copied: `embeddings/` (17 MB, 20 tracked files) and `figures/` (139 MB, ~180 tracked
PNGs). `make_figures.py` cannot run until `embeddings/` is copied across by hand — it is the
committed input, not a regenerable artifact, since regenerating it needs the netscratch
activations and a GPU. Whether those files are re-committed under the new path is
MIGRATION.md open question 10, so the decision was left to David.

`make_figures.py` needs `spade.figures.embedding_panels.class_cloud` / `continuous_cloud`
and `spade.figures.paper_style.{CATEGORY_STYLE, INK, RULE_STYLE, apply_style, rule_key,
save_figure}`, with `save_figure(fig, stem, camera_ready, dpi=)` and
`rule_key(fig, rules)`. The skeleton stubs name the panel helpers `class_panel` /
`confound_panel`; the figures agent's faithful port keeps the source names.

## Verification

`python -m py_compile` passes on all seven files (CPython 3.11.15, `spade_env`).
`PYTHONPATH=src python -c "import spade.spaces"` still fails, and only inside modules other
agents own — `spade.common.config`, `common.net`, `evaluation.karvonen_adapter`,
`saes.registry`, `othello.games`, `othello.taxonomy`, `figures.*` are all still pseudocode
stubs. With minimal fakes injected for those, every module in this zone imports and exposes
its public symbols. No run against real data was possible: no GPU, no cluster, and steps 1–3
need both.
