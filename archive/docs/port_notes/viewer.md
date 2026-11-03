# Port notes — viewer/ and src/spade/export/

Wave 7. What moved, what was dropped, what was decided, and what still needs David's
call. Source: `../Spade_Research_David_Sumedh_Ba/feature_viewer/`.

## What landed

| Source | Here | Change |
|---|---|---|
| `web/{index,main,model,sae,othello,weights_io}.*`, `package.json` | `viewer/web/` | byte-identical |
| `web/test_inference.mjs` | `viewer/web/` | one comment path |
| `web/README.md` (the HF Space card) | `viewer/web/README.md` | one link path |
| `static/index.html` | `viewer/static/` | byte-identical |
| `server.py` | `viewer/server.py` | imports, roster, `VIEWER_CKPT_DIR` removed |
| `web/export_weights.py` | `src/spade/export/viewer_weights.py` | imports, output path, roster |
| `GUIDE.md` | `viewer/GUIDE.md` | paths, roster paragraph, rescued pins |
| `build_space.py`, `space_src/`, `hf_space/` | not ported | dead Docker path |
| `README.md` | not copied | see *README* below |

`viewer/web/.gitkeep` is gone: it held an empty directory that now holds the app.

`.gitignore` was already correct for this — `viewer/web/weights/` is ignored and the
`*.html` negations keep both `viewer/web/index.html` and `viewer/static/index.html`
tracked. Verified with `git check-ignore` and `git ls-files`.

## Verification

`python -m py_compile` passes on both Python files (CPython 3.11.15, `spade_env`).

**The JavaScript is unverified.** There is no `node` on this machine — not on `PATH`, not
in `spade_env`, and no `nodejs` module on FASRC — so neither `node --check` nor
`test_inference.mjs` could run. The JS was read carefully instead, and it is byte-identical
to the deployed and previously-verified source apart from one comment, so the risk is
confined to whether the *export* still produces what the JS expects. That is exactly what
`node viewer/web/test_inference.mjs` answers, and it has to be run before any upload.

Nothing was exported and nothing was uploaded: no GPU, no HF credentials, and the live
Space was not touched.

## The two contracts

**Standardization** (SAEs trained on `scale*(x - mean)`) is implemented three times and all
three are untouched by this port:

1. `spade.evaluation.karvonen_adapter.StandardizedDictionary` — the research path.
2. `spade.export.viewer_weights.export_sae` — bakes `std_mean` into each `sae_*.bin` and
   `std_scale` into the manifest scalars.
3. `viewer/web/sae.js`, first line of `encode`: `xs[i] = scale * (x[i] - mean[i])`.

`test_inference.mjs` against the exported `reference.json` is the only thing that keeps
them in agreement, and `reference.json` itself is produced through (1). So the check is
(1) vs (3) with (2) as the transport — a bug shared between (1) and (3) would pass.

**`KIND_BY_CLASS`** still lists all six encoder kinds, and every one is reachable from the
new roster, so a redeploy exercises all six:

| class | kind | roster entries that load it |
|---|---|---|
| `AutoEncoder` | `standard` | PAnneal, Standard |
| `GatedAutoEncoder` | `gated` | GatedSAE, GatedAnneal |
| `AutoEncoderTopK` | `topk` | TopK |
| `BatchTopKSAE` | `batch_topk` | BatchTopK |
| `SpaDEAutoEncoder` | `spade` | SpaDE-lam1e-03-gam1e-04 |
| `CosineSpaDEAutoEncoder` | `cosine_spade` | CosineSpaDE |

`TENSORS_BY_KIND` covers the same six. The class names survive the port unchanged, which
is what makes the keying work; `spade_revival` reaches `SpaDEAutoEncoder` because the
adapter dispatches on the `SpaDE` substring of its `wandb_name` and matches `CosineSpaDE`
first.

Checked while reading: SpaDE's λ is exported as `float(ae.lambda_val)`, and `lambda_val` is
`softplus(lambda_pre)` where `lambda_pre` is an `nn.Parameter` and therefore restored from
the checkpoint. The export carries the *trained* λ, not the constructor default — which
matters now that the roster's SpaDE entry is a λ=1e-3 run rather than the default.

## The roster change — needs sign-off before it reaches the public page

`ROSTER` in `src/spade/export/viewer_weights.py` is an explicit list of eight config names.
The registry now returns **nine** seed-42 entries, and one of them is the excluded
γ=1e-2 checkpoint, so iterating it would ship nine including a checkpoint that
ARCHITECTURE.md says is excluded from the viewer. The explicit list also pins dropdown
order, which the old code got from dict insertion order in the registry.

`SpaDE-lam1e-03-gam1e-04` sits in slot 6, the slot `SpaDETrainer-othello` used to hold.

Consequences to be aware of:

- **The live Space is the old build.** It still serves the γ=1e-2 checkpoint, under the
  label `SpaDE`. Nothing here changed that; a redeploy would.
- The display name is derived as `config.replace("Trainer-othello", "")`, so the new entry
  appears in the dropdown as **`SpaDE-lam1e-03-gam1e-04`**, not `SpaDE`, and its weight file
  is `sae_SpaDE-lam1e-03-gam1e-04.bin`. Longer than every other label. Renaming it to
  `SpaDE` would need a special case in the display rule — not added, since a silent rename
  would make the new and old builds indistinguishable on the page.
- `viewer/web/README.md`, the Space card, lists the eight SAEs by name including `SpaDE`.
  It is accurate for what is deployed and becomes wrong the moment a redeploy happens. It
  was left alone deliberately: the card ships to HF, and editing it ahead of the upload
  makes the repo describe a build that does not exist. **Update it in the same change as
  the redeploy.**
- MIGRATION.md open question 13 (re-export and redeploy, or leave the Space frozen) is
  still open and this port does not answer it.

## Judgment calls

**`SAEEntry.checkpoint_dir`.** The source calls the field `ckpt_dir`; the registry skeleton
at `src/spade/saes/registry.py` declares `checkpoint_dir`. The exporter and the server use
`checkpoint_dir`, following the skeleton, since `saes/` is owned by another agent in this
wave. If that agent ports the field as `ckpt_dir`, two lines in `export_sae`, one in
`export_reference` and one in `server.py` need the rename — an `AttributeError` on first
run, not a silent wrong answer.

The port also assumes the adapter keeps `device=` on `load_karvonen_sae` and
`load_standardization_params`; the skeleton's pseudocode shows only `checkpoint_dir`.

**`server.py` shares the exporter's roster.** It imports `roster()` from
`spade.export.viewer_weights` rather than calling `seed42_entries()` itself. Otherwise the
cluster viewer would serve the excluded checkpoint while the browser build does not, which
is the exact split ARCHITECTURE.md says the exclusion decision exists to end. The direction
is a little odd — a deployable importing an exporter — and the alternative is to move
`ROSTER` somewhere more neutral. Flagging rather than deciding.

`ROSTER` is a positive list, not a restatement of `spade.spaces.EXCLUDED_SAE`; it fixes
which eight ship and in what order, which the exclusion set does not express. The two are
not duplicated rules, but they do have to agree, and nothing enforces that.

**`checkpoints()` was inlined.** With the `VIEWER_CKPT_DIR` branch gone the function was a
one-line dict comprehension with one caller, so it became the `ENTRIES` assignment.

**README.** The source's `feature_viewer/README.md` was not copied: `viewer/README.md`
already exists, is protected from editing, and covers the same ground better. One thing the
old README had that the new one does not is the local-server invocation, so it was rescued
into `viewer/GUIDE.md` verbatim:

```
python viewer/server.py                   # spade_env; open http://localhost:8420
ssh -L 8420:localhost:8420 <login-node>   # tunnel from your machine
```

**The one JS edit.** `test_inference.mjs` line 2 had `Run: node
feature_viewer/web/test_inference.mjs`. Applying the same old-directory rule the Space card
got, it now reads `viewer/web/test_inference.mjs`. Nothing executable changed; `diff`
against the source shows exactly one line here and one in `README.md`.

**Stale references left in place**, because they are provenance comments inside files that
were to be ported unchanged:

- `test_inference.mjs` line 1–2 credits `export_weights.py`, now
  `spade/export/viewer_weights.py`.
- `sae.js` line 1–2 says the encoders were ported from `circuits/dictionary_learning` and
  `src/models`; `src/models` is now `src/spade/saes`.

## Ported as-is, worth knowing

None of these were changed. They are in the deployed and verified build.

- `main.js` prints the L0 denominator as `manifest.saes[0].dict_size` — the *first* SAE's
  dictionary size, not the selected one's. Harmless today because all eight are 8192, wrong
  the day one is not.
- `main.js` hardcodes `CTX_LEN = 59` and `TOP_K = 20`, duplicating
  `spade.common.config.CTX_LEN` and the server's `DEFAULT_TOP_K`. No mechanism keeps them
  in step; the exporter does not write either into the manifest.
- `sae.js` `sparsemax` and `topFeatures` each sort all 8192 features per call, as does the
  `topk` branch. Fine at this size, but it is three full sorts per move.
- `export_reference` calls `roster()` once per SAE inside its loop, and the registry
  verifies every checkpoint on construction, so the export re-stats every checkpoint eight
  times. It also re-loads each `ae.pt` that `export_sae` already loaded. Slow, not wrong;
  faithful to the source.
- `export_model` uses bare `assert` for the model-config preconditions, which vanish under
  `python -O`.
- The manifest has no format version, so an old cached `manifest.json` against new JS fails
  by `undefined` rather than by a clear error.
- `server.py` computes `(feats > 0).sum()` twice and takes `k = min(top_k, L0)`, so a fully
  dense encode still returns only `top_k` bars.
- `viewer/static/index.html` is a second, divergent copy of the UI that talks to FastAPI.
  MIGRATION.md open question 12 asks whether it and `server.py` are kept at all; this port
  keeps them, per ARCHITECTURE.md's tree.

## Rescued before deletion

`space_src/requirements.txt` — the only written record of a verified CPU-only runtime — is
now quoted in full in `viewer/GUIDE.md`. That closes MIGRATION.md open question 11's
precondition; the deletion itself is the user's call and the source repo is untouched.
