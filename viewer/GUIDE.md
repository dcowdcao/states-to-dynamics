# Feature viewer: how it works, how to play, how to redeploy

## Play with it (anyone, zero setup)

Open **https://dcaodcow-othello-sae-viewer.static.hf.space** (Space page:
https://huggingface.co/spaces/dcaodcow/othello-sae-viewer).

1. Wait for the model download (~38 MB, progress shown top-right of the
   features panel).
2. Pick an SAE from the dropdown (8 seed-42 checkpoints; each downloads
   ~8 MB on first selection).
3. Click any highlighted square to place a piece — legal Othello moves only.
   The bars update per move: top-20 feature activations of the selected SAE
   at layer-5 resid_post, plus L0. Undo/Reset to explore alternate lines.

Everything runs in the browser — after the weight downloads there is no
server and no network traffic.

## The approach (why client-side)

OthelloGPT is small enough (25M params) that a hand-rolled JS forward pass is
interactive: layers 0–5 only (the hook point is `blocks.5.hook_resid_post`),
KV-cached so each placed piece computes one token (~100 ms). SAE encoders
(Standard/PAnneal, Gated, TopK, BatchTopK, SpaDE, CosineSpaDE) are ported to
JS with standardization baked in. Weights are fp16 exports of the real
checkpoints. HF hosts it as a **static Space** — free, unlike Docker/Gradio
Spaces which now require a PRO subscription (the reason a FastAPI Space was
abandoned; `server.py` remains for cluster use).

Correctness is checked against torch, not by eye — see Verify below.

The roster is deliberately the 8 **baseline** seed-42 configs, not sweep-best
(user decision, 2026-07-22): the distance-SpaDE on the site (frac_rec 0.92,
coverage 0.18) is far below the tuned sweep configs (e.g. lam1e-07-gam5e-03:
0.99/0.32, seeds 43+ on netscratch), and CosineSpaDE's gamma sweep is flat so
its γ=1.0 entry is representative.

**The roster changed on 2026-08-03 and the change is not deployed.** The SpaDE
γ=1e-2 seed-42 checkpoint is now excluded everywhere as degenerate, and the
default SpaDE configuration is λ=1e-3/γ=1e-4 (`checkpoints/spade_revival/`),
which takes its slot. The roster is still eight. `ROSTER` in
`src/spade/export/viewer_weights.py` holds it as an explicit list of config
names — that list, not the registry, decides what ships and in what order;
the registry now carries nine seed-42 entries. Change the roster there, then
re-export, re-verify and redeploy. **The live Space is still the old build and
still serves the γ=1e-2 checkpoint under the name `SpaDE`.**

## Redeploy after retraining or code changes

From the repo root, in `spade_env`:

```
python -m spade.export.viewer_weights          # regenerate viewer/web/weights/ (~105 MB)
node viewer/web/test_inference.mjs             # must be all PASS before uploading
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(
    folder_path='viewer/web', repo_id='dcaodcow/othello-sae-viewer',
    repo_type='space', ignore_patterns=['__pycache__/*'])"
```

### HF token (one-time per machine)

Uploading needs a **write** token owned by the Space owner (`dcaodcow`):
huggingface.co/settings/tokens → Create new token → type *Write* → then
`huggingface-cli login --token hf_...`. The token is cached in
`~/.cache/huggingface/token`; never commit it. Gotcha that cost us a round
trip: a *read* token passes `whoami` but 403s on repo creation/upload — if
uploads fail with 403/402, check the token type first (and note 402 on
`create_repo` for a Docker Space means the paywall, not a token problem).

## Verify (after any export or JS change)

`test_inference.mjs` compares the JS pipeline to torch reference outputs
written by the export: resid rel err ~7e-4 (fp16-limited), per-SAE top-10
rel err ~2e-4, exact L0s, 10/10 end-to-end top-k overlap. The JS board
engine was additionally cross-checked against the Python engine on 200
random games, and the KV cache verified bit-identical after undo/redo.

## Local server variant (cluster)

```
python viewer/server.py                   # spade_env; open http://localhost:8420
ssh -L 8420:localhost:8420 <login-node>   # tunnel from your machine
```

Same science, a second and divergent UI: `static/index.html` talks to FastAPI
instead of running the model in the browser. It is the repo's only consumer of
fastapi/uvicorn.

## CPU runtime pins, rescued from the abandoned Docker Space

The Docker path (`build_space.py`, `space_src/`, `hf_space/`) is not ported. Its
`requirements.txt` was the only written record of a CPU-only runtime verified to
run this pipeline end to end, so the pins are kept here:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
transformer-lens==1.17.0
transformers==4.57.6
sparsemax==0.1.9
einops==0.8.2
numpy==1.26.4
fastapi==0.139.2
uvicorn==0.51.0
matplotlib==3.10.8
tqdm==4.67.3
```

These are a CPU image's pins, not the research environment's — `spade_env` runs
the CUDA build of the same torch. `transformer-lens==1.17.0` is the load-bearing
one either way: 2.x is incompatible with the vendored `circuits`.
