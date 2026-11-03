# Feature viewer

Place discs on an Othello board and watch the top SAE feature activations update per move.

`web/` is the live deployment: a static Hugging Face Space with **no Python at runtime**.
The forward pass through layers 0–5, the six SAE encoder kinds and the Othello engine are
all hand-written JavaScript reading fp16 weights exported from the checkpoints. `web/`
*is* the Space root, so `web/README.md` is the Space card — its YAML front matter is
load-bearing, not documentation.

`server.py` + `static/` is the older FastAPI variant, kept for running against checkpoints
on the cluster over an SSH tunnel. It is the only consumer of fastapi/uvicorn in the repo.

The one seam into the research code is `spade.export.viewer_weights`, which regenerates
`web/weights/` (gitignored, ~100 MB) and `reference.json`. `web/test_inference.mjs` checks
the JavaScript against that reference and is the gate before any upload.

The Docker-Space path (`build_space.py`, `space_src/`, `hf_space/`) is dead and is not
ported. Its pinned CPU runtime versions are recorded in `docs/` before deletion.

Board legality is enforced: a freely-placed position is out of distribution for the model
and unrepresentable in the move-sequence tokenization.
