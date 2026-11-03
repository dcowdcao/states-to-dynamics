"""Interactive Othello board + live SAE feature viewer.

Serves static/index.html: the user places pieces on a legality-enforced board;
each move runs OthelloGPT, takes the layer-5 resid_post activation at the
latest position, encodes it with the selected seed-42 SAE, and returns the
top-k feature activations.

Run: python viewer/server.py   then open http://localhost:8420
"""

import os
from pathlib import Path

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from spade.common.config import CTX_LEN, DICT_SIZE, HF_REVISIONS, HOOK_POINT, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.evaluation.karvonen_adapter import (
    StandardizedDictionary, load_karvonen_sae, load_standardization_params)
from spade.export.viewer_weights import roster
from circuits.othello_engine_utils import OthelloBoardState, stoi

DEFAULT_TOP_K = 20

# The same pinned roster the browser build ships — see spade.export.viewer_weights.ROSTER.
ENTRIES = {e.config.replace("Trainer-othello", ""): e.ckpt_dir for e in roster()}
DEVICE = select_device()

_model = None
_saes: dict[str, StandardizedDictionary] = {}


def model():
    global _model
    if _model is None:
        from transformer_lens import HookedTransformer
        force_ipv4()
        torch.set_grad_enabled(False)
        _model = HookedTransformer.from_pretrained(
            MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=DEVICE)
        _model.eval()
    return _model


def sae(config: str) -> StandardizedDictionary:
    if config not in _saes:
        ckpt_dir = str(ENTRIES[config])
        ae, _ = load_karvonen_sae(ckpt_dir, device=DEVICE)
        params = load_standardization_params(ckpt_dir, device=DEVICE)
        mean = torch.tensor(params["mean"], device=DEVICE)
        _saes[config] = StandardizedDictionary(ae, mean, params["scale"]).to(DEVICE)
    return _saes[config]


def replay(moves: list[int]) -> OthelloBoardState:
    board = OthelloBoardState()
    for m in moves:
        if not (0 <= m < 64) or board.tentative_move(m) == 0:
            raise HTTPException(400, f"illegal move {m} after {board.history}")
        board.umpire(m)
    return board


class StateRequest(BaseModel):
    moves: list[int]          # board positions 0-63 in play order
    sae: str
    top_k: int = DEFAULT_TOP_K


app = FastAPI()


@app.get("/api/saes")
def list_saes():
    return [{"id": name, "display": name} for name in ENTRIES]


@app.post("/api/state")
def state(req: StateRequest):
    if req.sae not in ENTRIES:
        raise HTTPException(400, f"unknown SAE {req.sae!r}")
    board = replay(req.moves)

    valids = board.get_valid_moves()
    next_color = board.next_hand_color
    if valids and board.tentative_move(valids[0]) == 2:
        next_color *= -1  # forfeit: get_valid_moves() returned the opponent's moves
    resp = {
        "board": [int(v) for v in board.state.flatten()],  # 1 black, -1 white, 0 empty
        "next_player": "black" if next_color == 1 else "white",
        "legal_moves": valids,
        "features": [],
        "l0": None,
        "dict_size": DICT_SIZE,
    }
    if not req.moves:
        return resp

    # Model context is 59; only the 60th move of a finished game overflows, in
    # which case we drop the oldest move so the latest one stays the last token.
    tokens = torch.tensor([[stoi[m] for m in req.moves[-CTX_LEN:]]], device=DEVICE)
    _, cache = model().run_with_cache(tokens, names_filter=HOOK_POINT)
    act = cache[HOOK_POINT][0, -1]

    feats = sae(req.sae).encode(act[None]).squeeze(0)
    k = min(req.top_k, int((feats > 0).sum()))
    top = torch.topk(feats, k)
    resp["features"] = [{"index": i, "activation": v}
                        for i, v in zip(top.indices.tolist(), top.values.tolist())]
    resp["l0"] = int((feats > 0).sum())
    return resp


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8420")))
