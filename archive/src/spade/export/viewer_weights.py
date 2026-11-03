"""Export OthelloGPT (layers 0-5) + the 8 viewer SAEs to fp16 binaries for the browser
build, plus reference activations for test_inference.mjs.

Only layers 0-5 are exported: the viewer reads blocks.5.hook_resid_post, so layers 6-7 and
the unembed never run. Writes viewer/web/weights/{model.bin, sae_*.bin, manifest.json,
reference.json}.

Two invariants. KIND_BY_CLASS keys on the autoencoder *class name*, so a class rename
fails here with a KeyError rather than exporting a wrong number. And standardization is
baked in here and re-applied in web/sae.js: three independent implementations of
scale*(x-mean) exist (adapter, exporter, JS), and viewer/web/test_inference.mjs against
the exported reference.json is the only thing keeping them in agreement. Run it before
every upload.

Run: python -m spade.export.viewer_weights
"""

import json
from pathlib import Path

import torch

from spade.common.paths import add_karvonen_to_path, repo_root

add_karvonen_to_path()

from spade.common.config import CTX_LEN, HF_REVISIONS, HOOK_POINT, LAYER, MODEL_NAME
from spade.common.net import force_ipv4
from spade.evaluation.karvonen_adapter import (
    StandardizedDictionary, load_karvonen_sae, load_standardization_params)
from spade.saes.registry import seed42_entries
from circuits.othello_engine_utils import OthelloBoardState, stoi

OUT_DIR = repo_root() / "viewer" / "web" / "weights"
N_REF_MOVES = 30
REF_TOP_K = 10

# The deployed roster, in dropdown order: the 8 baseline seed-42 configs with the excluded
# SpaDE gamma=1e-2 checkpoint replaced, in its own slot, by the default
# SpaDE-lam1e-03-gam1e-04 (checkpoints/spade_revival/). Pinned by name rather than read off
# seed42_entries(), which returns nine.
ROSTER = [
    "PAnnealTrainer-othello",
    "StandardTrainer-othello",
    "GatedSAETrainer-othello",
    "TopKTrainer-othello",
    "GatedAnnealTrainer-othello",
    "SpaDE-lam1e-03-gam1e-04",
    "BatchTopKTrainer-othello",
    "CosineSpaDETrainer-othello",
]

KIND_BY_CLASS = {
    "AutoEncoder": "standard",
    "GatedAutoEncoder": "gated",
    "AutoEncoderTopK": "topk",
    "BatchTopKSAE": "batch_topk",
    "SpaDEAutoEncoder": "spade",
    "CosineSpaDEAutoEncoder": "cosine_spade",
}

TENSORS_BY_KIND = {
    "standard": ["encoder.weight", "encoder.bias", "bias"],
    "gated": ["encoder.weight", "decoder_bias", "r_mag", "gate_bias", "mag_bias"],
    "topk": ["encoder.weight", "encoder.bias", "b_dec"],
    "batch_topk": ["encoder.weight", "encoder.bias", "b_dec"],
    "spade": ["Ae"],
    "cosine_spade": ["Ae"],
}


def roster():
    """ROSTER, resolved against the registry in ROSTER order. KeyError if one is missing."""
    by_config = {e.config: e for e in seed42_entries()}
    return [by_config[config] for config in ROSTER]


def write_bin(path: Path, tensors: dict[str, torch.Tensor]) -> dict:
    """Concatenate tensors as little-endian fp16; return name -> {offset, shape}."""
    index, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        blob = tensor.detach().to(torch.float32).numpy().astype("<f2").tobytes()
        index[name] = {"offset": offset, "shape": list(tensor.shape)}
        blobs.append(blob)
        offset += len(blob)
    path.write_bytes(b"".join(blobs))
    return index


def export_model(model) -> dict:
    cfg = model.cfg
    assert cfg.normalization_type == "LNPre" and cfg.act_fn == "gelu"
    assert cfg.positional_embedding_type == "standard" and cfg.use_attn_scale
    sd = model.state_dict()
    names = ["embed.W_E", "pos_embed.W_pos"] + [
        f"blocks.{l}.{p}"
        for l in range(LAYER + 1)
        for p in ["attn.W_Q", "attn.b_Q", "attn.W_K", "attn.b_K", "attn.W_V",
                  "attn.b_V", "attn.W_O", "attn.b_O",
                  "mlp.W_in", "mlp.b_in", "mlp.W_out", "mlp.b_out"]]
    index = write_bin(OUT_DIR / "model.bin", {n: sd[n] for n in names})
    return {
        "file": "model.bin",
        "tensors": index,
        "cfg": {"n_layers": LAYER + 1, "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                "d_head": cfg.d_head, "d_mlp": cfg.d_mlp, "n_ctx": cfg.n_ctx,
                "d_vocab": cfg.d_vocab, "eps": cfg.eps},
    }


def export_sae(entry) -> dict:
    ae, _ = load_karvonen_sae(str(entry.ckpt_dir), device="cpu")
    kind = KIND_BY_CLASS[type(ae).__name__]
    std = load_standardization_params(str(entry.ckpt_dir), device="cpu")
    display = entry.config.replace("Trainer-othello", "")

    sd = ae.state_dict()
    tensors = {n: sd[n] for n in TENSORS_BY_KIND[kind]}
    tensors["std_mean"] = torch.tensor(std["mean"])
    index = write_bin(OUT_DIR / f"sae_{display}.bin", tensors)

    scalars = {"std_scale": std["scale"]}
    if kind == "topk":
        scalars["k"] = int(ae.k)
    if kind == "batch_topk":
        scalars["threshold"] = float(ae.threshold)
    if kind == "spade":
        scalars["lambda"] = float(ae.lambda_val)
    return {"id": entry.config, "display": display, "kind": kind,
            "file": f"sae_{display}.bin", "tensors": index, "scalars": scalars,
            "dict_size": ae.dict_size}


def reference_game() -> list[int]:
    board, moves = OthelloBoardState(), []
    while board.get_valid_moves() and len(moves) < N_REF_MOVES:
        m = board.get_valid_moves()[0]
        board.umpire(m)
        moves.append(m)
    return moves


def export_reference(model, sae_manifests) -> dict:
    moves = reference_game()
    tokens = torch.tensor([[stoi[m] for m in moves[-CTX_LEN:]]])
    _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
    resid = cache[HOOK_POINT][0, -1]

    saes = {}
    for m in sae_manifests:
        entry = next(e for e in roster() if e.config == m["id"])
        ae, _ = load_karvonen_sae(str(entry.ckpt_dir), device="cpu")
        std = load_standardization_params(str(entry.ckpt_dir), device="cpu")
        wrapped = StandardizedDictionary(ae, torch.tensor(std["mean"]), std["scale"])
        feats = wrapped.encode(resid[None]).squeeze(0)
        top = torch.topk(feats, REF_TOP_K)
        saes[m["display"]] = {
            "l0": int((feats > 0).sum()),
            "top": [[i, v] for i, v in zip(top.indices.tolist(), top.values.tolist())],
        }
    return {"moves": moves, "tokens": tokens[0].tolist(),
            "resid_post5": resid.tolist(), "saes": saes}


def main():
    from transformer_lens import HookedTransformer

    force_ipv4()
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device="cpu")
    model.eval()
    manifest = {"model": export_model(model),
                "saes": [export_sae(e) for e in roster()],
                "cos_eps": 1e-8}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest))
    (OUT_DIR / "reference.json").write_text(json.dumps(export_reference(model, manifest["saes"])))

    for f in sorted(OUT_DIR.glob("*")):
        print(f"{f.name:28} {f.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
