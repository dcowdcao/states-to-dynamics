"""`ae.encode()` inside a real nnsight trace, for one checkpoint of each provenance.

This is the exact call Karvonen's classifier stage makes — `circuits/utils.py`,
`collect_activations_batch` encodes the layer-5 residual inside `model.trace(...)` and
indexes the result as `[:, :, dims]`. Encoding outside a trace proves nothing: the
`sparsemax` package only fails on a proxy tensor, so a SpaDE checkpoint that passes every
plain-tensor test still dies the moment his stage runs.

The test also pins the replacement projection to the package the checkpoints were trained
under: same support, same values, same L0, on the same real activations.

    python -m spade.sae_loader.test_trace_safe_encode
"""

import argparse
import os

import torch as t
from sparsemax import Sparsemax

from spade.common.paths import add_karvonen_to_path
from spade.sae_loader.load_sae import load_sae
from spade.sae_loader.trace_safe_spade import sparsemax

add_karvonen_to_path()
import circuits.eval_sae_as_classifier as eval_sae  # noqa: E402
import circuits.utils as cutils  # noqa: E402

MODEL_NAME = "Baidicoot/Othello-GPT-Transformer-Lens"
LAYER = 5
N_GAMES = 25
# His stage indexes the code by a list of alive features; any list exercises the same path.
N_DIMS = 8

_MULTISEED = os.environ.get("SPADE_MULTISEED_DIR", "/n/netscratch/ba_lab/Lab/dcao/spade_multiseed")
_RELEASED = os.environ.get("SPADE_KARVONEN_RELEASED",
                           "/n/netscratch/ba_lab/Lab/dcao/karvonen_released")

DEFAULTS = {
    "karvonen": f"{_RELEASED}/othello-trained_model-layer_5-2024-07-10_topk"
                f"/othello-trained_model-layer_5-top_k/trainer_0",
    "spade": f"{_MULTISEED}/seed_43/autoencoders/trainer_5",
    "cosine": f"{_MULTISEED}/seed_43/autoencoders_cosine_spade_sweep_highgamma/trainer_0",
}


def othello_tokens(n_games: int, device: str) -> t.Tensor:
    """Real held-out games, from his own dataset builder. [n_games, 59]."""
    data = eval_sae.construct_dataset(True, [], n_games, split="test", device=device,
                                      precompute_dataset=False)
    return t.tensor(data["encoded_inputs"], device=device)


def encode_in_trace(model, submodule, ae, tokens: t.Tensor, dims: t.Tensor):
    """Karvonen's `collect_activations_batch`, minus the parts that do not touch the SAE."""
    with model.trace(tokens):
        acts = submodule.output.save()
        code = ae.encode(submodule.output)
        indexed = code[:, :, dims].save()
    return acts.value, indexed.value


def check(name: str, checkpoint_dir: str, model, submodule, tokens: t.Tensor, device: str):
    ae, meta = load_sae(checkpoint_dir, device)
    dims = t.arange(N_DIMS, device=device)

    acts_BLD, indexed_BLF = encode_in_trace(model, submodule, ae, tokens, dims)
    assert len(indexed_BLF.shape) == 3, indexed_BLF.shape          # his own assertion
    assert indexed_BLF.shape == (*tokens.shape, N_DIMS), indexed_BLF.shape
    assert acts_BLD.shape == (*tokens.shape, meta.activation_dim), acts_BLD.shape

    with t.no_grad():
        code_BLF = ae.encode(acts_BLD)
    assert code_BLF.shape == (*tokens.shape, meta.dict_size), code_BLF.shape
    assert t.equal(code_BLF[:, :, dims], indexed_BLF), "in-trace code differs from out-of-trace"

    l0 = (code_BLF != 0).float().sum(-1).mean().item()
    l1 = code_BLF.norm(p=1, dim=-1).mean().item()
    print(f"{name:12} {meta.dict_class:24} standardized={str(meta.standardized):5} "
          f"acts {tuple(acts_BLD.shape)} -> code {tuple(code_BLF.shape)} -> "
          f"indexed {tuple(indexed_BLF.shape)}")
    print(f"{'':12} L0 = {l0:.2f}   l1_loss = {l1:.6f}")
    return ae, meta, acts_BLD, code_BLF, l0


def check_against_reference(name: str, ae, acts_BLD: t.Tensor, code_BLF: t.Tensor, l0: float):
    """The replacement projection must be the package's, on these very activations.

    Same scores, so the only thing under test is the projection the checkpoint was
    trained under against the one `load_sae` evaluates it with.
    """
    with t.no_grad():
        scores = ae.dictionary._scores(ae.scale * (acts_BLD - ae.mean)).float()
        reference = Sparsemax(dim=-1)(scores)
    ref_l0 = (reference != 0).float().sum(-1).mean().item()
    max_abs = (reference - code_BLF).abs().max().item()
    same_support = t.equal(reference != 0, code_BLF != 0)

    assert same_support, "support differs from the sparsemax package"
    assert max_abs < 1e-6, max_abs
    assert abs(ref_l0 - l0) < 1e-9, (ref_l0, l0)
    print(f"{name:12} vs sparsemax package: max|diff| = {max_abs:.3e}  "
          f"support identical = {same_support}  L0 {l0:.4f} vs {ref_l0:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        p.add_argument(f"--{key}", default=default)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    tokens = othello_tokens(N_GAMES, args.device)
    model = cutils.get_model(MODEL_NAME, args.device)
    submodule = cutils.get_submodule(MODEL_NAME, LAYER, model)
    print(f"tokens {tuple(tokens.shape)} on {args.device}\n")

    for name in DEFAULTS:
        checkpoint_dir = getattr(args, name)
        ae, meta, acts, code, l0 = check(name, checkpoint_dir, model, submodule, tokens,
                                         args.device)
        if meta.standardized:
            check_against_reference(name, ae, acts, code, l0)
        print()

    print("OK")


if __name__ == "__main__":
    main()
