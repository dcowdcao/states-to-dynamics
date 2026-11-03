"""Shared constants for the OthelloGPT SpaDE experiment.

Single source of truth for the model/dataset/geometry so the training,
extraction, evaluation, and diagnostics scripts can't drift apart.
"""

from importlib.metadata import version

MODEL_NAME = "Baidicoot/Othello-GPT-Transformer-Lens"
GAMES_DATASET = "adamkarvonen/othello_45MB_games"
DATASET = "taufeeque/othellogpt"
# The HF commits every load pins: the weights, and the corpus whose stream order is the game set
# (docs/game_corpus.md). A name missing here is a KeyError at the load, not a silent `main`.
HF_REVISIONS = {
    MODEL_NAME: "30e9176f812fc31580671bc93ba9936bf2b7e08a",
    GAMES_DATASET: "caafaff9fd273cd1783e146d039e262f97c9f5f8",
}
LAYER = 5
HOOK_POINT = "blocks.5.hook_resid_post"
D_MODEL = 512
CTX_LEN = 59              # tokens per Othello game
EXPANSION = 16           # dictionary expansion factor
DICT_SIZE = EXPANSION * D_MODEL   # 8192

DEFAULT_LR = 3e-4
DEFAULT_WARMUP_STEPS = 1000


def provenance() -> dict:
    """What every manifest records about its inputs: the pinned HF commits and the libraries
    that fix the stream order (datasets), the activations (torch, transformer-lens) and the RNG."""
    return {"hf_revisions": HF_REVISIONS,
            "versions": {p: version(p) for p in ("datasets", "numpy", "torch", "transformer-lens")}}
