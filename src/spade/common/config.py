"""Shared constants for the OthelloGPT SpaDE experiment.

Single source of truth for the model/dataset/geometry so the training,
extraction, evaluation, and diagnostics scripts can't drift apart.
"""

MODEL_NAME = "Baidicoot/Othello-GPT-Transformer-Lens"
DATASET = "taufeeque/othellogpt"
LAYER = 5
HOOK_POINT = "blocks.5.hook_resid_post"
D_MODEL = 512
CTX_LEN = 59              # tokens per Othello game
EXPANSION = 16           # dictionary expansion factor
DICT_SIZE = EXPANSION * D_MODEL   # 8192

DEFAULT_LR = 3e-4
DEFAULT_WARMUP_STEPS = 1000
