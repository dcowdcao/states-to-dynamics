"""Single source of truth for model / dataset / geometry constants.

Everything downstream reads from here so a value like the layer index or dict size is
defined once. Port the exact values from the original src/common/config.py.
"""

# MODEL_NAME = "..."          # OthelloGPT (Li et al. TransformerLens port)
# DATASET = "..."
# LAYER = 5                    # residual stream block index
# HOOK_POINT = "blocks.5.hook_resid_post"
# D_MODEL = 512
# CTX_LEN = ...
# EXPANSION = 16
# DICT_SIZE = D_MODEL * EXPANSION   # 8192
# DEFAULT_LR = ...
# DEFAULT_WARMUP_STEPS = ...
