"""Single device-selection policy: cuda (clusters) → mps (mac) → cpu."""

import torch


def select_device(prefer: str = "auto") -> str:
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
