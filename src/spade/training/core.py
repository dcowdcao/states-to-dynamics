"""Shared training core for every SAE type.

The per-type entrypoints (`train_<type>.py`) each build their own trainer-config dict and
call `run_training(config, ...)`. Everything type-agnostic lives here so the entrypoints
stay thin and can't drift: activation streaming, Anthropic-style standardization
(center+scale), optional data-sampled prototype init (SpaDE/CosineSpaDE only), the single
Karvonen `trainSAE()` call, multi-GPU device assignment, and checkpoint/config writing.

KEEP SHARED LOGIC HERE, NOT IN THE ENTRYPOINTS (see PROCESS.md decision).
"""

# class StandardizedBuffer:                # wraps NNsightActivationBuffer with center+scale
#     ...

# def sample_init_activations(buffer, n):  # data-sampled prototypes for SpaDE-family init
#     ...

# def run_training(config: dict, save_dir: Path, device: str, seed: int | None = None,
#                  smoke_test: bool = False, no_wandb: bool = False):
#     # t.manual_seed(seed); build buffer + standardization; inject data-init for SpaDE types;
#     # call circuits.trainSAE(...); write ae.pt + config.json + standardization_params.json
#     ...

# --- baseline trainer configs come from circuits.dictionary_learning; SpaDE/Cosine from spade.saes ---
