"""Shared training core for every SAE type.

The per-type entrypoints (`train_<type>.py`) each build their own trainer-config dict and
hand it to `run_training`; `sweep.py` hands it a grid. Everything type-agnostic lives here
so the entrypoints stay thin and can't drift: activation streaming, Anthropic-style
standardization (center+scale), data-sampled prototype init (SpaDE/CosineSpaDE only), the
single Karvonen `trainSAE()` call, multi-GPU device assignment, and checkpoint/config
writing.

KEEP SHARED LOGIC HERE, NOT IN THE ENTRYPOINTS (see PROCESS.md decision).
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch as t
import torch.multiprocessing as mp
from tqdm import tqdm

from spade.common.config import (
    CTX_LEN,
    D_MODEL,
    DATASET,
    DEFAULT_WARMUP_STEPS,
    EXPANSION,
    LAYER,
    MODEL_NAME,
)
from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path

force_ipv4()
add_karvonen_to_path()

from circuits.utils import (
    othello_hf_dataset_to_generator,
    get_model,
    get_submodule,
    SubmoduleType,
)
from circuits.dictionary_learning.training import trainSAE
from circuits.dictionary_learning.buffer import NNsightActivationBuffer

from spade.saes.spade import SpaDETrainer
from spade.saes.spade_cosine import CosineSpaDETrainer

SAE_BATCH_SIZE = 8192
NUM_TOKENS = 300_000_000
WANDB_PROJECT = "spade-vs-karvonen-othello"

# p-annealing schedule, shared by PAnnealTrainer and GatedAnnealTrainer.
P_ANNEAL = {
    "sparsity_function": "Lp^p",
    "p_start": 1.0,
    "p_end": 0.2,
    "anneal_start": 10000,
    "anneal_end": None,  # defaults to steps
    "sparsity_queue_length": 10,
    "n_sparsity_updates": 10,
}


@dataclass(frozen=True)
class TrainParams:
    """What a trainer config is built from. `run_training` fills it in and passes it down."""

    steps: int
    activation_dim: int
    model_name: str
    layer: int
    device: str
    seed: int
    warmup_steps: int

    @property
    def dict_size(self) -> int:
        return EXPANSION * self.activation_dim

    @property
    def sparsity_warmup_steps(self) -> int:
        # Upstream default is 2000; scaled down so smoke tests (steps=100) pass the
        # sparsity_warmup_steps < steps assertion. Real runs (36k steps) still get 2000.
        return min(2000, max(1, self.steps // 2))


BuildConfigs = Callable[[TrainParams], list[dict]]


# ---------------------------------------------------------------------------
# StandardizedBuffer: wraps NNsightActivationBuffer with centering + scaling
# ---------------------------------------------------------------------------
class StandardizedBuffer:
    """
    Wraps an activation buffer to apply standardization:
        act_out = scale * (act - mean)
    where:
        mean   = E[act]                           (computed over calibration batches)
        scale  = sqrt(d / E[||act - mean||^2])    (gives unit variance per dimension)

    This is the input standardization of Anthropic's April 2024 circuits update.
    """

    def __init__(self, buffer, d_model: int, n_calibration_batches: int = 100):
        self.buffer = buffer
        self.d_model = d_model
        self.n_calibration_batches = n_calibration_batches
        self.mean = None
        self.scale = None
        self._calibration_cache = []

    def calibrate(self):
        """Compute mean and scale from calibration batches. Must be called before iteration."""
        print(f"Calibrating standardization over {self.n_calibration_batches} batches...")
        sum_act = None
        sum_sq_norm = 0.0
        total_samples = 0

        for i, act in enumerate(tqdm(self.buffer, total=self.n_calibration_batches, desc="Calibrating")):
            if i >= self.n_calibration_batches:
                break
            self._calibration_cache.append(act)
            batch_size = act.shape[0]
            total_samples += batch_size

            if sum_act is None:
                sum_act = act.sum(dim=0)
            else:
                sum_act += act.sum(dim=0)

        # Compute mean
        self.mean = (sum_act / total_samples).to(self._calibration_cache[0].device)

        # Compute E[||act - mean||^2]
        for act in self._calibration_cache:
            centered = act - self.mean
            sum_sq_norm += centered.pow(2).sum().item()

        mean_sq_norm = sum_sq_norm / total_samples
        self.scale = (self.d_model / mean_sq_norm) ** 0.5

        print(f"  Mean norm: {self.mean.norm().item():.4f}")
        print(f"  E[||x-mean||^2]: {mean_sq_norm:.4f}")
        print(f"  Scale factor: {self.scale:.6f}")
        print(f"  After standardization: E[||x'||^2] ≈ {self.d_model}")

    def __iter__(self):
        return self

    def __next__(self):
        # First yield cached calibration batches
        if self._calibration_cache:
            act = self._calibration_cache.pop(0)
        else:
            act = next(self.buffer)

        return self.scale * (act - self.mean)

    @property
    def config(self):
        """Expose buffer config for trainSAE's config saving."""
        try:
            cfg = self.buffer.config
        except AttributeError:
            cfg = {}
        cfg["standardization_mean_norm"] = self.mean.norm().item() if self.mean is not None else None
        cfg["standardization_scale"] = self.scale
        return cfg


# ---------------------------------------------------------------------------
# Data-sampled init for SpaDE only.
#
# We seed SpaDE's prototypes (`Aₑ` rows) with random standardized activations
# to fix the sparsemax-distance encoder's "no signal at step 0" pathology
# (randn rows have norm ≈ √d ≈ 22.6, well outside the standardized activation
# bulk at mean norm ≈ 14, so all squared distances are huge and sparsemax
# saturates). The 5 linear-encoder SAEs use their upstream Kaiming-style init
# for a fair cross-architecture comparison — they're well-tuned for that
# regime and data-init at full activation scale hurts their early dynamics.
# ---------------------------------------------------------------------------
def sample_init_activations(std_buffer, n_samples, d_model):
    """Pull n_samples standardized activations from std_buffer to seed dictionary atoms.

    Note: consumes batches via next(). The buffer is effectively infinite from
    NNsightActivationBuffer, so we are spending the first few yielded batches on init,
    NOT skipping training data.
    """
    collected = []
    total = 0
    while total < n_samples:
        batch = next(std_buffer)            # already standardized: scale*(act - mean)
        collected.append(batch.detach().cpu())
        total += batch.shape[0]
    pool = t.cat(collected, dim=0)
    assert pool.shape[1] == d_model, f"expected d_model={d_model}, got {pool.shape[1]}"
    perm = t.randperm(pool.shape[0])[:n_samples]
    return pool[perm].contiguous()          # [n_samples, d_model]


def run_training(
    build_configs: BuildConfigs,
    save_dir: str | Path,
    device: str = "cuda:0",
    seed: int = 42,
    smoke_test: bool = False,
    no_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = WANDB_PROJECT,
):
    """Stream OthelloGPT activations through trainSAE for whatever configs `build_configs` returns."""
    mp.set_start_method("spawn", force=True)  # the W&B loggers are subprocesses

    # ---- Model and data setup (matching Karvonen exactly) ----
    activation_dim = D_MODEL
    warmup_steps = DEFAULT_WARMUP_STEPS
    num_tokens = NUM_TOKENS

    # Global seed: covers init_data sampling (the SpaDE/CosineSpaDE weight init,
    # spade_base.py overwrites Ae/Ad with it) and buffer batch order — neither is
    # reachable by the per-trainer manual_seed, which fires later in trainSAE.
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

    if smoke_test:
        num_tokens = 100 * SAE_BATCH_SIZE  # 100 steps
        warmup_steps = 10  # reduce warmup for smoke test
        print(f"=== SMOKE TEST: {num_tokens // SAE_BATCH_SIZE} steps ===")

    steps = num_tokens // SAE_BATCH_SIZE
    log_steps = 5 if not smoke_test else 10

    print(f"Training SAEs for {steps} steps ({num_tokens:,} tokens)")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print(f"Seed: {seed}")

    # ---- Load model and create activation buffer ----
    print("Loading OthelloGPT model...")
    data = othello_hf_dataset_to_generator(
        DATASET, context_length=CTX_LEN, split="train", streaming=True
    )
    model = get_model(MODEL_NAME, device)
    submodule = get_submodule(MODEL_NAME, LAYER, model, SubmoduleType.resid_post)

    llm_batch_size = 64 if not smoke_test else 16
    n_ctxs = 1e3 if not smoke_test else 128
    n_calibration_batches = 100 if not smoke_test else 10

    activation_buffer = NNsightActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=n_ctxs,
        ctx_len=CTX_LEN,
        refresh_batch_size=llm_batch_size,
        out_batch_size=SAE_BATCH_SIZE,
        io="out",
        d_submodule=activation_dim,
        device=device,
    )

    # ---- Wrap with StandardizedBuffer (user's formula: center + scale) ----
    standardized_buffer = StandardizedBuffer(
        activation_buffer, d_model=activation_dim, n_calibration_batches=n_calibration_batches
    )
    standardized_buffer.calibrate()

    # ---- Sample standardized activations to seed every SAE's dictionary atoms ----
    spade_dict_size = EXPANSION * activation_dim
    init_data = sample_init_activations(standardized_buffer, spade_dict_size, activation_dim)
    print(
        f"Sampled init_data: shape={tuple(init_data.shape)}, "
        f"mean_norm={init_data.norm(dim=-1).mean().item():.4f}, "
        f"max_norm={init_data.norm(dim=-1).max().item():.4f} "
        f"(Jensen ceiling sqrt(d) ≈ {activation_dim**0.5:.4f})"
    )

    # ---- Build trainer configs ----
    trainer_configs = build_configs(TrainParams(
        steps=steps,
        activation_dim=activation_dim,
        model_name=MODEL_NAME,
        layer=LAYER,
        device=device,
        seed=seed,
        warmup_steps=warmup_steps,
    ))

    # ---- Inject init_data into SpaDE-family trainers ----
    # The 5 linear-encoder SAEs keep their upstream Kaiming-style init for a
    # fair cross-architecture comparison; the SpaDE variants get sampled
    # standardized activations as their dictionary atoms.
    for cfg in trainer_configs:
        if cfg["trainer"] is SpaDETrainer or cfg["trainer"] is CosineSpaDETrainer:
            cfg["init_data"] = init_data

    # ---- Device-parallel trainers: round-robin across visible GPUs ----
    # The buffer/LM stay on `device`; each trainer's update()/loss() moves
    # the shared batch to its own device, so updates overlap across GPUs.
    n_gpus = t.cuda.device_count() if device.startswith("cuda") else 0
    if n_gpus > 1:
        for i, cfg in enumerate(trainer_configs):
            cfg["device"] = f"cuda:{i % n_gpus}"
        print(f"Device-parallel: {len(trainer_configs)} trainers across {n_gpus} GPUs "
              f"(buffer/LM on {device})")

    # ---- Checkpointing: save at 25%, 50%, 75% of training ----
    save_steps = [steps // 4, steps // 2, 3 * steps // 4] if not smoke_test else None

    # ---- Run training ----
    print(f"\nStarting training with {len(trainer_configs)} trainers...")
    trainSAE(
        data=standardized_buffer,
        trainer_configs=trainer_configs,
        steps=steps,
        use_wandb=not no_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        save_steps=save_steps,
        save_dir=save_dir,
        log_steps=log_steps,
        normalize_activations=False,  # handled by StandardizedBuffer
        device=device,
    )

    print("\nTraining complete!")
    print(f"SAEs saved to: {save_dir}")

    # ---- Save standardization parameters for eval-time use ----
    norm_info = {
        "mean": standardized_buffer.mean.cpu().tolist(),
        "scale": standardized_buffer.scale,
        "d_model": activation_dim,
        "formula": "act_standardized = scale * (act - mean)",
    }
    norm_path = os.path.join(save_dir, "standardization_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_info, f, indent=2)
    print(f"Standardization parameters saved to: {norm_path}")


def arg_parser(description: str) -> argparse.ArgumentParser:
    """The flags every training entrypoint takes; sweep.py adds its grid flags on top."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save trained SAEs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight init, SpaDE init_data sampling, and data order")
    parser.add_argument("--smoke_test", action="store_true", help="Run 100 steps for quick validation")
    parser.add_argument(
        "--wandb_entity", type=str, default="", help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=WANDB_PROJECT, help="W&B project name"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser


def run_cli(config: Callable[[TrainParams], dict], description: str):
    """A per-type entrypoint's whole `main()`: parse the shared flags, train that one SAE."""
    args = arg_parser(description).parse_args()
    run_training(
        lambda p: [config(p)],
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        smoke_test=args.smoke_test,
        no_wandb=args.no_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
