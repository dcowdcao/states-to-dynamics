"""Train the PAnneal SAE.  Baseline from circuits.dictionary_learning.PAnnealTrainer.

Karvonen trainer33: L0=48.0, expansion=16x, initial_sparsity_penalty ≈ 0.042105
(sweep index 33 = expansion=16, sp_index=13 of linspace(0.025, 0.05, 20)).
Verified: 4597 alive features in Karvonen results → dict_size must be > 4096.

Entrypoint: `python -m spade.training.train_panneal --save_dir checkpoints/main --device cuda:0`
"""

from spade.training.core import P_ANNEAL, TrainParams, run_cli  # also puts circuits on sys.path

from circuits.dictionary_learning.dictionary import AutoEncoder
from circuits.dictionary_learning.trainers.p_anneal import PAnnealTrainer


def config(p: TrainParams) -> dict:
    return {
        "trainer": PAnnealTrainer,
        "dict_class": AutoEncoder,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "lr": 3e-4,
        "initial_sparsity_penalty": 0.042105,
        "sparsity_warmup_steps": p.sparsity_warmup_steps,
        **P_ANNEAL,
        "warmup_steps": p.warmup_steps,
        "resample_steps": None,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": "PAnnealTrainer-othello",
        "device": p.device,
    }


def main():
    run_cli(config, __doc__)


if __name__ == "__main__":
    main()
