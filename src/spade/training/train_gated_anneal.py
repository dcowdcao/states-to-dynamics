"""Train the GatedAnneal SAE.  Baseline from circuits.dictionary_learning.GatedAnnealTrainer.

Karvonen trainer38: L0=47.4, expansion=16x, initial_sparsity_penalty ≈ 0.955263
(sweep index 38 = expansion=16, sp_index=18 of linspace(0.15, 1.0, 20)).

Entrypoint: `python -m spade.training.train_gated_anneal --save_dir checkpoints/main --device cuda:0`
"""

from spade.training.core import P_ANNEAL, TrainParams, run_cli  # also puts circuits on sys.path

from circuits.dictionary_learning.dictionary import GatedAutoEncoder
from circuits.dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer


def config(p: TrainParams) -> dict:
    return {
        "trainer": GatedAnnealTrainer,
        "dict_class": GatedAutoEncoder,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "lr": 3e-4,
        "initial_sparsity_penalty": 0.955263,
        "sparsity_warmup_steps": p.sparsity_warmup_steps,
        **P_ANNEAL,
        "warmup_steps": p.warmup_steps,
        "resample_steps": None,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": "GatedAnnealTrainer-othello",
        "device": p.device,
    }


def main():
    run_cli(config, __doc__)


if __name__ == "__main__":
    main()
