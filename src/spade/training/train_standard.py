"""Train the Standard (ReLU) SAE.  Baseline from circuits.dictionary_learning.StandardTrainer.

Karvonen trainer30-v2: L0=131.7, expansion=16x, l1_penalty ≈ 0.058684
(index 10 of torch.linspace(0.035, 0.08, 20)).

Entrypoint: `python -m spade.training.train_standard --save_dir checkpoints/main --device cuda:0`
"""

from spade.training.core import TrainParams, run_cli  # also puts circuits on sys.path

from circuits.dictionary_learning.dictionary import AutoEncoder
from circuits.dictionary_learning.trainers.standard import StandardTrainer


L1_PENALTY = 0.058684
# The canonical run keeps this name; spade.training.sweep's grid passes a swept
# one, so a swept point cannot be mistaken for the baseline row that
# the acceptance table, saes.registry and figures.paper_style match on.
WANDB_NAME = "StandardTrainer-othello"


def config(p: TrainParams, l1_penalty: float = L1_PENALTY,
           wandb_name: str = WANDB_NAME) -> dict:
    return {
        "trainer": StandardTrainer,
        "dict_class": AutoEncoder,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "lr": 3e-4,
        "l1_penalty": l1_penalty,
        "sparsity_warmup_steps": p.sparsity_warmup_steps,
        "warmup_steps": p.warmup_steps,
        "resample_steps": None,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": wandb_name,
        "device": p.device,
    }


def main():
    run_cli(config, __doc__)


if __name__ == "__main__":
    main()
