"""Train the Gated SAE.  Baseline from circuits.dictionary_learning.GatedSAETrainer.

Karvonen trainer27-v2: L0=94.4, expansion=16x, l1_penalty ≈ 1.368421
(index 7 of torch.linspace(1.0, 2.0, 20)).

Entrypoint: `python -m spade.training.train_gated --save_dir checkpoints/main --device cuda:0`
"""

from spade.training.core import TrainParams, run_cli  # also puts circuits on sys.path

from circuits.dictionary_learning.dictionary import GatedAutoEncoder
from circuits.dictionary_learning.trainers.gdm import GatedSAETrainer


L1_PENALTY = 1.368421
# The canonical run keeps this name; spade.training.sweep's grid passes a swept
# one, so a swept point cannot be mistaken for the baseline row that
# results/acceptance, saes.registry and figures.paper_style match on.
WANDB_NAME = "GatedSAETrainer-othello"


def config(p: TrainParams, l1_penalty: float = L1_PENALTY,
           wandb_name: str = WANDB_NAME) -> dict:
    return {
        "trainer": GatedSAETrainer,
        "dict_class": GatedAutoEncoder,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "lr": 3e-4,
        "l1_penalty": l1_penalty,
        "sparsity_warmup_steps": p.sparsity_warmup_steps,
        "warmup_steps": p.warmup_steps,
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
