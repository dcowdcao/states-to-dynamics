"""Train the TopK SAE (k=80).  Baseline from circuits.dictionary_learning.TopKTrainer.

Karvonen trainer_47: L0=80.0, expansion=16x, k=80
(sweep index 47 = expansion=16, k_index=7 of linspace(10, 400, 40)).
lr is omitted — the trainer auto-computes 2e-4 / sqrt(8192/16384) ≈ 2.83e-4.

Entrypoint: `python -m spade.training.train_topk --save_dir checkpoints/main --device cuda:0`
"""

from spade.training.core import TrainParams, run_cli  # also puts circuits on sys.path

from circuits.dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer


K = 80
# The canonical run keeps this name; spade.training.sweep's grid passes a swept
# one, so a swept point cannot be mistaken for the baseline row that
# the acceptance table, saes.registry and figures.paper_style match on.
WANDB_NAME = "TopKTrainer-othello"


def config(p: TrainParams, k: int = K,
           wandb_name: str = WANDB_NAME) -> dict:
    return {
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "k": k,
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
