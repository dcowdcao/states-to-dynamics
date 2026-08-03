"""Train the CosineSpaDE SAE.  Novel — spade.saes.spade_cosine.CosineSpaDETrainer.

SpaDE with cosine similarity in place of negative squared distance. No lambda (cos is
bounded and scale-invariant); gamma is the only sparsity hyperparameter. gamma_reg = 1.0
default: the cosine misalignment term (1 - cos) is O(1) per atom vs. SpaDE's ||x - d||^2 ~
O(d_model), so the natural gamma scale is ~500x larger than distance-SpaDE's. Uses
data-sampled prototype init, injected by core. Off-default γ points are the sweep's job
(spade.training.sweep), which names them `CosineSpaDE-gam..`; this entrypoint only ever
trains the canonical `CosineSpaDETrainer-othello` config.

Entrypoint: `python -m spade.training.train_cosine_spade --save_dir checkpoints/cosine_spade --device cuda:0`
"""

from spade.saes.spade_cosine import DEFAULT_COSINE_GAMMA, CosineSpaDETrainer
from spade.training.core import TrainParams, run_cli


def config(p: TrainParams) -> dict:
    return {
        "trainer": CosineSpaDETrainer,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "gamma_reg": DEFAULT_COSINE_GAMMA,
        "lr": 3e-4,
        "warmup_steps": p.warmup_steps,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": "CosineSpaDETrainer-othello",
        "device": p.device,
    }


def main():
    run_cli(config, __doc__)


if __name__ == "__main__":
    main()
