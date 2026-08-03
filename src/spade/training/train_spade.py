"""Train the SpaDE (distance) SAE.  Novel — spade.saes.spade.SpaDETrainer.

lambda_init = 1e-3 ≈ 1/(2*512) (the theoretical default), gamma_reg = 1e-4 (the
distance-weighted L1 coefficient), expansion=16x to match all Karvonen SAEs for fair
comparison. Uses data-sampled prototype init, injected by core. Off-default λ/γ points
are the sweep's job (spade.training.sweep), which names them `SpaDE-lam..-gam..`; this
entrypoint only ever trains the canonical `SpaDETrainer-othello` config.

Entrypoint: `python -m spade.training.train_spade --save_dir checkpoints/spade_revival --device cuda:0`
"""

from spade.saes.spade import DEFAULT_LAMBDA_INIT, DEFAULT_SPADE_GAMMA, SpaDETrainer
from spade.training.core import TrainParams, run_cli


def config(p: TrainParams) -> dict:
    return {
        "trainer": SpaDETrainer,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "lambda_init": DEFAULT_LAMBDA_INIT,
        "gamma_reg": DEFAULT_SPADE_GAMMA,
        "lr": 3e-4,
        "warmup_steps": p.warmup_steps,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": "SpaDETrainer-othello",
        "device": p.device,
    }


def main():
    run_cli(config, __doc__)


if __name__ == "__main__":
    main()
